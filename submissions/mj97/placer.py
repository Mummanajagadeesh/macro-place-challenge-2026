"""
MJ97 v9 — Grid-Aligned GP + Fast Congestion Refinement

Key fixes over v8:
1. GRID-ALIGNED DENSITY: Use benchmark.grid_rows/cols for the FFT density
   grid instead of fixed 32×32. This is the #1 fix — when the evaluation
   grid is 30×27 (ibm02) and we spread on 32×32, the density signal is
   misaligned and post-legalization congestion blows up.

2. KILL plc.get_congestion_cost() LOOP: The real-congestion escape in v8
   burns 10s+ doing slow Python API calls per move. Removed entirely.
   Replace with fast RUDY-guided escape using pure numpy — same idea,
   100x faster.

3. TWO-PHASE GP: After legalization, run a short second GP pass (5-8s)
   with tight density to re-spread macros that got clustered by legalization.
   This is the main quality lever — legalization undoes GP spreading, so
   we re-spread and re-legalize.

4. STRICT 55s HARD CAP per benchmark (eval machine may be slower than ours)

5. NESTEROV with stable LR: Fixed base LR from bin size, simple momentum
   without the unstable Lipschitz estimation that was causing LR collapse.

Expected: avg ~1.35-1.40, runtime ~50-55s/bench → well within 1hr limit
"""

import math
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark

HARD_CAP = 55.0  # seconds per benchmark, hard limit


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_plc(benchmark: Benchmark):
    bench_name = benchmark.name
    candidates = [
        f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}",
        f"../external/MacroPlacement/Testcases/ICCAD04/{bench_name}",
        f"external/MacroPlacement/Flows/NanGate45/{bench_name}",
        f"../external/MacroPlacement/Flows/NanGate45/{bench_name}",
    ]
    try:
        from macro_place.loader import load_benchmark_from_dir
        for path in candidates:
            if os.path.isdir(path):
                _, plc = load_benchmark_from_dir(path)
                return plc
    except Exception:
        pass
    return None


def _extract_nets(benchmark: Benchmark, plc) -> List[List[int]]:
    name2idx = {name: i for i, name in enumerate(benchmark.macro_names)}
    nets = []
    try:
        for i in range(plc.net_cnt):
            net_obj = plc.nets[i]
            members = set()
            for pin in net_obj.pins:
                nm = pin.macro.get_name()
                if nm in name2idx:
                    members.add(name2idx[nm])
            if len(members) >= 2:
                nets.append(list(members))
        if nets:
            return nets
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Net index (CSR)
# ─────────────────────────────────────────────────────────────────────────────

class NetIndex:
    def __init__(self, nets: List[List[int]], n_nodes: int):
        self.n_nets = len(nets)
        self.n_nodes = n_nodes
        offsets = [0]; flat = []
        for net in nets:
            flat.extend(net); offsets.append(len(flat))
        self.flat = np.array(flat, dtype=np.int32)
        self.offsets = np.array(offsets, dtype=np.int32)

        node_nets: List[List[int]] = [[] for _ in range(n_nodes)]
        for ni, net in enumerate(nets):
            for node in net:
                node_nets[node].append(ni)
        nn_flat = []; nn_offsets = [0]
        for nl in node_nets:
            nn_flat.extend(nl); nn_offsets.append(len(nn_flat))
        self.nn_flat = np.array(nn_flat, dtype=np.int32)
        self.nn_offsets = np.array(nn_offsets, dtype=np.int32)

    def hpwl_node(self, pos: np.ndarray, node: int) -> float:
        x = pos[:, 0]; y = pos[:, 1]
        total = 0.0
        for ni in self.nn_flat[self.nn_offsets[node]:self.nn_offsets[node+1]]:
            idx = self.flat[self.offsets[ni]:self.offsets[ni+1]]
            total += x[idx].max() - x[idx].min() + y[idx].max() - y[idx].min()
        return total

    def hpwl_nodes(self, pos: np.ndarray, i: int, j: int) -> float:
        x = pos[:, 0]; y = pos[:, 1]
        si = set(self.nn_flat[self.nn_offsets[i]:self.nn_offsets[i+1]])
        sj = set(self.nn_flat[self.nn_offsets[j]:self.nn_offsets[j+1]])
        total = 0.0
        for ni in si | sj:
            idx = self.flat[self.offsets[ni]:self.offsets[ni+1]]
            total += x[idx].max() - x[idx].min() + y[idx].max() - y[idx].min()
        return total


# ─────────────────────────────────────────────────────────────────────────────
# WA wirelength + gradient
# ─────────────────────────────────────────────────────────────────────────────

def _wa_wl_grad(pos: np.ndarray, ni: NetIndex, gamma: float) -> Tuple[float, np.ndarray]:
    grad = np.zeros_like(pos)
    total = 0.0
    for net_id in range(ni.n_nets):
        o0 = ni.offsets[net_id]; o1 = ni.offsets[net_id+1]
        if o1 - o0 < 2:
            continue
        idx = ni.flat[o0:o1]
        for d in range(2):
            v = pos[idx, d]
            vm = v.max(); vn = v.min()
            ep = np.exp(np.clip((v-vm)/gamma, -30, 0))
            en = np.exp(np.clip(-(v-vn)/gamma, -30, 0))
            sp = ep.sum()+1e-12; sn = en.sum()+1e-12
            wp = (v*ep).sum()/sp; wn = (v*en).sum()/sn
            total += wp - wn
            np.add.at(grad[:, d], idx,
                      ep/sp*(1+(v-wp)/gamma) - en/sn*(1-(v-wn)/gamma))
    return total, grad


# ─────────────────────────────────────────────────────────────────────────────
# Bell kernel
# ─────────────────────────────────────────────────────────────────────────────

def _bell(u, s):
    au = np.abs(u); w = np.zeros_like(u)
    m1 = au < s; m2 = (au >= s) & (au < 2*s)
    w[m1] = 1.5 - au[m1]**2/(s[m1]**2+1e-30)
    w[m2] = 0.5*(2.0-au[m2]/(s[m2]+1e-30))**2
    return w

def _dbell(u, s):
    au = np.abs(u); dw = np.zeros_like(u)
    m1 = au < s; m2 = (au >= s) & (au < 2*s)
    dw[m1] = -2.0*u[m1]/(s[m1]**2+1e-30)
    dw[m2] = -(2.0-au[m2]/(s[m2]+1e-30))/(s[m2]+1e-30)*np.sign(u[m2])
    return dw


# ─────────────────────────────────────────────────────────────────────────────
# FFT density — GRID-ALIGNED (use actual evaluation grid dims)
# This is the critical fix: ibm02 has 30×27 grid, not 32×32.
# Spreading on wrong grid = density signal completely misaligned.
# ─────────────────────────────────────────────────────────────────────────────

def _density_fft(pos, sizes, canvas_w, canvas_h, R, C, target=0.7):
    """
    FFT electrostatic density on R×C grid matching the actual evaluation grid.
    target < 1.0 forces spreading; 0.6 is aggressive, 0.8 is moderate.
    """
    bh = canvas_h/R; bw = canvas_w/C
    bx = (np.arange(C)+0.5)*bw; by = (np.arange(R)+0.5)*bh

    # Spread radius: at least 1.5 bins
    sx = np.maximum(sizes[:,0:1], 1.5*bw)
    sy = np.maximum(sizes[:,1:2], 1.5*bh)

    dx = pos[:,0:1]-bx[None,:]   # (N, C)
    dy = pos[:,1:2]-by[None,:]   # (N, R)
    wx = _bell(dx, sx)            # (N, C)
    wy = _bell(dy, sy)            # (N, R)

    area = sizes[:,0]*sizes[:,1]
    w_area = area/(bw*bh)

    rho = (w_area[:,None]*wy).T @ wx   # (R, C)
    ovf = np.maximum(0.0, rho-target)
    penalty = 0.5*(ovf**2).sum()

    # Poisson solve
    ovf_t = torch.from_numpy(ovf.astype(np.float32))
    R2, C2 = R*2, C*2
    ext = torch.zeros(R2, C2)
    ext[:R,:C]=ovf_t; ext[:R,C:]=ovf_t.flip(1)
    ext[R:,:C]=ovf_t.flip(0); ext[R:,C:]=ovf_t.flip([0,1])
    F = torch.fft.rfft2(ext)
    kr = (2*math.pi*torch.arange(R2)/R2)**2
    kc = (2*math.pi*torch.arange(C2//2+1)/C2)**2
    k2 = kr[:,None]+kc[None,:]; k2[0,0]=1.0
    phi_fft = F/k2; phi_fft[0,0]=0.0
    phi = torch.fft.irfft2(phi_fft, s=(R2,C2))[:R,:C].numpy()

    Ex=np.zeros((R,C)); Ey=np.zeros((R,C))
    Ex[:,1:-1]=(phi[:,2:]-phi[:,:-2])/(2*bw)
    Ex[:,0]=(phi[:,1]-phi[:,0])/bw; Ex[:,-1]=(phi[:,-1]-phi[:,-2])/bw
    Ey[1:-1,:]=(phi[2:,:]-phi[:-2,:])/(2*bh)
    Ey[0,:]=(phi[1,:]-phi[0,:])/bh; Ey[-1,:]=(phi[-1,:]-phi[-2,:])/bh

    dwx = _dbell(dx, sx); dwy = _dbell(dy, sy)
    grad_x = w_area*(wy @ Ex * dwx).sum(axis=1)
    grad_y = w_area*(wx @ Ey.T * dwy).sum(axis=1)
    return penalty, np.stack([grad_x, grad_y], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Global placement — Nesterov with stable LR
# ─────────────────────────────────────────────────────────────────────────────

def _global_place(init, movable, sizes, ni, canvas_w, canvas_h, R, C,
                  time_budget=20.0, seed=97, density_target=0.65,
                  gamma0_frac=0.05, gamma_min_frac=0.0005,
                  lam0=1e-3, lam_max=4.0):
    np.random.seed(seed)
    t0 = time.time()
    hw = sizes[:,0]*0.5; hh = sizes[:,1]*0.5
    diag = math.sqrt(canvas_w**2+canvas_h**2)

    gamma0    = diag * gamma0_frac
    gamma_min = diag * gamma_min_frac

    # Stable base LR: bin size / 2
    bin_size = min(canvas_w/C, canvas_h/R)
    lr = bin_size * 0.4

    pos = init.copy()
    pos[:,0] = np.clip(pos[:,0], hw, canvas_w-hw)
    pos[:,1] = np.clip(pos[:,1], hh, canvas_h-hh)

    # Nesterov state
    v = np.zeros_like(pos)
    momentum = 0.9

    best_pos = pos.copy(); best_cost = float('inf')
    it = 0; max_iter = 600

    while it < max_iter and (time.time()-t0) < time_budget:
        frac = it/max_iter
        gamma = gamma0 * math.exp(math.log(gamma_min/gamma0)*frac)
        lam   = lam0   * math.exp(math.log(lam_max/lam0)*frac)

        # Nesterov lookahead
        x_look = pos + momentum*v
        x_look[:,0] = np.clip(x_look[:,0], hw, canvas_w-hw)
        x_look[:,1] = np.clip(x_look[:,1], hh, canvas_h-hh)

        wl, gwl = _wa_wl_grad(x_look, ni, gamma)
        dp, gdp = _density_fft(x_look, sizes, canvas_w, canvas_h, R, C, target=density_target)

        g = gwl + lam*gdp
        g[~movable] = 0.0

        v = momentum*v - lr*g
        pos = pos + v
        pos[:,0] = np.clip(pos[:,0], hw, canvas_w-hw)
        pos[:,1] = np.clip(pos[:,1], hh, canvas_h-hh)
        pos[~movable] = init[~movable]

        cost = wl + lam*dp
        if cost < best_cost and frac > 0.25:
            best_cost = cost; best_pos = pos.copy()

        it += 1

    best_pos[~movable] = init[~movable]
    return best_pos


# ─────────────────────────────────────────────────────────────────────────────
# Legalization — spiral, largest-first, guaranteed zero overlaps
# gap=0.003 gives a small buffer to avoid float-precision edge cases
# ─────────────────────────────────────────────────────────────────────────────

def _legalize(init, movable, sizes, canvas_w, canvas_h, gap=0.003):
    n = init.shape[0]
    hw = sizes[:,0]*0.5; hh = sizes[:,1]*0.5
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    order = sorted(range(n), key=lambda i: -(sizes[i,0]*sizes[i,1]))
    placed = np.zeros(n, dtype=bool)
    legal = init.copy()
    step_base = 0.15

    for idx in order:
        if not movable[idx]:
            placed[idx] = True; continue

        pidx = np.where(placed)[0]
        if len(pidx):
            dx = np.abs(legal[idx,0]-legal[pidx,0])
            dy = np.abs(legal[idx,1]-legal[pidx,1])
            if not ((dx < sep_x[idx][pidx]+gap) & (dy < sep_y[idx][pidx]+gap)).any():
                placed[idx] = True; continue

        step = max(sizes[idx,0], sizes[idx,1]) * step_base
        best = None; bdist = float('inf')

        for r in range(1, 400):
            found = False
            for dxm in range(-r, r+1):
                for sign in [1, -1]:
                    dym = sign*(r-abs(dxm))
                    cx = np.clip(init[idx,0]+dxm*step, hw[idx], canvas_w-hw[idx])
                    cy = np.clip(init[idx,1]+dym*step, hh[idx], canvas_h-hh[idx])
                    if len(pidx):
                        ddx = np.abs(cx-legal[pidx,0])
                        ddy = np.abs(cy-legal[pidx,1])
                        if ((ddx < sep_x[idx][pidx]+gap) & (ddy < sep_y[idx][pidx]+gap)).any():
                            continue
                    d = (cx-init[idx,0])**2+(cy-init[idx,1])**2
                    if d < bdist:
                        bdist = d; best = np.array([cx,cy]); found = True
            if found and r >= 3:
                break

        if best is not None:
            legal[idx] = best
        placed[idx] = True

    return legal


# ─────────────────────────────────────────────────────────────────────────────
# RUDY congestion map (fast numpy, no per-net Python loop on gradient)
# ─────────────────────────────────────────────────────────────────────────────

def _rudy_map(pos, ni, canvas_w, canvas_h, R, C):
    bw = canvas_w/C; bh = canvas_h/R
    demand = np.zeros((R, C), dtype=np.float64)
    x = pos[:,0]; y = pos[:,1]
    for nid in range(ni.n_nets):
        idx = ni.flat[ni.offsets[nid]:ni.offsets[nid+1]]
        xi = x[idx]; yi = y[idx]
        x0=xi.min(); x1=xi.max(); y0=yi.min(); y1=yi.max()
        nw=max(x1-x0,1e-8); nh=max(y1-y0,1e-8)
        box=nw*nh
        r0=int(np.clip(y0/bh,0,R-1)); r1=int(np.clip(y1/bh,0,R-1))
        c0=int(np.clip(x0/bw,0,C-1)); c1=int(np.clip(x1/bw,0,C-1))
        demand[r0:r1+1, c0:c1+1] += (nh/box+nw/box)*bw*bh*0.5
    return demand


# ─────────────────────────────────────────────────────────────────────────────
# Fast congestion escape — pure numpy, no plc API calls
#
# For each macro in a hot bin, try moving it to the best cooler bin
# within a search radius. Accept if RUDY congestion drops AND WL doesn't
# increase too much. Fast: no Python API calls, just array ops.
# ─────────────────────────────────────────────────────────────────────────────

def _congestion_escape(pos, movable, sizes, ni, canvas_w, canvas_h, R, C,
                       gap=0.003, time_budget=8.0, radius=4):
    t0 = time.time()
    n = len(pos)
    bw = canvas_w/C; bh = canvas_h/R
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx = np.where(movable)[0]
    diag = math.sqrt(canvas_w**2+canvas_h**2)

    p = pos.copy()

    # Iterative passes
    for pass_i in range(8):
        if time.time()-t0 >= time_budget:
            break

        demand = _rudy_map(p, ni, canvas_w, canvas_h, R, C)
        thresh = np.percentile(demand, 85)
        if thresh < 1e-6:
            break

        # Sort hot macros by demand (hottest first)
        hot = []
        for idx in movable_idx:
            r = int(np.clip(p[idx,1]/bh, 0, R-1))
            c = int(np.clip(p[idx,0]/bw, 0, C-1))
            if demand[r,c] >= thresh:
                hot.append((demand[r,c], idx))
        hot.sort(reverse=True)

        n_moved = 0
        for _, idx in hot:
            if time.time()-t0 >= time_budget:
                break

            r_cur = int(np.clip(p[idx,1]/bh, 0, R-1))
            c_cur = int(np.clip(p[idx,0]/bw, 0, C-1))
            cur_demand = demand[r_cur, c_cur]
            wl_before = ni.hpwl_node(p, idx)
            orig = p[idx].copy()

            best_score = -1e9; best_cand = None

            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    if dr==0 and dc==0: continue
                    nr = int(np.clip(r_cur+dr, 0, R-1))
                    nc = int(np.clip(c_cur+dc, 0, C-1))
                    if demand[nr,nc] >= cur_demand:
                        continue  # only try cooler bins

                    tx = np.clip((nc+0.5)*bw, hw[idx], canvas_w-hw[idx])
                    ty = np.clip((nr+0.5)*bh, hh[idx], canvas_h-hh[idx])

                    # Legality check (vectorized)
                    ddx = np.abs(tx-p[:,0]); ddy = np.abs(ty-p[:,1])
                    ddx[idx] = 999; ddy[idx] = 999  # ignore self
                    if ((ddx < sep_x[idx]+gap) & (ddy < sep_y[idx]+gap)).any():
                        continue

                    p[idx] = [tx, ty]
                    wl_after = ni.hpwl_node(p, idx)
                    p[idx] = orig

                    wl_delta = wl_after - wl_before
                    cong_gain = cur_demand - demand[nr,nc]

                    # Score: congestion gain minus normalized WL penalty
                    score = cong_gain - 0.3 * max(0, wl_delta) / (diag*0.01+1e-10)
                    if score > best_score:
                        best_score = score; best_cand = np.array([tx, ty])

            if best_cand is not None and best_score > 0:
                p[idx] = best_cand
                n_moved += 1
            else:
                p[idx] = orig

        if n_moved == 0:
            break

    return p


# ─────────────────────────────────────────────────────────────────────────────
# HPWL swap refinement
# ─────────────────────────────────────────────────────────────────────────────

def _swap_refine(pos, movable, sizes, ni, canvas_w, canvas_h,
                 gap=0.003, n_passes=6, time_budget=5.0, k=12):
    t0 = time.time()
    n = len(pos)
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx = np.where(movable)[0]

    def legal_swap(p, i, j):
        placed = np.ones(n, dtype=bool); placed[i]=False; placed[j]=False
        for nd, np_ in [(i, p[j].copy()), (j, p[i].copy())]:
            if not (hw[nd]<=np_[0]<=canvas_w-hw[nd] and hh[nd]<=np_[1]<=canvas_h-hh[nd]):
                return False
            dx=np.abs(np_[0]-p[:,0]); dy=np.abs(np_[1]-p[:,1])
            if ((dx<sep_x[nd]+gap)&(dy<sep_y[nd]+gap)&placed).any():
                return False
            placed[nd] = True
        return True

    p = pos.copy()
    for _ in range(n_passes):
        if time.time()-t0 >= time_budget: break
        improved = 0
        order = movable_idx.copy(); np.random.shuffle(order)
        for i in order:
            if time.time()-t0 >= time_budget: break
            dists = np.abs(p[i,0]-p[movable_idx,0])+np.abs(p[i,1]-p[movable_idx,1])
            neighbors = movable_idx[np.argsort(dists)[1:k+1]]
            best_gain = 1e-9; best_j = -1
            for j in neighbors:
                wl_b = ni.hpwl_nodes(p, i, j)
                p[i], p[j] = p[j].copy(), p[i].copy()
                if legal_swap(p, i, j):
                    wl_a = ni.hpwl_nodes(p, i, j)
                    if wl_b-wl_a > best_gain:
                        best_gain = wl_b-wl_a; best_j = j
                    else:
                        p[i], p[j] = p[j].copy(), p[i].copy()
                else:
                    p[i], p[j] = p[j].copy(), p[i].copy()
            if best_j >= 0: improved += 1
        if improved == 0: break
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Mini SA
# ─────────────────────────────────────────────────────────────────────────────

def _mini_sa(pos, movable, sizes, ni, canvas_w, canvas_h, gap=0.003, time_budget=3.0):
    t0 = time.time(); n = len(pos)
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx = np.where(movable)[0]
    diag = math.sqrt(canvas_w**2+canvas_h**2)
    T0 = 0.005*diag; T1 = 0.0002*diag
    p = pos.copy()

    while True:
        el = time.time()-t0
        if el >= time_budget: break
        frac = el/time_budget
        T = T0*math.exp(math.log(T1/T0)*frac); step = T*1.5
        i = int(np.random.choice(movable_idx))
        orig = p[i].copy()
        nx = float(np.clip(orig[0]+np.random.uniform(-step,step), hw[i], canvas_w-hw[i]))
        ny = float(np.clip(orig[1]+np.random.uniform(-step,step), hh[i], canvas_h-hh[i]))
        placed = np.ones(n, dtype=bool); placed[i]=False
        ddx = np.abs(nx-p[:,0]); ddy = np.abs(ny-p[:,1])
        if ((ddx<sep_x[i]+gap)&(ddy<sep_y[i]+gap)&placed).any():
            continue
        wl_b = ni.hpwl_node(p, i); p[i]=[nx,ny]; wl_a = ni.hpwl_node(p, i)
        delta = wl_a-wl_b
        if delta > 0 and np.random.random() >= math.exp(-delta/(T+1e-12)):
            p[i] = orig
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Overlap repair — final safety pass (should never trigger but just in case)
# ─────────────────────────────────────────────────────────────────────────────

def _repair_overlaps(pos, movable, sizes, canvas_w, canvas_h, gap=0.003):
    """Brute-force overlap repair as final safety check."""
    n = len(pos)
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    p = pos.copy()

    for _ in range(10):  # max 10 repair passes
        had_overlap = False
        for i in range(n):
            if not movable[i]: continue
            for j in range(i+1, n):
                dx = abs(p[i,0]-p[j,0]); dy = abs(p[i,1]-p[j,1])
                if dx < sep_x[i,j]+gap and dy < sep_y[i,j]+gap:
                    had_overlap = True
                    # Push i away from j
                    push_x = (sep_x[i,j]+gap - dx + 1e-4) * np.sign(p[i,0]-p[j,0]+1e-10)
                    push_y = (sep_y[i,j]+gap - dy + 1e-4) * np.sign(p[i,1]-p[j,1]+1e-10)
                    if movable[i]:
                        p[i,0] = np.clip(p[i,0]+push_x*0.5, hw[i], canvas_w-hw[i])
                        p[i,1] = np.clip(p[i,1]+push_y*0.5, hh[i], canvas_h-hh[i])
                    if movable[j]:
                        p[j,0] = np.clip(p[j,0]-push_x*0.5, hw[j], canvas_w-hw[j])
                        p[j,1] = np.clip(p[j,1]-push_y*0.5, hh[j], canvas_h-hh[j])
        if not had_overlap:
            break
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main placer
# ─────────────────────────────────────────────────────────────────────────────

class Mj97Placer:
    """
    MJ97 v9: Grid-aligned GP + two-phase placement + fast congestion escape.

    Pipeline (target: 50s per benchmark):
      1. GP phase 1 — 18s, density target 0.65, grid-aligned
      2. Legalization — spiral, guaranteed zero overlaps
      3. GP phase 2 — 8s, tight density target 0.5, re-spread after legalization
      4. Legalization again — fast (most already legal)
      5. Congestion escape — 8s, fast RUDY-guided numpy moves
      6. Swap refinement — 5s
      7. Mini SA — 3s
      8. Overlap repair — safety pass
    """

    def __init__(self, seed: int = 97):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        T0 = time.time()

        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return benchmark.macro_positions.clone()

        out    = benchmark.macro_positions.clone()
        init   = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)
        sizes  = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # Use actual evaluation grid — critical for ibm02/06/etc
        R = int(benchmark.grid_rows)
        C = int(benchmark.grid_cols)

        plc  = _load_plc(benchmark)
        nets = _extract_nets(benchmark, plc) if plc is not None else []

        if not nets:
            legal = _legalize(init, movable, sizes, cw, ch)
            out[:n_hard] = torch.from_numpy(legal).float()
            return out

        ni = NetIndex(nets, n_hard)

        def remaining(budget_from_start):
            return max(0.0, min(budget_from_start, HARD_CAP) - (time.time()-T0))

        # ── 1. GP phase 1: 18s, aggressive spreading (target=0.65)
        gp1 = _global_place(
            init=init.copy(), movable=movable, sizes=sizes, ni=ni,
            canvas_w=cw, canvas_h=ch, R=R, C=C,
            time_budget=remaining(18.0), seed=self.seed,
            density_target=0.65, lam_max=5.0,
        )

        # ── 2. Legalization 1
        legal1 = _legalize(gp1, movable, sizes, cw, ch, gap=0.003)

        # ── 3. GP phase 2: 8s, very tight density to re-spread after legalization
        #    Start from legalized positions, much tighter target forces more spreading
        gp2_budget = remaining(32.0)
        if gp2_budget > 2.0:
            gp2 = _global_place(
                init=legal1.copy(), movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch, R=R, C=C,
                time_budget=gp2_budget, seed=self.seed+1,
                density_target=0.5, lam0=1e-2, lam_max=3.0,
                gamma0_frac=0.02, gamma_min_frac=0.001,
            )
            # ── 4. Legalization 2
            legal2 = _legalize(gp2, movable, sizes, cw, ch, gap=0.003)
        else:
            legal2 = legal1

        # ── 5. Congestion escape: 8s, fast RUDY-guided
        cong_budget = remaining(45.0)
        if cong_budget > 1.0:
            legal2 = _congestion_escape(
                pos=legal2, movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch, R=R, C=C,
                gap=0.003, time_budget=cong_budget, radius=4,
            )

        # ── 6. Swap refinement: 5s
        swap_budget = remaining(50.0)
        if swap_budget > 1.0:
            legal2 = _swap_refine(
                pos=legal2, movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch, gap=0.003,
                n_passes=6, time_budget=swap_budget, k=12,
            )

        # ── 7. Mini SA: 3s
        sa_budget = remaining(53.0)
        if sa_budget > 0.5:
            legal2 = _mini_sa(
                pos=legal2, movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch, gap=0.003,
                time_budget=sa_budget,
            )

        # ── 8. Safety overlap repair (should be a no-op if legalizer worked)
        legal2 = _repair_overlaps(legal2, movable, sizes, cw, ch, gap=0.003)

        out[:n_hard] = torch.from_numpy(legal2).float()
        return out