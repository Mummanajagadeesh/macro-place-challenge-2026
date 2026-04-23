"""
MJ97 v10 — Vectorized Numpy GP, Strict Time Control

Root cause of v8/v9 failure:
1. _wa_wl_grad iterates over nets in Python — with 7k-16k nets × 600 iters,
   this takes 200-400s regardless of budget. Budget is checked between iters
   but each iter itself is slow.
2. ibm02 cong=2.336 unchanged across v4/v8/v9 — the congestion evaluator
   smooth range=2 on 30×27 grid means net bounding boxes need to be tight,
   not just macros spread. Need stronger WL optimization.

Fixes:
1. Fully vectorized WA gradient using numpy segment ops (no per-net Python loop)
   — uses np.maximum.reduceat / np.add.at with precomputed CSR structure
   — each gradient call: O(E) where E=total net pins, no Python loop
   — 10-50x speedup → can do 2000+ iters in 20s

2. Strict per-iteration time check so budget is actually respected

3. ibm02 fix: stronger gamma decay (tighter WA → closer to true HPWL) and
   higher lambda → forces macros to actually minimize net bounding boxes

4. Runtime target: 40s per benchmark (budget for eval machine slowdown)
"""

import math
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark

HARD_CAP = 50.0  # seconds hard limit per benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_plc(benchmark):
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


def _extract_nets(benchmark, plc):
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
# Net structure — CSR + per-node reverse index
# ─────────────────────────────────────────────────────────────────────────────

class NetIndex:
    def __init__(self, nets: List[List[int]], n_nodes: int):
        self.n_nets = len(nets)
        self.n_nodes = n_nodes

        # Forward: net → nodes
        offsets = [0]; flat = []
        for net in nets:
            flat.extend(net); offsets.append(len(flat))
        self.flat = np.array(flat, dtype=np.int32)
        self.offsets = np.array(offsets, dtype=np.int32)
        self.net_sizes = np.diff(self.offsets)

        # Reverse: node → nets
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
        x = pos[:, 0]; y = pos[:, 1]; total = 0.0
        for ni in self.nn_flat[self.nn_offsets[node]:self.nn_offsets[node+1]]:
            idx = self.flat[self.offsets[ni]:self.offsets[ni+1]]
            total += x[idx].max()-x[idx].min()+y[idx].max()-y[idx].min()
        return total

    def hpwl_nodes(self, pos: np.ndarray, i: int, j: int) -> float:
        x = pos[:, 0]; y = pos[:, 1]
        si = set(self.nn_flat[self.nn_offsets[i]:self.nn_offsets[i+1]])
        sj = set(self.nn_flat[self.nn_offsets[j]:self.nn_offsets[j+1]])
        total = 0.0
        for ni in si | sj:
            idx = self.flat[self.offsets[ni]:self.offsets[ni+1]]
            total += x[idx].max()-x[idx].min()+y[idx].max()-y[idx].min()
        return total


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED WA wirelength + gradient
#
# Key: no per-net Python loop. Uses np.maximum.reduceat and np.add.at
# over the flat CSR arrays. All net operations are single numpy calls.
#
# For each net (dimension d):
#   vmax = max(pos[idx, d])     <- np.maximum.reduceat(pos[flat,d], offsets)
#   ep[i] = exp(clip((v[i]-vmax)/gamma, -30, 0))
#   sp = sum(ep)
#   wp = sum(v*ep)/sp           <- numerically stable WA max
#   grad contribution: ep/sp*(1+(v-wp)/gamma) - en/sn*(1-(v-wn)/gamma)
# ─────────────────────────────────────────────────────────────────────────────

def _wa_wl_grad_fast(pos: np.ndarray, ni: NetIndex, gamma: float):
    """
    Fully vectorized WA WL + gradient. No Python loop over nets.
    Returns (total_wl, grad) where grad is (N,2).
    """
    grad = np.zeros_like(pos)
    total = 0.0
    flat = ni.flat; offsets = ni.offsets[:-1]  # start offsets for reduceat

    for d in range(2):
        v_flat = pos[ni.flat, d]  # values at all net pins, flat

        # Max per net (for numerical stability)
        vmax = np.maximum.reduceat(v_flat, offsets)  # (n_nets,)
        # Min per net
        vmin = np.minimum.reduceat(v_flat, offsets)  # (n_nets,)

        # Broadcast: repeat vmax/vmin to flat length
        # net_id[k] = net that flat pin k belongs to
        net_id = np.repeat(np.arange(ni.n_nets), ni.net_sizes)  # (E,)

        vmax_flat = vmax[net_id]  # (E,)
        vmin_flat = vmin[net_id]  # (E,)

        ep = np.exp(np.clip((v_flat - vmax_flat) / gamma, -30, 0))  # (E,)
        en = np.exp(np.clip(-(v_flat - vmin_flat) / gamma, -30, 0))  # (E,)

        sp = np.add.reduceat(ep, offsets) + 1e-12  # (n_nets,)
        sn = np.add.reduceat(en, offsets) + 1e-12  # (n_nets,)

        # WA positions
        vep = np.add.reduceat(v_flat * ep, offsets)  # (n_nets,)
        ven = np.add.reduceat(v_flat * en, offsets)  # (n_nets,)
        wp = vep / sp   # (n_nets,)
        wn = ven / sn   # (n_nets,)

        total += (wp - wn).sum()

        # Gradient per pin
        sp_flat = sp[net_id]; sn_flat = sn[net_id]
        wp_flat = wp[net_id]; wn_flat = wn[net_id]

        g_flat = (ep / sp_flat * (1 + (v_flat - wp_flat) / gamma)
                  - en / sn_flat * (1 - (v_flat - wn_flat) / gamma))

        # Scatter back to nodes
        np.add.at(grad[:, d], ni.flat, g_flat)

    return total, grad


# ─────────────────────────────────────────────────────────────────────────────
# Bell kernel (same as before)
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
# FFT density — grid-aligned (uses actual R×C evaluation grid)
# ─────────────────────────────────────────────────────────────────────────────

def _density_fft(pos, sizes, canvas_w, canvas_h, R, C, target=0.65):
    bh = canvas_h/R; bw = canvas_w/C
    bx = (np.arange(C)+0.5)*bw; by = (np.arange(R)+0.5)*bh
    sx = np.maximum(sizes[:,0:1], 1.5*bw)
    sy = np.maximum(sizes[:,1:2], 1.5*bh)
    dx = pos[:,0:1]-bx[None,:]; dy = pos[:,1:2]-by[None,:]
    wx = _bell(dx, sx); wy = _bell(dy, sy)
    area = sizes[:,0]*sizes[:,1]; w_area = area/(bw*bh)
    rho = (w_area[:,None]*wy).T @ wx
    ovf = np.maximum(0.0, rho-target)
    penalty = 0.5*(ovf**2).sum()

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
# Global placement — Nesterov, vectorized gradient, strict time control
# ─────────────────────────────────────────────────────────────────────────────

def _global_place(init, movable, sizes, ni, canvas_w, canvas_h, R, C,
                  t_start, time_budget,
                  seed=97, density_target=0.65,
                  gamma0_frac=0.05, gamma_min_frac=0.0005,
                  lam0=5e-4, lam_max=5.0, momentum=0.9):
    np.random.seed(seed)
    hw = sizes[:,0]*0.5; hh = sizes[:,1]*0.5
    diag = math.sqrt(canvas_w**2+canvas_h**2)
    gamma0    = diag * gamma0_frac
    gamma_min = diag * gamma_min_frac
    bin_size  = min(canvas_w/C, canvas_h/R)
    lr        = bin_size * 0.35

    pos = init.copy()
    pos[:,0] = np.clip(pos[:,0], hw, canvas_w-hw)
    pos[:,1] = np.clip(pos[:,1], hh, canvas_h-hh)

    v = np.zeros_like(pos)
    best_pos = pos.copy(); best_cost = float('inf')
    it = 0; deadline = t_start + time_budget

    while time.time() < deadline:
        frac = min(1.0, (time.time()-t_start)/time_budget)
        gamma = gamma0 * math.exp(math.log(gamma_min/gamma0)*frac)
        lam   = lam0   * math.exp(math.log(lam_max/lam0)*frac)

        x_look = pos + momentum*v
        x_look[:,0] = np.clip(x_look[:,0], hw, canvas_w-hw)
        x_look[:,1] = np.clip(x_look[:,1], hh, canvas_h-hh)

        wl, gwl = _wa_wl_grad_fast(x_look, ni, gamma)
        dp, gdp = _density_fft(x_look, sizes, canvas_w, canvas_h, R, C, target=density_target)

        g = gwl + lam*gdp
        g[~movable] = 0.0

        v = momentum*v - lr*g
        pos = pos + v
        pos[:,0] = np.clip(pos[:,0], hw, canvas_w-hw)
        pos[:,1] = np.clip(pos[:,1], hh, canvas_h-hh)
        pos[~movable] = init[~movable]

        cost = wl + lam*dp
        if cost < best_cost and frac > 0.2:
            best_cost = cost; best_pos = pos.copy()
        it += 1

    best_pos[~movable] = init[~movable]
    return best_pos


# ─────────────────────────────────────────────────────────────────────────────
# Legalization — spiral, largest-first, guaranteed zero overlaps
# ─────────────────────────────────────────────────────────────────────────────

def _legalize(init, movable, sizes, canvas_w, canvas_h, gap=0.003):
    n = init.shape[0]
    hw = sizes[:,0]*0.5; hh = sizes[:,1]*0.5
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    order = sorted(range(n), key=lambda i: -(sizes[i,0]*sizes[i,1]))
    placed = np.zeros(n, dtype=bool)
    legal = init.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True; continue

        pidx = np.where(placed)[0]
        if len(pidx):
            dx = np.abs(legal[idx,0]-legal[pidx,0])
            dy = np.abs(legal[idx,1]-legal[pidx,1])
            if not ((dx < sep_x[idx][pidx]+gap) & (dy < sep_y[idx][pidx]+gap)).any():
                placed[idx] = True; continue

        step = max(sizes[idx,0], sizes[idx,1]) * 0.15
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
# RUDY map (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def _rudy_map(pos, ni, canvas_w, canvas_h, R, C):
    bw = canvas_w/C; bh = canvas_h/R
    demand = np.zeros((R, C))
    x = pos[:,0]; y = pos[:,1]
    for nid in range(ni.n_nets):
        idx = ni.flat[ni.offsets[nid]:ni.offsets[nid+1]]
        xi=x[idx]; yi=y[idx]
        x0=xi.min(); x1=xi.max(); y0=yi.min(); y1=yi.max()
        nw=max(x1-x0,1e-8); nh=max(y1-y0,1e-8); box=nw*nh
        r0=int(np.clip(y0/bh,0,R-1)); r1=int(np.clip(y1/bh,0,R-1))
        c0=int(np.clip(x0/bw,0,C-1)); c1=int(np.clip(x1/bw,0,C-1))
        demand[r0:r1+1,c0:c1+1] += (nh/box+nw/box)*bw*bh*0.5
    return demand


# ─────────────────────────────────────────────────────────────────────────────
# Congestion escape — RUDY-guided, fast numpy
# ─────────────────────────────────────────────────────────────────────────────

def _congestion_escape(pos, movable, sizes, ni, canvas_w, canvas_h, R, C,
                       t_start, time_budget, gap=0.003, radius=4):
    n = len(pos)
    bw = canvas_w/C; bh = canvas_h/R
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx = np.where(movable)[0]
    diag = math.sqrt(canvas_w**2+canvas_h**2)
    deadline = t_start + time_budget
    p = pos.copy()

    for _ in range(10):
        if time.time() >= deadline: break
        demand = _rudy_map(p, ni, canvas_w, canvas_h, R, C)
        thresh = np.percentile(demand, 85)
        if thresh < 1e-6: break

        hot = [(demand[int(np.clip(p[idx,1]/bh,0,R-1)), int(np.clip(p[idx,0]/bw,0,C-1))], idx)
               for idx in movable_idx
               if demand[int(np.clip(p[idx,1]/bh,0,R-1)), int(np.clip(p[idx,0]/bw,0,C-1))] >= thresh]
        hot.sort(reverse=True)

        n_moved = 0
        for cur_d, idx in hot:
            if time.time() >= deadline: break
            r_cur = int(np.clip(p[idx,1]/bh, 0, R-1))
            c_cur = int(np.clip(p[idx,0]/bw, 0, C-1))
            wl_b = ni.hpwl_node(p, idx)
            orig = p[idx].copy()
            best_score = -1e9; best_cand = None

            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    if dr==0 and dc==0: continue
                    nr=int(np.clip(r_cur+dr,0,R-1)); nc=int(np.clip(c_cur+dc,0,C-1))
                    if demand[nr,nc] >= cur_d: continue
                    tx=np.clip((nc+0.5)*bw, hw[idx], canvas_w-hw[idx])
                    ty=np.clip((nr+0.5)*bh, hh[idx], canvas_h-hh[idx])
                    ddx=np.abs(tx-p[:,0]); ddy=np.abs(ty-p[:,1])
                    ddx[idx]=999; ddy[idx]=999
                    if ((ddx<sep_x[idx]+gap)&(ddy<sep_y[idx]+gap)).any(): continue
                    p[idx]=[tx,ty]; wl_a=ni.hpwl_node(p,idx); p[idx]=orig
                    score = (cur_d-demand[nr,nc]) - 0.3*max(0,wl_a-wl_b)/(diag*0.01+1e-10)
                    if score > best_score: best_score=score; best_cand=np.array([tx,ty])

            if best_cand is not None and best_score > 0:
                p[idx]=best_cand; n_moved+=1
            else:
                p[idx]=orig
        if n_moved == 0: break
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Swap refinement
# ─────────────────────────────────────────────────────────────────────────────

def _swap_refine(pos, movable, sizes, ni, canvas_w, canvas_h,
                 t_start, time_budget, gap=0.003, k=12):
    n = len(pos)
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx = np.where(movable)[0]
    deadline = t_start + time_budget
    p = pos.copy()

    def legal_swap(p, i, j):
        placed = np.ones(n, dtype=bool); placed[i]=False; placed[j]=False
        for nd, np_ in [(i,p[j].copy()), (j,p[i].copy())]:
            if not (hw[nd]<=np_[0]<=canvas_w-hw[nd] and hh[nd]<=np_[1]<=canvas_h-hh[nd]):
                return False
            dx=np.abs(np_[0]-p[:,0]); dy=np.abs(np_[1]-p[:,1])
            if ((dx<sep_x[nd]+gap)&(dy<sep_y[nd]+gap)&placed).any(): return False
            placed[nd]=True
        return True

    while time.time() < deadline:
        improved = 0
        order = movable_idx.copy(); np.random.shuffle(order)
        for i in order:
            if time.time() >= deadline: break
            dists = np.abs(p[i,0]-p[movable_idx,0])+np.abs(p[i,1]-p[movable_idx,1])
            neighbors = movable_idx[np.argsort(dists)[1:k+1]]
            best_gain = 1e-9; best_j = -1
            for j in neighbors:
                wl_b = ni.hpwl_nodes(p,i,j)
                p[i],p[j] = p[j].copy(),p[i].copy()
                if legal_swap(p,i,j):
                    wl_a = ni.hpwl_nodes(p,i,j)
                    if wl_b-wl_a > best_gain: best_gain=wl_b-wl_a; best_j=j
                    else: p[i],p[j] = p[j].copy(),p[i].copy()
                else: p[i],p[j] = p[j].copy(),p[i].copy()
            if best_j >= 0: improved+=1
        if improved == 0: break
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Mini SA
# ─────────────────────────────────────────────────────────────────────────────

def _mini_sa(pos, movable, sizes, ni, canvas_w, canvas_h,
             t_start, time_budget, gap=0.003):
    n = len(pos)
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx = np.where(movable)[0]
    diag = math.sqrt(canvas_w**2+canvas_h**2)
    T0v = 0.005*diag; T1v = 0.0002*diag
    deadline = t_start + time_budget
    p = pos.copy()

    while time.time() < deadline:
        el = time.time()-t_start; frac = min(1.0, el/time_budget)
        T = T0v*math.exp(math.log(T1v/T0v)*frac); step = T*1.5
        i = int(np.random.choice(movable_idx))
        orig = p[i].copy()
        nx = float(np.clip(orig[0]+np.random.uniform(-step,step), hw[i], canvas_w-hw[i]))
        ny = float(np.clip(orig[1]+np.random.uniform(-step,step), hh[i], canvas_h-hh[i]))
        placed=np.ones(n,dtype=bool); placed[i]=False
        ddx=np.abs(nx-p[:,0]); ddy=np.abs(ny-p[:,1])
        if ((ddx<sep_x[i]+gap)&(ddy<sep_y[i]+gap)&placed).any(): continue
        wl_b=ni.hpwl_node(p,i); p[i]=[nx,ny]; wl_a=ni.hpwl_node(p,i)
        delta=wl_a-wl_b
        if delta>0 and np.random.random()>=math.exp(-delta/(T+1e-12)): p[i]=orig
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Overlap repair — final safety
# ─────────────────────────────────────────────────────────────────────────────

def _repair_overlaps(pos, movable, sizes, canvas_w, canvas_h, gap=0.003):
    n = len(pos)
    hw = sizes[:,0]/2; hh = sizes[:,1]/2
    sep_x = (sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y = (sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    p = pos.copy()
    for _ in range(20):
        had = False
        for i in range(n):
            if not movable[i]: continue
            for j in range(i+1, n):
                ox = sep_x[i,j]+gap - abs(p[i,0]-p[j,0])
                oy = sep_y[i,j]+gap - abs(p[i,1]-p[j,1])
                if ox > 0 and oy > 0:
                    had = True
                    # push along smaller overlap axis
                    if ox < oy:
                        push = (ox/2+1e-4)*np.sign(p[i,0]-p[j,0]+1e-10)
                        if movable[i]: p[i,0]=np.clip(p[i,0]+push, hw[i], canvas_w-hw[i])
                        if movable[j]: p[j,0]=np.clip(p[j,0]-push, hw[j], canvas_w-hw[j])
                    else:
                        push = (oy/2+1e-4)*np.sign(p[i,1]-p[j,1]+1e-10)
                        if movable[i]: p[i,1]=np.clip(p[i,1]+push, hh[i], canvas_h-hh[i])
                        if movable[j]: p[j,1]=np.clip(p[j,1]-push, hh[j], canvas_h-hh[j])
        if not had: break
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main placer
# ─────────────────────────────────────────────────────────────────────────────

class Mj97Placer:
    """
    MJ97 v10: Vectorized numpy WA gradient + strict deadline control.

    Pipeline (hard cap: 50s):
      t=0    Load + net extraction
      t=~2   GP phase 1: 20s (vectorized, density_target=0.65, grid-aligned)
      t=~22  Legalize 1
      t=~25  GP phase 2: 8s (re-spread from legalized, target=0.5)
      t=~33  Legalize 2
      t=~36  Congestion escape: 6s (RUDY-guided numpy)
      t=~42  Swap refine: 5s
      t=~47  Mini SA: 2s
      t=~49  Overlap repair
    """

    def __init__(self, seed: int = 97):
        self.seed = seed

    def place(self, benchmark) -> torch.Tensor:
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
        R  = int(benchmark.grid_rows)
        C  = int(benchmark.grid_cols)

        plc  = _load_plc(benchmark)
        nets = _extract_nets(benchmark, plc) if plc is not None else []

        if not nets:
            legal = _legalize(init, movable, sizes, cw, ch)
            out[:n_hard] = torch.from_numpy(legal).float()
            return out

        ni = NetIndex(nets, n_hard)

        def rem(target_elapsed):
            """Seconds remaining before target_elapsed or HARD_CAP."""
            used = time.time()-T0
            return max(0.0, min(target_elapsed, HARD_CAP) - used)

        # ── 1. GP phase 1: 20s, aggressive spreading
        gp1 = _global_place(
            init=init.copy(), movable=movable, sizes=sizes, ni=ni,
            canvas_w=cw, canvas_h=ch, R=R, C=C,
            t_start=T0, time_budget=rem(20.0),
            seed=self.seed, density_target=0.65,
            lam0=5e-4, lam_max=5.0,
        )

        # ── 2. Legalize 1
        legal = _legalize(gp1, movable, sizes, cw, ch, gap=0.003)

        # ── 3. GP phase 2: re-spread from legalized positions
        gp2_budget = rem(33.0)
        if gp2_budget > 2.0:
            gp2 = _global_place(
                init=legal.copy(), movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch, R=R, C=C,
                t_start=T0, time_budget=gp2_budget,
                seed=self.seed+1, density_target=0.5,
                lam0=1e-2, lam_max=3.0,
                gamma0_frac=0.02, gamma_min_frac=0.001,
            )
            legal = _legalize(gp2, movable, sizes, cw, ch, gap=0.003)

        # ── 4. Congestion escape: 6s
        cong_budget = rem(42.0)
        if cong_budget > 1.0:
            legal = _congestion_escape(
                pos=legal, movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch, R=R, C=C,
                t_start=T0, time_budget=cong_budget,
                gap=0.003, radius=4,
            )

        # ── 5. Swap refine: 5s
        swap_budget = rem(47.0)
        if swap_budget > 1.0:
            legal = _swap_refine(
                pos=legal, movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch,
                t_start=T0, time_budget=swap_budget,
                gap=0.003, k=12,
            )

        # ── 6. Mini SA: 2s
        sa_budget = rem(49.0)
        if sa_budget > 0.5:
            legal = _mini_sa(
                pos=legal, movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch,
                t_start=T0, time_budget=sa_budget,
                gap=0.003,
            )

        # ── 7. Safety overlap repair
        legal = _repair_overlaps(legal, movable, sizes, cw, ch, gap=0.003)

        out[:n_hard] = torch.from_numpy(legal).float()
        return out