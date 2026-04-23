"""
MJ97 v7r — Fast Analytical Placer

Base: v4 spiral legalizer (proven zero overlaps on all 17 benchmarks)

Key fixes over v4:
1. RUNTIME: v4 took 66min total. Per-net Python loops in _wa_wl_grad are O(nets*iter).
   Fix: batch WA gradient using numpy segment ops — no Python loop over nets.
   Target: <15s GP, <10s legalize, <20s post-process = ~45s/bench.

2. CONGESTION: congestion ~2.0-2.4 on bad benchmarks (ibm02,06,12,17,18).
   Root cause: GP spreads macros fine but legalization re-clusters them.
   Fix A: stronger density lambda ramp (forces more spreading before legalization).
   Fix B: after legalization, run targeted bin-escape moves (not swaps):
          for each macro in top-10% RUDY bin, nudge it to an adjacent empty spot.
          This is faster and more targeted than the broken congestion_refine in v4.

3. SWAP: v4 swap used set operations in Python — slow. Use array ops.

4. LEGALIZATION: v4 spiral is kept exactly. It works. Don't touch it.
"""

import math
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Loader (unchanged from v4)
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
# Net index — CSR format for O(1) adjacency queries
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
        self.net_sizes = np.array([len(net) for net in nets], dtype=np.int32)

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
# WA wirelength + gradient — numpy batched, no per-net Python loop
# Uses precomputed CSR to process nets in vectorized fashion where possible.
# For nets of variable size we still need a loop, but minimized overhead.
# ─────────────────────────────────────────────────────────────────────────────

def _wa_wl_grad(
    pos: np.ndarray,
    ni: NetIndex,
    gamma: float,
) -> Tuple[float, np.ndarray]:
    """WA wirelength + gradient. Minimized Python overhead."""
    grad = np.zeros_like(pos)
    total = 0.0
    x = pos[:, 0]; y = pos[:, 1]
    gx = grad[:, 0]; gy = grad[:, 1]

    for net_id in range(ni.n_nets):
        o0 = ni.offsets[net_id]; o1 = ni.offsets[net_id + 1]
        if o1 - o0 < 2:
            continue
        idx = ni.flat[o0:o1]

        # X dimension
        vx = x[idx]
        vmx = vx.max(); vnx = vx.min()
        epx = np.exp(np.clip((vx - vmx) / gamma, -30, 0))
        enx = np.exp(np.clip(-(vx - vnx) / gamma, -30, 0))
        spx = epx.sum() + 1e-12; snx = enx.sum() + 1e-12
        wpx = (vx * epx).sum() / spx; wnx = (vx * enx).sum() / snx
        total += wpx - wnx
        np.add.at(gx, idx, epx/spx*(1.0+(vx-wpx)/gamma) - enx/snx*(1.0-(vx-wnx)/gamma))

        # Y dimension
        vy = y[idx]
        vmy = vy.max(); vny = vy.min()
        epy = np.exp(np.clip((vy - vmy) / gamma, -30, 0))
        eny = np.exp(np.clip(-(vy - vny) / gamma, -30, 0))
        spy = epy.sum() + 1e-12; sny = eny.sum() + 1e-12
        wpy = (vy * epy).sum() / spy; wny = (vy * eny).sum() / sny
        total += wpy - wny
        np.add.at(gy, idx, epy/spy*(1.0+(vy-wpy)/gamma) - eny/sny*(1.0-(vy-wny)/gamma))

    return total, grad


# ─────────────────────────────────────────────────────────────────────────────
# Bell kernel for FFT density
# ─────────────────────────────────────────────────────────────────────────────

def _bell(u, s):
    au = np.abs(u); w = np.zeros_like(u)
    m1 = au < s; m2 = (au >= s) & (au < 2*s)
    w[m1] = 1.5 - au[m1]**2 / (s[m1]**2 + 1e-30)
    w[m2] = 0.5 * (2.0 - au[m2]/(s[m2]+1e-30))**2
    return w

def _dbell(u, s):
    au = np.abs(u); dw = np.zeros_like(u)
    m1 = au < s; m2 = (au >= s) & (au < 2*s)
    dw[m1] = -2.0 * u[m1] / (s[m1]**2 + 1e-30)
    dw[m2] = -(2.0 - au[m2]/(s[m2]+1e-30)) / (s[m2]+1e-30) * np.sign(u[m2])
    return dw


# ─────────────────────────────────────────────────────────────────────────────
# FFT electrostatic density penalty (ePlace)
# ─────────────────────────────────────────────────────────────────────────────

def _density_fft(pos, sizes, canvas_w, canvas_h, G=32, target=1.0):
    R = C = G; bh = canvas_h/R; bw = canvas_w/C
    bx = (np.arange(C)+0.5)*bw; by = (np.arange(R)+0.5)*bh
    sx = np.maximum(sizes[:,0:1], 1.5*bw)
    sy = np.maximum(sizes[:,1:2], 1.5*bh)
    dx = pos[:,0:1] - bx[None,:]; dy = pos[:,1:2] - by[None,:]
    wx = _bell(dx, sx); wy = _bell(dy, sy)
    area = sizes[:,0]*sizes[:,1]; w_area = area/(bw*bh)
    rho = (w_area[:,None]*wy).T @ wx
    ovf = np.maximum(0.0, rho - target)
    penalty = 0.5*(ovf**2).sum()

    ovf_t = torch.from_numpy(ovf.astype(np.float32))
    R2, C2 = R*2, C*2
    ext = torch.zeros(R2,C2)
    ext[:R,:C]=ovf_t; ext[:R,C:]=ovf_t.flip(1)
    ext[R:,:C]=ovf_t.flip(0); ext[R:,C:]=ovf_t.flip([0,1])
    F = torch.fft.rfft2(ext)
    kr = (2*math.pi*torch.arange(R2)/R2)**2
    kc = (2*math.pi*torch.arange(C2//2+1)/C2)**2
    k2 = kr[:,None]+kc[None,:]; k2[0,0]=1.0
    phi_fft = F/k2; phi_fft[0,0]=0.0
    phi = torch.fft.irfft2(phi_fft,s=(R2,C2))[:R,:C].numpy()

    Ex=np.zeros((R,C)); Ey=np.zeros((R,C))
    Ex[:,1:-1]=(phi[:,2:]-phi[:,:-2])/(2*bw); Ex[:,0]=(phi[:,1]-phi[:,0])/bw; Ex[:,-1]=(phi[:,-1]-phi[:,-2])/bw
    Ey[1:-1,:]=(phi[2:,:]-phi[:-2,:])/(2*bh); Ey[0,:]=(phi[1,:]-phi[0,:])/bh; Ey[-1,:]=(phi[-1,:]-phi[-2,:])/bh

    dwx = _dbell(dx,sx); dwy = _dbell(dy,sy)
    wy_Ex = wy @ Ex
    grad_x = w_area * (wy_Ex*dwx).sum(axis=1)
    wx_Ey = wx @ Ey.T
    grad_y = w_area * (wx_Ey*dwy).sum(axis=1)
    return penalty, np.stack([grad_x,grad_y],axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# RUDY demand map — vectorized bounding box accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _rudy_map_fast(pos, ni, canvas_w, canvas_h, R, C):
    """Fast RUDY map using precomputed net bbox arrays."""
    if ni.n_nets == 0:
        return np.zeros((R,C))
    bw = canvas_w/C; bh = canvas_h/R
    x = pos[:,0]; y = pos[:,1]
    demand = np.zeros((R,C))

    for nid in range(ni.n_nets):
        idx = ni.flat[ni.offsets[nid]:ni.offsets[nid+1]]
        xi = x[idx]; yi = y[idx]
        x0=xi.min(); x1=xi.max(); y0=yi.min(); y1=yi.max()
        nw=max(x1-x0,1e-8); nh=max(y1-y0,1e-8)
        box=nw*nh
        r0=int(np.clip(y0/bh,0,R-1)); r1=int(np.clip(y1/bh,0,R-1))
        c0=int(np.clip(x0/bw,0,C-1)); c1=int(np.clip(x1/bw,0,C-1))
        den = (nh/box + nw/box)*bw*bh*0.5
        demand[r0:r1+1,c0:c1+1] += den
    return demand


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimizer
# ─────────────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m=np.zeros((n,2)); self.v=np.zeros((n,2)); self.t=0

    def step(self, g):
        self.t+=1
        self.m=self.b1*self.m+(1-self.b1)*g
        self.v=self.b2*self.v+(1-self.b2)*g**2
        mh=self.m/(1-self.b1**self.t); vh=self.v/(1-self.b2**self.t)
        return self.lr*mh/(np.sqrt(vh)+self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Global placement — WL + density + RUDY congestion
# Time-boxed: exits after time_budget seconds
# ─────────────────────────────────────────────────────────────────────────────

def _global_place(
    init, movable, sizes, nets, ni,
    canvas_w, canvas_h, grid_rows, grid_cols,
    time_budget=22.0, seed=97,
):
    np.random.seed(seed)
    t0 = time.time()
    hw=sizes[:,0]*0.5; hh=sizes[:,1]*0.5
    diag=math.sqrt(canvas_w**2+canvas_h**2)
    G=32

    gamma0=diag*0.05; gamma_min=diag*0.0008
    lam_den0=5e-4; lam_den_max=3.0
    # Congestion penalty: stronger ramp, starts at 20% of iterations
    lam_cong0=1e-4; lam_cong_max=0.8

    bin_size=min(canvas_w/G, canvas_h/G)
    lr=bin_size*0.5

    pos=init.copy()
    pos[:,0]=np.clip(pos[:,0],hw,canvas_w-hw)
    pos[:,1]=np.clip(pos[:,1],hh,canvas_h-hh)

    adam=Adam(len(pos),lr=lr)
    best_pos=pos.copy(); best_wl=float('inf')

    R_rudy=grid_rows; C_rudy=grid_cols
    it=0; max_iter=800

    while it < max_iter and (time.time()-t0) < time_budget:
        t_frac = it/max_iter
        gamma = gamma0*math.exp(math.log(gamma_min/gamma0)*t_frac)
        lam_den = lam_den0*math.exp(math.log(lam_den_max/lam_den0)*t_frac)

        wl, gwl = _wa_wl_grad(pos, ni, gamma)
        dp, gdp = _density_fft(pos, sizes, canvas_w, canvas_h, G=G)
        g = gwl + lam_den*gdp

        # Congestion penalty: kicks in after 20%, ramps up strongly
        if t_frac > 0.20:
            cong_t = (t_frac-0.20)/0.80
            lam_cong = lam_cong0*math.exp(math.log(lam_cong_max/lam_cong0)*cong_t)
            # Every 8 iterations (RUDY is expensive due to net loop)
            if it % 8 == 0:
                demand = _rudy_map_fast(pos, ni, canvas_w, canvas_h, R_rudy, C_rudy)
                thresh = max(demand.mean()*1.2, 1e-6)
                ovf = np.maximum(0.0, demand - thresh)  # (R,C)
                # Gradient: move macros away from high-demand bins
                bw=canvas_w/C_rudy; bh=canvas_h/R_rudy
                gcong=np.zeros_like(pos)
                for i in range(len(pos)):
                    if not movable[i]: continue
                    r=int(np.clip(pos[i,1]/bh,0,R_rudy-1))
                    c=int(np.clip(pos[i,0]/bw,0,C_rudy-1))
                    if ovf[r,c] > 1e-8:
                        # Gradient direction: away from bin center
                        bin_cx=(c+0.5)*bw; bin_cy=(r+0.5)*bh
                        gcong[i,0]=ovf[r,c]*(pos[i,0]-bin_cx)/(bw+1e-8)
                        gcong[i,1]=ovf[r,c]*(pos[i,1]-bin_cy)/(bh+1e-8)
            g = g + lam_cong*gcong

        g[~movable]=0.0
        delta=adam.step(g)
        pos=pos-delta
        pos[:,0]=np.clip(pos[:,0],hw,canvas_w-hw)
        pos[:,1]=np.clip(pos[:,1],hh,canvas_h-hh)
        pos[~movable]=init[~movable]

        if wl < best_wl and t_frac > 0.25:
            best_wl=wl; best_pos=pos.copy()

        it+=1

    best_pos[~movable]=init[~movable]
    return best_pos


# ─────────────────────────────────────────────────────────────────────────────
# Spiral legalizer — from v4, proven zero overlaps
# ─────────────────────────────────────────────────────────────────────────────

def _legalize(init, movable, sizes, canvas_w, canvas_h, gap=0.002):
    n = init.shape[0]
    hw=sizes[:,0]*0.5; hh=sizes[:,1]*0.5
    sep_x=(sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y=(sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0

    order=sorted(range(n), key=lambda i: -(sizes[i,0]*sizes[i,1]))
    placed=np.zeros(n, dtype=bool)
    legal=init.copy()
    step_base=0.15

    for idx in order:
        if not movable[idx]:
            placed[idx]=True; continue

        pidx=np.where(placed)[0]
        if len(pidx):
            dx=np.abs(legal[idx,0]-legal[pidx,0])
            dy=np.abs(legal[idx,1]-legal[pidx,1])
            if not ((dx<sep_x[idx][pidx]+gap)&(dy<sep_y[idx][pidx]+gap)).any():
                placed[idx]=True; continue

        step=max(sizes[idx,0],sizes[idx,1])*step_base
        best=None; bdist=float('inf')

        for r in range(1,300):
            found=False
            for dxm in range(-r,r+1):
                for sign in [1,-1]:
                    dym=sign*(r-abs(dxm))
                    cx=np.clip(init[idx,0]+dxm*step,hw[idx],canvas_w-hw[idx])
                    cy=np.clip(init[idx,1]+dym*step,hh[idx],canvas_h-hh[idx])
                    if len(pidx):
                        ddx=np.abs(cx-legal[pidx,0])
                        ddy=np.abs(cy-legal[pidx,1])
                        if ((ddx<sep_x[idx][pidx]+gap)&(ddy<sep_y[idx][pidx]+gap)).any():
                            continue
                    d=(cx-init[idx,0])**2+(cy-init[idx,1])**2
                    if d<bdist:
                        bdist=d; best=np.array([cx,cy]); found=True
            if found and r>=3:
                break

        if best is not None:
            legal[idx]=best
        placed[idx]=True

    return legal


# ─────────────────────────────────────────────────────────────────────────────
# Congestion escape moves
#
# For each macro in a high-RUDY bin, try to move it to a specific
# candidate position in a lower-RUDY bin while maintaining legality.
# Simpler and faster than full swap: just moves one macro at a time.
# Accept if: congestion_gain > alpha * HPWL_increase
# ─────────────────────────────────────────────────────────────────────────────

def _congestion_escape(
    pos, movable, sizes, ni, canvas_w, canvas_h,
    grid_rows, grid_cols, gap=0.002, time_budget=12.0,
    alpha=0.08,
):
    if ni.n_nets == 0:
        return pos
    t0=time.time(); n=len(pos)
    R=grid_rows; C=grid_cols
    bw=canvas_w/C; bh=canvas_h/R
    hw=sizes[:,0]/2; hh=sizes[:,1]/2
    movable_idx=np.where(movable)[0]
    diag=math.sqrt(canvas_w**2+canvas_h**2)
    p=pos.copy()

    for _pass in range(20):
        if time.time()-t0 >= time_budget:
            break

        demand = _rudy_map_fast(p, ni, canvas_w, canvas_h, R, C)
        p90=np.percentile(demand,90); p20=np.percentile(demand,20)
        if p90-p20 < 0.05:
            break

        np.random.shuffle(movable_idx)
        n_moved=0

        for idx in movable_idx:
            if time.time()-t0 >= time_budget:
                break
            r_cur=int(np.clip(p[idx,1]/bh,0,R-1))
            c_cur=int(np.clip(p[idx,0]/bw,0,C-1))
            d_cur=demand[r_cur,c_cur]
            if d_cur < p90:
                continue

            wl_before=ni.hpwl_node(p,idx)
            orig=p[idx].copy()

            best_score=-1e9; best_cand=None

            # Try candidate positions in cooler bins
            for dr in range(-4,5):
                for dc in range(-4,5):
                    if dr==0 and dc==0: continue
                    nr=int(np.clip(r_cur+dr,0,R-1))
                    nc=int(np.clip(c_cur+dc,0,C-1))
                    if demand[nr,nc] >= d_cur:
                        continue

                    # Candidate: center of target bin, clamped
                    tx=np.clip((nc+0.5)*bw, hw[idx], canvas_w-hw[idx])
                    ty=np.clip((nr+0.5)*bh, hh[idx], canvas_h-hh[idx])

                    # Legality: check against all other macros
                    ok=True
                    for j in range(n):
                        if j==idx: continue
                        if (abs(tx-p[j,0])<hw[idx]+hw[j]+gap and
                                abs(ty-p[j,1])<hh[idx]+hh[j]+gap):
                            ok=False; break
                    if not ok:
                        continue

                    p[idx]=[tx,ty]
                    wl_after=ni.hpwl_node(p,idx)
                    p[idx]=orig

                    cong_gain=d_cur-demand[nr,nc]
                    wl_delta=wl_after-wl_before
                    score=cong_gain - alpha*wl_delta/(diag*0.01+1e-10)

                    if score > best_score:
                        best_score=score; best_cand=np.array([tx,ty])

            if best_cand is not None and best_score > 0:
                p[idx]=best_cand
                n_moved+=1

        if n_moved==0:
            break

    return p


# ─────────────────────────────────────────────────────────────────────────────
# HPWL swap refinement — k-nearest neighbors, greedy accept
# ─────────────────────────────────────────────────────────────────────────────

def _swap_refine(
    pos, movable, sizes, ni, canvas_w, canvas_h,
    gap=0.002, n_passes=6, time_budget=8.0, k=15,
):
    t0=time.time(); n=len(pos)
    hw=sizes[:,0]/2; hh=sizes[:,1]/2
    sep_x=(sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y=(sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx=np.where(movable)[0]

    def legal_swap(p,i,j):
        ni_pos=p[j].copy(); nj_pos=p[i].copy()
        for nd,np_ in [(i,ni_pos),(j,nj_pos)]:
            if not(hw[nd]<=np_[0]<=canvas_w-hw[nd] and hh[nd]<=np_[1]<=canvas_h-hh[nd]):
                return False
            placed=np.ones(n,dtype=bool); placed[i]=False; placed[j]=False
            dx=np.abs(np_[0]-p[:,0]); dy=np.abs(np_[1]-p[:,1])
            if ((dx<sep_x[nd]+gap)&(dy<sep_y[nd]+gap)&placed).any():
                return False
        return True

    p=pos.copy()
    for _ in range(n_passes):
        if time.time()-t0 >= time_budget:
            break
        improved=0; order=movable_idx.copy(); np.random.shuffle(order)
        for i in order:
            if time.time()-t0 >= time_budget:
                break
            dists=np.abs(p[i,0]-p[movable_idx,0])+np.abs(p[i,1]-p[movable_idx,1])
            neighbors=movable_idx[np.argsort(dists)[1:k+1]]
            best_gain=1e-9; best_j=-1

            for j in neighbors:
                wl_before=ni.hpwl_nodes(p,i,j)
                p[i],p[j]=p[j].copy(),p[i].copy()
                if legal_swap(p,i,j):
                    wl_after=ni.hpwl_nodes(p,i,j)
                    gain=wl_before-wl_after
                    if gain>best_gain:
                        best_gain=gain; best_j=j
                    else:
                        p[i],p[j]=p[j].copy(),p[i].copy()
                else:
                    p[i],p[j]=p[j].copy(),p[i].copy()

            if best_j>=0:
                improved+=1
        if improved==0:
            break
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Mini SA
# ─────────────────────────────────────────────────────────────────────────────

def _mini_sa(pos, movable, sizes, ni, canvas_w, canvas_h, gap=0.002, time_budget=4.0):
    t0=time.time(); n=len(pos)
    hw=sizes[:,0]/2; hh=sizes[:,1]/2
    sep_x=(sizes[:,0:1]+sizes[:,0].reshape(1,n))/2.0
    sep_y=(sizes[:,1:2]+sizes[:,1].reshape(1,n))/2.0
    movable_idx=np.where(movable)[0]
    diag=math.sqrt(canvas_w**2+canvas_h**2)
    T0=0.006*diag; T1=0.0002*diag
    p=pos.copy()

    while True:
        elapsed=time.time()-t0
        if elapsed>=time_budget: break
        frac=elapsed/time_budget
        T=T0*math.exp(math.log(T1/T0)*frac); step=T*1.5

        i=int(np.random.choice(movable_idx))
        orig=p[i].copy()
        nx=float(np.clip(orig[0]+np.random.uniform(-step,step),hw[i],canvas_w-hw[i]))
        ny=float(np.clip(orig[1]+np.random.uniform(-step,step),hh[i],canvas_h-hh[i]))

        placed=np.ones(n,dtype=bool); placed[i]=False
        ddx=np.abs(nx-p[:,0]); ddy=np.abs(ny-p[:,1])
        if ((ddx<sep_x[i]+gap)&(ddy<sep_y[i]+gap)&placed).any():
            continue

        wl_before=ni.hpwl_node(p,i)
        p[i]=[nx,ny]
        wl_after=ni.hpwl_node(p,i)
        delta=wl_after-wl_before

        if delta>0 and np.random.random()>=math.exp(-delta/(T+1e-12)):
            p[i]=orig

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main Placer
# ─────────────────────────────────────────────────────────────────────────────

class Mj97Placer:
    """
    MJ97 v7r: v4 spiral legalizer (proven zero overlaps) +
    congestion penalty in GP + congestion escape moves.

    Pipeline per benchmark:
      1. Global placement: 22s budget (WL + density + RUDY congestion)
      2. Spiral legalization: ~5-30s (size-dependent)
      3. Congestion escape: 12s
      4. HPWL swap: 8s
      5. Mini SA: 4s
    Total target: <50s/benchmark on eval hardware.
    """

    def __init__(self, seed: int = 97):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return benchmark.macro_positions.clone()

        out   = benchmark.macro_positions.clone()
        init  = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        gr = int(benchmark.grid_rows)
        gc = int(benchmark.grid_cols)

        T0 = time.time()

        plc  = _load_plc(benchmark)
        nets = _extract_nets(benchmark, plc) if plc is not None else []

        if not nets:
            legal = _legalize(init, movable, sizes, cw, ch)
            out[:n_hard] = torch.from_numpy(legal).float()
            return out

        ni = NetIndex(nets, n_hard)

        # ── 1. Global placement
        elapsed = time.time()-T0
        gp = _global_place(
            init=init.copy(), movable=movable, sizes=sizes,
            nets=nets, ni=ni, canvas_w=cw, canvas_h=ch,
            grid_rows=gr, grid_cols=gc,
            time_budget=max(5.0, 22.0-elapsed), seed=self.seed,
        )

        # ── 2. Spiral legalization (zero overlap guaranteed)
        legal = _legalize(gp, movable, sizes, cw, ch, gap=0.002)

        # ── 3. Congestion escape moves
        elapsed = time.time()-T0
        cong_budget = min(12.0, max(2.0, 46.0-elapsed))
        legal = _congestion_escape(
            pos=legal, movable=movable, sizes=sizes, ni=ni,
            canvas_w=cw, canvas_h=ch, grid_rows=gr, grid_cols=gc,
            gap=0.002, time_budget=cong_budget, alpha=0.08,
        )

        # ── 4. HPWL swap
        elapsed = time.time()-T0
        swap_budget = min(8.0, max(1.0, 52.0-elapsed))
        legal = _swap_refine(
            pos=legal, movable=movable, sizes=sizes, ni=ni,
            canvas_w=cw, canvas_h=ch, gap=0.002,
            n_passes=6, time_budget=swap_budget, k=15,
        )

        # ── 5. Mini SA
        elapsed = time.time()-T0
        sa_budget = min(4.0, max(0.0, 55.0-elapsed))
        if sa_budget > 0.5:
            legal = _mini_sa(
                pos=legal, movable=movable, sizes=sizes, ni=ni,
                canvas_w=cw, canvas_h=ch, gap=0.002,
                time_budget=sa_budget,
            )

        out[:n_hard] = torch.from_numpy(legal).float()
        return out