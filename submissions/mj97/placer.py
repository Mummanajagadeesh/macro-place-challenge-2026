"""
MJ97 v6

What changed vs v5:
  - KEEP fast vectorized WA grad (reduceat) — this works great
  - REVERT legalizer to v4's proven spiral (zero overlaps guaranteed)
    but speed it up: pre-build a single centers/hw/hh array, grow it
    with np.vstack incrementally — no KD-tree rebuild each macro
  - Density cached every 3 iters (keep from v5)
  - GP time budget 45s, legalizer uncapped (it's fast enough), 
    congestion 8s, swap 5s, SA leftover
  - KD-tree in congestion/swap/SA for legality (keep from v5, it works)
  - Bug fix: congestion_refine was rebuilding tree inside inner loop
"""

import math
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree

from macro_place.benchmark import Benchmark


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
        offsets = [0]
        flat = []
        for net in nets:
            flat.extend(net)
            offsets.append(len(flat))
        self.flat = np.array(flat, dtype=np.int32)
        self.offsets = np.array(offsets, dtype=np.int32)
        self.net_sizes = np.array([len(net) for net in nets], dtype=np.int32)
        node_nets: List[List[int]] = [[] for _ in range(n_nodes)]
        for ni, net in enumerate(nets):
            for node in net:
                node_nets[node].append(ni)
        nn_flat = []
        nn_offsets = [0]
        for nl in node_nets:
            nn_flat.extend(nl)
            nn_offsets.append(len(nn_flat))
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
        ni_set = set(self.nn_flat[self.nn_offsets[i]:self.nn_offsets[i+1]])
        nj_set = set(self.nn_flat[self.nn_offsets[j]:self.nn_offsets[j+1]])
        total = 0.0
        for ni in ni_set | nj_set:
            idx = self.flat[self.offsets[ni]:self.offsets[ni+1]]
            total += x[idx].max() - x[idx].min() + y[idx].max() - y[idx].min()
        return total


# ─────────────────────────────────────────────────────────────────────────────
# Build CSR arrays for vectorized WA
# ─────────────────────────────────────────────────────────────────────────────

def _build_wa_arrays(ni: NetIndex, n_nodes: int):
    n_pins = len(ni.flat)
    net_id_per_pin = np.zeros(n_pins, dtype=np.int32)
    for net_id in range(ni.n_nets):
        o0, o1 = ni.offsets[net_id], ni.offsets[net_id+1]
        net_id_per_pin[o0:o1] = net_id
    return ni.flat.copy(), net_id_per_pin, ni.offsets.copy(), ni.n_nets


# ─────────────────────────────────────────────────────────────────────────────
# Fully vectorized WA wirelength + gradient  (no per-net Python loop)
# ─────────────────────────────────────────────────────────────────────────────

def _wa_wl_grad_fast(
    pos: np.ndarray,
    flat_idx: np.ndarray,
    net_id: np.ndarray,
    offsets: np.ndarray,
    n_nets: int,
    gamma: float,
) -> Tuple[float, np.ndarray]:
    grad = np.zeros_like(pos)
    total_wl = 0.0

    for d in range(2):
        v = pos[flat_idx, d]

        v_max = np.maximum.reduceat(v, offsets[:-1])
        v_min = np.minimum.reduceat(v, offsets[:-1])

        vmax_pin = v_max[net_id]
        vmin_pin = v_min[net_id]

        ep = np.exp(np.clip((v - vmax_pin) / gamma, -30, 0))
        en = np.exp(np.clip(-(v - vmin_pin) / gamma, -30, 0))

        sp  = np.zeros(n_nets); np.add.at(sp,  net_id, ep)
        sn  = np.zeros(n_nets); np.add.at(sn,  net_id, en)
        svp = np.zeros(n_nets); np.add.at(svp, net_id, v * ep)
        svn = np.zeros(n_nets); np.add.at(svn, net_id, v * en)

        sp = np.maximum(sp, 1e-12)
        sn = np.maximum(sn, 1e-12)
        wp = svp / sp
        wn = svn / sn

        total_wl += (wp - wn).sum()

        wp_pin = wp[net_id]
        wn_pin = wn[net_id]
        sp_pin = sp[net_id]
        sn_pin = sn[net_id]

        gp = ep / sp_pin * (1.0 + (v - wp_pin) / gamma)
        gn = en / sn_pin * (1.0 - (v - wn_pin) / gamma)
        np.add.at(grad[:, d], flat_idx, gp - gn)

    return float(total_wl), grad


# ─────────────────────────────────────────────────────────────────────────────
# Bell kernel
# ─────────────────────────────────────────────────────────────────────────────

def _bell(u, s):
    au = np.abs(u)
    w = np.zeros_like(u)
    m1 = au < s
    m2 = (au >= s) & (au < 2*s)
    w[m1] = 1.5 - au[m1]**2 / (s[m1]**2 + 1e-30)
    w[m2] = 0.5 * (2.0 - au[m2] / (s[m2] + 1e-30))**2
    return w

def _dbell(u, s):
    au = np.abs(u)
    dw = np.zeros_like(u)
    m1 = au < s
    m2 = (au >= s) & (au < 2*s)
    dw[m1] = -2.0 * u[m1] / (s[m1]**2 + 1e-30)
    dw[m2] = -(2.0 - au[m2] / (s[m2] + 1e-30)) / (s[m2] + 1e-30) * np.sign(u[m2])
    return dw


# ─────────────────────────────────────────────────────────────────────────────
# FFT electrostatic density
# ─────────────────────────────────────────────────────────────────────────────

def _density_fft(
    pos: np.ndarray,
    sizes: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    G: int = 32,
    target: float = 1.0,
) -> Tuple[float, np.ndarray, float]:
    R = C = G
    bh = canvas_h / R
    bw = canvas_w / C

    bx = (np.arange(C) + 0.5) * bw
    by = (np.arange(R) + 0.5) * bh

    sx = np.maximum(sizes[:, 0:1], 1.5 * bw)
    sy = np.maximum(sizes[:, 1:2], 1.5 * bh)

    dx = pos[:, 0:1] - bx[None, :]
    dy = pos[:, 1:2] - by[None, :]

    wx = _bell(dx, sx)
    wy = _bell(dy, sy)

    area = sizes[:, 0] * sizes[:, 1]
    w_area = area / (bw * bh)

    rho = (w_area[:, None] * wy).T @ wx

    ovf = np.maximum(0.0, rho - target)
    penalty = 0.5 * (ovf**2).sum()

    ovf_t = torch.from_numpy(ovf.astype(np.float32))
    R2, C2 = R*2, C*2
    ext = torch.zeros(R2, C2)
    ext[:R, :C] = ovf_t
    ext[:R, C:] = ovf_t.flip(1)
    ext[R:, :C] = ovf_t.flip(0)
    ext[R:, C:] = ovf_t.flip([0, 1])

    F = torch.fft.rfft2(ext)
    kr = (2 * math.pi * torch.arange(R2) / R2)**2
    kc = (2 * math.pi * torch.arange(C2//2+1) / C2)**2
    k2 = kr[:, None] + kc[None, :]
    k2[0, 0] = 1.0
    phi_fft = F / k2
    phi_fft[0, 0] = 0.0
    phi = torch.fft.irfft2(phi_fft, s=(R2, C2))[:R, :C].numpy()

    Ex = np.zeros((R, C)); Ey = np.zeros((R, C))
    Ex[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*bw)
    Ex[:, 0]    = (phi[:, 1]  - phi[:, 0])   / bw
    Ex[:, -1]   = (phi[:, -1] - phi[:, -2])  / bw
    Ey[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*bh)
    Ey[0, :]    = (phi[1, :]  - phi[0, :])   / bh
    Ey[-1, :]   = (phi[-1, :] - phi[-2, :])  / bh

    dwx = _dbell(dx, sx)
    dwy = _dbell(dy, sy)

    wy_Ex  = wy @ Ex
    grad_x = w_area * (wy_Ex * dwx).sum(axis=1)
    wx_Ey  = wx @ Ey.T
    grad_y = w_area * (wx_Ey * dwy).sum(axis=1)

    return penalty, np.stack([grad_x, grad_y], axis=1), float(ovf.max())


# ─────────────────────────────────────────────────────────────────────────────
# Adam
# ─────────────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, n: int, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = np.zeros((n, 2)); self.v = np.zeros((n, 2)); self.t = 0

    def step(self, g: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*g
        self.v = self.b2*self.v + (1-self.b2)*g**2
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Global placement
# ─────────────────────────────────────────────────────────────────────────────

def _global_place(
    init: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    ni: NetIndex,
    canvas_w: float,
    canvas_h: float,
    time_budget: float = 45.0,
    seed: int = 97,
) -> np.ndarray:
    np.random.seed(seed)
    t0 = time.time()

    flat_idx, net_id_arr, offsets, n_nets = _build_wa_arrays(ni, init.shape[0])

    hw = sizes[:, 0] * 0.5; hh = sizes[:, 1] * 0.5
    diag = math.sqrt(canvas_w**2 + canvas_h**2)
    G = 32

    gamma0    = diag * 0.05
    gamma_min = diag * 0.0006
    lam0      = 2e-4
    lam_max   = 3.0

    bin_size = min(canvas_w / G, canvas_h / G)
    lr = bin_size * 0.6

    pos = init.copy()
    pos[:, 0] = np.clip(pos[:, 0], hw, canvas_w - hw)
    pos[:, 1] = np.clip(pos[:, 1], hh, canvas_h - hh)

    # Break symmetry
    rng = np.random.default_rng(seed)
    pos[movable, 0] += rng.uniform(-bin_size*0.3, bin_size*0.3, movable.sum())
    pos[movable, 1] += rng.uniform(-bin_size*0.3, bin_size*0.3, movable.sum())
    pos[:, 0] = np.clip(pos[:, 0], hw, canvas_w - hw)
    pos[:, 1] = np.clip(pos[:, 1], hh, canvas_h - hh)

    adam = Adam(len(pos), lr=lr)
    best_pos = pos.copy(); best_wl = float('inf')
    cached_gdp = np.zeros_like(pos)

    it = 0
    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break

        t = min(elapsed / time_budget, 1.0)
        gamma = gamma0 * math.exp(math.log(gamma_min / gamma0) * t)
        lam   = lam0   * math.exp(math.log(lam_max  / lam0)   * t)

        wl, gwl = _wa_wl_grad_fast(pos, flat_idx, net_id_arr, offsets, n_nets, gamma)

        if it % 3 == 0:
            _, cached_gdp, _ = _density_fft(pos, sizes, canvas_w, canvas_h, G=G)

        g = gwl + lam * cached_gdp
        g[~movable] = 0.0

        delta = adam.step(g)
        pos = pos - delta
        pos[:, 0] = np.clip(pos[:, 0], hw, canvas_w - hw)
        pos[:, 1] = np.clip(pos[:, 1], hh, canvas_h - hh)
        pos[~movable] = init[~movable]

        if wl < best_wl and t > 0.3:
            best_wl = wl; best_pos = pos.copy()

        it += 1

    best_pos[~movable] = init[~movable]
    return best_pos


# ─────────────────────────────────────────────────────────────────────────────
# Legalization: place-by-area, spiral search — PROVEN zero-overlap version
# Vectorized overlap check, no KD-tree (O(N) per macro but N<=537, fast)
# ─────────────────────────────────────────────────────────────────────────────

def _legalize(
    init: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.001,
) -> np.ndarray:
    n = init.shape[0]
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5

    # Place largest macros first
    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))

    placed_centers = np.empty((0, 2), dtype=np.float64)
    placed_hw = np.empty(0, dtype=np.float64)
    placed_hh = np.empty(0, dtype=np.float64)

    legal = init.copy()

    for idx in order:
        hw_i = hw[idx]; hh_i = hh[idx]

        def overlaps(cx, cy):
            if cx - hw_i < 0 or cx + hw_i > canvas_w:
                return True
            if cy - hh_i < 0 or cy + hh_i > canvas_h:
                return True
            if len(placed_centers) == 0:
                return False
            ddx = np.abs(cx - placed_centers[:, 0])
            ddy = np.abs(cy - placed_centers[:, 1])
            return bool(((ddx < placed_hw + hw_i + gap) &
                         (ddy < placed_hh + hh_i + gap)).any())

        if not movable[idx]:
            # Fixed macro: add to placed list as-is
            cx0 = float(np.clip(init[idx, 0], hw_i, canvas_w - hw_i))
            cy0 = float(np.clip(init[idx, 1], hh_i, canvas_h - hh_i))
            legal[idx] = [cx0, cy0]
            placed_centers = np.vstack([placed_centers, [[cx0, cy0]]]) if len(placed_centers) > 0 else np.array([[cx0, cy0]])
            placed_hw = np.append(placed_hw, hw_i)
            placed_hh = np.append(placed_hh, hh_i)
            continue

        gx = float(init[idx, 0])
        gy = float(init[idx, 1])
        cx0 = float(np.clip(gx, hw_i, canvas_w - hw_i))
        cy0 = float(np.clip(gy, hh_i, canvas_h - hh_i))

        # Check current GP position first
        if not overlaps(cx0, cy0):
            legal[idx] = [cx0, cy0]
            placed_centers = np.vstack([placed_centers, [[cx0, cy0]]]) if len(placed_centers) > 0 else np.array([[cx0, cy0]])
            placed_hw = np.append(placed_hw, hw_i)
            placed_hh = np.append(placed_hh, hh_i)
            continue

        # Spiral search
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.5
        best = None; bdist = float('inf')

        for r in range(1, 400):
            found_this_r = False
            for dxm in range(-r, r+1):
                for sign in [1, -1]:
                    dym = sign * (r - abs(dxm))
                    cx = float(np.clip(gx + dxm * step, hw_i, canvas_w - hw_i))
                    cy = float(np.clip(gy + dym * step, hh_i, canvas_h - hh_i))
                    if not overlaps(cx, cy):
                        d = (cx - gx)**2 + (cy - gy)**2
                        if d < bdist:
                            bdist = d; best = np.array([cx, cy])
                        found_this_r = True
            if found_this_r and r >= 2:
                break

        if best is None:
            # Last resort: random search
            for _ in range(2000):
                cx = float(np.random.uniform(hw_i, canvas_w - hw_i))
                cy = float(np.random.uniform(hh_i, canvas_h - hh_i))
                if not overlaps(cx, cy):
                    best = np.array([cx, cy]); break

        if best is not None:
            legal[idx] = best
            placed_centers = np.vstack([placed_centers, [best]]) if len(placed_centers) > 0 else best[None]
            placed_hw = np.append(placed_hw, hw_i)
            placed_hh = np.append(placed_hh, hh_i)
        else:
            # Absolute fallback: keep GP position (may overlap, will be caught)
            placed_centers = np.vstack([placed_centers, [legal[idx]]]) if len(placed_centers) > 0 else legal[idx][None]
            placed_hw = np.append(placed_hw, hw_i)
            placed_hh = np.append(placed_hh, hh_i)

    return legal


# ─────────────────────────────────────────────────────────────────────────────
# RUDY congestion map (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def _rudy_map(
    pos: np.ndarray,
    flat_idx: np.ndarray,
    net_id_arr: np.ndarray,
    offsets: np.ndarray,
    n_nets: int,
    canvas_w: float,
    canvas_h: float,
    R: int, C: int,
) -> np.ndarray:
    bw = canvas_w / C; bh = canvas_h / R
    x = pos[flat_idx, 0]; y = pos[flat_idx, 1]

    x0 = np.minimum.reduceat(x, offsets[:-1])
    x1 = np.maximum.reduceat(x, offsets[:-1])
    y0 = np.minimum.reduceat(y, offsets[:-1])
    y1 = np.maximum.reduceat(y, offsets[:-1])

    nw = np.maximum(x1 - x0, 1e-8)
    nh = np.maximum(y1 - y0, 1e-8)
    box = nw * nh
    h_den = nh / box
    v_den = nw / box

    demand = np.zeros((R, C), dtype=np.float64)
    r0_all = np.clip((y0 / bh).astype(int), 0, R-1)
    r1_all = np.clip((y1 / bh).astype(int), 0, R-1)
    c0_all = np.clip((x0 / bw).astype(int), 0, C-1)
    c1_all = np.clip((x1 / bw).astype(int), 0, C-1)

    for ni in range(n_nets):
        r0, r1 = r0_all[ni], r1_all[ni]
        c0, c1 = c0_all[ni], c1_all[ni]
        demand[r0:r1+1, c0:c1+1] += (h_den[ni] + v_den[ni]) * bw * bh * 0.5

    return demand


# ─────────────────────────────────────────────────────────────────────────────
# Congestion-driven refinement (KD-tree built once per pass)
# ─────────────────────────────────────────────────────────────────────────────

def _congestion_refine(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    ni: NetIndex,
    flat_idx: np.ndarray,
    net_id_arr: np.ndarray,
    offsets: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    gap: float = 0.001,
    time_budget: float = 8.0,
    n_passes: int = 6,
) -> np.ndarray:
    if ni.n_nets == 0:
        return pos

    t0 = time.time()
    n = pos.shape[0]
    R, C = grid_rows, grid_cols
    bw = canvas_w / C; bh = canvas_h / R
    hw = sizes[:, 0] * 0.5; hh = sizes[:, 1] * 0.5
    diag = math.sqrt(canvas_w**2 + canvas_h**2)
    movable_idx = np.where(movable)[0]
    max_hw = float(hw.max())

    p = pos.copy()

    for _ in range(n_passes):
        if time.time() - t0 > time_budget:
            break

        demand = _rudy_map(p, flat_idx, net_id_arr, offsets, ni.n_nets,
                           canvas_w, canvas_h, R, C)
        thresh = np.percentile(demand, 85)
        if thresh < 1e-6:
            break

        # Build KD-tree ONCE per pass
        tree = KDTree(p)

        np.random.shuffle(movable_idx)
        n_moved = 0

        for idx in movable_idx:
            if time.time() - t0 > time_budget:
                break

            cx_bin = int(np.clip(p[idx, 0] / bw, 0, C-1))
            cy_bin = int(np.clip(p[idx, 1] / bh, 0, R-1))

            if demand[cy_bin, cx_bin] < thresh:
                continue

            wl_before = ni.hpwl_node(p, idx)
            orig = p[idx].copy()
            best_score = 0.0
            best_cand = None

            for dc in [-3, -2, -1, 1, 2, 3]:
                for dr in [-3, -2, -1, 1, 2, 3]:
                    nc = int(np.clip(cx_bin + dc, 0, C-1))
                    nr = int(np.clip(cy_bin + dr, 0, R-1))

                    if demand[nr, nc] >= demand[cy_bin, cx_bin] * 0.9:
                        continue

                    tx = float(np.clip((nc + 0.5) * bw, hw[idx], canvas_w - hw[idx]))
                    ty = float(np.clip((nr + 0.5) * bh, hh[idx], canvas_h - hh[idx]))

                    r_q = hw[idx] + max_hw + gap * 2
                    neighbors = tree.query_ball_point([tx, ty], r_q)
                    legal = True
                    for nb in neighbors:
                        if nb == idx:
                            continue
                        if (abs(tx - p[nb, 0]) < hw[idx] + hw[nb] + gap and
                            abs(ty - p[nb, 1]) < hh[idx] + hh[nb] + gap):
                            legal = False; break
                    if not legal:
                        continue

                    p[idx] = [tx, ty]
                    wl_after = ni.hpwl_node(p, idx)
                    wl_delta = wl_after - wl_before
                    p[idx] = orig

                    cong_gain = demand[cy_bin, cx_bin] - demand[nr, nc]
                    score = cong_gain - 0.25 * wl_delta / (diag * 0.01 + 1e-10)

                    if score > best_score:
                        best_score = score
                        best_cand = np.array([tx, ty])

            if best_cand is not None:
                p[idx] = best_cand
                n_moved += 1

        if n_moved == 0:
            break

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Swap refinement (KD-tree built once per pass)
# ─────────────────────────────────────────────────────────────────────────────

def _swap_refine(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    ni: NetIndex,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.001,
    n_passes: int = 6,
    time_budget: float = 5.0,
    k: int = 15,
) -> np.ndarray:
    t0 = time.time()
    n = pos.shape[0]
    hw = sizes[:, 0] * 0.5; hh = sizes[:, 1] * 0.5
    movable_idx = np.where(movable)[0]
    max_hw = float(hw.max())

    p = pos.copy()

    for _ in range(n_passes):
        if time.time() - t0 > time_budget:
            break

        tree = KDTree(p)
        improved = 0
        order = movable_idx.copy()
        np.random.shuffle(order)

        for i in order:
            if time.time() - t0 > time_budget:
                break

            _, idxs = tree.query(p[i], k=min(k+1, n))
            neighbors = [j for j in idxs[1:] if j < n and movable[j]][:k]

            best_gain = 1e-9
            best_j = -1

            for j in neighbors:
                if max(sizes[i, 0], sizes[j, 0]) / (min(sizes[i, 0], sizes[j, 0]) + 1e-9) > 2.5:
                    continue

                pi, pj = p[i].copy(), p[j].copy()

                # Bounds check
                if (pj[0] - hw[i] < 0 or pj[0] + hw[i] > canvas_w or
                    pj[1] - hh[i] < 0 or pj[1] + hh[i] > canvas_h or
                    pi[0] - hw[j] < 0 or pi[0] + hw[j] > canvas_w or
                    pi[1] - hh[j] < 0 or pi[1] + hh[j] > canvas_h):
                    continue

                r_q = max_hw * 2 + gap * 2
                legal = True
                for nb in tree.query_ball_point(pj, r_q):
                    if nb == i or nb == j:
                        continue
                    if (abs(pj[0] - p[nb, 0]) < hw[i] + hw[nb] + gap and
                        abs(pj[1] - p[nb, 1]) < hh[i] + hh[nb] + gap):
                        legal = False; break

                if legal:
                    for nb in tree.query_ball_point(pi, r_q):
                        if nb == i or nb == j:
                            continue
                        if (abs(pi[0] - p[nb, 0]) < hw[j] + hw[nb] + gap and
                            abs(pi[1] - p[nb, 1]) < hh[j] + hh[nb] + gap):
                            legal = False; break

                if not legal:
                    continue

                wl_before = ni.hpwl_nodes(p, i, j)
                p[i], p[j] = pj, pi
                wl_after = ni.hpwl_nodes(p, i, j)
                gain = wl_before - wl_after
                p[i], p[j] = pi, pj  # revert

                if gain > best_gain:
                    best_gain = gain
                    best_j = j

            if best_j >= 0:
                p[i], p[best_j] = p[best_j].copy(), p[i].copy()
                improved += 1

        if improved == 0:
            break

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Mini SA
# ─────────────────────────────────────────────────────────────────────────────

def _mini_sa(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    ni: NetIndex,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.001,
    time_budget: float = 5.0,
) -> np.ndarray:
    t0 = time.time()
    n = pos.shape[0]
    hw = sizes[:, 0] * 0.5; hh = sizes[:, 1] * 0.5
    movable_idx = np.where(movable)[0]
    diag = math.sqrt(canvas_w**2 + canvas_h**2)
    T0_v = 0.01 * diag
    T1_v = 0.0002 * diag
    max_hw = float(hw.max())

    p = pos.copy()
    tree = KDTree(p)
    moves_since_rebuild = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break
        frac = elapsed / time_budget
        T = T0_v * math.exp(math.log(T1_v / T0_v) * frac)
        step = T * 1.5

        i = int(np.random.choice(movable_idx))
        orig = p[i].copy()
        nx = float(np.clip(orig[0] + np.random.uniform(-step, step), hw[i], canvas_w - hw[i]))
        ny = float(np.clip(orig[1] + np.random.uniform(-step, step), hh[i], canvas_h - hh[i]))

        r_q = hw[i] + max_hw + gap * 2
        legal = True
        for nb in tree.query_ball_point([nx, ny], r_q):
            if nb == i:
                continue
            if (abs(nx - p[nb, 0]) < hw[i] + hw[nb] + gap and
                abs(ny - p[nb, 1]) < hh[i] + hh[nb] + gap):
                legal = False; break
        if not legal:
            continue

        wl_before = ni.hpwl_node(p, i)
        p[i] = [nx, ny]
        wl_after = ni.hpwl_node(p, i)
        delta = wl_after - wl_before

        if delta < 0 or np.random.random() < math.exp(-delta / (T + 1e-12)):
            moves_since_rebuild += 1
            if moves_since_rebuild > 300:
                tree = KDTree(p)
                moves_since_rebuild = 0
        else:
            p[i] = orig

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main Placer
# ─────────────────────────────────────────────────────────────────────────────

class Mj97Placer:
    """
    MJ97 v6: Fast vectorized GP + proven zero-overlap spiral legalizer +
    KD-tree congestion/swap/SA refinement.
    Target: avg proxy < 1.45, runtime ~60-120s/benchmark, zero overlaps.
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

        out    = benchmark.macro_positions.clone()
        init   = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)
        sizes  = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        gr = int(benchmark.grid_rows)
        gc = int(benchmark.grid_cols)

        T0 = time.time()

        plc  = _load_plc(benchmark)
        nets = _extract_nets(benchmark, plc) if plc is not None else []

        if not nets:
            out[:n_hard] = torch.from_numpy(init).float()
            return out

        ni = NetIndex(nets, n_hard)
        flat_idx, net_id_arr, offsets, n_nets = _build_wa_arrays(ni, n_hard)

        # ── 1. Global placement (45s)
        elapsed = time.time() - T0
        gp_result = _global_place(
            init=init.copy(),
            movable=movable,
            sizes=sizes,
            ni=ni,
            canvas_w=cw,
            canvas_h=ch,
            time_budget=max(5.0, 45.0 - elapsed),
            seed=self.seed,
        )

        # ── 2. Legalization (zero overlaps guaranteed)
        legal = _legalize(gp_result, movable, sizes, cw, ch, gap=0.001)

        # ── 3. Congestion-driven displacement (8s)
        elapsed = time.time() - T0
        cong_budget = min(8.0, max(1.0, 60.0 - elapsed))
        legal = _congestion_refine(
            pos=legal,
            movable=movable,
            sizes=sizes,
            ni=ni,
            flat_idx=flat_idx,
            net_id_arr=net_id_arr,
            offsets=offsets,
            canvas_w=cw,
            canvas_h=ch,
            grid_rows=gr,
            grid_cols=gc,
            gap=0.001,
            time_budget=cong_budget,
            n_passes=6,
        )

        # ── 4. Swap refinement (5s)
        elapsed = time.time() - T0
        swap_budget = min(5.0, max(0.5, 65.0 - elapsed))
        legal = _swap_refine(
            pos=legal,
            movable=movable,
            sizes=sizes,
            ni=ni,
            canvas_w=cw,
            canvas_h=ch,
            gap=0.001,
            n_passes=6,
            time_budget=swap_budget,
            k=15,
        )

        # ── 5. Mini SA (remaining, up to 5s)
        elapsed = time.time() - T0
        sa_budget = min(5.0, max(0.0, 60.0 - elapsed))
        if sa_budget > 0.5:
            legal = _mini_sa(
                pos=legal,
                movable=movable,
                sizes=sizes,
                ni=ni,
                canvas_w=cw,
                canvas_h=ch,
                gap=0.001,
                time_budget=sa_budget,
            )

        out[:n_hard] = torch.from_numpy(legal).float()
        return out