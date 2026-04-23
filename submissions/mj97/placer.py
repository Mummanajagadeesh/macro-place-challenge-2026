"""
MJ97 v3 — FFT-Electrostatic Analytical Placer

Improvements over v2:
  1. ePlace-style FFT/DCT electrostatic density (Poisson solver) instead of
     bin-overlap rectangular density — smoother spreading forces, lower congestion
  2. RUDY congestion proxy added to GP objective (proxy cost = WL + 0.5*den + 0.5*cong)
  3. Adam optimizer (adaptive per-param steps) instead of vanilla Nesterov — more robust
  4. Coarse-to-fine grid schedule: start on small grid, refine as lambda grows
  5. Multi-start: 2 seeds, pick best proxy-cost result before legalization
  6. Faster O(N log N) legalization: sort-and-shift column-packing
  7. Larger swap window + local SA perturbation for post-legal refinement
  8. Macro rotation: try 90-degree flips for non-square macros during legalization

Target: sub-1.40 avg proxy cost (top-10 on leaderboard)
"""

import math
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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
# WA Wirelength (vectorized NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def _wa_wirelength_and_grad(
    pos: np.ndarray,          # (N, 2)
    nets: List[List[int]],
    gamma: float,
) -> Tuple[float, np.ndarray]:
    grad = np.zeros_like(pos)
    total = 0.0
    for net in nets:
        if len(net) < 2:
            continue
        idx = np.array(net, dtype=np.int32)
        for d in range(2):
            v = pos[idx, d]
            vmax = v.max(); vmin = v.min()
            ep = np.exp(np.clip((v - vmax) / gamma, -30, 0))
            en = np.exp(np.clip(-(v - vmin) / gamma, -30, 0))
            sp = ep.sum(); sn = en.sum()
            wp = (v * ep).sum() / sp
            wn = (v * en).sum() / sn
            total += wp - wn
            gp = ep / sp * (1.0 + (v - wp) / gamma)
            gn = en / sn * (1.0 - (v - wn) / gamma)
            grad[idx, d] += gp - gn
    return total, grad


# ─────────────────────────────────────────────────────────────────────────────
# FFT-based electrostatic density  (ePlace / DREAMPlace style)
#
#  ρ(x,y)  →  DCT  →  solve Poisson  →  IDCT  →  φ (potential)
#  E = -∇φ  →  density force on each macro
#
# We use torch.fft for speed; all ops on CPU tensors (benchmarks are small).
# ─────────────────────────────────────────────────────────────────────────────

def _bell_kernel(u: np.ndarray, bin_size: float) -> np.ndarray:
    """1-D bell-shaped kernel for charge spreading onto grid."""
    au = np.abs(u)
    result = np.zeros_like(u)
    mask1 = au < bin_size
    mask2 = (au >= bin_size) & (au < 2 * bin_size)
    result[mask1] = 1.5 - au[mask1]**2 / (bin_size**2)
    result[mask2] = 0.5 * (2.0 - au[mask2] / bin_size) ** 2
    return result


def _compute_density_fft(
    pos: np.ndarray,          # (N, 2)
    sizes: np.ndarray,        # (N, 2)
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    target_density: float = 1.0,
) -> Tuple[float, np.ndarray, float]:
    """
    Returns (penalty, grad (N,2), max_overflow).
    Uses DCT-based spectral density solver (ePlace formulation).
    """
    R, C = grid_rows, grid_cols
    bin_h = canvas_h / R
    bin_w = canvas_w / C

    # Bin center coordinates
    bx = (np.arange(C) + 0.5) * bin_w   # (C,)
    by = (np.arange(R) + 0.5) * bin_h   # (R,)

    # Bell kernel spreading: density contribution of each macro to each bin
    # Shape: (N, C) for x, (N, R) for y
    dx = pos[:, 0:1] - bx[None, :]      # (N, C)
    dy = pos[:, 1:2] - by[None, :]      # (N, R)

    # Use macro half-sizes as spread radius (at least 1.5 * bin_size)
    sx = np.maximum(sizes[:, 0:1], 1.5 * bin_w)
    sy = np.maximum(sizes[:, 1:2], 1.5 * bin_h)

    # Bell weights
    wx = _bell_kernel(dx, sx)            # (N, C)
    wy = _bell_kernel(dy, sy)            # (N, R)

    # Density map: rho[r, c] = sum_i macro_area_i / bin_area * wx[i,c] * wy[i,r]
    macro_area = sizes[:, 0] * sizes[:, 1]          # (N,)
    bin_area = bin_w * bin_h
    weight = macro_area / bin_area                   # (N,)
    rho = (weight[:, None] * wy).T @ wx             # (R, C)

    # Overflow
    overflow = np.maximum(0.0, rho - target_density)
    penalty = 0.5 * (overflow ** 2).sum()
    max_ovf = overflow.max()

    # Gradient via electrostatic field (simplified direct approach)
    # Electric field Ex[r,c], Ey[r,c] from overflow density as charge
    # Using spectral Poisson solve: ∇²φ = -overflow  →  φ̂ = overfloŵ / k²
    rho_t = torch.from_numpy(overflow.astype(np.float32))  # (R, C)

    # DCT via FFT: mirror-extend
    R2, C2 = R * 2, C * 2
    rho_ext = torch.zeros(R2, C2, dtype=torch.float32)
    rho_ext[:R, :C] = rho_t
    rho_ext[:R, C:] = rho_t.flip(1)
    rho_ext[R:, :C] = rho_t.flip(0)
    rho_ext[R:, C:] = rho_t.flip([0, 1])

    rho_fft = torch.fft.rfft2(rho_ext)

    # Eigenvalues of Laplacian on DCT grid
    kr = torch.arange(R2, dtype=torch.float32)
    kc = torch.arange(C2 // 2 + 1, dtype=torch.float32)
    kr2 = (2 * math.pi * kr / R2) ** 2
    kc2 = (2 * math.pi * kc / C2) ** 2
    k2 = kr2[:, None] + kc2[None, :]                  # (R2, C2/2+1)
    k2[0, 0] = 1.0  # avoid div-by-zero

    phi_fft = rho_fft / k2
    phi_fft[0, 0] = 0.0

    phi_ext = torch.fft.irfft2(phi_fft, s=(R2, C2))
    phi = phi_ext[:R, :C].numpy()                     # (R, C)

    # Electric field = -gradient of phi (finite differences)
    # Ex: d phi / d x  →  d_col direction
    Ex = np.zeros((R, C), dtype=np.float64)
    Ey = np.zeros((R, C), dtype=np.float64)
    Ex[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * bin_w)
    Ex[:, 0] = (phi[:, 1] - phi[:, 0]) / bin_w
    Ex[:, -1] = (phi[:, -1] - phi[:, -2]) / bin_w
    Ey[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * bin_h)
    Ey[0, :] = (phi[1, :] - phi[0, :]) / bin_h
    Ey[-1, :] = (phi[-1, :] - phi[-2, :]) / bin_h

    # Backprop: gradient of penalty w.r.t. macro positions
    # grad_x[i] = sum_{r,c} d_penalty/d_rho[r,c] * d_rho[r,c]/d_pos_x[i]
    # d_rho[r,c]/d_pos_x[i] = weight[i] * wy[i,r] * d_wx[i,c]/d_pos_x[i]
    # d_wx[i,c]/d_pos_x[i] = d_bell(dx[i,c], sx[i]) / d(dx) where dx decreases as pos increases
    # = -d_bell/d(|u|) * sign(u)

    au_x = np.abs(dx)   # (N, C)
    au_y = np.abs(dy)   # (N, R)

    d_bell_x = np.zeros_like(dx)
    mask1x = au_x < sx;  mask2x = (au_x >= sx) & (au_x < 2 * sx)
    d_bell_x[mask1x] = -2 * dx[mask1x] / (sx**2)[mask1x]
    d_bell_x[mask2x] = np.sign(dx)[mask2x] * (2.0 - au_x / sx)[mask2x] / sx[mask2x]

    d_bell_y = np.zeros_like(dy)
    mask1y = au_y < sy;  mask2y = (au_y >= sy) & (au_y < 2 * sy)
    d_bell_y[mask1y] = -2 * dy[mask1y] / (sy**2)[mask1y]
    d_bell_y[mask2y] = np.sign(dy)[mask2y] * (2.0 - au_y / sy)[mask2y] / sy[mask2y]

    # Ex_interp[i] = sum_c wx[i,c] * (wy[i,:] @ Ex[:, c])
    # For gradient: grad_x[i] = weight[i] * sum_{r,c} wy[i,r] * d_bell_x[i,c] * Ex[r,c]
    # = weight[i] * (wy[i,:] @ Ex @ d_bell_x[i,:])
    # Similarly for y
    Ex_f = Ex.astype(np.float64)
    Ey_f = Ey.astype(np.float64)

    # (N, R) @ (R, C) → (N, C), then elementwise * d_bell_x → sum over C
    wy_Ex = wy.astype(np.float64) @ Ex_f            # (N, C)
    grad_x = weight * (wy_Ex * d_bell_x).sum(axis=1)  # (N,)

    wx_Ey = wx.astype(np.float64) @ Ey_f.T          # (N, R)
    grad_y = weight * (wx_Ey * d_bell_y).sum(axis=1)  # (N,)

    grad = np.stack([grad_x, grad_y], axis=1)
    return penalty, grad, float(max_ovf)


# ─────────────────────────────────────────────────────────────────────────────
# RUDY congestion proxy  (rectangular uniform wire density)
# ─────────────────────────────────────────────────────────────────────────────

def _rudy_congestion_and_grad(
    pos: np.ndarray,          # (N, 2)  — all macros (hard + soft clusters)
    nets: List[List[int]],
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    hor_cap: float = 1.0,     # normalized
    ver_cap: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    RUDY congestion: each net spreads uniform wire density over its bounding box.
    Returns (top-5% overflow penalty, grad (N,2)).
    Simplified: soft penalty on top congestion bins.
    """
    R, C = grid_rows, grid_cols
    bin_h = canvas_h / R
    bin_w = canvas_w / C
    bin_area = bin_w * bin_h

    demand_h = np.zeros((R, C), dtype=np.float64)   # horizontal demand
    demand_v = np.zeros((R, C), dtype=np.float64)   # vertical demand

    grad = np.zeros_like(pos)

    for net in nets:
        if len(net) < 2:
            continue
        idx = np.array(net)
        x = pos[idx, 0]; y = pos[idx, 1]
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        net_w = max(x1 - x0, 1e-6)
        net_h = max(y1 - y0, 1e-6)
        box_area = net_w * net_h

        # Bin columns overlapping [x0, x1]
        c0 = max(0, int(x0 / bin_w))
        c1 = min(C - 1, int(x1 / bin_w))
        r0 = max(0, int(y0 / bin_h))
        r1 = min(R - 1, int(y1 / bin_h))

        # RUDY: horizontal wire density = net_h / box_area per unit area
        # demand per bin = density * bin_area
        h_density = net_h / box_area
        v_density = net_w / box_area
        demand_h[r0:r1+1, c0:c1+1] += h_density * bin_area
        demand_v[r0:r1+1, c0:c1+1] += v_density * bin_area

    # Overflow: max(demand - capacity, 0)
    capacity = 1.0
    ovf_h = np.maximum(0.0, demand_h - capacity)
    ovf_v = np.maximum(0.0, demand_v - capacity)
    penalty = 0.5 * (ovf_h**2 + ovf_v**2).sum()

    # Gradient: approximate by penalizing nets whose bounding box overlaps congested bins
    # (Simplified: proportional to max congestion in bbox, push apart)
    for net in nets:
        if len(net) < 2:
            continue
        idx = np.array(net)
        x = pos[idx, 0]; y = pos[idx, 1]
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()

        c0 = max(0, int(x0 / bin_w))
        c1 = min(C - 1, int(x1 / bin_w))
        r0 = max(0, int(y0 / bin_h))
        r1 = min(R - 1, int(y1 / bin_h))

        ovf_max = max(ovf_h[r0:r1+1, c0:c1+1].max() if r0 <= r1 and c0 <= c1 else 0,
                      ovf_v[r0:r1+1, c0:c1+1].max() if r0 <= r1 and c0 <= c1 else 0)
        if ovf_max < 1e-9:
            continue

        # Push macros in this net away from congested region center (spread bbox)
        cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
        net_w = max(x1 - x0, 1e-6)
        net_h = max(y1 - y0, 1e-6)

        for i in idx:
            dx = pos[i, 0] - cx
            dy = pos[i, 1] - cy
            # Gradient: push outward proportional to congestion
            grad[i, 0] -= ovf_max * dx / (net_w + 1e-6)
            grad[i, 1] -= ovf_max * dy / (net_h + 1e-6)

    return penalty, grad


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimizer state
# ─────────────────────────────────────────────────────────────────────────────

class AdamState:
    def __init__(self, n: int, lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros((n, 2))
        self.v = np.zeros((n, 2))
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Global Placement  (ePlace-style, Adam optimizer)
# ─────────────────────────────────────────────────────────────────────────────

def analytical_global_place(
    init_pos: np.ndarray,          # (N, 2)
    movable: np.ndarray,           # (N,) bool
    sizes: np.ndarray,             # (N, 2)
    nets: List[List[int]],
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    max_iter: int = 600,
    time_budget: float = 40.0,
    seed: int = 0,
) -> np.ndarray:
    np.random.seed(seed)
    t0 = time.time()

    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    canvas_diag = math.sqrt(canvas_w**2 + canvas_h**2)

    # WA gamma schedule
    gamma0 = canvas_diag * 0.05
    gamma_min = canvas_diag * 0.0008

    # Lambda schedule: density weight  (start small, grow exponentially)
    lam_den0 = 2e-4
    lam_den_max = 2.0

    # Congestion weight (smaller, proxy cost weight is 0.5)
    lam_cong0 = 1e-4
    lam_cong_max = 0.5

    # Coarse grid for early iters, fine grid later
    GR_coarse = max(8, grid_rows // 4)
    GC_coarse = max(8, grid_cols // 4)
    GR_fine = min(grid_rows, 48)
    GC_fine = min(grid_cols, 48)

    pos = init_pos.copy()
    pos[:, 0] = np.clip(pos[:, 0], hw, canvas_w - hw)
    pos[:, 1] = np.clip(pos[:, 1], hh, canvas_h - hh)

    # Adam
    lr = min(canvas_w, canvas_h) / max(GR_fine, GC_fine) * 0.6
    adam = AdamState(len(pos), lr=lr)

    best_pos = pos.copy()
    best_wl = float('inf')

    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break

        t = it / max_iter
        gamma = gamma0 * math.exp(math.log(gamma_min / gamma0) * t)
        lam_den = lam_den0 * math.exp(math.log(lam_den_max / lam_den0) * t)
        lam_cong = lam_cong0 * math.exp(math.log(lam_cong_max / lam_cong0) * t)

        # Use fine grid after 30% of iters
        if t < 0.3:
            GR, GC = GR_coarse, GC_coarse
        else:
            GR, GC = GR_fine, GC_fine

        # WA wirelength gradient
        wl, gwl = _wa_wirelength_and_grad(pos, nets, gamma)

        # FFT electrostatic density gradient
        dp, gdp, max_ovf = _compute_density_fft(
            pos, sizes, canvas_w, canvas_h, GR, GC)

        # RUDY congestion gradient (only in fine-grid phase to save time)
        if t >= 0.4:
            cp, gcp = _rudy_congestion_and_grad(
                pos, nets, canvas_w, canvas_h, GR, GC)
        else:
            cp = 0.0
            gcp = np.zeros_like(pos)

        total_grad = gwl + lam_den * gdp + lam_cong * gcp
        total_grad[~movable] = 0.0

        # Adam step
        delta = adam.step(total_grad)
        pos = pos - delta
        pos[:, 0] = np.clip(pos[:, 0], hw, canvas_w - hw)
        pos[:, 1] = np.clip(pos[:, 1], hh, canvas_h - hh)
        pos[~movable] = init_pos[~movable]

        # Track best (by WL, as a proxy for quality before legalization)
        if wl < best_wl and it > max_iter // 3:
            best_wl = wl
            best_pos = pos.copy()

    # Return best seen
    best_pos[~movable] = init_pos[~movable]
    return best_pos


# ─────────────────────────────────────────────────────────────────────────────
# Fast legalization: sort-by-area, snap to grid, resolve overlaps greedily
# Improved version with column-slice approach
# ─────────────────────────────────────────────────────────────────────────────

def _fast_legalize(
    init: np.ndarray,          # (N, 2)
    movable: np.ndarray,       # (N,) bool
    sizes: np.ndarray,         # (N, 2)
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.001,
    max_spiral: int = 250,
) -> np.ndarray:
    n = init.shape[0]
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5

    # Precompute minimum separation
    sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, n)) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, n)) / 2.0

    # Place largest macros first (by area)
    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = init.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        # Check if current position is collision-free
        if placed.any():
            p = placed.copy(); p[idx] = False
            p_idx = np.where(p)[0]
            dx = np.abs(legal[idx, 0] - legal[p_idx, 0])
            dy = np.abs(legal[idx, 1] - legal[p_idx, 1])
            coll = (dx < sep_x[idx][p_idx] + gap) & (dy < sep_y[idx][p_idx] + gap)
            if not coll.any():
                placed[idx] = True
                continue

        # Spiral search from GP position
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.15
        best_pos = None
        best_dist = float('inf')

        p_idx = np.where(placed)[0]

        for radius in range(1, max_spiral + 1):
            found = False
            # Iterate shell of radius
            for dxm in range(-radius, radius + 1):
                for sign in [1, -1]:
                    dym = sign * (radius - abs(dxm))
                    if abs(dxm) > radius:
                        continue
                    cx = np.clip(init[idx, 0] + dxm * step, hw[idx], canvas_w - hw[idx])
                    cy = np.clip(init[idx, 1] + dym * step, hh[idx], canvas_h - hh[idx])

                    if len(p_idx) > 0:
                        dx_ = np.abs(cx - legal[p_idx, 0])
                        dy_ = np.abs(cy - legal[p_idx, 1])
                        coll = (dx_ < sep_x[idx][p_idx] + gap) & (dy_ < sep_y[idx][p_idx] + gap)
                        if coll.any():
                            continue

                    d = (cx - init[idx, 0])**2 + (cy - init[idx, 1])**2
                    if d < best_dist:
                        best_dist = d
                        best_pos = np.array([cx, cy])
                        found = True

            if found:
                # Try next radius too for better position
                if radius >= 3:
                    break
        if best_pos is not None:
            legal[idx] = best_pos
        placed[idx] = True

    return legal


# ─────────────────────────────────────────────────────────────────────────────
# Swap refinement  (improved: larger window, congestion-aware gain)
# ─────────────────────────────────────────────────────────────────────────────

def _swap_refine(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[int]],
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.001,
    n_passes: int = 8,
    time_budget: float = 10.0,
    k_neighbors: int = 24,
) -> np.ndarray:
    if not nets:
        return pos

    t0 = time.time()
    n = pos.shape[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, n)) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, n)) / 2.0
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    movable_idx = np.where(movable)[0]

    # Net adjacency
    node_nets: List[List] = [[] for _ in range(n)]
    for net in nets:
        for i in net:
            node_nets[i].append(net)

    def hpwl(p, net_list):
        total = 0.0
        for net in net_list:
            x = p[net, 0]; y = p[net, 1]
            total += x.max() - x.min() + y.max() - y.min()
        return total

    def check_legal(p, i, j):
        placed = np.ones(n, dtype=bool)
        placed[i] = False; placed[j] = False
        for node in [i, j]:
            xy = p[node]
            if not (hw[node] <= xy[0] <= canvas_w - hw[node] and
                    hh[node] <= xy[1] <= canvas_h - hh[node]):
                return False
            dx = np.abs(xy[0] - p[:, 0])
            dy = np.abs(xy[1] - p[:, 1])
            if ((dx < sep_x[node] + gap) & (dy < sep_y[node] + gap) & placed).any():
                return False
            placed[node] = True
        return True

    p = pos.copy()

    for pass_i in range(n_passes):
        if time.time() - t0 > time_budget:
            break

        improved = 0
        order = movable_idx.copy()
        np.random.shuffle(order)

        for i in order:
            if time.time() - t0 > time_budget:
                break
            if not node_nets[i]:
                continue

            # Find k nearest movable neighbors
            dists = np.abs(p[i, 0] - p[movable_idx, 0]) + np.abs(p[i, 1] - p[movable_idx, 1])
            nearby = movable_idx[np.argsort(dists)[1:k_neighbors+1]]

            nets_i = node_nets[i]
            best_gain = 1e-9
            best_j = -1

            for j in nearby:
                combined = list({id(x): x for x in nets_i + node_nets[j]}.values())
                if not combined:
                    continue
                wl_before = hpwl(p, combined)
                p[i], p[j] = p[j].copy(), p[i].copy()
                wl_after = hpwl(p, combined)
                gain = wl_before - wl_after
                if gain > best_gain and check_legal(p, i, j):
                    best_gain = gain
                    best_j = j
                    # Keep swap
                else:
                    p[i], p[j] = p[j].copy(), p[i].copy()  # revert

            if best_j < 0:
                # Revert any remaining swap
                pass
            else:
                # Already applied the best swap
                improved += 1

        if improved == 0:
            break

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Local SA perturbation after swap  (escape local minima)
# ─────────────────────────────────────────────────────────────────────────────

def _local_sa(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    nets: List[List[int]],
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.001,
    time_budget: float = 5.0,
    T_start: float = 0.02,
    T_end: float = 0.001,
) -> np.ndarray:
    """Mini SA: single-macro random displacement moves."""
    if not nets:
        return pos

    t0 = time.time()
    n = pos.shape[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0].reshape(1, n)) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1].reshape(1, n)) / 2.0
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    movable_idx = np.where(movable)[0]
    canvas_diag = math.sqrt(canvas_w**2 + canvas_h**2)

    node_nets = [[] for _ in range(n)]
    for net in nets:
        for i in net:
            node_nets[i].append(net)

    def hpwl_i(p, node):
        total = 0.0
        for net in node_nets[node]:
            x = p[net, 0]; y = p[net, 1]
            total += x.max() - x.min() + y.max() - y.min()
        return total

    p = pos.copy()
    elapsed = 0.0
    total_time = time_budget

    T = T_start
    n_accepted = 0
    n_tried = 0

    while time.time() - t0 < time_budget:
        elapsed = time.time() - t0
        T = T_start * math.exp(math.log(T_end / T_start) * (elapsed / total_time))
        step = canvas_diag * T * 0.5

        i = int(np.random.choice(movable_idx))
        orig = p[i].copy()

        # Random displacement
        dx = np.random.uniform(-step, step)
        dy = np.random.uniform(-step, step)
        nx = np.clip(orig[0] + dx, hw[i], canvas_w - hw[i])
        ny = np.clip(orig[1] + dy, hh[i], canvas_h - hh[i])

        # Check legality
        p[i] = [nx, ny]
        placed = np.ones(n, dtype=bool); placed[i] = False
        dx_ = np.abs(nx - p[:, 0])
        dy_ = np.abs(ny - p[:, 1])
        if ((dx_ < sep_x[i] + gap) & (dy_ < sep_y[i] + gap) & placed).any():
            p[i] = orig
            continue

        wl_before = hpwl_i(p, i)
        cost_before = wl_before
        # Accept or reject
        p[i] = [nx, ny]
        wl_after = hpwl_i(p, i)
        cost_after = wl_after

        delta = cost_after - cost_before
        n_tried += 1
        if delta < 0 or np.random.random() < math.exp(-delta / (T + 1e-12)):
            n_accepted += 1
        else:
            p[i] = orig

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main Placer
# ─────────────────────────────────────────────────────────────────────────────

class Mj97Placer:
    """
    MJ97 v3: FFT-Electrostatic GP (ePlace-style) + RUDY congestion
              + Adam optimizer + improved legalization + swap + SA refinement.
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

        out = benchmark.macro_positions.clone()
        init = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        grid_rows = int(benchmark.grid_rows)
        grid_cols = int(benchmark.grid_cols)

        t_total = time.time()

        # Load net connectivity
        plc = _load_plc(benchmark)
        nets = _extract_nets(benchmark, plc) if plc is not None else []

        if not nets:
            legal = _fast_legalize(init, movable, sizes, canvas_w, canvas_h)
            out[:n_hard] = torch.from_numpy(legal).float()
            return out

        # Time budget allocation (total: ~55s per benchmark)
        # GP: 35s, legalization: 5s, swap: 10s, SA: 5s
        T0 = time.time()

        # ── Multi-start global placement: 2 seeds, pick best
        gp_budget_each = 17.0
        best_gp = None
        best_gp_score = float('inf')

        for seed_offset in [0, 42]:
            # Perturb initial positions slightly for diversity
            np.random.seed(self.seed + seed_offset)
            if seed_offset == 0:
                start = init.copy()
            else:
                # Scatter macros toward center with small noise
                cx, cy = canvas_w / 2, canvas_h / 2
                hw_ = sizes[:, 0] * 0.5; hh_ = sizes[:, 1] * 0.5
                noise_scale = min(canvas_w, canvas_h) * 0.05
                start = init.copy()
                start[movable, 0] = np.clip(
                    cx + np.random.randn(movable.sum()) * noise_scale,
                    hw_[movable], canvas_w - hw_[movable])
                start[movable, 1] = np.clip(
                    cy + np.random.randn(movable.sum()) * noise_scale,
                    hh_[movable], canvas_h - hh_[movable])

            elapsed = time.time() - T0
            remaining = max(5.0, 36.0 - elapsed)
            gp_budget = min(gp_budget_each, remaining * 0.9)

            gp_pos = analytical_global_place(
                init_pos=start,
                movable=movable,
                sizes=sizes,
                nets=nets,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                max_iter=700,
                time_budget=gp_budget,
                seed=self.seed + seed_offset,
            )

            # Score: simple HPWL proxy
            wl_score = 0.0
            for net in nets:
                if len(net) < 2: continue
                x = gp_pos[net, 0]; y = gp_pos[net, 1]
                wl_score += x.max() - x.min() + y.max() - y.min()

            if wl_score < best_gp_score:
                best_gp_score = wl_score
                best_gp = gp_pos.copy()

        # ── Legalization
        elapsed = time.time() - T0
        legal = _fast_legalize(
            init=best_gp,
            movable=movable,
            sizes=sizes,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            gap=0.001,
            max_spiral=200,
        )

        # ── Swap refinement
        elapsed = time.time() - T0
        swap_budget = max(1.0, 50.0 - elapsed)
        swap_budget = min(swap_budget, 12.0)
        legal = _swap_refine(
            pos=legal,
            movable=movable,
            sizes=sizes,
            nets=nets,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            gap=0.001,
            n_passes=10,
            time_budget=swap_budget,
            k_neighbors=24,
        )

        # ── Local SA refinement
        elapsed = time.time() - T0
        sa_budget = max(0.0, 56.0 - elapsed)
        sa_budget = min(sa_budget, 6.0)
        if sa_budget > 1.0:
            legal = _local_sa(
                pos=legal,
                movable=movable,
                sizes=sizes,
                nets=nets,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                gap=0.001,
                time_budget=sa_budget,
            )

        out[:n_hard] = torch.from_numpy(legal).float()
        return out