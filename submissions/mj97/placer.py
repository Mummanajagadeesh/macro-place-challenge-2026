"""
MJ97 v2 — Analytical Global Placer

Pipeline:
  1. Load net connectivity from PlacementCost (via benchmark name)
  2. Analytical Global Placement:
       - Weighted-Average (WA) wirelength (ePlace/RePlAce style)
       - Bin density penalty (vectorized)
       - Nesterov's accelerated gradient descent
  3. Minimal-displacement legalization (v1, preserved)
  4. Post-legalization pairwise swap refinement

Target: beat RePlAce baseline (1.4578 avg proxy cost)
"""

import os
import random
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Net extraction from PlacementCost
# ─────────────────────────────────────────────────────────────────────────────

def _extract_nets(benchmark: Benchmark, plc) -> list:
    """Returns nets as list of lists of tensor-index ints."""
    name2idx = {name: i for i, name in enumerate(benchmark.macro_names)}
    nets = []

    # Style A: plc.nets[i].pins[j].macro.get_name()
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

    return nets


def _load_plc(benchmark: Benchmark):
    """Load the PlacementCost object for this benchmark."""
    try:
        from macro_place.loader import load_benchmark_from_dir
        bench_name = benchmark.name
        candidates = [
            f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}",
            f"../external/MacroPlacement/Testcases/ICCAD04/{bench_name}",
            f"external/MacroPlacement/Flows/NanGate45/{bench_name}",
            f"../external/MacroPlacement/Flows/NanGate45/{bench_name}",
        ]
        for path in candidates:
            if os.path.isdir(path):
                _, plc = load_benchmark_from_dir(path)
                return plc
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# WA Wirelength + gradient  (ePlace/RePlAce style)
# ─────────────────────────────────────────────────────────────────────────────

def _wa_wirelength_grad(pos: np.ndarray, nets: list, gamma: float):
    """
    Weighted-average wirelength and gradient.
    Uses LSE approximation: WL_e ~ log(sum exp(x/gamma)) - log(sum exp(-x/gamma))
    """
    grad = np.zeros_like(pos)
    total_wl = 0.0

    for net in nets:
        if len(net) < 2:
            continue
        idx = np.array(net, dtype=np.int32)
        for dim in range(2):
            v = pos[idx, dim]
            vmax = v.max()
            vmin = v.min()
            ep = np.exp((v - vmax) / gamma)
            en = np.exp(-(v - vmin) / gamma)
            sp = ep.sum()
            sn = en.sum()
            wp = (v * ep).sum() / sp
            wn = (v * en).sum() / sn
            total_wl += wp - wn
            # WA gradient
            gp = ep / sp * (1.0 + (v - wp) / gamma)
            gn = en / sn * (1.0 - (v - wn) / gamma)
            grad[idx, dim] += gp - gn

    return total_wl, grad


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized Bin Density Penalty + gradient
# ─────────────────────────────────────────────────────────────────────────────

def _density_grad(pos: np.ndarray, sizes: np.ndarray,
                  canvas_w: float, canvas_h: float,
                  grid_rows: int, grid_cols: int,
                  target_density: float = 1.0):
    """
    Computes bin density overflow penalty and gradient via rectangular overlap.
    Returns (penalty, grad (N,2), max_density)
    """
    bin_w = canvas_w / grid_cols
    bin_h = canvas_h / grid_rows
    bin_area = bin_w * bin_h

    px = pos[:, 0]
    py = pos[:, 1]
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5

    bx = (np.arange(grid_cols) + 0.5) * bin_w   # (C,)
    by_ = (np.arange(grid_rows) + 0.5) * bin_h  # (R,)

    # Overlap x: (N, C)
    lx = np.maximum(px[:, None] - hw[:, None], bx[None, :] - bin_w * 0.5)
    rx = np.minimum(px[:, None] + hw[:, None], bx[None, :] + bin_w * 0.5)
    ox = np.maximum(0.0, rx - lx)

    # Overlap y: (N, R)
    ly = np.maximum(py[:, None] - hh[:, None], by_[None, :] - bin_h * 0.5)
    ry_ = np.minimum(py[:, None] + hh[:, None], by_[None, :] + bin_h * 0.5)
    oy = np.maximum(0.0, ry_ - ly)

    # Density: (R, C)
    density = (oy.T @ ox) / bin_area

    overflow = np.maximum(0.0, density - target_density)
    penalty = 0.5 * (overflow ** 2).sum()

    # Gradient
    drx = ((px[:, None] + hw[:, None]) < (bx[None, :] + bin_w * 0.5 - 1e-12)).astype(np.float64)
    dlx = ((px[:, None] - hw[:, None]) > (bx[None, :] - bin_w * 0.5 + 1e-12)).astype(np.float64)
    dox = (drx - dlx) * (ox > 0)

    dry = ((py[:, None] + hh[:, None]) < (by_[None, :] + bin_h * 0.5 - 1e-12)).astype(np.float64)
    dly = ((py[:, None] - hh[:, None]) > (by_[None, :] - bin_h * 0.5 + 1e-12)).astype(np.float64)
    doy = (dry - dly) * (oy > 0)

    oy_ovf = oy @ overflow        # (N, C)
    grad_x = (dox * oy_ovf).sum(axis=1) / bin_area

    ox_ovf = ox @ overflow.T      # (N, R)
    grad_y = (doy * ox_ovf).sum(axis=1) / bin_area

    grad = np.stack([grad_x, grad_y], axis=1)
    return penalty, grad, density.max()


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Global Placement  (Nesterov's method)
# ─────────────────────────────────────────────────────────────────────────────

def analytical_global_place(
    init_pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    nets: list,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    max_iter: int = 500,
    time_budget: float = 45.0,
) -> np.ndarray:
    """
    Nesterov's accelerated gradient descent.
    Objective: WA_wirelength + lambda * density_penalty
    """
    t0 = time.time()
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5

    pos = init_pos.copy()
    pos[:, 0] = np.clip(pos[:, 0], hw, canvas_w - hw)
    pos[:, 1] = np.clip(pos[:, 1], hh, canvas_h - hh)

    canvas_diag = np.sqrt(canvas_w**2 + canvas_h**2)
    gamma0 = canvas_diag * 0.04
    gamma_min = canvas_diag * 0.001
    lam0 = 5e-4
    lam_max = 4.0

    GR = min(grid_rows, 32)
    GC = min(grid_cols, 32)

    step = min(canvas_w / GC, canvas_h / GR) * 0.4
    step_max = min(canvas_w / GC, canvas_h / GR) * 2.0

    v = pos.copy()
    alpha = 1.0

    def grad_fn(p, lam, gamma):
        wl, gwl = _wa_wirelength_grad(p, nets, gamma)
        dp, gd, max_d = _density_grad(p, sizes, canvas_w, canvas_h, GR, GC)
        g = gwl + lam * gd
        g[~movable] = 0.0
        return wl + lam * dp, g, max_d

    _, g_prev, _ = grad_fn(pos, lam0, gamma0)

    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break

        t = it / max_iter
        gamma = gamma0 * (1.0 - t) + gamma_min * t
        lam = lam0 * np.exp(np.log(lam_max / lam0) * t)

        _, gv, max_d = grad_fn(v, lam, gamma)

        new_pos = v - step * gv
        new_pos[:, 0] = np.clip(new_pos[:, 0], hw, canvas_w - hw)
        new_pos[:, 1] = np.clip(new_pos[:, 1], hh, canvas_h - hh)
        new_pos[~movable] = init_pos[~movable]

        alpha_new = (1.0 + np.sqrt(1.0 + 4.0 * alpha**2)) / 2.0
        beta = (alpha - 1.0) / alpha_new
        v = new_pos + beta * (new_pos - pos)
        v[:, 0] = np.clip(v[:, 0], hw, canvas_w - hw)
        v[:, 1] = np.clip(v[:, 1], hh, canvas_h - hh)
        v[~movable] = init_pos[~movable]

        pos = new_pos
        alpha = alpha_new

        _, g_new, _ = grad_fn(pos, lam, gamma)
        dot = (g_new * g_prev).sum()
        if dot < 0:
            step = max(step * 0.80, step * 0.1)
        else:
            step = min(step * 1.04, step_max)
        g_prev = g_new

    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Legalization (v1, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _legalize(
    init: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.0008,
    step_mul: float = 0.18,
    max_radius: int = 300,
) -> np.ndarray:
    n = init.shape[0]
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = init.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            coll = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
            coll[idx] = False
            if not coll.any():
                placed[idx] = True
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * step_mul
        best_pos = legal[idx].copy()
        best_dist = float("inf")

        for radius in range(1, max_radius + 1):
            found = False
            for dxm in range(-radius, radius + 1):
                for dym in range(-radius, radius + 1):
                    if abs(dxm) != radius and abs(dym) != radius:
                        continue
                    cx = np.clip(init[idx, 0] + dxm * step, half_w[idx], canvas_w - half_w[idx])
                    cy = np.clip(init[idx, 1] + dym * step, half_h[idx], canvas_h - half_h[idx])
                    if placed.any():
                        dx_ = np.abs(cx - legal[:, 0])
                        dy_ = np.abs(cy - legal[:, 1])
                        coll = (dx_ < sep_x[idx] + gap) & (dy_ < sep_y[idx] + gap) & placed
                        coll[idx] = False
                        if coll.any():
                            continue
                    d = (cx - init[idx, 0])**2 + (cy - init[idx, 1])**2
                    if d < best_dist:
                        best_dist = d
                        best_pos = np.array([cx, cy], dtype=np.float64)
                        found = True
            if found:
                break

        legal[idx] = best_pos
        placed[idx] = True

    return legal


# ─────────────────────────────────────────────────────────────────────────────
# Post-legalization swap refinement
# ─────────────────────────────────────────────────────────────────────────────

def _swap_refine(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    nets: list,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.0008,
    n_passes: int = 5,
    time_budget: float = 8.0,
) -> np.ndarray:
    """
    Greedy pairwise swap: swap i,j if HPWL decreases and result stays legal.
    """
    if not nets:
        return pos

    t0 = time.time()
    n = pos.shape[0]
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    movable_idx = np.where(movable)[0]

    # net membership per node
    node_nets = [[] for _ in range(n)]
    for net in nets:
        for i in net:
            node_nets[i].append(net)

    def hpwl_nets(p, net_list):
        total = 0.0
        for net in net_list:
            x = p[net, 0]
            y = p[net, 1]
            total += x.max() - x.min() + y.max() - y.min()
        return total

    def legal_swap(p, i, j):
        placed = np.ones(n, dtype=bool)
        placed[i] = False
        placed[j] = False
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

    for _ in range(n_passes):
        if time.time() - t0 > time_budget:
            break
        order = movable_idx.copy()
        np.random.shuffle(order)

        for i in order:
            if time.time() - t0 > time_budget:
                break

            dists = np.abs(p[i, 0] - p[movable_idx, 0]) + np.abs(p[i, 1] - p[movable_idx, 1])
            nearby_k = np.argsort(dists)[1:16]
            neighbors = movable_idx[nearby_k]

            nets_i = node_nets[i]
            best_gain = 1e-9
            best_j = -1

            for j in neighbors:
                all_nets = list({id(x): x for x in (nets_i + node_nets[j])}.values())
                if not all_nets:
                    continue

                wl_before = hpwl_nets(p, all_nets)
                p_tmp = p.copy()
                p_tmp[i], p_tmp[j] = p[j].copy(), p[i].copy()
                wl_after = hpwl_nets(p_tmp, all_nets)

                gain = wl_before - wl_after
                if gain > best_gain and legal_swap(p_tmp, i, j):
                    best_gain = gain
                    best_j = j

            if best_j >= 0:
                p[i], p[best_j] = p[best_j].copy(), p[i].copy()

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main Placer
# ─────────────────────────────────────────────────────────────────────────────

class Mj97Placer:
    """
    MJ97 v2: Analytical GP (Nesterov/WA/density) + legalization + swap refinement.
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

        # 1. Load net connectivity
        plc = _load_plc(benchmark)
        nets = _extract_nets(benchmark, plc) if plc is not None else []

        if not nets:
            # Fallback: v1 legalizer only (no connectivity)
            legal = _legalize(init, movable, sizes, canvas_w, canvas_h)
            out[:n_hard] = torch.from_numpy(legal).to(dtype=torch.float32)
            return out

        # 2. Analytical global placement
        gp_budget = max(5.0, 50.0 - (time.time() - t_total) - 12.0)
        gp_pos = analytical_global_place(
            init_pos=init.copy(),
            movable=movable,
            sizes=sizes,
            nets=nets,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            max_iter=600,
            time_budget=gp_budget,
        )

        # 3. Legalization
        legal = _legalize(
            init=gp_pos,
            movable=movable,
            sizes=sizes,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            gap=0.0008,
            step_mul=0.18,
            max_radius=300,
        )

        # 4. Swap refinement
        elapsed = time.time() - t_total
        swap_budget = max(0.0, 58.0 - elapsed)
        if swap_budget > 1.0:
            legal = _swap_refine(
                pos=legal,
                movable=movable,
                sizes=sizes,
                nets=nets,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                gap=0.0008,
                n_passes=5,
                time_budget=swap_budget,
            )

        out[:n_hard] = torch.from_numpy(legal).to(dtype=torch.float32)
        return out