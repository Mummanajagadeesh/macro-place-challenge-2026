"""
MJ97 Placer - Aggressive ePlace-Lite style legalizer.

Keeps the strong initial analytical placement and performs a very
small-displacement hard-macro legalization pass.

Usage:
    uv run evaluate submissions/mj97/placer.py
    uv run evaluate submissions/mj97/placer.py --all
"""

import random

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class Mj97Placer:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return benchmark.macro_positions.clone()

        placement = benchmark.macro_positions.clone()
        pos = placement[:n_hard].cpu().numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        pos = self._legalize_min_displacement(
            pos,
            movable,
            sizes,
            half_w,
            half_h,
            canvas_w,
            canvas_h,
        )

        placement[:n_hard] = torch.from_numpy(pos).to(dtype=torch.float32)
        return placement

    def _legalize_min_displacement(
        self,
        pos: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        canvas_w: float,
        canvas_h: float,
    ) -> np.ndarray:
        n = pos.shape[0]
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0

        order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()
        gap = 0.001
        step_mul = 0.20
        max_radius = 220

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

                        cand_x = np.clip(
                            pos[idx, 0] + dxm * step,
                            half_w[idx],
                            canvas_w - half_w[idx],
                        )
                        cand_y = np.clip(
                            pos[idx, 1] + dym * step,
                            half_h[idx],
                            canvas_h - half_h[idx],
                        )

                        if placed.any():
                            dx = np.abs(cand_x - legal[:, 0])
                            dy = np.abs(cand_y - legal[:, 1])
                            coll = (
                                (dx < sep_x[idx] + gap)
                                & (dy < sep_y[idx] + gap)
                                & placed
                            )
                            coll[idx] = False
                            if coll.any():
                                continue

                        dist = (cand_x - pos[idx, 0]) ** 2 + (cand_y - pos[idx, 1]) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = np.array([cand_x, cand_y], dtype=np.float64)
                            found = True

                if found:
                    break

            legal[idx] = best_pos
            placed[idx] = True

        return legal
