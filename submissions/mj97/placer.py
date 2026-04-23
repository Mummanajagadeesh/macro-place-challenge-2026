"""
MJ97 Placer - Minimal displacement legalizer.

This version preserves the strong analytical initial placement while enforcing
strict hard-macro legality with very small displacement.
"""

import random

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class Mj97Placer:
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
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        legal = self._legalize(
            init,
            movable,
            sizes,
            half_w,
            half_h,
            canvas_w,
            canvas_h,
            gap=0.0008,
            step_mul=0.18,
            max_radius=240,
        )

        out[:n_hard] = torch.from_numpy(legal).to(dtype=torch.float32)
        return out

    def _legalize(
        self,
        init: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        canvas_w: float,
        canvas_h: float,
        gap: float,
        step_mul: float,
        max_radius: int,
    ) -> np.ndarray:
        n = init.shape[0]
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

                        cx = np.clip(
                            init[idx, 0] + dxm * step,
                            half_w[idx],
                            canvas_w - half_w[idx],
                        )
                        cy = np.clip(
                            init[idx, 1] + dym * step,
                            half_h[idx],
                            canvas_h - half_h[idx],
                        )

                        if placed.any():
                            dx = np.abs(cx - legal[:, 0])
                            dy = np.abs(cy - legal[:, 1])
                            coll = (
                                (dx < sep_x[idx] + gap)
                                & (dy < sep_y[idx] + gap)
                                & placed
                            )
                            coll[idx] = False
                            if coll.any():
                                continue

                        d = (cx - init[idx, 0]) ** 2 + (cy - init[idx, 1]) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_pos = np.array([cx, cy], dtype=np.float64)
                            found = True

                if found:
                    break

            legal[idx] = best_pos
            placed[idx] = True

        return legal
