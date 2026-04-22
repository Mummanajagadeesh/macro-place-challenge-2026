"""
MJ97 Placer - Multi-start legal SA selector.

Runs several legal SA candidates and picks the best one using a fast,
benchmark-derived connectivity surrogate cost.

Usage:
    uv run evaluate submissions/mj97/placer.py
    uv run evaluate submissions/mj97/placer.py --all
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _load_will_seed_placer_class():
    src = Path(__file__).resolve().parents[1] / "will_seed" / "placer.py"
    spec = importlib.util.spec_from_file_location("will_seed_placer_module", str(src))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load placer module from {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.WillSeedPlacer


class Mj97Placer:
    def __init__(self):
        will_seed = _load_will_seed_placer_class()
        self._candidates = [
            will_seed(seed=42, refine_iters=1800),
            will_seed(seed=97, refine_iters=1800),
            will_seed(seed=7, refine_iters=1800),
        ]

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        scorer = _SurrogateScorer(benchmark)
        best_score = float("inf")
        best_placement = None

        for placer in self._candidates:
            placement = placer.place(benchmark)
            score = scorer.score(placement)
            if score < best_score:
                best_score = score
                best_placement = placement

        if best_placement is None:
            return benchmark.macro_positions.clone()
        return best_placement


class _SurrogateScorer:
    def __init__(self, benchmark: Benchmark):
        self.num_hard = benchmark.num_hard_macros
        self.half_w = (
            benchmark.macro_sizes[: self.num_hard, 0].cpu().numpy() * 0.5
        ).astype(np.float64)
        self.half_h = (
            benchmark.macro_sizes[: self.num_hard, 1].cpu().numpy() * 0.5
        ).astype(np.float64)
        self.edge_i, self.edge_j, self.edge_w, self.anchor_pos, self.anchor_w = (
            self._build_graph(benchmark)
        )

    def _build_graph(self, benchmark: Benchmark):
        edge_map = {}
        anchor_sum = np.zeros((self.num_hard, 2), dtype=np.float64)
        anchor_w = np.zeros(self.num_hard, dtype=np.float64)

        macro_pos = benchmark.macro_positions.cpu().numpy()
        port_pos = benchmark.port_positions.cpu().numpy()
        total_nodes = benchmark.num_macros + port_pos.shape[0]

        for net_idx, nodes_tensor in enumerate(benchmark.net_nodes):
            nodes = [int(v) for v in nodes_tensor.tolist() if 0 <= int(v) < total_nodes]
            if len(nodes) < 2:
                continue

            hard_nodes = [v for v in nodes if v < self.num_hard]
            if not hard_nodes:
                continue

            weight = (
                float(benchmark.net_weights[net_idx].item())
                if net_idx < benchmark.net_weights.shape[0]
                else 1.0
            )
            weight = max(weight, 1e-6)

            if len(hard_nodes) >= 2:
                pair_w = weight / max(float(len(hard_nodes) - 1), 1.0)
                for ii in range(len(hard_nodes)):
                    hi = hard_nodes[ii]
                    for jj in range(ii + 1, len(hard_nodes)):
                        hj = hard_nodes[jj]
                        key = (hi, hj) if hi < hj else (hj, hi)
                        edge_map[key] = edge_map.get(key, 0.0) + pair_w

            anchors = []
            for v in nodes:
                if v < self.num_hard:
                    continue
                if v < benchmark.num_macros:
                    anchors.append(macro_pos[v])
                else:
                    pidx = v - benchmark.num_macros
                    if 0 <= pidx < port_pos.shape[0]:
                        anchors.append(port_pos[pidx])

            if anchors:
                center = np.mean(np.asarray(anchors, dtype=np.float64), axis=0)
                aw = weight * (1.0 + 0.25 * len(anchors))
                for hi in hard_nodes:
                    anchor_sum[hi] += aw * center
                    anchor_w[hi] += aw

        if edge_map:
            pairs = np.asarray(list(edge_map.keys()), dtype=np.int64)
            edge_i = pairs[:, 0]
            edge_j = pairs[:, 1]
            edge_w = np.asarray([edge_map[p] for p in edge_map], dtype=np.float64)
        else:
            edge_i = np.zeros(0, dtype=np.int64)
            edge_j = np.zeros(0, dtype=np.int64)
            edge_w = np.zeros(0, dtype=np.float64)

        anchor_pos = np.zeros((self.num_hard, 2), dtype=np.float64)
        nz = anchor_w > 0.0
        if np.any(nz):
            anchor_pos[nz] = anchor_sum[nz] / anchor_w[nz, None]

        return edge_i, edge_j, edge_w, anchor_pos, anchor_w

    def score(self, placement: torch.Tensor) -> float:
        pos = placement[: self.num_hard].cpu().numpy().astype(np.float64)

        if self.edge_w.size > 0:
            dx = np.abs(pos[self.edge_i, 0] - pos[self.edge_j, 0])
            dy = np.abs(pos[self.edge_i, 1] - pos[self.edge_j, 1])
            edge_cost = float(np.sum(self.edge_w * (dx + dy)))
        else:
            edge_cost = 0.0

        adx = np.abs(pos[:, 0] - self.anchor_pos[:, 0])
        ady = np.abs(pos[:, 1] - self.anchor_pos[:, 1])
        anchor_cost = float(np.sum(self.anchor_w * (adx + ady)))

        overlap_penalty = float(self._overlap_count(pos)) * 1e8
        return edge_cost + 0.7 * anchor_cost + overlap_penalty

    def _overlap_count(self, pos: np.ndarray) -> int:
        dx = np.abs(pos[:, 0][:, None] - pos[:, 0][None, :])
        dy = np.abs(pos[:, 1][:, None] - pos[:, 1][None, :])
        req_x = self.half_w[:, None] + self.half_w[None, :] + 0.001
        req_y = self.half_h[:, None] + self.half_h[None, :] + 0.001
        ov = (dx < req_x) & (dy < req_y)
        np.fill_diagonal(ov, False)
        return int(np.count_nonzero(np.triu(ov, k=1)))
