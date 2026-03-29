"""
Will's Seed Attempt v2 — FD + SA with Real Net Connectivity

Key improvement over v1: extracts actual net connectivity from PlacementCost
instead of using fake uniform edges. This gives the force-directed placement
and SA real information about which macros should be close together.

Usage:
    uv run evaluate submissions/will_seed/placer.py
    uv run evaluate submissions/will_seed/placer.py --all
    uv run evaluate submissions/will_seed/placer.py --ng45
"""

import math
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

from macro_place.benchmark import Benchmark


# Known testcase directories
TESTCASE_ROOTS = [
    Path("external/MacroPlacement/Testcases/ICCAD04"),
    Path("external/MacroPlacement/Flows/NanGate45"),
]


def _find_plc(benchmark_name: str):
    """Try to load PlacementCost for this benchmark to get net connectivity."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    # IBM benchmarks
    for root in TESTCASE_ROOTS:
        candidate = root / benchmark_name
        if candidate.exists():
            try:
                _, plc = load_benchmark_from_dir(str(candidate))
                return plc
            except Exception:
                pass

    # NG45 benchmarks (name like ariane133_ng45)
    ng45_name_map = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    design = ng45_name_map.get(benchmark_name)
    if design:
        ng45_root = Path("external/MacroPlacement/Flows/NanGate45")
        netlist = ng45_root / design / "netlist" / "output_CT_Grouping" / "netlist.pb.txt"
        plc_file = ng45_root / design / "netlist" / "output_CT_Grouping" / "initial.plc"
        if netlist.exists() and plc_file.exists():
            try:
                _, plc = load_benchmark(str(netlist), str(plc_file))
                return plc
            except Exception:
                pass

    return None


def _extract_edges(benchmark: Benchmark, plc) -> Tuple[np.ndarray, np.ndarray]:
    """Extract real macro-macro connectivity from PlacementCost nets."""
    n = benchmark.num_macros
    mods = plc.modules_w_pins
    hm_indices = plc.hard_macro_indices

    # Build plc_name -> benchmark_index mapping
    name_to_bidx = {}
    for bidx, hm_idx in enumerate(hm_indices):
        name = mods[hm_idx].get_name()
        name_to_bidx[name] = bidx

    # Also try benchmark.macro_names (may differ in ordering)
    for bidx, name in enumerate(benchmark.macro_names):
        if name not in name_to_bidx:
            name_to_bidx[name] = bidx

    edge_dict = {}
    for driver, sinks in plc.nets.items():
        all_pins = [driver] + sinks
        macro_set = set()
        for pin in all_pins:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macro_set.add(name_to_bidx[parent])
        if len(macro_set) >= 2:
            macro_list = sorted(macro_set)
            w = 1.0 / (len(macro_list) - 1)
            for i in range(len(macro_list)):
                for j in range(i + 1, len(macro_list)):
                    pair = (macro_list[i], macro_list[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w

    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)

    edges_list = list(edge_dict.keys())
    weights_list = [edge_dict[e] for e in edges_list]
    return np.array(edges_list, dtype=np.int32), np.array(weights_list, dtype=np.float64)


class WillSeedPlacer:
    """Force-directed placement with real-connectivity SA refinement."""

    def __init__(self, seed: int = 42, fd_iters: int = 200, sa_iters: int = 5000):
        self.seed = seed
        self.fd_iters = fd_iters
        self.sa_iters = sa_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)

        n = benchmark.num_macros
        movable = benchmark.get_movable_mask().numpy()
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # Try to get real net connectivity
        plc = _find_plc(benchmark.name) if benchmark.name else None
        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges, edge_weights = self._build_fallback_edges(benchmark)

        if len(edges) == 0:
            edges, edge_weights = self._build_fallback_edges(benchmark)

        # Phase 1: Force-directed placement
        pos = self._force_directed(benchmark, edges, edge_weights, movable, sizes, cw, ch)

        # Phase 2: Legalize
        pos = self._legalize(pos, movable, sizes, cw, ch)

        # Phase 3: SA refinement
        pos = self._sa_refine(pos, benchmark, edges, edge_weights, movable, sizes, cw, ch)

        return torch.tensor(pos, dtype=torch.float32)

    def _build_fallback_edges(self, benchmark):
        """Fallback: weak uniform edges when no net data available."""
        n = benchmark.num_macros
        edges_list = []
        weights_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edges_list.append((i, j))
                weights_list.append(0.01)
        return np.array(edges_list, dtype=np.int32), np.array(weights_list, dtype=np.float64)

    def _force_directed(self, benchmark, edges, edge_weights, movable, sizes, cw, ch):
        """Vectorized force-directed placement."""
        n = benchmark.num_macros
        pos = benchmark.macro_positions.numpy().copy().astype(np.float64)

        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        for i in range(n):
            if not movable[i]:
                continue
            pos[i, 0] = np.clip(
                cw / 2 + np.random.uniform(-cw * 0.3, cw * 0.3),
                half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(
                ch / 2 + np.random.uniform(-ch * 0.3, ch * 0.3),
                half_h[i], ch - half_h[i])

        movable_mask = movable.astype(np.float64)
        ei = edges[:, 0]
        ej = edges[:, 1]
        areas = sizes[:, 0] * sizes[:, 1]
        max_area = areas.max()

        for iteration in range(self.fd_iters):
            t = 1.0 - iteration / self.fd_iters
            lr = max(0.3, 3.0 * t)

            forces = np.zeros((n, 2), dtype=np.float64)

            # --- Attraction (net springs) ---
            dx = pos[ej, 0] - pos[ei, 0]
            dy = pos[ej, 1] - pos[ei, 1]
            dist = np.maximum(np.sqrt(dx * dx + dy * dy), 0.1)
            # Spring force proportional to distance × weight
            f = edge_weights * dist * 0.005
            fx = f * dx / dist
            fy = f * dy / dist
            np.add.at(forces[:, 0], ei, fx * movable_mask[ei])
            np.add.at(forces[:, 1], ei, fy * movable_mask[ei])
            np.add.at(forces[:, 0], ej, -fx * movable_mask[ej])
            np.add.at(forces[:, 1], ej, -fy * movable_mask[ej])

            # --- Repulsion (all pairs, vectorized) ---
            dx_all = pos[:, 0:1] - pos[:, 0:1].T
            dy_all = pos[:, 1:2] - pos[:, 1:2].T
            sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
            sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
            overlap_x = np.maximum(0, sep_x - np.abs(dx_all))
            overlap_y = np.maximum(0, sep_y - np.abs(dy_all))
            overlapping = (overlap_x > 0) & (overlap_y > 0)
            np.fill_diagonal(overlapping, False)

            dist_all = np.maximum(np.sqrt(dx_all ** 2 + dy_all ** 2), 0.01)

            # Strong overlap repulsion
            repel_mag = 40.0 * overlap_x * overlap_y / max_area * overlapping
            forces[:, 0] += (repel_mag * dx_all / dist_all).sum(axis=1) * movable_mask
            forces[:, 1] += (repel_mag * dy_all / dist_all).sum(axis=1) * movable_mask

            # Density spreading for non-overlapping nearby macros
            spread_radius = max(cw, ch) * 0.2
            nearby = (~overlapping) & (dist_all < spread_radius)
            np.fill_diagonal(nearby, False)
            spread_mag = 0.3 * t / np.maximum(dist_all, 0.1) * nearby
            forces[:, 0] += (spread_mag * dx_all / dist_all).sum(axis=1) * movable_mask
            forces[:, 1] += (spread_mag * dy_all / dist_all).sum(axis=1) * movable_mask

            # Mild center gravity
            forces[:, 0] += (cw / 2 - pos[:, 0]) * 0.001 * movable_mask
            forces[:, 1] += (ch / 2 - pos[:, 1]) * 0.001 * movable_mask

            # Clamp and apply
            f_mag = np.sqrt(forces[:, 0] ** 2 + forces[:, 1] ** 2)
            max_step = lr * np.maximum(sizes[:, 0], sizes[:, 1])
            scale = np.where(f_mag > max_step, max_step / np.maximum(f_mag, 1e-10), 1.0)
            pos[:, 0] += forces[:, 0] * scale
            pos[:, 1] += forces[:, 1] * scale
            pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

        return pos

    def _legalize(self, pos, movable, sizes, cw, ch):
        """Remove overlaps with greedy displacement — largest first."""
        n = len(pos)
        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal_pos = pos.copy()
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue

            if not self._has_overlap_vec(idx, legal_pos[idx], legal_pos, sizes, placed):
                placed[idx] = True
                continue

            w, h = sizes[idx]
            step = max(w, h) * 0.4
            best_pos = legal_pos[idx].copy()
            best_dist = float('inf')

            for radius in range(1, 100):
                found = False
                r = radius
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r:
                            continue
                        cx = np.clip(pos[idx, 0] + dx * step, half_w[idx], cw - half_w[idx])
                        cy = np.clip(pos[idx, 1] + dy * step, half_h[idx], ch - half_h[idx])
                        candidate = np.array([cx, cy])
                        if not self._has_overlap_vec(idx, candidate, legal_pos, sizes, placed):
                            dist = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best_pos = candidate.copy()
                                found = True
                if found:
                    break

            legal_pos[idx] = best_pos
            placed[idx] = True

        return legal_pos

    def _has_overlap_vec(self, idx, candidate, all_pos, sizes, placed):
        """Vectorized overlap check with small gap."""
        gap = 0.01
        dx = np.abs(candidate[0] - all_pos[:, 0])
        dy = np.abs(candidate[1] - all_pos[:, 1])
        sep_x = (sizes[idx, 0] + sizes[:, 0]) / 2 + gap
        sep_y = (sizes[idx, 1] + sizes[:, 1]) / 2 + gap
        overlaps = (dx < sep_x) & (dy < sep_y) & placed
        overlaps[idx] = False
        return overlaps.any()

    def _sa_refine(self, pos, benchmark, edges, edge_weights, movable, sizes, cw, ch):
        """SA refinement with wirelength + overlap cost."""
        n = benchmark.num_macros
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        pos = pos.copy()
        legalized_pos = pos.copy()
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        # Build per-macro edge lists for connectivity-guided moves
        macro_edges: List[List[Tuple[int, int, float]]] = [[] for _ in range(n)]
        for k, (i, j) in enumerate(edges):
            macro_edges[i].append((j, k, edge_weights[k]))
            macro_edges[j].append((i, k, edge_weights[k]))

        # Precompute
        sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        def wirelength_cost():
            dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
            dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
            return (edge_weights * (dx + dy)).sum()

        def overlap_cost():
            dx = np.abs(pos[:, 0:1] - pos[:, 0:1].T)
            dy = np.abs(pos[:, 1:2] - pos[:, 1:2].T)
            ox = np.maximum(0, sep_x_mat - dx)
            oy = np.maximum(0, sep_y_mat - dy)
            ov = ox * oy
            np.fill_diagonal(ov, 0)
            return ov.sum() * 250.0

        current_cost = wirelength_cost() + overlap_cost()
        best_pos = legalized_pos.copy()
        best_cost = float('inf')

        T_start = max(cw, ch) * 0.3
        T_end = max(cw, ch) * 0.0005

        for step in range(self.sa_iters):
            frac = step / self.sa_iters
            T = T_start * (T_end / T_start) ** frac

            move_type = random.random()
            old_pos = pos.copy()

            if move_type < 0.45:
                # SHIFT
                i = random.choice(movable_idx)
                shift = T * (0.3 + 0.7 * (1 - frac))
                pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift),
                                    half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift),
                                    half_h[i], ch - half_h[i])
            elif move_type < 0.75:
                # SWAP (prefer connected macros)
                i = random.choice(movable_idx)
                if macro_edges[i] and random.random() < 0.7:
                    candidates = [j for j, _, _ in macro_edges[i] if movable[j]]
                    j = random.choice(candidates) if candidates else random.choice(movable_idx)
                else:
                    j = random.choice(movable_idx)
                if i != j:
                    pi = np.clip(old_pos[j].copy(),
                                 [half_w[i], half_h[i]], [cw - half_w[i], ch - half_h[i]])
                    pj = np.clip(old_pos[i].copy(),
                                 [half_w[j], half_h[j]], [cw - half_w[j], ch - half_h[j]])
                    pos[i] = pi
                    pos[j] = pj
            else:
                # MOVE TOWARD NEIGHBOR
                i = random.choice(movable_idx)
                if macro_edges[i]:
                    j, _, _ = random.choice(macro_edges[i])
                    alpha = random.uniform(0.05, 0.4)
                    pos[i, 0] = np.clip(
                        pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]),
                        half_w[i], cw - half_w[i])
                    pos[i, 1] = np.clip(
                        pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]),
                        half_h[i], ch - half_h[i])

            new_cost = wirelength_cost() + overlap_cost()
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                ov = overlap_cost()
                if ov == 0 and new_cost < best_cost:
                    best_cost = new_cost
                    best_pos = pos.copy()
            else:
                pos = old_pos

        if best_cost == float('inf'):
            return legalized_pos
        return best_pos
