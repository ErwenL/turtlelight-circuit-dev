"""S-parameter reduction solver for photonic networks.

Builds a block-diagonal network S-matrix from component S-parameters and
eliminates connected internal ports using S-parameter reduction.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import numpy as np

from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid


class SParamSolver:
    """Compute reduced network S-parameters from a connected photonic network."""

    def solve(self, network: Network, freq: FrequencyGrid) -> np.ndarray:
        """Solve network S-parameters on a frequency grid.

        Args:
            network: Photonic network containing components and port links.
            freq: Frequency grid for simulation.

        Returns:
            Reduced S-matrix with shape ``(n_freq, n_external_ports, n_external_ports)``.
        """
        n_freq = len(freq.values)
        if not network.components:
            return np.zeros((n_freq, 0, 0), dtype=np.complex128)

        port_index_map, ordered_port_names = self._build_port_index_map(network)
        s_total = self._build_block_diagonal_s(network, freq)

        links = dict(network.connections._links)
        internal_port_names = [
            port_name
            for port_name in ordered_port_names
            if port_name in links and links[port_name] in port_index_map
        ]
        external_indices = [
            idx for idx, port_name in enumerate(ordered_port_names) if port_name not in internal_port_names
        ]

        if not internal_port_names:
            return s_total[:, external_indices][:, :, external_indices]

        internal_indices = [port_index_map[name] for name in internal_port_names]
        permutation = self._build_connection_permutation(internal_port_names, links)
        identity = np.eye(len(internal_indices), dtype=np.complex128)

        reduced = np.zeros(
            (n_freq, len(external_indices), len(external_indices)), dtype=np.complex128
        )

        for i in range(n_freq):
            s_f = s_total[i]
            s_ee = s_f[np.ix_(external_indices, external_indices)]
            s_ei = s_f[np.ix_(external_indices, internal_indices)]
            s_ie = s_f[np.ix_(internal_indices, external_indices)]
            s_ii = s_f[np.ix_(internal_indices, internal_indices)]

            s_ei_connected = s_ei @ permutation
            s_ii_connected = s_ii @ permutation
            reduced[i] = s_ee + s_ei_connected @ np.linalg.inv(identity - s_ii_connected) @ s_ie

        return reduced

    @staticmethod
    def _build_port_index_map(network: Network) -> tuple[dict[str, int], list[str]]:
        """Map each component port name to global block-diagonal index."""
        ordered_names: list[str] = []
        for component in network.components:
            for port in component.ports:
                ordered_names.append(port.name)
        return {name: idx for idx, name in enumerate(ordered_names)}, ordered_names

    @staticmethod
    def _build_block_diagonal_s(network: Network, freq: FrequencyGrid) -> np.ndarray:
        """Assemble block-diagonal S-matrix from all component S-matrices."""
        n_freq = len(freq.values)
        s_blocks = [component.get_s(freq).astype(np.complex128) for component in network.components]

        for block, component in zip(s_blocks, network.components):
            expected_shape = (n_freq, component.num_of_ports, component.num_of_ports)
            if block.shape != expected_shape:
                raise ValueError(
                    f"Component '{component.name}' returned S shape {block.shape}, "
                    f"expected {expected_shape}"
                )

        port_sizes = [block.shape[1] for block in s_blocks]
        total_ports = int(sum(port_sizes))
        s_total = np.zeros((n_freq, total_ports, total_ports), dtype=np.complex128)

        for f_idx in range(n_freq):
            matrix_blocks = []
            for row_idx, row_block in enumerate(s_blocks):
                row_entries = []
                for col_idx, col_block in enumerate(s_blocks):
                    if row_idx == col_idx:
                        row_entries.append(row_block[f_idx])
                    else:
                        row_entries.append(
                            np.zeros((row_block.shape[1], col_block.shape[2]), dtype=np.complex128)
                        )
                matrix_blocks.append(row_entries)
            s_total[f_idx] = np.block(matrix_blocks)

        return s_total

    @staticmethod
    def _build_connection_permutation(
        internal_port_names: list[str], links: dict[str, str]
    ) -> np.ndarray:
        """Build permutation matrix mapping each internal port to its linked peer."""
        size = len(internal_port_names)
        index_map = {name: idx for idx, name in enumerate(internal_port_names)}
        permutation = np.zeros((size, size), dtype=np.complex128)

        for port_name, row_idx in index_map.items():
            peer_name = links.get(port_name)
            if peer_name is None or peer_name not in index_map:
                raise ValueError(f"Internal port '{port_name}' has no valid connected peer")
            permutation[row_idx, index_map[peer_name]] = 1.0

        return permutation
