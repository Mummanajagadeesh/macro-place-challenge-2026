"""
Submission template for macro placement competition.

All submissions must inherit from BasePlacerInterface and implement the place() method.
"""

from abc import ABC, abstractmethod
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from marco_place.data.tensor_schema import CircuitTensorData


class BasePlacerInterface(ABC):
    """
    Base interface for macro placement algorithms.

    All submissions must inherit from this class and implement the place() method.
    Optionally, implement place_cells() for two-stage placement (macros + standard cells).

    Your placer will be evaluated on:
    1. Placement legality (no overlaps, within boundaries)
    2. Proxy cost (wirelength + density + congestion)
    3. Runtime (must complete within 1 hour per benchmark)

    Prize Eligibility:
    - The $20K prize is awarded ONLY if your submission beats the Circuit Training
      baseline on aggregate across all benchmarks
    - Aggregate score must be > 0 (positive improvement over baseline)
    """

    @abstractmethod
    def place(self, circuit_data: CircuitTensorData):
        """
        Compute placement for the given circuit.

        Args:
            circuit_data: CircuitTensorData object containing:
                - metadata: dict with design_name, num_macros, num_stdcells, canvas dimensions
                - macro_sizes: [num_macros, 2] tensor of (width, height)
                - stdcell_sizes: [num_stdcells, 2] tensor of (width, height) [OPTIONAL]
                - net_to_nodes: list of tensors containing node indices for each net
                - net_weights: [num_nets] tensor of net weights
                - Other optional fields (see tensor_schema.py for full specification)

        Returns:
            One of:
            1. macro_placement: torch.Tensor [num_macros, 2] - if only placing macros
            2. (macro_placement, cell_placement): tuple of tensors - if placing both
               - macro_placement: [num_macros, 2]
               - cell_placement: [num_stdcells, 2] or None

        Constraints:
        - Placements must be within canvas boundaries
        - No overlaps between macros
        - No overlaps between macros and cells
        - Runtime limit: 1 hour (3600 seconds) per benchmark

        Notes:
        - You can place only macros (return single tensor)
        - Or place both macros and cells (return tuple)
        - If you don't place cells, they keep initial positions
        - Standard cells are pre-clustered (typically 800-1000 clusters per design)

        Examples:
            >>> # Option 1: Macro-only placement
            >>> def place(self, circuit_data):
            >>>     macro_placement = your_macro_algorithm(circuit_data)
            >>>     return macro_placement  # [num_macros, 2]

            >>> # Option 2: Place both (single-stage)
            >>> def place(self, circuit_data):
            >>>     macro_placement, cell_placement = your_joint_algorithm(circuit_data)
            >>>     return macro_placement, cell_placement

            >>> # Option 3: Place both (two-stage internally)
            >>> def place(self, circuit_data):
            >>>     macro_placement = place_macros_first(circuit_data)
            >>>     cell_placement = place_cells_around_macros(circuit_data, macro_placement)
            >>>     return macro_placement, cell_placement

            >>> # Option 4: Macro-only (explicit)
            >>> def place(self, circuit_data):
            >>>     macro_placement = your_macro_algorithm(circuit_data)
            >>>     return macro_placement, None  # None = don't place cells
        """
        pass


class TemplatePlacer(BasePlacerInterface):
    """
    Template placer implementation.

    Replace this with your own algorithm!

    This is a simple grid-based placement that serves as a starting point.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize your placer.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Simple grid-based placement.

        Replace this with your own algorithm!
        """
        torch.manual_seed(self.seed)

        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        # Create grid
        cols = int(torch.ceil(torch.sqrt(torch.tensor(num_macros, dtype=torch.float32))))
        rows = int(torch.ceil(torch.tensor(num_macros, dtype=torch.float32) / cols))

        cell_width = canvas_width / cols
        cell_height = canvas_height / rows

        # Place macros in grid
        placement = torch.zeros(num_macros, 2)

        for i in range(num_macros):
            row = i // cols
            col = i % cols

            # Center of grid cell
            placement[i, 0] = (col + 0.5) * cell_width
            placement[i, 1] = (row + 0.5) * cell_height

        return placement


if __name__ == "__main__":
    # Simple test
    print("Testing TemplatePlacer...")

    # Create dummy circuit data
    metadata = {
        'design_name': 'test',
        'num_macros': 4,
        'canvas_width': 1000.0,
        'canvas_height': 1000.0,
    }

    macro_sizes = torch.tensor([
        [100.0, 100.0],
        [150.0, 100.0],
        [100.0, 150.0],
        [120.0, 120.0],
    ])

    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=torch.zeros(4, 2),
        macro_sizes=macro_sizes,
    )

    # Test placer
    placer = TemplatePlacer()
    placement = placer.place(circuit_data)

    print(f"Generated placement for {circuit_data.num_macros} macros:")
    print(placement)
    print("\n✓ Template placer works!")
