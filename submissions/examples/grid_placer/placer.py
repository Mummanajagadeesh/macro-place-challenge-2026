"""
Example grid-based placer demonstrating flexible placement interface.

This placer demonstrates both:
1. Macro-only placement (default)
2. Joint macro + cell placement (when place_cells=True)
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from marco_place.data.tensor_schema import CircuitTensorData


class GridPlacer:
    """
    Grid-based placer that arranges macros (and optionally cells) in a regular grid.

    This is a simple baseline that doesn't optimize for wirelength/congestion,
    but produces valid placements quickly.
    """

    def __init__(self, seed: int = 42, place_cells: bool = False):
        """
        Initialize GridPlacer.

        Args:
            seed: Random seed for reproducibility
            place_cells: If True, also place standard cells; if False, macro-only
        """
        self.seed = seed
        self.place_cells = place_cells

    def place(self, circuit_data: CircuitTensorData):
        """
        Place macros (and optionally cells) in a grid pattern.

        Returns:
            - If place_cells=False: macro_placement tensor [num_macros, 2]
            - If place_cells=True: (macro_placement, cell_placement) tuple
        """
        torch.manual_seed(self.seed)

        # Place macros
        macro_placement = self._place_macros(circuit_data)

        if not self.place_cells or circuit_data.num_stdcells == 0:
            # Macro-only placement
            return macro_placement

        # Joint placement: also place cells
        cell_placement = self._place_cells(circuit_data, macro_placement)
        return macro_placement, cell_placement

    def _place_macros(self, circuit_data: CircuitTensorData) -> torch.Tensor:
        """
        Place macros in a grid pattern.

        Strategy: Row-packing with height-based sorting for better utilization.
        """
        num_macros = circuit_data.num_macros
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height
        macro_sizes = circuit_data.macro_sizes

        # Sort macros by height (descending) for better packing
        heights = macro_sizes[:, 1]
        sorted_indices = torch.argsort(heights, descending=True)

        placement = torch.zeros(num_macros, 2)

        # Row packing
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0

        for idx in sorted_indices:
            width = macro_sizes[idx, 0].item()
            height = macro_sizes[idx, 1].item()

            # Check if macro fits in current row
            if current_x + width > canvas_width:
                # Move to next row
                current_x = 0.0
                current_y += row_height
                row_height = 0.0

            # Place macro (center position)
            placement[idx, 0] = current_x + width / 2
            placement[idx, 1] = current_y + height / 2

            # Update position
            current_x += width
            row_height = max(row_height, height)

        return placement

    def _place_cells(
        self,
        circuit_data: CircuitTensorData,
        macro_placement: torch.Tensor
    ) -> torch.Tensor:
        """
        Place standard cells in available space around macros.

        Strategy: Simple grid in regions not occupied by macros.
        """
        num_cells = circuit_data.num_stdcells
        canvas_width = circuit_data.canvas_width
        canvas_height = circuit_data.canvas_height

        # Create simple grid for cells
        # This is a coarse placement - a real placer would do detailed placement
        cols = int(torch.ceil(torch.sqrt(torch.tensor(num_cells, dtype=torch.float32))))
        rows = int(torch.ceil(torch.tensor(num_cells, dtype=torch.float32) / cols))

        cell_width = canvas_width / cols
        cell_height = canvas_height / rows

        cell_placement = torch.zeros(num_cells, 2)

        for i in range(num_cells):
            row = i // cols
            col = i % cols

            # Center of grid cell
            cell_placement[i, 0] = (col + 0.5) * cell_width
            cell_placement[i, 1] = (row + 0.5) * cell_height

        return cell_placement


if __name__ == "__main__":
    # Simple test of both modes
    print("Testing GridPlacer...")

    # Create dummy circuit data
    metadata = {
        'design_name': 'test',
        'num_macros': 4,
        'num_stdcells': 10,
        'canvas_width': 1000.0,
        'canvas_height': 1000.0,
    }

    macro_sizes = torch.tensor([
        [100.0, 100.0],
        [150.0, 100.0],
        [100.0, 150.0],
        [120.0, 120.0],
    ])

    stdcell_sizes = torch.tensor([
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
        [20.0, 20.0],
    ])

    circuit_data = CircuitTensorData(
        metadata=metadata,
        macro_positions=torch.zeros(4, 2),
        macro_sizes=macro_sizes,
        stdcell_positions=torch.zeros(10, 2),
        stdcell_sizes=stdcell_sizes,
    )

    # Test 1: Macro-only placement
    print("\n--- Test 1: Macro-only placement ---")
    placer_macro_only = GridPlacer(place_cells=False)
    result = placer_macro_only.place(circuit_data)

    if isinstance(result, tuple):
        print(f"ERROR: Expected single tensor, got tuple")
    else:
        print(f"✓ Returned macro placement: {result.shape}")
        print(f"  {result}")

    # Test 2: Joint placement
    print("\n--- Test 2: Joint macro + cell placement ---")
    placer_joint = GridPlacer(place_cells=True)
    result = placer_joint.place(circuit_data)

    if isinstance(result, tuple):
        macro_placement, cell_placement = result
        print(f"✓ Returned tuple:")
        print(f"  Macro placement: {macro_placement.shape}")
        print(f"  Cell placement: {cell_placement.shape}")
    else:
        print(f"ERROR: Expected tuple, got single tensor")

    print("\n✓ GridPlacer tests passed!")
