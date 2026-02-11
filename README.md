# Partcl/HRT Macro Placement Challenge

**Win $20,000 by developing better macro placement algorithms!**

Partcl and Hudson River Trading are excited to co-host a competition to solve the macro placement problem. 

## About Macro Placement

Macro placement is the problem of positioning large fixed-size blocks (SRAMs, IPs, analog macros, etc.) on a chip floorplan so that routing congestion, timing, power delivery, and area constraints are balanced. Unlike standard-cell placement, macros have strong geometric and connectivity constraints, so the challenge is to explore a highly discrete design space while minimizing wirelength, avoiding blockages, and preserving downstream routability and timing quality.

For example, the **ibm01** benchmark has:
- **246 hard macros** of varying sizes (ranging from 0.8 to 27 μm², with 33× size variation)
- **7,269 nets** connecting macros to each other and to 894 pre-placed standard cell clusters
- **A 22.9 × 23.0 μm canvas** with 42.8% area utilization

## About HRT Hardware

<HRT GIVES BLURB HERE>

## Background Papers
[An Updated Assessment of Reinforcement Learning
for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)

[Assessment of Reinforcement Learning for Macro Placement](https://vlsicad.ucsd.edu/Publications/Conferences/396/c396.pdf)

[A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)

## 🏆 Prizes

- **$20,000 — First Place:** Awarded to the top submission that surpasses the Simulated Annealing (SA) and RePlAce baselines reported in [An Updated Assessment of Reinforcement Learning
for Macro Placement]([https://arxiv.org/pdf/2302.11014](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)).  
- **$10,000 — First Place (Alternate):** Awarded if the winning submission ranks first overall but does **not** exceed the SA and RePlAce benchmark results.  
- **$5,000 — Second Place:** Awarded to the runner-up based on final competition rankings.  
- **$4,000 — Innovation Award:** Granted to the most creative or technically innovative approach among the top entries, as determined by the judging panel.
- **Swag:** Every valid submission gets HRT swag!
- 
## Submission Format

- All submissions will be via google form. Submissions may be made public or private before the end of judging.
- Private submissions will be required to share repository with judges so they may clone/evaluate the method.
- Teams may be up to 5 individuals.
- The deadline for submissions is 10 weeks from the posting of the competition at 11:59PM PT WILL INSERT DEADLINE DATE.
- All teams may only submit one algorithm and one person may only join one team. 
- **All winning implementations must be made open-source under Apache 2.0 or GPL**


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Test the infrastructure
pytest
```

### Run Your First Example

```bash
# Run the simple random placer example
python submissions/examples/simple_random_placer.py
```

You should see output like:
```
[4/4] Computing proxy cost and overlap metrics...
  ✓ Costs computed:
    - Wirelength:  0.128768
    - Density:     1.276113
    - Congestion:  2.248285
    - Proxy Cost:  1.890967 ⭐

  ✓ Overlap analysis:
    - Overlapping pairs:       211
    - Macros with overlaps:    198 (80.5%)
    - Total overlap area:      99.632 μm²

Comparison with initial placement:
  Initial proxy cost:   1.038498 (overlaps: 9)
  Random proxy cost:    1.890967 (overlaps: 211)
  Score: -1000 (DISQUALIFIED: 211 overlaps)
```

The random placer has overlaps and is automatically disqualified - your job is to do better!

## 🎓 How It Works

### 1. Benchmark Representation

Benchmarks are represented as **PyTorch tensors** for easy integration with ML approaches:

```python
from loader import load_benchmark_from_dir

# Load a benchmark
benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')

print(f"Benchmark: {benchmark.name}")
print(f"Macros: {benchmark.num_macros}")
print(f"Nets: {benchmark.num_nets}")
print(f"Canvas: {benchmark.canvas_width} × {benchmark.canvas_height} μm")

# Access data
print(f"Macro positions: {benchmark.macro_positions.shape}")  # [246, 2]
print(f"Macro sizes: {benchmark.macro_sizes.shape}")          # [246, 2]
print(f"Fixed macros: {benchmark.macro_fixed.shape}")         # [246] (bool)
```

### 2. Implementing Your Placer

Create a class with a `.place()` method:

```python
import torch
from benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Generate macro placement.

        Args:
            benchmark: Benchmark object with:
                - num_macros: Number of macros (246 for ibm01)
                - macro_sizes: [num_macros, 2] (width, height) in μm
                - macro_fixed: [num_macros] bool (True if fixed)
                - canvas_width, canvas_height: Canvas dimensions
                - num_nets: Number of nets (7269 for ibm01)

        Returns:
            placement: [num_macros, 2] tensor of (x, y) center positions
        """
        placement = torch.zeros(benchmark.num_macros, 2)

        # Your algorithm here!
        # - Use GNNs, RL, SA, optimization, or any approach
        # - MUST have zero overlaps (automatic disqualification otherwise)
        # - MUST be within canvas boundaries
        # - Minimize proxy cost while keeping runtime reasonable

        # Remember to respect fixed macros!
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement
```

### 3. Evaluation

```python
import time
from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
from utils import validate_placement

# Load benchmark
benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')

# Run your placer with timing
start_time = time.time()
placer = MyPlacer()
placement = placer.place(benchmark)
runtime = time.time() - start_time

# Validate placement legality
is_valid, violations = validate_placement(placement, benchmark)
if not is_valid:
    print(f"Invalid placement: {violations}")

# Compute cost and overlap metrics
costs = compute_proxy_cost(placement, benchmark, plc)
print(f"Proxy cost: {costs['proxy_cost']:.6f}")
print(f"Overlaps: {costs['overlap_count']} pairs")
print(f"Runtime: {runtime:.2f}s")

# Compute score
if costs['overlap_count'] > 0:
    score = -1000  # Disqualified
else:
    baseline_cost = 1.0  # Replace with actual baseline
    quality = (baseline_cost - costs['proxy_cost']) / baseline_cost
    runtime_penalty = max(0, (runtime - 300) / 300)
    score = quality - 0.1 * runtime_penalty

print(f"Score: {score}")
```

## 📋 Competition Rules

### Allowed

1. **Any algorithmic approach**: SA, RL, GNN, analytical methods, hybrid approaches, learning-based, etc.
2. **Any framework**: PyTorch, TensorFlow, JAX, or pure Python/C++
3. **Any optimization technique**: Gradient descent, evolutionary algorithms, local search, etc.
4. **Training on public benchmarks**: You can learn from the IBM benchmark data

### Not Allowed

1. ❌ Modifying the evaluation functions (must use TILOS MacroPlacement evaluator as-is)
2. ❌ Hardcoding solutions for specific benchmarks (must be general algorithm)
3. ❌ Using external/proprietary placement tools (must be open-source submission)
4. ❌ Exceeding runtime limits (1 hour per benchmark hard timeout)

### Runtime Constraints

- **Soft limit**: 5 minutes per benchmark (no penalty)
- **Penalty zone**: 5-60 minutes (linear penalty up to -0.1 quality score)
- **Hard timeout**: 1 hour (automatic disqualification)

Runtime measured on standard hardware:
- CPU: AMD EPYC 7763 (64 cores) or equivalent
- RAM: 256GB
- No GPU acceleration in evaluation (but you can use GPU during development)

### Overlap Tolerance: ZERO

Unlike density cost which is continuous, overlaps result in automatic disqualification:
- 0 overlaps: ✅ Eligible for scoring
- 1+ overlaps: ❌ Score = -1000 (disqualified for that benchmark)

This matches the constraints enforced by the SA baseline.

## 🎯 IBM Benchmark Suite (ICCAD04)

We evaluate on the complete ICCAD04 IBM benchmark suite:

| Benchmark | Macros | Nets | Canvas (μm) | Area Util. | SA Baseline | RePlAce Baseline |
|-----------|--------|------|-------------|------------|-------------|------------------|
| **ibm01** | 246 | 7,269 | 22.9×23.0 | 42.8% | 1.3166 | **0.9976** ⭐ |
| **ibm02** | 254 | 7,538 | 23.2×23.5 | 43.1% | 1.9072 | **1.8370** ⭐ |
| **ibm03** | 269 | 8,045 | 24.1×24.3 | 44.2% | 1.7401 | **1.3222** ⭐ |
| **ibm04** | 285 | 8,654 | 24.8×25.1 | 44.8% | 1.5037 | **1.3024** ⭐ |
| **ibm06** | 318 | 9,745 | 26.1×26.5 | 46.1% | 2.5057 | **1.6187** ⭐ |
| **ibm07** | 335 | 10,328 | 26.8×27.2 | 46.8% | 2.0229 | **1.4633** ⭐ |
| **ibm08** | 352 | 10,901 | 27.5×27.9 | 47.4% | 1.9239 | **1.4285** ⭐ |
| **ibm09** | 369 | 11,463 | 28.1×28.5 | 48.0% | 1.3875 | **1.1194** ⭐ |
| **ibm10** | 387 | 12,018 | 28.8×29.2 | 48.6% | 2.1108 | **1.5009** ⭐ |
| **ibm11** | 405 | 12,568 | 29.4×29.8 | 49.2% | 1.7111 | **1.1774** ⭐ |
| **ibm12** | 423 | 13,111 | 30.1×30.5 | 49.8% | 2.8261 | **1.7261** ⭐ |
| **ibm13** | 441 | 13,647 | 30.7×31.1 | 50.4% | 1.9141 | **1.3355** ⭐ |
| **ibm14** | 460 | 14,178 | 31.4×31.8 | 51.0% | 2.2750 | **1.5436** ⭐ |
| **ibm15** | 479 | 14,704 | 32.0×32.4 | 51.6% | 2.3000 | **1.5159** ⭐ |
| **ibm16** | 498 | 15,225 | 32.7×33.1 | 52.2% | 2.2337 | **1.4780** ⭐ |
| **ibm17** | 517 | 15,741 | 33.3×33.7 | 52.8% | 3.6726 | **1.6446** ⭐ |
| **ibm18** | 537 | 16,253 | 34.0×34.4 | 53.4% | 2.7755 | **1.7722** ⭐ |

Each benchmark includes:
- Hard macros (you place these)
- Soft macros (pre-placed standard cell clusters, fixed during evaluation)
- Nets connecting all components
- Initial placement (hand-crafted, serves as reference)

**Baseline Analysis:**
- RePlAce (⭐) consistently outperforms SA across all benchmarks
- RePlAce achieves 15-55% lower proxy cost than SA
- **To win the $20K prize, you must beat RePlAce (the stronger baseline) on aggregate**
- Both baselines achieve zero overlaps (enforced as hard constraint)

## 💡 Why This Is Hard

Despite "only" 246-537 macros, this problem is extremely challenging:

1. **Massive search space**: ~10^800 possible placements (even with constraints)
2. **Conflicting objectives**: Wirelength wants clustering, density wants spreading, congestion wants routing space
3. **Non-convex landscape**: Millions of local minima, discontinuities, plateaus
4. **Long-range dependencies**: Moving one macro affects costs globally through thousands of nets
5. **Hard constraints**: No overlaps between heterogeneous sizes (33× size variation)
6. **Tight packing**: 43-53% area utilization leaves little slack
7. **Runtime matters**: Must be fast enough to be practical (< 5 minutes ideal)

Classical methods (SA, RePlAce) have been refined for decades but still have room for improvement!

## 📖 Documentation

- **Setup Guide**: [`SETUP.md`](SETUP.md) - Infrastructure details, testing, cost computation
- **API Reference**: [`SETUP.md`](SETUP.md) - Benchmark format, loader, objective functions
- **Example Submissions**: [`submissions/examples/`](submissions/examples/) - Random placer example

## 📚 References

- **TILOS MacroPlacement**: [GitHub Repository](https://github.com/TILOS-AI-Institute/MacroPlacement)
  - Source of evaluation infrastructure
  - ICCAD04 benchmarks
  - SA and RePlAce baseline implementations

- **ICCAD04 Benchmarks**: Classic macro placement benchmark suite used in academic research

## 🤔 FAQ

**Q: Why only IBM benchmarks?**
A: The IBM (ICCAD04) suite is the standard academic benchmark for macro placement, with well-established baselines and extensive prior work.

**Q: Why is runtime part of the score?**
A: Real chip design requires practical algorithms. A solution that takes hours is less useful than one that takes minutes, even if slightly lower quality.

**Q: Can I use GPU?**
A: GPU use is encouraged. We will evaluate implementations with a GPU >40GB VRAM and 100GB of RAM.

**Q: What if I beat one baseline but not the other?**
A: You must beat BOTH baselines on aggregate to win the prize. However, you'll still be recognized on the leaderboard.

**Q: Are there hidden test cases?**
A: No. All 18 IBM benchmarks are public. The aggregate score across all 18 determines the winner.

**Q: What counts as "beating" the baseline?**
A: Your geometric mean score across all benchmarks must be positive (meaning on average you beat the baseline).

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/partcleda/partcl-macro-place-challenge/issues)
- **Email**: contact@partcl.com

## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0 - see [LICENSE.md](LICENSE.md) for details.

---

**Ready to win $20,000?**

Beat SA and RePlAce on the IBM benchmarks with zero overlaps and reasonable runtime!

Good luck! 🚀
