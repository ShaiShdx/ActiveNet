# Active Network Simulation - Refactored

A modular Python simulation of Active Filament Networks with motor proteins.

## Requirements

- Python 3.7+
- NumPy
- SciPy  
- Matplotlib
- NetworkX
- PyTorch (optional)

```bash
pip install numpy scipy matplotlib networkx (torch torchvision)
```

**Note**: PyTorch is optional. The code uses NumPy arrays in a torch-compatible structure.

## Structure

```
./
├── setup.py                      # Installs the dependencies
├── examples.py                   # requirements: dependencies
├── README.md                     # README
├── LICENSE                       # MIT LICENSE
├── BUILD_GUIDE.md                # Usage examples
│
├────────────────────────── MINI VERSION CODE ───────────────────────────
│
├── actnet_mini/                  # A mini version of the code
│   ├── active_contraction.py     # Plotting functions    
│   ├── shape_conversion.py       # Physics and time evolution with CUDA
│   └── benchmark_torch.py        # Helper utilities with CUDA
│
├────────────────────────── BODY OF THE CODE ────────────────────────────
│
├── main.py                       # Main entry point
├── main_cuda.py                  # Main entry point for CUDA
├── examples.py                   # Usage examples
│
└── AN_utils/
    ├── config.py                 # Configuration management (dataclasses)
    ├── network.py                # Network generation and topology
    ├── simulation.py             # Physics and time evolution
    ├── helpers.py                # Helper utilities
    ├── visualization.py          # Plotting functions    
    ├── simulation_cuda.py        # Physics and time evolution with CUDA
    └── helpers_cuda.py           # Helper utilities with CUDA

```
# Build System Guide

## Quick Start

```bash
# One-command setup
./setup.sh
source venv/bin/activate
python main.py
```

## Files Included

1. **setup.sh** - Automated setup script
2. **requirements.txt** - Python dependencies

---

## Shell Script

```bash
# Make executable (first time only)
chmod +x setup.sh

# Run setup
./setup.sh
```
---

## Dependencies

**Required:**
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- networkx >= 2.5

---

## Troubleshooting

**"Permission denied: ./setup.sh"**
```bash
chmod +x setup.sh
```

**"venv already exists"**
```bash
rm -rf venv
./setup.sh
```

**"make: command not found"**
Use `./setup.sh` instead or install make.

**"CUDA not available after setup"**
Reinstall PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---


## Quick Start

### Basic Usage (examples.py)

```python
from main import run_simulation

# Run with default settings
sim, network = run_simulation()
```

### Custom Configuration

```python
from config import Config
from main import run_simulation

# Create and customize config
cfg = Config()
cfg.sim.T_tot = 200          # Longer simulation
cfg.sim.dt = 0.005           # Smaller timestep
cfg.net.dens0 = 2.0          # Higher density
cfg.motor.c_mm = 0.2         # More minus-minus motors
cfg.phys.actv_on = True      # Enable activity
cfg.viz.save_img = False     # Don't save frames

# Run simulation
sim, network = run_simulation(cfg)
```

## Configuration Groups

### SimulationConfig
- `T_tot`: Total simulation time
- `dt`: Time step
- `N_frame`: Number of frames to plot
- `seed`: Random seed

### NetworkConfig
- `XY_lens`: Domain size [Lx, Ly]
- `dens0`: Particle density
- `unit_cell`: 'square' or 'triangular'
- `mpf`: Max motors per filament
- `Rc_0`: Cutoff radius

### MotorConfig
- `c_el`, `c_mm`, `c_pp`: Motor concentrations (auto-normalized)
- `V_el`, `V_mm`, `V_pp`: Motor velocities
- `k_mm`, `k_pp`: Spring constants

### PhysicsConfig
- `actv_on`, `elas_on`, `drag_on`, `nois_on`: Feature flags
- `M0`, `I0`: Mass and moment of inertia
- `mu`: Viscosity
- `Temp_t`, `Temp_a`: Noise temperatures

### VisualizationConfig
- `plot_img`, `save_img`: Display/save options
- `fsize`: Figure size
- Colors and styling

## Logging

Set logging level:

```python
from helpers import setup_logging
import logging

setup_logging(level=logging.DEBUG)  # Verbose output
setup_logging(level=logging.INFO)   # Normal output
setup_logging(level=logging.WARNING)  # Minimal output
```

## Examples

See `examples.py` for:
- Basic simulation
- Parameter sweep
- Custom force implementation
- Different initial conditions

## Key Classes

### FilamentNetwork
Manages network geometry and topology:
- `initialize()`: Create lattice and connectivity
- `update_connectivity()`: Rebuild network (for rearrangement)

### ActiveNetworkSimulation
Runs time evolution:
- `initialize_state()`: Set initial conditions
- `run(callback)`: Execute simulation with optional callback

### NetworkVisualizer
Handles all plotting:
- `plot_rods()`: Draw filaments as rods
- `plot_network()`: Draw as point-edge network
- `plot_timeseries()`: Time series analysis


# CUDA Architecture Diagram

## File Structure Comparison

```
ORIGINAL (CPU only)              WITH CUDA (GPU accelerated)
━━━━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━━━━━━━━━━━

main.py                          main.py 
  ↓                              main_cuda.py
simulation.py                      ↓
  ↓                              simulation.py 
ForceCalculator                  simulation_cuda.py
  ↓                                ↓
NumPy/SciPy                      cuda_helpers.py
                                   ↓
CPU only                         CPU + GPU hybrid
```

## Execution Flow

```
┌──────────────────────────────────────────────────────┐
│                   User Code                          │
│  from main_cuda import run_simulation                │
│  sim, network = run_simulation(use_cuda=True)        │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│            Device Detection (cuda_helpers)           │
│  ┌──────────────────────────────────────────────┐    │
│  │ CUDA Available? ─YES→ Use GPU (cuda:0)       │    │
│  │                 └NO→  Use CPU (NumPy)        │    │
│  └──────────────────────────────────────────────┘    │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│           Network Initialization (CPU)               │
│  ┌──────────────────────────────────────────────┐    │
│  │ • Generate lattice points                    │    │
│  │ • Build sparse connectivity matrices         │    │
│  │ • Assign motor types                         │    │
│  │ → Stays on CPU (scipy.sparse)                │    │
│  └──────────────────────────────────────────────┘    │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│         State Initialization (GPU if available)      │
│   ┌──────────────────────────────────────────────┐   │
│   │ Positions   ─────→ GPU                       │   │
│   │ Velocities  ─────→ GPU                       │   │
│   │ Orientations ────→ GPU                       │   │
│   └──────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│               Time Loop (Hybrid CPU/GPU)             │
│   ┌──────────────────────────────────────────────┐   │
│   │ For each timestep:                           │   │
│   │                                              │   │
│   │ 1. Rearrangement (every N steps)             │   │
│   │    GPU→CPU: Transfer positions               │   │
│   │    CPU: Rebuild sparse matrices              │   │
│   │                                              │   │
│   │ 2. Compute Forces (GPU)                      │   │
│   │    CPU: Sparse matrix ops (B12.dot)          │   │
│   │    GPU: Dense ops (exp, sqrt, *)             │   │
│   │    CPU→GPU: Transfer results                 │   │
│   │                                              │   │
│   │ 3. Update State (GPU)                        │   │
│   │    GPU: ax = F/M                             │   │
│   │    GPU: x += v*dt + a*dt²/2                  │   │
│   │    GPU: v += a*dt                            │   │
│   │                                              │   │
│   │ 4. Visualization (every M steps)             │   │
│   │    GPU→CPU: Transfer for plotting            │   │
│   │    CPU: Matplotlib rendering                 │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## Memory Layout

```
    CPU Memory                    GPU Memory
    ━━━━━━━━━━                    ━━━━━━━━━━

┌─────────────────┐            ┌─────────────────┐
│ Sparse Matrices │            │   Positions     │
│  - B12 (scipy)  │            │   xCtr, yCtr    │
│  - B21          │            │   (Ntot × 1)    │
│  - Bas          │            │                 │
│  - incid_el     │            ├─────────────────┤
│                 │            │   Velocities    │
│ (Stay on CPU)   │            │   Vx, Vy, av    │
└─────────────────┘            │   (Ntot × 1)    │
                               │                 │
┌─────────────────┐            ├─────────────────┤
│ Network Graph   │            │   Forces        │
│  - edges        │            │   Fax, Fay      │
│  - topology     │            │   Fpx, Fpy      │
│                 │            │   Fdx, Fdy      │
│ (Stay on CPU)   │            │   (Ntot × 1)    │
└─────────────────┘            │                 │
                               ├─────────────────┤
┌─────────────────┐            │  Orientations   │
│  Plot Buffers   │ ←Transfer─ │   phi, px, py   │
│  (for viz)      │            │   (Ntot × 1)    │
└─────────────────┘            └─────────────────┘
```

## Performance Profile

```
Operation                CPU Time    GPU Time    Speedup
━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━    ━━━━━━━━    ━━━━━━━

Network Setup            5%          5%           1.0x
(sparse matrices)        CPU only    CPU only    

Force Calculation        60%         15%          4.0x
(dense ops)              Heavy       Fast        

State Update             20%          5%          4.0x
(integration)            Slow        Fast        

Memory Transfer          0%           5%          N/A
                         None        Small       

Visualization            15%         10%          1.5x
(plotting)               Moderate    Transfer    
━━━━━━━━━━━━━━━━━━━━━━   ━━━━━━━━    ━━━━━━━━    ━━━━━━━
TOTAL                    100%         40%         2.5x

* Times for 900 filaments on RTX 3080
```

## Data Transfer Points

```
           CPU ←─────────────→ GPU
                   Transfer

1. Initialization:
   CPU: Create network
   CPU→GPU: Initial positions/velocities [ONE TIME]

2. Each Timestep:
   CPU: B12.dot(Pr) [sparse]
   CPU→GPU: Result [SMALL, ~1KB]
   GPU: Force calculations [FAST]
   GPU: State updates [FAST]

3. Rearrangement (every ~100 steps):
   GPU→CPU: Positions [MODERATE, ~10KB]
   CPU: Rebuild network
   CPU→GPU: New connectivity [SMALL, ~1KB]

4. Visualization (every ~50 steps):
   GPU→CPU: Full state [MODERATE, ~10KB]
   CPU: Matplotlib plot

Total Transfer: ~1% of computation time
```

## Component Breakdown

```
┌────────────────────────────────────────────────────┐
│ cuda_helpers.py                                    │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                    │
│ DeviceManager                                      │
│   • CUDA detection                                 │
│   • Tensor conversions (to_tensor, to_numpy)       │
│   • Device operations (zeros, randn, sqrt, exp)    │
│   • Automatic fallback to NumPy                    │
│                                                    │
│ Global helpers                                     │
│   • get_device_manager()                           │
│   • reset_device_manager()                         │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ simulation_cuda.py                                 │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                    │
│ ForceCalculatorCUDA                                │
│   • compute_active_forces() - GPU-accelerated      │
│   • compute_elastic_forces() - GPU-accelerated     │
│   • compute_drag_forces() - GPU-accelerated        │
│                                                    │
│ ActiveNetworkSimulationCUDA                        │
│   • initialize_state() - Move to GPU               │
│   • run() - Hybrid CPU/GPU loop                    │
│   • Same interface as original                     │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ main_cuda.py                                       │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                    │
│ run_simulation()                                   │
│   • Device initialization                          │
│   • GPU↔CPU transfer for visualization             │
│   • Same API as main.py                            │
│                                                    │
│ main()                                             │
│   • Command-line argument parsing                  │
│   • --cpu, --verbose flags                         │
└────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
