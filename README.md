# PDE Framework — Modular Finite Difference Solver

## Features

- **JIT-Compiled Kernels**: Pure numerical kernels optimized with Numba for performance
- **Flexible API**: Extensible interfaces (IOperator, IEquation, IIntegrator, IBoundaryCondition, ...)
- **Multiple Schemes**: Euler, RK4, Crank-Nicolson integrators; Dirichlet, Neumann, Periodic, Robin BCs
- **Well-Tested**: 78 unit tests covering nearly all components
- **Interactive Examples**: Jupyter notebooks demonstrating Heat and Schrödinger equations
- **Type Safe**: Full type hints and NumPy-style docstrings throughout
- **Active Development**: Mid-stage framework with expanding features, approaching feature-complete

## Project Structure

```
pde_framework/
├── calculation_core/       # Numerical kernels
│   ├── laplacian_kernels.py
│   ├── gradient_kernels.py
│   ├── divergence_kernels.py
│   ├── bc_kernels.py
│   ├── time_kernels.py
│   └── matrix_kernels.py
│
├── grid/                   # Grid structures
│   └── Grid1D.py          # 1D uniform grid
│
├── field/                  # Fields
│   ├── ScalarField.py     # Scalar field
│   └── VectorField.py     # N-component vector field
│
├── boundary_conditions/    # Boundary conditions
│   ├── IBoundaryCondition.py
│   └── clients/
│       ├── DirichletBC.py      # Fixed value boundaries
│       ├── NeumannBC.py        # Fixed gradient boundaries
│       ├── PeriodicBC.py       # Periodic boundaries
│       └── RobinBC.py          # Mixed boundaries
│
├── operators/              # Differential operators
│   ├── IOperator.py
│   └── clients/
│       ├── LaplacianOperator.py
│       ├── GradientOperator.py
│       └── DivergenceOperator.py
│
├── equations/              # PDE systems
│   ├── IEquation.py
│   └── clients/
│       ├── HeatEquation.py      
│       └── SchrodingerEquation.py  
│
├── integrators/            # Time integration schemes
│   ├── IIntegrator.py
│   └── clients/
│       ├── ExplicitEulerIntegrator.py
│       ├── RK4Integrator.py
│       ├── CrankNicolson1DIntegrator.py
│       └── SpatialIntegrator.py
│
├── solvers/                # Solvers orchestration
│   └── Solver.py           # Time stepper with snapshot management
│
├── utils/                  # Utilities and validators
│   ├── validators.py       # Grid and field validation, CFL checks
│   └── array_utils.py      # Helper functions (gaussian_1d, etc.)
│
├── visualization/          # Visualization utilities (extensible)
└── config/                 # Configuration constants
```

### Architecture Principles

1. **Kernel Layer** (`calculation_core/`): Numerical kernels
2. **Data Layer** (`grid/`, `field/`): Immutable, passive data structures
3. **Operator Layer** (`operators/`, `boundary_conditions/`): Interfaces with concrete implementations
4. **Equation Layer** (`equations/`): PDE systems combining operators and BCs
5. **Integrator Layer** (`integrators/`): Temporal discretization schemes
6. **Solver Layer** (`solvers/`): Orchestration

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pde_kodik

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install in development mode with dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check package imports
python -c "from pde_framework import Grid1D, ScalarField, Solver; print('✓ Package imported successfully')"
```

### Run Examples

```bash
# Start Jupyter and open notebooks
jupyter notebook notebooks/

# Or run specific notebook
jupyter notebook notebooks/02_heat_equation_solver.ipynb
```

## Usage Examples

### Example 1: Simple 1D Heat Equation

```python
import numpy as np
from pde_framework import (
    Grid1D, ScalarField, DirichletBC,
    LaplacianOperator, HeatEquation,
    ExplicitEulerIntegrator, Solver
)

# Create domain: [-1, 1] with 101 points
grid = Grid1D(x_min=-1.0, x_max=1.0, n_points=101)

# Initial condition: Gaussian
u0 = ScalarField(grid)
u0.apply_function(lambda x: np.exp(-x**2 / 0.1))

# Boundary conditions: homogeneous Dirichlet (u=0 at boundaries)
bc = DirichletBC(left_value=0.0, right_value=0.0)

# PDE: ∂u/∂t = α·∇²u (Heat equation)
laplacian = LaplacianOperator()
equation = HeatEquation(alpha=0.1, laplacian=laplacian, boundary_conditions=bc)

# Time integration: Explicit Euler
integrator = ExplicitEulerIntegrator()

# Solver
solver = Solver(integrator=integrator, equation=equation)

# Run simulation: dt=0.001, 1000 steps, save every 10 steps
snapshots = solver.run(
    initial=u0,
    dt=0.001,
    n_steps=1000,
    save_every=10
)

print(f"Computed {len(snapshots)} snapshots")
print(f"Final state norm: {snapshots[-1].norm():.6f}")
```

### Example 2: Schrödinger Equation in Potential Well

```python
from pde_framework import (
    Grid1D, ScalarField, DirichletBC,
    SchrodingerEquation, CrankNicolson1DIntegrator, Solver
)

# Domain: [-10, 10]
grid = Grid1D(x_min=-10.0, x_max=10.0, n_points=201)

# Initial state: Gaussian packet (complex)
psi0 = ScalarField(grid)
psi0.apply_function(lambda x: np.exp(-x**2 / 4 + 1j * x))

# Boundary conditions: homogeneous Dirichlet
bc = DirichletBC(left_value=0.0, right_value=0.0)

# Schrödinger equation: i·∂ψ/∂t = -½·∇²ψ + V(x)·ψ
# With harmonic potential: V(x) = ½·x²
potential = lambda x: 0.5 * x**2

equation = SchrodingerEquation(
    laplacian=LaplacianOperator(),
    potential=potential,
    boundary_conditions=bc
)

# Implicit time stepping (Crank-Nicolson) for better stability
integrator = CrankNicolson1DIntegrator()

solver = Solver(integrator=integrator, equation=equation)

# Simulate: dt=0.01, t_end=10
snapshots = solver.run(
    initial=psi0,
    dt=0.01,
    t_end=10.0,
    save_every=5
)
```

### Example 3: Different Boundary Conditions

```python
from pde_framework import (
    Grid1D, ScalarField,
    NeumannBC, PeriodicBC, RobinBC,
    LaplacianOperator, HeatEquation,
    RK4Integrator, Solver
)

grid = Grid1D(x_min=0.0, x_max=1.0, n_points=51)

# Neumann BC: ∂u/∂x = 0 at boundaries (insulated)
bc_neumann = NeumannBC(left_gradient=0.0, right_gradient=0.0)

# Periodic BC: u(0) = u(1), ∇u(0) = ∇u(1)
bc_periodic = PeriodicBC()

# Robin BC: u + λ·∂u/∂n = 0 at boundaries
bc_robin = RobinBC(alpha=1.0, beta=0.1, gamma=0.0)

# Choose BC and solve
equation = HeatEquation(
    alpha=0.1,
    laplacian=LaplacianOperator(),
    boundary_conditions=bc_neumann  # Try different BCs
)

initial = ScalarField(grid)
initial.apply_function(lambda x: np.sin(np.pi * x))

solver = Solver(integrator=RK4Integrator(), equation=equation)
snapshots = solver.run(initial, dt=0.01, n_steps=100, save_every=10)
```

## Advanced Features

### Custom Validation and Time Step Control

```python
# HeatEquation has built-in CFL stability check
from pde_framework import validate_heat_cfl_1d

# Check if time step is stable (raises ValueError if unstable)
validate_heat_cfl_1d(grid, dt, alpha=0.1, raise_error=True)

# Or just warn instead of raising
validate_heat_cfl_1d(grid, dt, alpha=0.1, raise_error=False)
```

### Field Operations

```python
# Arithmetic
u = ScalarField(grid)
v = ScalarField(grid)

u.apply_function(lambda x: np.sin(x))
v.apply_function(lambda x: np.cos(x))

w = u + v  # Addition
d = u - v  # Subtraction
s = u * 2.5  # Scalar multiplication

# Norms and metrics
norm_u = u.norm()  # L² norm
norm_inf = np.max(np.abs(u.data))  # L∞ norm

# Deep copy
u_copy = u.copy()
```

### Custom Equations

To implement a custom PDE, subclass `IEquation`:

```python
from pde_framework.equations import IEquation
from pde_framework.field import ScalarField
import numpy as np

class AdvectionEquation(IEquation):
    """Simple advection: ∂u/∂t + c·∂u/∂x = 0"""
    
    def __init__(self, velocity, gradient_op):
        self.c = velocity
        self.gradient = gradient_op
    
    def rhs(self, state: ScalarField) -> np.ndarray:
        """Return du/dt = -c·∇u"""
        grad = self.gradient.apply(state)
        return -self.c * grad.data
    
    def __repr__(self):
        return f"AdvectionEquation(c={self.c})"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=pde_framework --cov-report=html

# Run specific test suite
pytest tests/calculation_core/ -v
pytest tests/integrators/ -v

# Run specific test
pytest tests/test_import_package.py::test_import_package -v
```

## Architecture Decisions

### Why Separation of Concerns?

1. **calculation_core** is pure Python+NumPy+Numba with no OOP — allows:
   - Easy testing of mathematical correctness
   - Potential GPU acceleration (Numba CUDA)
   - Reuse from non-Python tools
   - Clear performance characteristics

2. **Field/Grid** structures are passive data holders — allows:
   - Serialization and debugging
   - Memory-efficient operations
   - Type safety without overhead

3. **IOperator/IEquation/IIntegrator** interfaces — allows:
   - Swapping implementations easily
   - Testing with mock objects
   - Future extensions (2D, 3D grids)

### Performance Considerations

- If possible, numerical kernels use `@njit(cache=True)`
- `ScalarField` uses complex dtype internally to allow for quantum simulations without separate classes
- `Grid1D` uses `@dataclass(frozen=True, slots=True)` for memory efficiency
- Lazy operations where possible (no unnecessary copies)

## Requirements

- **Python:** ≥3.11
- **Core dependencies:**
  - numpy ≥1.20 — array operations
  - numba ≥0.55 — JIT compilation
  - matplotlib ≥3.5 — visualization
  - jupyter ≥1.0 — interactive notebooks
- **Development:**
  - pytest ≥6.0 — testing
  - pytest-cov ≥2.12 — coverage reporting
  - pyright ≥1.1.300 — static type checking
  - mypy ≥0.910 — type checking
  - ruff ≥0.15.0 — formatting and linting

## Documentation

- **API Documentation** — Available in docstrings (use `help()` or IDE tooltips)
- **Examples:**
  - `notebooks/01_grid_field_and_operators.ipynb` — Core concepts
  - `notebooks/02_heat_equation_solver.ipynb` — Heat equation tutorial
  - `notebooks/02_wave_equation_examples.ipynb` — Wave propagation
  - `notebooks/03_heat_equation_gaussian.ipynb` — Advanced example
  - `SE_assignment.ipynb` — Schrödinger equation assignment

## Known Limitations

1. **1D Only:** Current implementation focuses on 1D domains (Grid1D)
2. **Serial Computation:** No MPI or GPU support yet
3. **Explicit Euler:** Limited by CFL condition for parabolic PDEs
4. **No Adaptive Grids:** Fixed uniform grids only
5. **No Visualization:** Plotting left to matplotlib/user code

## Future Roadmap

- [ ] 2D grid support (`Grid2D`)
- [ ] 3D grid support (`Grid3D`)
- [ ] Implicit solvers for stiff systems
- [ ] GPU acceleration via Numba CUDA
- [ ] Visualization utilities module
- [ ] MPI parallelization
- [ ] Adaptive mesh refinement

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest tests/`
2. Type hints are complete: `mypy pde_framework/`
3. Code is formatted: `ruff format pde_framework/ tests/`
4. Code passes linting: `ruff check pde_framework/ tests/`
5. Docstrings follow NumPy style
6. New features include tests

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this framework in academic work, please cite:

```bibtex
@software{pde_kodik_2026,
  title={PDE Kodik: Modular Finite Difference Solver},
  author={\v{C}erve\v{n}an J.},
  year={2026},
  url={<repository-url>}
}
```

## Development

### Setting Up the Development Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dev dependencies
pip install -e ".[dev]"
```

### Code Quality

The project uses multiple tools for code quality and type checking:

**Ruff** — Fast Python linter combining flake8, isort, and more:
```bash
# Format code
ruff format pde_framework/ tests/

# Check code
ruff check pde_framework/ tests/

# Fix issues automatically
ruff check --fix pde_framework/ tests/
```

**Type Checking** — Pyright and Mypy:
```bash
# Run type checking
pyright pde_framework/ tests/
mypy pde_framework/ tests/
```

**Testing with Coverage**:
```bash
pytest --cov=pde_framework tests/
```

### Workflow Summary

1. Make changes
2. Format: `ruff format pde_framework/ tests/`
3. Lint: `ruff check --fix pde_framework/ tests/`
4. Type check: `pyright pde_framework/ tests/`
5. Test: `pytest tests/`

## API Design Principles

1. **Separation of concerns**: Pure kernels in `calculation_core` are independent of OOP layers
2. **Interface-driven**: All major components have abstract interfaces (IGrid, IField, IBoundaryCondition, etc.)
3. **Immutability by default**: Operators and integrators return new objects; in-place operations are explicit
4. **Type safety**: Full type annotations for all public APIs
5. **NumPy compatibility**: Native NumPy array integration throughout

## References

- Numba JIT Compilation: https://numba.readthedocs.io/
- NumPy API Reference: https://numpy.org/doc/stable/
- Ruff Documentation: https://docs.astral.sh/ruff/
- Pyright Documentation: https://github.com/microsoft/pyright

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation and examples
- Run tests to verify your environment: `pytest tests/ -v`

---

**Status:** Mid-stage development with core features implemented and tested. Actively adding new features and improving documentation. 
Contribution suggestions welcome! Reach out mail at jancervenan@gmail.com, GitHub issues, or discussions.
