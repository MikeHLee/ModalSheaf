# Handoff Document: PyDelt Documentation Upgrade

This document captures the plan for upgrading PyDelt's documentation to match the
pedagogical approach developed for ModalSheaf.

---

## Current State of PyDelt

**Location**: `/Users/Michaellee/Documents/Runes/pydelt/`

**Purpose**: Numerical differentiation and integration for time series data.

**Current Modules**:
- `derivatives.py` — Numerical differentiation
- `integrals.py` — Numerical integration
- `autodiff.py` — Automatic differentiation
- `interpolation.py` — Interpolation methods
- `multivariate.py` — Multivariate calculus
- `stochastic.py` — Stochastic calculus
- `tensor_derivatives.py` — Tensor calculus

---

## Proposed Documentation Structure

Following the ModalSheaf pattern, create educational documentation that teaches
the underlying mathematics gently.

### Theory Section (New)

```
docs/theory/
├── 00_why_calculus.md          # Why calculus for ML practitioners
├── 01_functions_and_limits.md  # Foundation: what is a function?
├── 02_derivatives_intuition.md # Rates of change, slopes, sensitivity
├── 03_differentiation_rules.md # Chain rule, product rule, etc.
├── 04_integration_intuition.md # Accumulation, area, inverse of derivative
├── 05_approximation_theory.md  # Taylor series, function approximation
├── 06_multivariate_calculus.md # Gradients, Jacobians, Hessians
├── 07_complex_analysis.md      # (If applicable) Analytic functions
└── 08_applications_to_ml.md    # Backprop, optimization, physics-informed NN
```

### Key Concepts to Cover

#### Why Derivatives Matter for ML

1. **Gradient descent**: Training neural networks
2. **Sensitivity analysis**: How inputs affect outputs
3. **Taylor approximation**: Local linear/quadratic models
4. **Physics-informed ML**: Differential equations as constraints

#### Why Integration Matters for ML

1. **Probability**: PDFs, CDFs, expectations
2. **Cumulative effects**: Time series analysis
3. **Energy/loss functions**: Integral formulations
4. **Neural ODEs**: Continuous-depth networks

---

## Pedagogical Approach

### From ModalSheaf (Apply to PyDelt)

1. **Start with intuition**: Real-world examples before formulas
2. **Build gradually**: Simple → complex
3. **Connect to ML**: Every concept linked to practical use
4. **Cite sources**: Academic rigor with accessibility
5. **Code examples**: Theory immediately applied

### Suggested Narrative Arc

```
Chapter 1: Why Calculus?
  "Calculus is the mathematics of change. In ML, everything changes:
   weights during training, predictions with inputs, loss over time."

Chapter 2: Functions and Limits
  "A function is a machine: input → output. A limit asks: what happens
   as we get infinitely close to something?"

Chapter 3: Derivatives
  "The derivative measures instantaneous rate of change. It's the slope
   of the tangent line. It tells you: if I nudge the input, how much
   does the output change?"

Chapter 4: Differentiation Rules
  "We don't compute derivatives from scratch. We use rules that let us
   differentiate complex functions by breaking them into simple parts."

Chapter 5: Integration
  "Integration is the inverse of differentiation. It accumulates change
   over time. It computes areas, volumes, and expectations."

Chapter 6: Approximation
  "Taylor series let us approximate any smooth function with polynomials.
   This is the foundation of numerical methods and neural network theory."

Chapter 7: Multivariate
  "Real ML has millions of parameters. Multivariate calculus extends
   derivatives to functions of many variables: gradients, Jacobians, Hessians."

Chapter 8: Applications
  "Backpropagation is just the chain rule. Optimization is gradient descent.
   Physics-informed networks use differential equations as loss functions."
```

---

## Potential Name Change

Current name: **PyDelt** (Python Derivatives/Deltas)

Considerations:
- Not immediately intuitive
- Could be confused with "delta" in other contexts

Possible alternatives:
- **calcupy** — Calculus for Python (check availability)
- **diffint** — Differentiation and Integration
- **smoothgrad** — Smooth gradients (but too specific)
- **funculus** — Function calculus (playful)
- **derivint** — Derivatives and Integrals

**Recommendation**: Keep PyDelt but add a tagline:
> "PyDelt: Practical Calculus for Data Scientists"

---

## Complex Analysis Fit

**Question**: Is complex analysis a good fit for PyDelt?

**Answer**: Partially.

### Relevant Parts

1. **Analytic functions**: Smooth, infinitely differentiable
2. **Contour integration**: Useful for some signal processing
3. **Residue theorem**: Computing certain integrals
4. **Conformal maps**: Preserving angles (relevant for some ML)

### Less Relevant Parts

1. **Riemann surfaces**: Too abstract for most ML
2. **Algebraic geometry**: Beyond scope
3. **Sheaves on complex manifolds**: (That's ModalSheaf territory!)

### Recommendation

Include a brief chapter on complex analysis focusing on:
- Complex numbers as 2D vectors
- Euler's formula and its applications
- Analytic functions and their special properties
- Connection to Fourier analysis

---

## Bibliography for PyDelt

### Foundational Texts

1. **Strang, G.** *Calculus*. (Free online, excellent for intuition)
2. **Spivak, M.** *Calculus*. (Rigorous but readable)
3. **Apostol, T.** *Calculus*. (Comprehensive reference)

### For ML Practitioners

4. **Goodfellow et al.** *Deep Learning*. Chapter 4: Numerical Computation.
5. **Boyd & Vandenberghe.** *Convex Optimization*. (Gradients, Hessians)
6. **Baydin et al.** "Automatic Differentiation in Machine Learning: A Survey."

### Numerical Methods

7. **Press et al.** *Numerical Recipes*. (Classic reference)
8. **Trefethen, L.N.** *Approximation Theory and Approximation Practice*.

### Complex Analysis (Optional)

9. **Needham, T.** *Visual Complex Analysis*. (Beautiful, intuitive)
10. **Ahlfors, L.** *Complex Analysis*. (Standard graduate text)

---

## Action Items for Next Session

1. [ ] Review current PyDelt documentation
2. [ ] Create `docs/theory/` directory structure
3. [ ] Write `00_why_calculus.md` following ModalSheaf style
4. [ ] Add ReadTheDocs configuration
5. [ ] Consider name change (gather feedback)
6. [ ] Add bibliography with proper citations
7. [ ] Create example notebooks linking theory to code

---

## Connection Between Projects

ModalSheaf and PyDelt are complementary:

| PyDelt | ModalSheaf |
|--------|------------|
| Local analysis (derivatives) | Global analysis (cohomology) |
| Single functions | Networks of functions |
| Rates of change | Consistency of change |
| Approximation | Assembly |

**Potential integration**: PyDelt could provide the numerical differentiation
for computing sheaf Laplacians and diffusion in ModalSheaf.

---

## Notes

- Both projects target ML practitioners who want to understand the math
- Educational documentation is a key differentiator
- Cite sources rigorously but explain accessibly
- Code examples should immediately follow theory
- Build intuition before formalism
