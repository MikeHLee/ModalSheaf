# Bibliography & Sources

This document cites the key sources that informed the design and theory behind ModalSheaf.

---

## Primary Sources

### Foundational Sheaf Theory

1. **Robinson, M. (2014)**. *Topological Signal Processing*. Springer.
   - The foundational text connecting sheaf theory to signal processing
   - Introduces sheaf Laplacian and diffusion
   - Source for our consistency checking approach
   - [Springer Link](https://link.springer.com/book/10.1007/978-3-642-36104-3)

2. **Robinson, M. (2017)**. "Sheaves are the canonical data structure for sensor integration."
   *Information Fusion*, 36, 208-224.
   - Key paper arguing sheaves are natural for sensor fusion
   - Source for our coordinate frame gluing
   - [arXiv:1603.01446](https://arxiv.org/abs/1603.01446)

3. **Curry, J. (2014)**. *Sheaves, Cosheaves and Applications*. PhD Thesis, University of Pennsylvania.
   - Comprehensive treatment of computational sheaf theory
   - Introduces cellular sheaves for computation
   - [arXiv:1303.3255](https://arxiv.org/abs/1303.3255)

### Neural Sheaf Diffusion

4. **Bodnar, C., et al. (2022)**. "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs."
   *NeurIPS 2022*.
   - Applies sheaf theory to graph neural networks
   - Source for our sheaf Laplacian implementation
   - [arXiv:2202.04579](https://arxiv.org/abs/2202.04579)

5. **Hansen, J. & Ghrist, R. (2019)**. "Toward a spectral theory of cellular sheaves."
   *Journal of Applied and Computational Topology*, 3(4), 315-358.
   - Rigorous spectral theory for sheaves
   - Mathematical foundation for diffusion
   - [arXiv:1808.01513](https://arxiv.org/abs/1808.01513)

### Sheaves in Machine Learning

6. **Ayzenberg, A. (2025)**. "Sheaf theory: from deep geometry to deep learning."
   - Survey connecting sheaf theory to modern deep learning
   - Excellent overview of the field
   - [arXiv:2502.15476](https://arxiv.org/abs/2502.15476)

7. **Barbero, F., et al. (2022)**. "Sheaf Neural Networks with Connection Laplacians."
   *ICML 2022 Workshop on Topology, Algebra, and Geometry in ML*.
   - Practical neural network architectures using sheaves
   - [arXiv:2206.08702](https://arxiv.org/abs/2206.08702)

### Topological Data Analysis

8. **Carlsson, G. (2009)**. "Topology and data."
   *Bulletin of the American Mathematical Society*, 46(2), 255-308.
   - Foundational paper on topological data analysis
   - Introduces persistent homology
   - [AMS Link](https://www.ams.org/journals/bull/2009-46-02/S0273-0979-09-01249-X/)

9. **Ghrist, R. (2008)**. "Barcodes: The persistent topology of data."
   *Bulletin of the American Mathematical Society*, 45(1), 61-75.
   - Accessible introduction to persistent homology
   - [AMS Link](https://www.ams.org/journals/bull/2008-45-01/S0273-0979-07-01191-3/)

### Category Theory Background

10. **Mac Lane, S. & Moerdijk, I. (1994)**. *Sheaves in Geometry and Logic*.
    Springer.
    - The classic graduate text on sheaves
    - Rigorous categorical treatment
    - [Springer Link](https://link.springer.com/book/10.1007/978-1-4612-0927-0)

11. **Fong, B. & Spivak, D. (2019)**. *An Invitation to Applied Category Theory*.
    Cambridge University Press.
    - Accessible introduction to category theory for applications
    - [arXiv:1803.05316](https://arxiv.org/abs/1803.05316)

---

## Accessible Introductions

### For Beginners

12. **Robinson, M. (2013)**. "Hunting for Foxes with Sheaves."
    *AMS Notices*, 60(8).
    - Very accessible introduction using a hunting metaphor
    - Highly recommended first read
    - [AMS PDF](https://www.ams.org/notices/201308/rnoti-p1012.pdf)

13. **Ghrist, R. (2014)**. *Elementary Applied Topology*.
    - Free online textbook
    - Covers homology, cohomology, and sheaves
    - [Free PDF](https://www.math.upenn.edu/~ghrist/notes.html)

### Video Lectures

14. **Robinson, M.** "Applied Sheaf Theory" lecture series.
    - DARPA-funded tutorial series
    - Available on YouTube
    - Excellent visual explanations

15. **The Bright Side of Mathematics** YouTube channel.
    - Topology playlist with intuitive explanations
    - Good for building geometric intuition

---

## Software & Implementations

### Related Libraries

16. **PySheaf** (Robinson, M.)
    - Original Python sheaf library
    - More mathematical, less ML-focused
    - [GitHub](https://github.com/kb1dds/pysheaf)

17. **GUDHI** (INRIA)
    - Comprehensive TDA library
    - Persistent homology, simplicial complexes
    - [GUDHI](https://gudhi.inria.fr/)

18. **Giotto-TDA**
    - Topological data analysis for ML
    - Scikit-learn compatible
    - [GitHub](https://github.com/giotto-ai/giotto-tda)

---

## Key Concepts by Source

| Concept | Primary Source | Section in ModalSheaf |
|---------|----------------|----------------------|
| Sheaf definition | Mac Lane & Moerdijk [10] | `theory/02_sheaves_intuition` |
| Restriction maps | Curry [3] | `modalities/transforms.py` |
| Gluing axiom | Robinson [1] | `gluing.py` |
| Sheaf Laplacian | Bodnar et al. [4] | `consistency.py` |
| Cohomology | Ghrist [13] | `theory/04_gluing_and_cohomology` |
| Sensor fusion | Robinson [2] | `examples/04_gluing.py` |
| Neural sheaves | Barbero et al. [7] | Future work |

---

## Citation

If you use ModalSheaf in your research, please cite:

```bibtex
@software{modalsheaf2024,
  title = {ModalSheaf: Practical Sheaf Theory for Multimodal ML},
  author = {Lee, Michael Harrison},
  year = {2024},
  url = {https://github.com/MikeHLee/modalsheaf}
}
```

---

## Acknowledgments

This library draws heavily on the pioneering work of:

- **Michael Robinson** (American University) — for making sheaf theory accessible and applicable
- **Robert Ghrist** (University of Pennsylvania) — for connecting topology to data science
- **Cristian Bodnar** (Cambridge) — for neural sheaf diffusion
- **Justin Curry** — for computational sheaf theory

The pedagogical approach is inspired by the "math for programmers" movement,
particularly the work of Jeremy Kun (*A Programmer's Introduction to Mathematics*)
and Grant Sanderson (3Blue1Brown).
