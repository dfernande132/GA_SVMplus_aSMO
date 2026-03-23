# aSMO: A Linear-Time SMO Solver for Large-Scale SVM+ Learning

This repository contains the official MATLAB implementation of **aSMO**, an efficient Sequential Minimal Optimization (SMO)-style solver designed for the **Learning Using Privileged Information (LUPI)** paradigm[cite: 9, 41]. 

[cite_start]Unlike generic Quadratic Programming (QP) solvers that typically scale as $O(n^3)$, this algorithm achieves a **linear-time complexity per iteration**, $O(n)$, enabling the training of SVM+ models with tens of thousands of samples[cite: 50, 406, 407].

## Key Features
* [cite_start]**Computational Efficiency**: Reduces the cost per iteration from $O(n^2)$ to a linear bound of $T_{iter} \le 4n$[cite: 50, 403].
* [cite_start]**Scalability**: Successfully processes up to 50,000 samples in less than four minutes[cite: 14, 635, 755].
* [cite_start]**Generalization**: Provides the first comprehensive implementation of an SMO-style solver for general binary SVM+ problems[cite: 15, 44, 761].
* [cite_start]**Numerical Precision**: Achieves objective function values virtually identical to exact QP solvers, with a relative difference below $1.2 \times 10^{-4}$[cite: 16, 601, 754].
* [cite_start]**Open Source**: The first publicly available tool of its kind to bridge the gap between SVM+ theory and practical deployment[cite: 15, 52, 765].

---

## Function Reference: `solve_asmo`

[cite_start]The core solver optimizes the dual objective function of SVM+ by operating on irreducible sets of variables[cite: 10, 184, 191].

### Input Parameters
| Parameter | Description |
| :--- | :--- |
| `fv` | [cite_start]Decision space feature matrix $X$ ($n \times d$)[cite: 153, 163, 861]. |
| `fvStar` | [cite_start]Privileged information feature matrix $X^*$ ($n \times d^*$)[cite: 153, 163, 861]. |
| `lbl` | [cite_start]Binary class labels $y_i \in \{-1, +1\}$[cite: 153, 75, 861]. |
| `C` | [cite_start]Regularization parameter balancing margin and errors[cite: 166, 73, 861]. |
| `gamma` | [cite_start]Hyperparameter $\gamma$ controlling the influence of privileged info[cite: 165, 102, 861]. |
| `sgmPlus` | [cite_start]Gaussian RBF kernel width for the decision space[cite: 81, 89, 861]. |
| `sgmStar` | [cite_start]Gaussian RBF kernel width for the privileged space[cite: 102, 585, 861]. |
| `opts` | [cite_start]Struct with `tol` (KKT threshold), `maxIter`, and `kappa` (min step)[cite: 591, 283]. |

### Output Parameters
| Parameter | Description |
| :--- | :--- |
| `z` | Concatenated vector of Lagrange multipliers $[\alpha; [cite_start]\beta]$[cite: 62, 172, 861]. |
| `fval` | [cite_start]Value of the dual objective function $J(\alpha, \beta)$ at convergence[cite: 173, 601, 861]. |
| `bPlus` | [cite_start]Bias term for the decision boundary[cite: 378, 379, 861]. |
| `bStar` | [cite_start]Bias term for the correcting function in privileged space[cite: 161, 378, 861]. |

---

## Usage Example

To reproduce the results presented in the paper using the MNIST benchmark[cite: 411, 431]:

1. **Prepare Data**: Load the subsampled MNIST dataset (digits 5 vs 8).
   ```matlab
   load('train_reducted.mat'); % Contains fv, fvStar, and lbl

## 📝 Citation
If you use this code or the aSMO algorithm in your research, please cite the following paper:

Fernández, J. D., Vega, J., & Dormido-Canto, S. (2026). aSMO: A Linear-Time SMO Solver for Large-Scale SVM+ Learning.

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.

