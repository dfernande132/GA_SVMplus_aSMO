# aSMO: A Linear-Time SMO Solver for Large-Scale SVM+ Learning

This repository contains the official MATLAB implementation of **aSMO**, an efficient Sequential Minimal Optimization (SMO)-style solver designed for the **Learning Using Privileged Information (LUPI)** paradigm. 

Unlike generic Quadratic Programming (QP) solvers that typically scale as $O(n^3)$, this algorithm achieves a **linear-time complexity per iteration**, $O(n)$, enabling the training of SVM+ models with tens of thousands of samples.

## Key Features
* **Computational Efficiency**: Reduces the cost per iteration from $O(n^2)$ to a linear bound of $T_{iter} \le 4n$.
* **Scalability**: Successfully processes up to 50,000 samples in less than four minutes.
* **Generalization**: Provides the first comprehensive implementation of an SMO-style solver for general binary SVM+ problems.
* **Numerical Precision**: Achieves objective function values virtually identical to exact QP solvers, with a relative difference below $1.2 \times 10^{-4}$.
* **Open Source**: The first publicly available tool of its kind to bridge the gap between SVM+ theory and practical deployment.

---

## Function Reference: `solve_asmo`

The core solver optimizes the dual objective function of SVM+ by operating on irreducible sets of variables.

### Input Parameters
| Parameter | Description |
| :--- | :--- |
| `fv` | [cite_start]Decision space feature matrix $X$ ($n \times d$). |
| `fvStar` | [cite_start]Privileged information feature matrix $X^*$ ($n \times d^*$). |
| `lbl` | [cite_start]Binary class labels $y_i \in \{-1, +1\}$. |
| `C` | [cite_start]Regularization parameter balancing margin and errors. |
| `gamma` | [cite_start]Hyperparameter $\gamma$ controlling the influence of privileged info. |
| `sgmPlus` | [cite_start]Gaussian RBF kernel width for the decision space. |
| `sgmStar` | [cite_start]Gaussian RBF kernel width for the privileged space. |
| `opts` | [cite_start]Struct with `tol` (KKT threshold), `maxIter`, and `kappa` (min step). |

### Output Parameters
| Parameter | Description |
| :--- | :--- |
| `z` | Concatenated vector of Lagrange multipliers $[\alpha; [cite_start]\beta]$. |
| `fval` | [cite_start]Value of the dual objective function $J(\alpha, \beta)$ at convergence. |
| `bPlus` | [cite_start]Bias term for the decision boundary. |
| `bStar` | [cite_start]Bias term for the correcting function in privileged space. |

---

## Usage Example

To reproduce the results presented in the paper using the MNIST benchmark:

1. **Prepare Data**: Load the subsampled MNIST dataset (digits 5 vs 8).
   ```matlab
   load('train_reducted.mat'); % Contains fv, fvStar, and lbl

## 📝 Citation
If you use this code or the aSMO algorithm in your research, please cite the following paper:

Fernández, J. D., Vega, J., & Dormido-Canto, S. (2026). aSMO: A Linear-Time SMO Solver for Large-Scale SVM+ Learning.

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
