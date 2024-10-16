# cocoa

COCOA (COllection of Continuous Optimization Algorithms) is a large suite of efficient algorithms written in C++ for the optimization of continuous 
black-box functions (mostly without using derivative information). 

Main advantages:
- provides a single unified interface for all algorithms
- provides a variety of classical algorithms with state-of-the-art improvements 
(e.g. automatic parameter adaptation) 
- convenient wrappers for Python with a user-friendly API

# Installation

To use this library in a Python project, you will need a C++ compiler for Python and git (e.g. ``conda install git``). 
Then install directly:

```
pip install git+https://github.com/mike-gimelfarb/cocoa
```

# Algorithms Supported

The following algorithms are currently fully supported with Python wrappers:

* Univariate:
    * [Branch and Bound](https://eudml.org/doc/287965)
    * Brent Methods:
        * [Local Brent](https://books.google.ca/books/about/Algorithms_for_Minimization_Without_Deri.html?id=AITCAgAAQBAJ&redir_esc=y)
        * [Global Brent](https://books.google.ca/books/about/Algorithms_for_Minimization_Without_Deri.html?id=AITCAgAAQBAJ&redir_esc=y)
    * [Calvin Method](https://dl.acm.org/doi/abs/10.5555/2699214.2699215)
    * [Davies-Swann-Campey Method](https://link.springer.com/book/10.1007/978-1-0716-0843-2)
    * [Fibonacci Algorithm](https://en.wikipedia.org/wiki/Fibonacci_search_technique)
    * [Golden Section Search](https://en.wikipedia.org/wiki/Golden-section_search)
    * [Piyavskii Method](https://epubs.siam.org/doi/10.1137/110859129)
* Multivariate:
    * Unconstrained:
        * [Adaptive Coordinate Descent (ACD)](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_21)
        * [AMaLGaM IDEA](https://dl.acm.org/doi/10.1145/1570256.1570313)
        * [Basin Hopping](https://pubs.acs.org/doi/10.1021/jp970984n)
        * [Controlled Random Search (CRS)](https://link.springer.com/article/10.1007/s10957-006-9101-0)
        * Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES):
            * [Vanilla CMA-ES](https://ieeexplore.ieee.org/document/6790628/)
            * [Active CMA-ES](https://ieeexplore.ieee.org/document/1688662)
            * [Cholesky CMA-ES](https://papers.nips.cc/paper_files/paper/2016/file/289dff07669d7a23de0ef88d2f7129e7-Paper.pdf)
            * [Limited Memory CMA-ES](https://dl.acm.org/doi/10.1145/2576768.2598294)
            * [Separable CMA-ES](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30)
            * [IPOP CMA-ES](https://ieeexplore.ieee.org/document/1554902)
            * [BIPOP CMA-ES](https://link.springer.com/chapter/10.1007/978-3-642-32937-1_30)
        * Differential Evolution (DE):
            * [JADE](https://ieeexplore.ieee.org/document/4424751)
            * [L-SHADE](https://ieeexplore.ieee.org/abstract/document/6900380)
            * [SANSDE](https://ieeexplore.ieee.org/document/4630935/)
            * [SSDE](https://www.mdpi.com/2227-7390/10/23/4519#B47-mathematics-10-04519)
        * [Differential Search (DSA)](https://www.sciencedirect.com/science/article/abs/pii/S0098300411004353)
        * [Exponential Natural Evolution Strategy (xNES)](https://dl.acm.org/doi/10.1145/1830483.1830557)
        * [Novel Self-Adaptive Harmony Search (NSHS)](https://onlinelibrary.wiley.com/doi/10.1155/2013/653749)
        * [Hessian Evolutionary Strategy (HE-ES)](https://link.springer.com/chapter/10.1007/978-3-030-58112-1_41)
        * [Self-Adaptive Multi-Population JAYA](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8640077)
        * [Adaptive Nelder-Mead Method](https://www.tandfonline.com/doi/full/10.1080/0305215X.2019.1688315)
        * Particle Swarm Optimization (PSO):
            * [Adaptive PSO (APSO)](https://ieeexplore.ieee.org/document/4812104)
            * [Competitive PSO (CSO)](https://link.springer.com/chapter/10.1007/978-981-13-0761-4_9)
            * [Cooperative Co-Evolving PSO (CCPSO)](https://ieeexplore.ieee.org/document/5910380)
            * [Self-Learning PSO (SLPSO)](https://ieeexplore.ieee.org/document/6069879)
        * Powell's Methods:
            * [BOBYQA](https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf)
            * [NEWUOA](https://link.springer.com/chapter/10.1007/0-387-30065-1_16)
        * [PRAXIS](https://link.springer.com/article/10.3758/BF03203605)
        * [Rosenbrock Method](https://academic.oup.com/comjnl/article/12/1/69/311651)
       
# Usage

Simple example to optimize a univariate function:

```python
import numpy as np
from cocoaopt import Brent

# function to optimize
def fx(x):
    return np.sin(x) + np.sin(10 * x / 3)

alg = Brent(mfev=20000, atol=1e-6)
sol = alg.optimize(fx, lower=2.7, upper=7.5, guess=np.random.uniform(2.7, 7.5))
print(sol)
```

This will print the following output:

```
x*: 5.1457349293974861
calls to f: 10
converged: 1
```

Simple example to optimize a multivariate function:

```python
import numpy as np
from cocoaopt import ActiveCMAES

# function to optimize
def fx(x):
    total = 0.0
    for x1, x2 in zip(x[:-1], x[1:]):
        total += 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    return total

n = 10  # dimension of problem
alg = ActiveCMAES(mfev=10000, tol=1e-4, np=20)
sol = alg.optimize(fx, lower=-10 * np.ones(n), upper=10 * np.ones(n), guess=np.random.uniform(-10, 10, size=n))
print(sol)
```

This will print the following output:

```
x*: 0.999989 0.999999 1.000001 1.000007 1.000020 1.000029 1.000102 1.000183 1.000357 1.000689 
objective calls: 6980
constraint calls: 0
B/B constraint calls: 0
converged: yes
```

# Citation

To cite this repository, either use the link in the sidebar, or the following bibtext entry:

```
@software{gimelfarb2024cocoa,
author = {Gimelfarb, Michael},
license = {LGPL-2.1+},
title = {{COCOA: COllection of Continuous Optimization Algorithms}},
url = {https://github.com/mike-gimelfarb/cocoa},
year = {2024}
}
```

To cite the authors of the individual algorithms, please see the references in the comments heads of the respective C++ source files.
