Introduction
============

bboptpy is a large suite of efficient algorithms written in C++ for the optimization of black-box functions.

Main advantages:

- single unified interface for Python with a user-friendly API
- variety of classical and modern algorithms with state-of-the-art improvements that are robust and very fast
- transparent implementation that makes it easy to build upon.

Installation
-------------------

Install directly from pip:

.. code-block:: shell

    pip install bboptpy

Basic Univariate Example
-------------------

The following example optimizes a univariate sinusoidal function using the Brent method:

.. code-block:: python

	import numpy as np
	from bboptpy import Brent
	
	# function to optimize
	def fx(x):
	    return np.sin(x) + np.sin(10 * x / 3)
	
	alg = Brent(mfev=20000, atol=1e-6)
	sol = alg.optimize(fx, lower=2.7, upper=7.5, guess=np.random.uniform(2.7, 7.5))
	print(sol)

This will print the following output:

.. code-block:: shell

	x*: 5.1457349293974861
	calls to f: 10
	converged: 1

which indicates the algorithm has found a local minimum (in this case, also a global minimum).

Basic Multivariate Example
-------------------

The following example optimizes the 10-dimensional <Rosenbrock function `https://en.wikipedia.org/wiki/Rosenbrock_function`>_
using the active variant of the CMA-ES optimizer:

.. code-block:: python

    import numpy as np
    from bboptpy import ActiveCMAES

    # function to optimize
    def fx(x):
        return sum((100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2) for x1, x2 in zip(x[:-1], x[1:]))

    n = 10  # dimension of problem
    alg = ActiveCMAES(mfev=10000, tol=1e-4, np=20)
    sol = alg.optimize(fx,
                       lower=-10 * np.ones(n),
                       upper=10 * np.ones(n),
                       guess=np.random.uniform(low=-10., high=10., size=n))
    print(sol)

This will print the following output:

.. code-block:: shell

    x*: 0.999989 0.999999 1.000001 1.000007 1.000020 1.000029 1.000102 1.000183 1.000357 1.000689 
    objective calls: 6980
    constraint calls: 0
    B/B constraint calls: 0
    converged: yes

which indicates the algorithm has found a local minimum (in this case, also a global minimum).
