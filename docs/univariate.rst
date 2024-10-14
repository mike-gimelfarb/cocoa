COCOA Univariate Optimizers
====================================

The optimization problem for a univariate function on a closed interval can be described as

.. math::

	\min_{x} f(x), \quad x \in [lb, ub]. 
	
COCOA provides a number of algorithms for optimizing univariate functions.


Branch and Bound
-------------------

The branch and bound technique for univariate functions is described in this paper:

* Aaid, Djamel, Amel Noui, and Mohand Ouanes. "New technique for solving univariate global optimization." Archivum Mathematicum 53.1 (2017): 19-33.

In summary, branch and bound is a search technique that finds a global minimum of a univariate function,
which also enjoys strong theoretical convergence properties. It uses an iterative process of splitting the 
optimization interval into disjoint subintervals until the minimum has been found. A quadratic
underestimator of the objective is also fit to guide the division.

.. function:: BranchAndBound(mfev, tol, K, n=16)

   Initializes a new branch and bound optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the estimated error of the solution is lower than this value.
   :type tol: float
   :param K: Upper bound on the second derivative of the objective function.
   :type K: float
   :param n: Number of quadratic functions to use for the underestimator.
   :type n: int
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch
  
  
Brent's Methods
-------------------

The Brent's methods for local and global search are described in this book:

* Brent, Richard P. Algorithms for minimization without derivatives. Courier Corporation, 2013.

Both versions of his methods are particularly robust for optimizing univariate functions.
The local variant finds a local minimum of a univariate function by adaptive combining golden-section
search with parabolic interpolation.

.. function:: BrentSearch(mfev, atol, rtol = 1e-15)

   Initializes a new branch and bound optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param atol: Terminate when the estimated error of the solution is lower than this value (absolute tolerance). 
   :type atol: float
   :param rtol: Terminate when the estimated error of the solution is lower than this value (relative tolerance). 
   :type rtol: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch
  
The global variant finds a global minimum under the assumption that the second derivative of the objective is upper-bounded.
This is the translation of the [``GLOMIN`` Fortran subroutine by John Burkardt](https://people.math.sc.edu/Burkardt/f77_src/brent/brent.html).
  
.. function:: GlobalBrentSearch(mfev, tol, bound_on_hessian)

   Initializes a new branch and bound optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the estimated error of the solution is lower than this value. 
   :type tol: float
   :param bound_on_hessian: Upper bound on the second derivative of the objective function.
   :type bound: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch
