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

.. function:: Brent(mfev, atol, rtol = 1e-15)

   Initializes a new local Brent optimizer with the specified parameters.

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
  
.. function:: GlobalBrent(mfev, tol, bound_on_hessian)

   Initializes a new global Brent optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the estimated error of the solution is lower than this value. 
   :type tol: float
   :param bound_on_hessian: Upper bound on the second derivative of the objective function.
   :type bound_on_hessian: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch

  
Calvin's Method
-------------------

The Calvin's method for univariate functions is described in this paper:

* Calvin, James M., Yvonne Chen, and Antanas Žilinskas. "An adaptive univariate global optimization algorithm and its convergence rate for twice continuously differentiable functions." Journal of Optimization Theory and Applications 155 (2012): 628-636.

In summary, this is a global search method that uses an interval splitting technique to find the global minimum.
It enjoys strong theoretical asymptotic convergence properties for continuous functions sampled randomly according to the Wiener measure.

.. function:: Calvin(mfev, tol, lam = 16)

   Initializes a new Calvin optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the estimated error of the solution is lower than this value. 
   :type tol: float
   :param lam: The lambda parameter specified in the paper (must be at least 16).
   :type lam: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch

   
Davies-Swann-Campey Method
-------------------

This method is described in detail in this book:

* Antoniou, Andreas, and Wu-Sheng Lu. Practical optimization. Springer, 2007.

This method is not widely known or implemented, but it appears to be very robust 
in finding local minima of smooth univariate functions. It takes decreasingly 
smaller steps along a direction until a bracket for the minimum is located.

.. function:: DSC(mfev, tol, decay = 0.1)

   Initializes a new Davies-Swann-Campey optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the estimated error of the solution is lower than this value. 
   :type tol: float
   :param decay: How much to decay the step size in each iteration.
   :type decay: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch

   
Fibonacci Search
-------------------

The Fibonacci search is described in detail in this paper:

* Kiefer, J. (1953), "Sequential minimax search for a maximum", Proceedings of the American Mathematical Society, 4 (3): 502–506

The Fibonacci search is a simple algorithm for finding the local minimum of a 
strictly unimodal, but not necessarily continuous, function.
This overall strategy works by reducing the interval of uncertainty in every step, 
ultimately converging the interval, containing the minimizer, to a desired small size.
Specifically, the search interval is divided into two parts that have sizes 
proportional to two consecutive Fibonacci numbers.

.. function:: Fibonacci(mfev, atol, rtol = 1e-15)

   Initializes a new Fibonacci optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param atol: Terminate when the estimated error of the solution is lower than this value (absolute tolerance). 
   :type atol: float
   :param rtol: Terminate when the estimated error of the solution is lower than this value (relative tolerance). 
   :type rtol: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch
 
 
Golden Section Search
-------------------

The golden section search is described in detail in this paper:

* Kiefer, J. (1953), "Sequential minimax search for a maximum", Proceedings of the American Mathematical Society, 4 (3): 502–506

This algorithm can be seen as a limit of the Fibonacci search, in the sense that the ratio 
of two consecutive Fibonacci numbers approaches the golden ratio. Like the Fibonacci search,
this algorithm works when the objective function is strictly unimodal, but not necessarily continuous.

.. function:: GoldenSection(mfev, atol, rtol = 1e-15)

   Initializes a new golden section search optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param atol: Terminate when the estimated error of the solution is lower than this value (absolute tolerance). 
   :type atol: float
   :param rtol: Terminate when the estimated error of the solution is lower than this value (relative tolerance). 
   :type rtol: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch
 
 
Piyavskii's Method
-------------------

The Piyavskii method as implemented in this package is described in the following paper:

* Lera, Daniela, and Yaroslav D. Sergeyev. "Acceleration of univariate global optimization algorithms working with Lipschitz functions and Lipschitz first derivatives." SIAM Journal on Optimization 23.1 (2013): 508-529.
 
 
The Piyavskii method is suitable for finding the global minimum of a univariate Lipschitz-continuous function.
The original algorithm requires the Lipschitz constant to be properly estimated in order for the method to be effective.
It uses the Lipschitz property to define a piecewise linear support function over the search space that bounds the original objective.
The version implemented here estimates the Lipschitz constant adaptively, and thus does not require the constant to be specified a-priori.

.. function:: Piyavskii(mfev, tol, r = 1.4, xi = 1e-6)

   Initializes a new adaptive Piyavskii optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the estimated error of the solution is lower than this value. 
   :type tol: float
   :param r: The r parameter specified in the paper for estimating the Lipschitz constant.
   :type r: float
   :param xi: The xi parameter specified in the paper for estimating the Lipschitz constant.
   :type xi: float
   :returns: optimizer instance
   :rtype: object of type UnivariateSearch
   