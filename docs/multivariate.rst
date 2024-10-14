COCOA Multivariate Optimizers
====================================

The optimization problem for a multivariate function on a box-bounded support can be written as:

.. math::

	\min_{x_1, \dots x_n} f(x_1, \dots x_n), \quad x_i \in [lb_i, ub_i], \, i = 1 \dots n. 
	
COCOA provides a number of algorithms for optimizing multivariate functions.


Adaptive Coordinate Descent (ACD)
-------------------

This algorithm is described in the following paper:

* Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Adaptive coordinate descent." Proceedings of the 13th annual conference on Genetic and evolutionary computation. 2011.

ACD is a variation of coordinate descent that is more effective for non-separable objectives,
since it maintains a transformation of the coordinate system such that the variables are decorrelated
under the objective function. 

.. function:: ACD(mfev, xtol, ftol, ksucc = 2.0, kunsucc = 0.5)

   Initializes a new ACD optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param xtol: Terminate when the change in solution is less than this value.
   :type xtol: float
   :param ftol: Terminate when the change in objective value is less than this value.
   :type ftol: float
   :param ksucc: Increase factor for step size on successful update.
   :type ksucc: float
   :param kunsucc: Decrease factor for step size on unsuccessful update.
   :type kunsucc: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

AMaLGaM IDEA
-------------------

This algorithm is described in the following paper:

* Bosman, Peter AN, Jï¿½rn Grahl, and Dirk Thierens. "AMaLGaM IDEAs in
noiseless black-box optimization benchmarking." Proceedings of the 11th
Annual Conference Companion on Genetic and Evolutionary Computation
Conference: Late Breaking Papers. ACM, 2009.

AMaLGaM (Adapted Maximum-Likelihood Gaussian Model Iterated Density-Estimation Evolutionary Algorithm) 
is an estimation-of-distribution (EDA) algorithm that maintains a Gaussian
surrogate function of the optimization landscape that is updated via maximum-likelihood estimation.
It's particularly noted for being parameter-free, meaning it doesn't require manual tuning of parameters to work effectively

.. function:: AMALGAM(mfev, tol, stol, np = 0, iamalgam = True, noparam = True, print = True)

   Initializes a new AMaLGaM optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param stol: Terminate when the change in solution is less than this value.
   :type stol: float
   :param np: Population size (selected automatically when non-positive).
   :type np: int
   :param iamalgam: Whether to use the iAMaLGaM modification with increasing population size and restarts.
   :type iamalgam: bool
   :param noparam: Whether to use the no-parameter version described in the paper.
   :type noparam: bool
   :param print: Whether to print progress on local searches to the console when using the iAMaLGaM version.
   :type print: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Basin Hopping
-------------------

The basin hopping procedure was first outlined in the following paper:

* Wales, David J.; Doye, Jonathan P. K. (1997-07-10). "Global Optimization by Basin-Hopping
and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms".
The Journal of Physical Chemistry A. 101 (28): 5111–5116.
 
In summary, basin hopping combines two steps in alternation until it achieves a desired global minimum.
First, the algorithm randomly changes the coordinates of the current solution (perturbation).
It then performs a local optimization using some other algorithm. The new coordinates
are accepted if they result in a lower objective value, otherwise they are rejected.
The acceptance/rejection threshold is adjusted during optimization. This algorithm is particularly
useful for optimizing rugged landscapes.

.. function:: BasinHopping(minimizer, stepstrat, print = True, mit = 99, temp = 1.0)

   Initializes a new basin hopping optimizer with the specified parameters.

   :param minimizer: Local optimizer.
   :type minimizer: MultivariateSearch
   :param stepstrat: Strategy for varying step size or acceptance/rejection rate.
   :type stepstrat: BasinHopping_StepStrategy
   :param print: Whether to print progress on local searches to the console.
   :type print: bool
   :param mit: Maximum number of iterations of local search.
   :type mit: int
   :param temp: Initial annealing temperature.
   :type temp: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Controlled Random Search (CRS)
-------------------

This algorithm (and its other variants) are described in this paper:

* Kaelo, P., and M. M. Ali. "Some variants of the controlled random search
 algorithm for global optimization." Journal of optimization theory and
 applications 130.2 (2006): 253-264.

Controlled random search (CRS) works by maintaining a population of solution candidates, where
the worst solutions are replaced with new randomly-generated points. This variant
of CRS improves the local search behavior and efficiency significantly 
by applying a local mutation operator, termed CRS2 in the aforementioned paper.

.. function:: CRS(mfev, np, tol)

   Initializes a new controlled random search optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Population size.
   :type np: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

 