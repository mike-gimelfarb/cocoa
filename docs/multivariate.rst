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

* Bosman, Peter AN, Jï¿½rn Grahl, and Dirk Thierens. "AMaLGaM IDEAs in noiseless black-box optimization benchmarking." Proceedings of the 11th Annual Conference Companion on Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM, 2009.

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
   :param stol: Terminate when the standard deviation of population objective values is less than this value.
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

* Wales, David J.; Doye, Jonathan P. K. (1997-07-10). "Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms". The Journal of Physical Chemistry A. 101 (28): 5111–5116.
 
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


Controlled Random Search with Local Mutation (CRS)
-------------------

This algorithm (and its other variants) are described in this paper:

* Kaelo, P., and M. M. Ali. "Some variants of the controlled random search algorithm for global optimization." Journal of optimization theory and applications 130.2 (2006): 253-264.

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


Covariance Matrix Adaptation Evolutionary Strategies (CMA-ES)
-------------------
 
The covariance matrix adaptation evolution strategy (CMA-ES) is a broad family of algorithms
that adapt a mean and covariance matrix to the shape of the search space. It works
by maintaining a population of solution candidates from this distribution, whose best individuals are 
recombined to create new candidate solutions.
 
COCOA implements many of the best-performing CMA-ES variants.
 
 
Basic CMA-ES
~~~~~~~~
 
The basic CMA-ES was introduced in this paper:
 
* Hansen, Nikolaus, and Andreas Ostermeier. "Completely derandomized self-adaptation in evolution strategies." Evolutionary computation 9.2 (2001): 159-195.
 
The covariance update has ``O(n^3)`` time complexity when updated naively, 
making it scale poorly to higher dimensions. The COCOA implementation applies a 
simple trick of updating the covariance matrix once every ``O(n)`` iterations,
reducing the amortized time complexity to ``O(n^2)``. This allows the algorithm to 
perform well on the order of 100 decision variables.
 
.. function:: CMAES(mfev, tol, np, sigma0 = 2.0, bound = False, eigenrate = 0.25)

   Initializes a new CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param eigenrate: Rate at which the covariance matrix is updated.
   :type eigenrate: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Active CMA-ES
~~~~~~~~
 
The active CMA-ES algorithm was introduced in this paper:
 
* Jastrebski, Grahame A., and Dirk V. Arnold. "Improving evolution strategies through active covariance matrix adaptation." Evolutionary Computation, 2006. CEC 2006. IEEE Congress on. IEEE, 2006.
 
In Active CMA-ES, not only is the variance increased in directions 
that have proven successful (as in standard CMA-ES), but also the variance is 
decreased in directions that have been particularly unsuccessful. This greatly
improves the search efficiency.

.. function:: ActiveCMAES(mfev, tol, np, sigma0 = 2.0, bound = False, alphacov = 2.0, eigenrate = 0.25)

   Initializes a new active CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param alphacov: the alpha parameter as described in the paper
   :type alphacov: float
   :param eigenrate: Rate at which the covariance matrix is updated.
   :type eigenrate: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Cholesky CMA-ES
~~~~~~~~

The Cholesky variant of CMA-ES was described in this paper:

* Krause, Oswin, Dídac Rodríguez Arbonès, and Christian Igel. "CMA-ES with optimal covariance update and storage complexity." Advances in Neural Information Processing Systems. 2016.

The naive covariance update takes ``O(n^3)`` time. The Cholesky CMA-ES variant
reduces this complexity to ``O(n^2)`` by using the Cholesky decomposition instead
of the eigenvalue decomposition, which often results in a better quality update
than the periodic update of the basic CMA-ES.

.. function:: CholeskyCMAES(mfev, tol, stol, np, sigma0 = 2.0, bound = False)

   Initializes a new Cholesky CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param stol: Terminate when the standard deviation of the population is less than this value.
   :type stol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

 
Limited-Memory CMA-ES
~~~~~~~~

The limited memory variant of CMA-ES was described in this paper:

* Loshchilov, Ilya. "A computationally efficient limited memory CMA-ES for large scale optimization." Proceedings of the 2014 Annual Conference on Genetic and Evolutionary Computation. ACM, 2014. 

The limited-memory CMA-ES variant reduces the complexity of the update
by using a limited number of direction vectors to approximate the covariance matrix.
This allows the algorithm to scale to problems with 1000s or even millions of decision variables.

.. function:: LmCMAES(mfev, tol, np, memory = 0, sigma0 = 2.0, bound = False, rademacher = True, usenew = True)

   Initializes a new limited-memory CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param memory: Number of direction vectors in the covariance update (determined automatically when non-positive).
   :type memory: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param rademacher: Whether to sample candidates from Rademacher instead of Gaussian.
   :type rademacher: bool
   :param usenew: Whether to use the new variant of the algorithm from the paper.
   :type usenew: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

 
Separable CMA-ES
~~~~~~~~

This variant of the CMA-ES was introduced in this paper:

* Ros, Raymond, and Nikolaus Hansen. "A simple modification in CMA-ES achieving linear time and space complexity." International Conference on Parallel Problem Solving from Nature. Springer, Berlin, Heidelberg, 2008.

The separable CMA-ES variant reduces the complexity of the covariance update by
maintaining only a diagonal representation of the covariance matrix. This reduces the cost of the
covariance update to ``O(n)``, making it well-suited for optimizing problems with 1000s or millions
of decision variables. However, this simplification comes at the cost of being able to capture 
complex dependencies between decision variables, making it most suitable for separable objective functions.

.. function:: SepCMAES(mfev, tol, np, sigma0 = 2.0, bound = False, adjustlr = True)

   Initializes a new separable CMA-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the change in objective is less than this value.
   :type tol: float
   :param np: Population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param bound: Whether to clip sampled candidate solutions to the search space.
   :type bound: bool
   :param adjustlr: Whether to apply the empirical learning rate adjustment in the paper.
   :type adjustlr: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


IPOP-CMA-ES and NIPOP-CMA-ES
~~~~~~~~

The IPOP-CMA-ES and NIPOP-CMA-ES variants of the CMA-ES algorithm were described in these papers:

* Auger, Anne, and Nikolaus Hansen. "A restart CMA evolution strategy with increasing population size." Evolutionary Computation, 2005. The 2005 IEEE Congress on. Vol. 2. IEEE, 2005.
* Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference (GECCO-2012), ACM Press : 269-276. July 2012.

The Increasing Population CMA-ES (IPOP-CMA-ES) implements a multi-restart strategy
for CMA-ES, where the population is increased in each restart. The use of larger population
size makes the search more global, which helps to avoid local minima and to explore the
search space more thoroughly.

.. function:: IPopCMAES(base, mfev, print = false, sigma0 = 2.0, nipop = True, ksigmadec = 1.6, boundlambda = True)

   Initializes a new IPOP-CMA-ES optimizer with the specified parameters.

   :param base: CMA-ES variant to use to optimize in each restart.
   :type base: BaseCMAES
   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param print: Whether to print progress on restarts to the console.
   :type print: bool
   :param sigma0: Initial step size.
   :type sigma0: float
   :param nipop: Whether to use the NIPOP-CMA-ES variant described in the second reference.
   :type nipop: bool
   :param ksigmadec: Factor to increase step size in each restart.
   :type ksigmadec: float   
   :param boundlambda: Whether to apply optimal lambda cycling as described in the paper.
   :type boundlambda: bool
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


BIPOP-CMA-ES and NBIPOP-CMA-ES
~~~~~~~~

The BIPOP-CMA-ES and NBIPOP-CMA-ES variants of the CMA-ES algorithm were described in these papers:

* Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed." Proceedings of the 11th Annual Conference Companion on Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM, 2009.
* Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference (GECCO-2012), ACM Press : 269-276. July 2012.

The Bi-Population CMA-ES (BIPOP-CMA-ES) is a variant of the IPOP-CMA-ES that incorporates
two different population sizes to improve the optimization. It uses two restart regimes,
one with a small population size, and one with a larger increasing population
size. It dynamically switches between the two regimes based on the progress of the optimization.

.. function:: BiPopCMAES(base, mfev, print = false, sigma0 = 2.0, maxlargeruns = 9, nbipop = True, ksigmadec = 1.6, kbudget = 2.0)

   Initializes a new BIPOP-CMA-ES optimizer with the specified parameters.

   :param base: CMA-ES variant to use to optimize in each restart.
   :type base: BaseCMAES
   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param print: Whether to print progress on restarts to the console.
   :type print: bool
   :param sigma0: Initial step size.
   :type sigma0: float
   :param maxlargeruns: Maximum number of restarts with large population size.
   :type maxlargeruns: int
   :param nbipop: Whether to use the NBIPOP-CMA-ES variant described in the second reference.
   :type nbipop: bool
   :param ksigmadec: Factor to increase step size in each restart.
   :type ksigmadec: float   
   :param kbudget: How much budget in function evaluations to allocate to the large population size restarts.
   :type kbudget: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


Differential Evolution (DE)
-------------------

Differential evolution (DE) is a broad family of algorithms that is inspired by the replication
and mutation of DNA sequences in nature. It maintains a population of candidate solutions that are combined through
mutation and crossover operations to create new candidate solutions, which then replace the existing population through
a process of selection (to mimic the process of survival-of-the-fittest in nature).

COCOA implements many of the best-performing DE variants from the literature.


JADE
~~~~~~~~

The JADE algorithm was developed in the following series of papers:

* Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: Self-adaptive differential evolution with fast and reliable convergence performance." 2007 IEEE congress on evolutionary computation. IEEE, 2007.
* Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential evolution with optional external archive." IEEE Transactions on evolutionary computation 13.5 (2009): 945-958.
* Li, Jie, et al. "Power mean based crossover rate adaptive differential evolution." International Conference on Artificial Intelligence and Computational Intelligence. Springer, Berlin, Heidelberg, 2011.
* Gong, Wenyin, Zhihua Cai, and Yang Wang. "Repairing the crossover rate in adaptive differential evolution." Applied Soft Computing 15 (2014): 149-168.
 
JADE is an adaptive differential evolution that tunes the crossover and mutation 
rates based on values that performed well in the past. These values are maintained 
as long-run moving averages of the best-performing values from previous iterations.

The COCOA version of JADE implements the optional external archive as described 
in the second paper to improve the population diversity, uses a power mean adaptation for the 
crossover rate as suggested in the third paper, and repairs the crossover
rate as suggested in the fourth paper.

.. function:: JADE(mfev, np, tol, archive = True, repaircr = True, pelite = 0.05, cdamp = 0.1, sigma = 0.07)

   Initializes a new JADE optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Population size.
   :type np: int
   :param tol: Terminate when the std of the candidate solutions is less than this value.
   :type tol: float
   :param archive: Whether to maintain an optional archive.
   :type archive: bool
   :param repaircr: Whether to repair the crossover rate.
   :type repaircr: bool
   :param pelite: Top fraction of candidates to sample from for mutation.
   :type pelite: float
   :param cdamp: Exponential moving average coefficient for updating parameters.
   :type cdamp: float
   :param sigma: Lower bound on std for the power-mean crossover adaptation.
   :type sigma: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch


L-SHADE
~~~~~~~~

The L-SHADE algorithm was described in the following paper:

* Tanabe, Ryoji, and Alex S. Fukunaga. "Improving the search performance of SHADE using linear population size reduction." 2014 IEEE congress on evolutionary computation (CEC). IEEE, 2014.

In summary, L-SHADE is very similar to JADE (and in fact builds upon it), but the method of
updating the crossover and mutation rates is different. In L-SHADE, these parameters
are adapted by maintaining a history of H previous parameter values that resulted 
in candidates with good objective values. This history is updated periodically 
and values are drawn from it before each mutation and crossover operation.

The COCOA version implements the optional external archive as described for JADE,
as well as the linear population size reduction described in the aforementioned 
paper.

.. function:: SHADE(mfev, npinit, tol, archive = True, h = 100, npmin = 4)

   Initializes a new L-SHADE optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param npinit: Initial population size.
   :type npinit: int
   :param tol: Terminate when the std of the candidate solutions is less than this value.
   :type tol: float
   :param archive: Whether to maintain an optional archive.
   :type archive: bool
   :param h: Size of the history for updating crossover and mutation parameters.
   :type h: int
   :param npmin: When this value is strictly less than npinit, a linear population size reduction will be used.
   :type npmin: int
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
   

SANSDE
~~~~~~~~

The SANSDE algorithm was described in the following paper:

* Yang, Zhenyu, Ke Tang, and Xin Yao. "Self-adaptive differential evolution with neighborhood search." 2008 IEEE congress on evolutionary computation (IEEE World Congress on Computational Intelligence). IEEE, 2008.

Similar to JADE and L-SHADE, the SANSDE algorithm also implements its own variant
of parameter adaptation for the crossover and mutation. However, it implements two
different possible mutation strategies, and selects between them based on the strategy
that performed well in past iterations.

The COCOA version of this algorithm also repairs the crossover rate as suggested for JADE. 

.. function:: SANSDE(mfev, np, tol, repaircr = True, crref = 5, pupdate = 50, crupdate = 25)

   Initializes a new SANSDE optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Population size.
   :type np: int
   :param tol: Terminate when the std of the candidate solutions is less than this value.
   :type tol: float
   :param repaircr: Whether to repair the crossover rate.
   :type repaircr: bool
   :param crref: How often to generate a new crossover rate in iterations.
   :type crref: int
   :param pupdate: How often to update the mutation strategy selection parameter.
   :type pupdate: int
   :param crupdate: How often to update the crossover and mutation parameters.
   :type crupdate: int
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch

   
Differential Search (DSA)
-------------------

This algorithm is introduced in the following paper:

* P. Civicioglu, "Transforming Geocentric Cartesian Coordinates to Geodetic Coordinates by Using Differential Search Algorithm", Computers and Geosciences, 46, 229-247, 2012.

Similar to differential evolution, differential search (DSA) maintains a population of
candidate solutions that are combined to create new solutions using four different strategies. 
These strategies are inspired by the concept of stable motion and migration of 
superorganisms and are somewhat similar to mutation in DE. 

The current implementation is based on the original Matlab code. However, rather than
selecting among the different strategies randomly at uniform, the COCOA implementation
uses the non-stationary multi-armed bandit algorithm Rexp3 to favor the best-performing
strategy over time.
 
.. function:: DSA(mfev, tol, stol, np, adapt = True, nbatch = 100)

   Initializes a new DSA optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the range of objective values among the candidate solutions is less than this value.
   :type tol: float
   :param stol: Terminate when the std of the candidate solutions is less than this value.
   :type stol: float
   :param np: Population size.
   :type np: int
   :param adapt: Whether to adapt the strategy selection using Rexp3, or use random uniform sampling.
   :type adapt: bool
   :param nbatch: The batch size for the strategy selection algorithm.
   :type nbatch: int
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 

Exponential Natural Evolution Strategy (xNES)
-------------------

This algorithm is described in the following paper:

* Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J. (2010, July). Exponential natural evolution strategies. In Proceedings of the 12th annual conference on Genetic and evolutionary computation (pp. 393-400).

The Exponential Natural Evolution Strategy (xNES) is closely related to the CMA-ES
algorithm, in which the population is modeled using a multivariate Gaussian distribution.
The key idea of xNES is to adapt the Gaussian sampling distribution using the natural gradient.

.. function:: xNES(mfev, tol, a0 = 1.0, etamu = 1.0)

   Initializes a new xNES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the range of objective values among the candidate solutions is less than this value.
   :type tol: float
   :param a0: Initial diagonal values of the covariance.
   :type a0: float
   :param etamu: The learning rate for the mean.
   :type etamu: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 

Novel Self-Adaptive Harmony Search (NSHS)
-------------------

This version of harmony search is described in the following paper:

* Luo, Kaiping. "A Novel Self-Adaptive Harmony Search Algorithm." Journal of Applied Mathematics 2013.1 (2013): 653749.

Harmony search is loosely inspired by the musical improvisation process of musicians, 
and mimics the way musicians search for a perfect harmony by adjusting the pitch of their instruments.
It is population-based, like some of the previously-described algorithms, where each candidate solution is adjusted
based on a harmony memory. The Novel Self-Adaptive Harmony Search (NSHS) variant adapts the pitch and rate adjustment parameters
automatically based on historical information and is much more efficient than standard harmony search and many
of its variants.

.. function:: NSHS(mfev, hms, fstdmin = 0.0001)

   Initializes a new NSHS optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param hms: Harmony memory size.
   :type hms: int
   :param fstdmin: Lower bound on standard deviation as described in the paper (where it is hard coded as 0.0001).
   :type fstdmin: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 
 
Hessian Estimation Evolutionary Strategy (HE-ES)
-------------------

This algorithm was introduced in the following paper:

* Glasmachers, Tobias, and Oswin Krause. "The Hessian Estimation Evolution Strategy." International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2020.

Similar to CMA-ES and variants, the Hessian Estimation Evolution Strategy (HE-ES) maintains a Gaussian
sampling distribution by updating its mean and covariance parameters. Crucially in HE-ES,
the covariance matrix is updated using curvature information from the objective function's imputed Hessian.
In practice, HE-ES often performs superior to CMA-ES in terms of robustness and 
number of function evaluations, in many cases even when the objective function
is not twice-differentiable.

.. function:: HEES(mfev, tol, mres = 1, print = False, np = 0, sigma0 = 2.0)

   Initializes a new HE-ES optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param tol: Terminate when the range of objective values among the candidate solutions is less than this value.
   :type tol: float
   :param mres: Number of runs with increasing population size as suggested in the paper.
   :type mres: int
   :param print: Whether to print progress on local searches to the console when using HE-ES with multiple restarts.
   :type print: bool
   :param np: Initial population size.
   :type np: int
   :param sigma0: Initial step size.
   :type sigma0: float
   :param fstdmin: Lower bound on standard deviation as described in the paper (where it is hard coded as 0.0001).
   :type fstdmin: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 

Self-Adaptive Multi-Population JAYA
-------------------

This version of the JAYA search algorithm is described in detail in the following paper:

* Yu, Jiang-Tao, et al. "Jaya algorithm with self-adaptive multi-population and Lévy flights for solving economic load dispatch problems." IEEE Access 7 (2019): 21372-21384.

The JAYA algorithm is a population-based metaheuristic search algorithm for finding
the global minimum of a multivariate function. It is similar to differential evolution,
but the procedure for generating new candidate solutions includes not only an attraction 
term for moving towards the best candidate in the population, but also a repulsion
term for moving away from the worst candidate. The version implemented in COCOA
features a multi-population scheme for improving the search behavior of JAYA,
automatic parameter adaptation, and Levy flights and other mutation operators 
for global exploration of the search space.

.. function:: JAYA(mfev, tol, np, npmin, adapt = True, k0 = 2, mutation = JAYA_Mutation.logistic, scale = 0.01, beta = 1.5, kcheb = 2, temper = 10.0)

   Initializes a new adaptive JAYA optimizer with the specified parameters.

   :param mfev: Maximum number of function evaluations.
   :type mfev: int
   :param np: Total population size.
   :type np: int
   :param npmin: Minimum subpopulation size.
   :type npmin: int
   :param adapt: Whether to adapt the number of subpopulations.
   :type adapt: bool
   :param k0: Initial number of subpopulations.
   :type k0: int
   :param mutation: The type of mutation operator to use.
   :type mutation: JAYA_Mutation
   :param scale: Step size parameter used when the mutation operator is ``levy``.
   :type scale: float
   :param beta: Distributional parameter used when the mutation operator is ``levy``.
   :type beta: float
   :param kcheb: Currently unused.
   :type kcheb: int
   :param temper: Temperature parameter that controls parameter adaptation speed.
   :type temper: float
   :returns: optimizer instance
   :rtype: object of type MultivariateSearch
 