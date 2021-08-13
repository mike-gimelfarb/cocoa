/*
 Copyright (c) 2020 Mike Gimelfarb

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the > "Software"), to
 deal in the Software without restriction, including without limitation the
 rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, > subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ================================================================
 REFERENCES:

 [1] Hansen, Nikolaus. "Benchmarking a BI-population CMA-ES on the BBOB-2009
 function testbed." Proceedings of the 11th Annual Conference Companion on
 Genetic and Evolutionary Computation Conference: Late Breaking Papers. ACM,
 2009.

 [2] Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box
 Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the
 BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference
 (GECCO-2012), ACM Press : 269-276. July 2012.

 [3] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative
 restart strategies for CMA-ES." International Conference on Parallel Problem
 Solving from Nature. Springer, Berlin, Heidelberg, 2012.
 */

#ifndef MULTIVARIATE_CMAES_BIPOP_CMAES_H_
#define MULTIVARIATE_CMAES_BIPOP_CMAES_H_

#include <random>

#include "../../tabular.hpp"

#include "base_cmaes.h"

class BiPopCmaes: public MultivariateOptimizer {

protected:
	bool _print, _adaptbudget;
	int _n, _budgetfac, _mfev, _mrun, _lambdal, _lambdas, _lambda, _lambdaref,
			_budgetl, _budgets, _il, _is, _it, _evref, _lregime, _cregime,
			_bregime, _fev;
	double _tol, _sigmaref, _sigmadec, _sigmal, _sigma, _fx, _fxold, _fxbest;
	multivariate _f;
	BaseCmaes *_base;
	Tabular _table;
	std::vector<double> _lower, _upper, _x, _xguess, _x0, _xbest;

public:
	std::normal_distribution<> _Z { 0., 1. };

	BiPopCmaes(BaseCmaes *base, int mfev, double tol, bool print = false,
			double sigma0 = 2., double sigmadecay = 1.6, int maxlpruns = 9999,
			bool adaptbudget = true);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);

private:
	void runFirstRegime();

	void runSecondRegime();
};

#endif /* MULTIVARIATE_CMAES_BIPOP_CMAES_H_ */
