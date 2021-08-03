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

#include <cmath>
#include <iostream>

#include "bipop_cmaes.h"
#include "../../random.hpp"

using Random = effolkronium::random_static;

BiPopCmaes::BiPopCmaes(BaseCmaes *base, int mfev, double tol, double sigma0, // @suppress("Class members should be properly initialized")
		double sigmadecay, int maxlpruns, bool adaptbudget, bool print) {
	_tol = tol;
	_sigmaref = sigma0;
	_sigmadec = sigmadecay;
	_mfev = mfev;
	_mrun = maxlpruns;
	_print = print;
	_adaptbudget = adaptbudget;
	_base = base;
}

void BiPopCmaes::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {

	// initialize problem
	_f = f;
	_n = n;
	_lower = std::vector<double>(lower, lower + n);
	_upper = std::vector<double>(upper, upper + n);
	_xguess = std::vector<double>(guess, guess + n);

	// initialize lambda
	_lambdaref = 4 + (int) (3. * std::log(1. * _n));
	_lambda = _lambdal = _lambdaref;

	// initialize sigma
	_sigma = _sigmal = _sigmaref;

	// initialize evaluations
	_evref = (_n + 3) * (_n + 3);
	_evref = (int) (100. + 50. * _evref / std::sqrt(1. * _lambda)) * _lambda;
	_evref = std::min(_evref, _mfev);

	// set up the optimizer
	_x0 = std::vector<double>(_xguess.begin(), _xguess.end());
	_base->setParams(_lambda, _sigma, _evref);

	// first default run with small population size
	const auto &sol = _base->optimize(_f, _n, &_x0[0], &_lower[0], &_upper[0]);
	_x = std::vector<double>(sol._sol.begin(), sol._sol.end());
	_fx = _f(&_x[0]);

	// initialize counters - note we do first restart with first regime
	_fev = sol._fev + 1;
	_budgetl = _budgets = 0;
	_it = _il = _is = 0;
	_lregime = _cregime = _bregime = 0;
	_budgetfac = 2;

	// initialize best points
	_xbest = _x;
	_fxbest = _fx;
	_fxold = NAN;

	// print output
	if (_print) {
		std::cout << "Run\t" << "Mode\t" << "Run1\t" << "Run2\t" << "Budget1\t"
				<< "Budget2\t" << "MaxBudget\t" << "Pop\t" << "Sigma\t" << "F\t"
				<< "BestF" << std::endl;
		std::cout << _it << "\t" << 0 << "\t" << _il << "\t" << _is << "\t"
				<< _budgetl << "\t" << _budgets << "\t" << _evref << "\t"
				<< _lambda << "\t" << _sigma << "\t" << _fx << "\t" << _fxbest
				<< std::endl;
	}
}

void BiPopCmaes::iterate() {

	// evolve the initial guess using D-dim random walk
	for (int i = 0; i < _n; i++) {
		do {
			_x0[i] = _xguess[i] + _sigmaref * Random::get(_Z);
		} while (_x0[i] < _lower[i] || _x0[i] > _upper[i]);
	}

	// decide which strategy to run
	if (_it == 0) {

		// first run is with the large population size
		_cregime = 0;
	} else {
		if (_adaptbudget) {

			// in NBIPOP-aCMA-ES, use an adaptive strategy for each budget:
			// allocate twice the budget to the regime which yielded the best
			// solution in the recent runs
			if (_bregime == 0) {
				if (_budgetl <= _budgetfac * _budgets) {
					_cregime = 0;
				} else {
					_cregime = 1;
				}
			} else {
				if (_budgets <= _budgetfac * _budgetl) {
					_cregime = 1;
				} else {
					_cregime = 0;
				}
			}
		} else {

			// in BIPOP-CMA-ES we choose the regime with the lower budget
			if (_budgetl <= _budgets) {
				_cregime = 0;
			} else {
				_cregime = 1;
			}
		}
	}

	// apply the strategy with the lower budget
	if (_cregime == 0) {
		runFirstRegime();
	} else {
		runSecondRegime();
	}

	// update the best point so far
	if (_fx < _fxbest) {
		_xbest = _x;
		_fxbest = _fx;
		_bregime = _cregime;
	}

	// print output
	if (_print) {
		std::cout << _it << "\t" << 0 << "\t" << _il << "\t" << _is << "\t"
				<< _budgetl << "\t" << _budgets << "\t" << _evref << "\t"
				<< _lambda << "\t" << _sigma << "\t" << _fx << "\t" << _fxbest
				<< std::endl;
	}
	_it++;
}

multivariate_solution BiPopCmaes::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {
	init(f, n, guess, lower, upper);
	bool converged = false;
	while (true) {
		iterate();

		// check if reached max number of restarts or evals
		if (_il >= _mrun || _fev >= _mfev) {
			break;
		}

		// check for convergence
		if (_lregime == 0 && _fxold != _fx) {
			if (std::fabs(_fx - _fxold) <= _tol) {
				converged = true;
				break;
			}
			_fxold = _fx;
		}
	}
	return {_xbest, _fev, converged};
}

void BiPopCmaes::runFirstRegime() {

	// compute the new lambda
	_lambdal *= 2;
	_lambda = _lambdal;

	// adjust the sigma based on Loshchilov et al. (2012)
	_sigmal /= _sigmadec;
	_sigmal = std::max(0.01 * _sigmaref, _sigmal);
	_sigma = _sigmal;

	// number of function evaluations
	_evref = (_n + 3) * (_n + 3);
	_evref = (int) (100. + 50. * _evref / std::sqrt(1. * _lambda)) * _lambda;
	_evref = std::min(_evref, _mfev - _fev);

	// set up the optimizer
	_base->setParams(_lambdal, _sigmal, _evref);

	// run the CMAES with increasing population size
	const auto &sol = _base->optimize(_f, _n, &_x[0], &_lower[0], &_upper[0]);
	_x = std::vector<double>(sol._sol.begin(), sol._sol.end());
	_fx = _f(&_x[0]);

	// increment counters and adjust budget
	_fev += sol._fev + 1;
	_budgetl += sol._fev + 1;
	_il++;
	_lregime = 0;
}

void BiPopCmaes::runSecondRegime() {

	// compute new lambda
	const double u = Random::get(0., 1.);
	_lambdas = (int) (_lambdaref
			* std::pow((0.5 * _lambdal) / _lambdaref, u * u));
	_lambda = _lambdas;

	// compute new sigma
	_sigma = _sigmaref * std::pow(10., -2. * Random::get(0., 1.));

	// number of function evaluations
	if (_lregime == 0) {
		_evref = _budgetl >> 1;
	} else {
		_evref = (_n + 3) * (_n + 3);
		_evref = (int) (100. + 50. * _evref / std::sqrt(1. * _lambda))
				* _lambda;
	}
	_evref = std::min(_evref, _mfev - _fev);

	// set up the optimizer
	_base->setParams(_lambdas, _sigma, _evref);

	// run the CMAES with small population size
	const auto &sol = _base->optimize(_f, _n, &_x[0], &_lower[0], &_upper[0]);
	_x = std::vector<double>(sol._sol.begin(), sol._sol.end());
	_fx = _f(&_x[0]);

	// increment counters and adjust budget
	_fev += sol._fev + 1;
	_budgets += sol._fev + 1;
	_is++;
	_lregime = 1;
}
