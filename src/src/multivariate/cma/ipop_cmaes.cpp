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

 [1] Auger, Anne, and Nikolaus Hansen. "A restart CMA evolution strategy with
 increasing population size." Evolutionary Computation, 2005. The 2005 IEEE
 Congress on. Vol. 2. IEEE, 2005.

 [2] Ilya Loshchilov, Marc Schoenauer, and Michèle Sebag. "Black-box
 Optimization Benchmarking of NIPOP-aCMA-ES and NBIPOP-aCMA-ES on the
 BBOB-2012 Noiseless Testbed." Genetic and Evolutionary Computation Conference
 (GECCO-2012), ACM Press : 269-276. July 2012.

 [3] Loshchilov, Ilya, Marc Schoenauer, and Michele Sebag. "Alternative
 restart strategies for CMA-ES." International Conference on Parallel Problem
 Solving from Nature. Springer, Berlin, Heidelberg, 2012.

 [4] Liao, Tianjun, and Thomas Stützle. "Bounding the population size of
 IPOP-CMA-ES on the noiseless BBOB testbed." Proceedings of the 15th annual
 conference companion on Genetic and evolutionary computation. ACM, 2013.
 */

#include <cmath>

#include "../../random.hpp"

#include "ipop_cmaes.h"

using Random = effolkronium::random_static;

IPopCmaes::IPopCmaes(BaseCmaes *base, int mfev, double tol, bool print, // @suppress("Class members should be properly initialized")
		double sigma0, double sigmadecay, int npmax) {
	_tol = tol;
	_sigmaref = sigma0;
	_sigmadec = sigmadecay;
	_mfev = mfev;
	_lambdamax = npmax;
	_print = print;
	_base = base;
	_adaptlambda = !(_lambdamax > 0);
}

void IPopCmaes::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {

	// initialize domain
	_f = f;
	_n = n;
	_lower = std::vector<double>(lower, lower + n);
	_upper = std::vector<double>(upper, upper + n);
	_xguess = std::vector<double>(guess, guess + n);

	// initialize sigma
	_sigma = _sigmaref;

	// initialize lambda
	_lambda = 4 + (int) (3. * std::log(1. * _n));

	// initialize max lambda as per Liao et al. (2013)
	if (_adaptlambda) {
		_lambdamax = 10 * _n * _n;
	}

	// initialize number of function evaluations
	_maxfev = (int) (100. + 50. * (_n + 3) * (_n + 3) / std::sqrt(1. * _lambda))
			* _lambda;
	_maxfev = std::min(_maxfev, _mfev);

	// create new optimizer
	_xstart = std::vector<double>(_xguess.begin(), _xguess.end());
	_base->setParams(_lambda, _sigma, _maxfev);

	// run initial CMAES algorithm
	const auto &sol = _base->optimize(_f, _n, &_xstart[0], &_lower[0],
			&_upper[0]);
	_x = std::vector<double>(sol._sol.begin(), sol._sol.end());
	_fx = _f(&_x[0]);

	// initialize counters
	_fev = sol._fev + 1;
	_it = 1;

	// initialize best points
	_xbest = _x;
	_fbest = _fx;
	_fxold = NAN;

	// print output
	_table = Tabular();
	if (_print) {
		_table.setWidth( { 5, 10, 10, 5, 25, 25, 25 });
		_table.printRow("run", "budget", "max_budget", "pop", "sigma", "f*",
				"best f*");
		_table.printRow(_it, _fev, _maxfev, _lambda, _sigma, _fx, _fbest);
	}
}

void IPopCmaes::iterate() {

	// increase population size
	// we apply the maximum bound in Liao et al. (2013) and reset the lambda to
	// its initial value when reached
	_lambda *= 2;
	if (_lambda >= _lambdamax) {
		_lambda = 4 + (int) (3. * std::log(1. * _n));
	}

	// adjust the sigma based on Loshchilov et al. (2012)
	_sigma /= _sigmadec;
	_sigma = std::max(_sigma, 0.01 * _sigmaref);

	// set budget
	_maxfev = (int) (100. + 50. * (_n + 3) * (_n + 3) / std::sqrt(_lambda))
			* _lambda;
	_maxfev = std::min(_maxfev, _mfev - _fev);

	// set the guess
	for (int i = 0; i < _n; i++) {
		do {
			_xstart[i] = _xguess[i] + _sigmaref * Random::get(_Z);
		} while (_xstart[i] < _lower[i] || _xstart[i] > _upper[i]);
	}

	// create new optimizer
	_base->setParams(_lambda, _sigma, _maxfev);

	// run CMAES again
	const auto &sol = _base->optimize(_f, _n, &_xstart[0], &_lower[0],
			&_upper[0]);
	_x = std::vector<double>(sol._sol.begin(), sol._sol.end());
	_fx = _f(&_x[0]);

	// increment counters
	_fev += (sol._fev + 1);
	_it++;

	// update best point
	if (_fx < _fbest) {
		_fxold = _fbest;
		_xbest = _x;
		_fbest = _fx;
	}

	// print output
	if (_print) {
		_table.printRow(_it, _fev, _maxfev, _lambda, _sigma, _fx, _fbest);
	}
}

multivariate_solution IPopCmaes::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {
	init(f, n, guess, lower, upper);
	bool converged = false;
	while (_fev < _mfev) {
		iterate();

		// check convergence
		if (_fx != _fxold) {
			if (std::fabs(_fx - _fxold) <= _tol) {
				converged = true;
				break;
			}
		}
	}
	return {_xbest, _fev, converged};
}
