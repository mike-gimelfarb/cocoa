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

 [1] Birgin, Ernesto G., and José Mario Martínez. "Improving ultimate convergence
 of an augmented Lagrangian method." Optimization Methods and Software 23.2 (2008):
 177-195.
 */

#ifndef ALGENCAN_H_
#define ALGENCAN_H_

#include <vector>

#include "../multivariate.h"
#include "../../string_utils.h"
#include "../../tabular.hpp"

using equality_constraints = void (*)(const double*, double*);

using inequality_constraints = void (*)(const double*, double*);

class Algencan {

protected:
	bool _print, _localconv;
	int _n, _m, _p, _it, _fev, _cev, _mit;
	double _tol, _rho, _rhoold, _gamma, _tau, _holdnorm, _voldnorm, _lambdamin,
			_lambdamax, _mumax, _lambda0, _mu0;
	multivariate _f;
	equality_constraints _h;
	inequality_constraints _g;
	MultivariateOptimizer *_local;
	Tabular _table;
	std::vector<double> _lower, _upper, _x0, _hx, _gx, _mu, _lambda;

public:
	Algencan(MultivariateOptimizer *local, int mit, double tol, bool print =
			false, double tau = 0.5, double gamma = 10., double lambda0 = 0.,
			double mu0 = 0.);

	void init(multivariate f, equality_constraints h, inequality_constraints g,
			int n, int m, int p, double *guess, double *lower, double *upper);

	void iterate();

	constrained_solution optimize(multivariate f, equality_constraints h,
			inequality_constraints g, int n, int m, int p, double *guess,
			double *lower, double *upper);

	double lagrangian(const double *x);

private:
	double initialRho(const double *x);

	void solveLocal();

	void updateRho();

	void updateMultipliers();

	double icm();
};

#endif /* ALGENCAN_H_ */
