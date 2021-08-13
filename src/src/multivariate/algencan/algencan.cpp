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

#include <cmath>
#include <numeric>

#include "algencan.h"

Algencan::Algencan(MultivariateOptimizer *local, int mit, double tol, // @suppress("Class members should be properly initialized")
		bool print, double tau, double gamma, double lambda0, double mu0) {
	_local = local;
	_mit = mit;
	_print = print;
	_tau = tau;
	_gamma = gamma;
	_lambda0 = lambda0;
	_mu0 = mu0;
}

void Algencan::init(multivariate f, equality_constraints h,
		inequality_constraints g, int n, int m, int p, double *guess,
		double *lower, double *upper) {

	// define problem
	_f = f;
	_h = h;
	_g = g;
	_n = n;
	_m = m;
	_p = p;
	_lower = std::vector<double>(lower, lower + n);
	_upper = std::vector<double>(upper, upper + n);

	// define memory
	_hx = std::vector<double>(_m);
	_gx = std::vector<double>(_p);
	_x0 = std::vector<double>(guess, guess + n);
	_localconv = false;
	_table = Tabular();

	// other parameters
	_lambdamin = -1e20;
	_lambdamax = +1e20;
	_mumax = +1e20;
	_it = 1;
	_fev = 0;
	_cev = 0;
	_mu = std::vector<double>(_p, _mu0);
	_lambda = std::vector<double>(_m, _lambda0);

	// define initial trust region
	_rho = initialRho(&_x0[0]);
}

void Algencan::iterate() {
	solveLocal();
	updateRho();
	updateMultipliers();
	_it++;
}

constrained_solution Algencan::optimize(multivariate f, equality_constraints h,
		inequality_constraints g, int n, int m, int p, double *guess,
		double *lower, double *upper) {
	init(f, h, g, n, m, p, guess, lower, upper);

	// print header and initialization
	double xicm = icm();
	if (_print) {
		_table.setWidth( { 5, 25, 25, 25, 5, 10, 25 });
		_table.printRow("iter", "f*", "icm", "L", "conv", "fev", "rho");
		const double lag = lagrangian(&_x0[0]);
		_table.printRow(_it - 1, _f(&_x0[0]), xicm, lag, false, _fev, _rho);
	}

	// main loop
	bool conv = false;
	while (_it < _mit) {
		iterate();

		// print progress
		xicm = icm();
		if (_print) {
			const double lag = lagrangian(&_x0[0]);
			_table.printRow(_it - 1, _f(&_x0[0]), xicm, lag, _localconv, _fev,
					_rho);
		}

		// check convergence
		if (_it > 2 && xicm <= _tol) {
			conv = true;
			break;
		}
	}
	return {_x0, _fev, _cev, conv};
}

double Algencan::lagrangian(const double *x) {

	// evaluate function
	const double fx = _f(x);
	_fev++;

	// evaluate equality constraint
	_h(x, &_hx[0]);
	double hpen = 0.;
	for (int i = 0; i < _m; i++) {
		const double slack = _hx[i] + _lambda[i] / _rho;
		hpen += std::pow(slack, 2.);
	}
	hpen *= (_rho / 2.);

	// evaluate inequality constraint
	_g(x, &_gx[0]);
	_cev++;
	double gpen = 0.;
	for (int i = 0; i < _p; i++) {
		const double slack = _gx[i] + _mu[i] / _rho;
		gpen += std::pow(std::max(0., slack), 2.);
	}
	gpen *= (_rho / 2.);

	return fx + hpen + gpen;
}

double Algencan::initialRho(const double *x) {

	// evaluate function
	const double fx = _f(x);
	_fev++;

	// evaluate equality constraint
	_h(x, &_hx[0]);
	const double hnorm2 = std::inner_product(_hx.begin(), _hx.end(),
			_hx.begin(), 0.);

	// evaluate inequality constraint
	_g(x, &_gx[0]);
	_cev++;
	for (int i = 0; i < _p; i++) {
		_gx[i] = std::max(0., _gx[i]);
	}
	const double gnorm2 = std::inner_product(_gx.begin(), _gx.end(),
			_gx.begin(), 0.);

	// evaluate initial penalty using equation 7
	const double rho0 = 2. * std::fabs(fx) / (hnorm2 + gnorm2);
	return std::max(1e-6, std::min(10., rho0));
}

void Algencan::solveLocal() {

	// define the objective
	multivariate lag = std::bind(&Algencan::lagrangian, this,
			std::placeholders::_1);

	// optimize
	const auto &sol = _local->optimize(lag, _n, &_x0[0], &_lower[0], &_upper[0]);
	_x0 = sol._sol;

	// evaluate constraints at new solution
	_h(&_x0[0], &_hx[0]);
	_g(&_x0[0], &_gx[0]);
	_cev++;
	_localconv = sol._converged;
}

void Algencan::updateRho() {

	// equation 4
	double vnorm = 0.;
	for (int i = 0; i < _p; i++) {
		const double v = std::max(_gx[i], -_mu[i] / _rho);
		const double vabs = std::abs(v);
		vnorm = std::max(vnorm, vabs);
	}

	// compute the norm of h
	double hnorm = 0.;
	for (int i = 0; i < _m; i++) {
		const double habs = std::abs(_hx[i]);
		hnorm = std::max(hnorm, habs);
	}

	// update trust region
	_rhoold = _rho;
	if (_it != 1
			&& std::max(hnorm, vnorm)
					<= _tau * std::max(_holdnorm, _voldnorm)) {
		_rho *= _gamma;
	}
	_holdnorm = hnorm;
	_voldnorm = vnorm;
}

void Algencan::updateMultipliers() {

	// safeguarded estimate of lambda
	for (int i = 0; i < _m; i++) {
		const double step = _lambda[i] + _rhoold * _hx[i];
		_lambda[i] = std::max(_lambdamin, std::min(step, _lambdamax));
	}

	// safeguarded estimate of mu
	for (int i = 0; i < _p; i++) {
		const double step = _mu[i] + _rhoold * _gx[i];
		_mu[i] = std::max(0., std::min(step, _mumax));
	}
}

double Algencan::icm() {
	return std::max(_holdnorm, _voldnorm);
}
