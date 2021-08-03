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

 [1] Zhan, Zhang, and Chung. Adaptive particle swarm optimization, IEEE
 Transactions on Systems, Man, and Cybernetics, Part B: CyberneticsVolume 39,
 Issue 6, 2009, Pages 1362-1381 (2009)
 */

#include <stdexcept>
#include "random.hpp"
#include "pso.h"

using Random = effolkronium::random_static;

PsoSearch::PsoSearch(int mfev, double tol, double stol, int np, bool correct) { // @suppress("Class members should be properly initialized")
	_tol = tol;
	_stol = stol;
	_np = np;
	_mfev = mfev;
	_correct = correct;
}

void PsoSearch::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {

	// set problem
	_f = f;
	_n = n;
	_lower = std::vector<double>(lower, lower + _n);
	_upper = std::vector<double>(upper, upper + _n);

	// initialize parameters
	_w = 0.9;
	_c1 = _c2 = 2.;
	_it = _state = 0;
	_maxit = (int) std::round(_mfev / (1. + _np));

	// initialize swarm
	_swarm.clear();
	_fbest = std::numeric_limits<double>::infinity();
	int ibest = _worst = 0;
	_fev = 0;
	for (int i = 0; i < _np; i++) {

		// create particle
		std::vector<double> x(_n);
		std::vector<double> v(_n);
		std::vector<double> xb(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = xb[j] = Random::get(_lower[j], _upper[j]);
			v[j] = 0.;
		}
		const double fx = _f(&x[0]);
		auto pParticle = std::make_shared<pso_particle>(pso_particle { x, v, xb,
				fx, fx });
		_swarm.push_back(std::move(pParticle));
		_fev++;

		// update best and worst positions
		if (fx < _fbest) {
			_fbest = fx;
			ibest = i;
		}
		if (fx >= _swarm[_worst]->_f) {
			_worst = i;
		}
	}
	_xbest = std::vector<double>(_n);
	std::copy((_swarm[ibest]->_x).begin(), (_swarm[ibest]->_x).end(),
			_xbest.begin());

	// initialize work arrays
	_wp = std::vector<double>(_n);
	_ws = std::vector<double>(_np);
	_wmu = std::vector<double>(4);
}

void PsoSearch::iterate() {

	// perform elitist learning
	updateBest();

	// compute the evolutionary factor
	const double f = getF();
	const int state = nextState(f);

	// get the next state of exploration and update swarm parameters
	updateParams(f, state);

	// update swarm
	updateSwarm();

	// update counters
	_state = state;
	_it++;
}

multivariate_solution PsoSearch::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {

	// initialize parameters
	init(f, n, guess, lower, upper);

	// main iteration loop over generations
	bool converged = false;
	while (_it < _maxit && _fev < _mfev) {

		// perform a single generation
		iterate();

		// converge when distance in fitness between best and worst points
		// is below the given tolerance
		const double dy = std::fabs(_fbest - _swarm[_worst]->_f);
		if (dy <= _tol) {

			// compute standard deviation of swarm radiuses
			int count = 0;
			double mean = 0., m2 = 0.;
			for (auto &pt : _swarm) {
				const double x = std::sqrt(
						std::inner_product(pt->_x.begin(), pt->_x.end(),
								pt->_x.begin(), 0.));
				count++;
				const double delta = x - mean;
				mean += delta / count;
				const double delta2 = x - mean;
				m2 += delta * delta2;
			}

			// test convergence in standard deviation
			if (m2 <= (_np - 1) * _stol * _stol) {
				converged = true;
				break;
			}
		}
	}
	return {_xbest,_fev, converged};
}

void PsoSearch::updateParticle(pso_particle &particle) {

	// update the velocity and position of this particle (1)-(2)
	for (int i = 0; i < _n; i++) {
		const double r1 = Random::get(0., 1.);
		const double r2 = Random::get(0., 1.);
		particle._v[i] = particle._v[i] * _w
				+ _c1 * r1 * (particle._xb[i] - particle._x[i])
				+ _c2 * r2 * (_xbest[i] - particle._x[i]);
		particle._x[i] += particle._v[i];
	}

	// correct if out of box
	if (_correct) {
		for (int i = 0; i < _n; i++) {
			if (particle._x[i] < _lower[i]) {
				particle._x[i] = _lower[i];
			} else if (particle._x[i] > _upper[i]) {
				particle._x[i] = _upper[i];
			}
		}
	}

	// fitness re-evaluation and update best point so far
	particle._f = _f(&particle._x[0]);
	_fev++;
	if (particle._f < particle._fb) {
		std::copy((particle._x).begin(), (particle._x).end(),
				(particle._xb).begin());
		particle._fb = particle._f;
	}
}

void PsoSearch::updateBest() {

	// this subprocedure is based on Figure 7
	// set P = gbest;
	std::copy(_xbest.begin(), _xbest.end(), _wp.begin());

	// perturb P(d)
	const int d = Random::get(0, _n - 1);
	const double sigma = _smax - (_smax - _smin) * _it / _maxit;
	const double gaus = Random::get(_Z) * sigma;
	_wp[d] += (_upper[d] - _lower[d]) * gaus;

	// make sure P is in the range
	if (_correct) {
		if (_wp[d] < _lower[d]) {
			_wp[d] = _lower[d];
		} else if (_wp[d] > _upper[d]) {
			_wp[d] = _upper[d];
		}
	}

	// evaluate P and replace best or worst point if necessary
	const double nu = _f(&_wp[0]);
	_fev++;
	if (nu < _fbest) {
		std::copy(_wp.begin(), _wp.end(), _xbest.begin());
		_fbest = nu;
	} else {
		auto &worst = _swarm[_worst];
		worst->_f = nu;
		std::copy(_wp.begin(), _wp.end(), (worst->_x).begin());
		if (worst->_f < worst->_fb) {
			std::copy((worst->_x).begin(), (worst->_x).end(),
					(worst->_xb).begin());
			worst->_fb = worst->_f;
		}
	}
}

void PsoSearch::updateSwarm() {

	// update the swarm
	for (auto &p : _swarm) {
		updateParticle(*p);
	}

	// compute the new global best and worst
	int ibest = -1;
	_worst = 0;
	_fbest = std::numeric_limits<double>::infinity();
	for (int i = 0; i < _np; i++) {
		auto &p = _swarm[i];
		if (p->_f <= _fbest) {
			_fbest = p->_f;
			ibest = i;
		}
		if (p->_f >= _swarm[_worst]->_f) {
			_worst = i;
		}
	}

	// if mutation resulted in improvement in best position record it
	if (ibest >= 0) {
		auto &best = _swarm[ibest];
		std::copy((best->_x).begin(), (best->_x).end(), _xbest.begin());
	}
}

void PsoSearch::updateParams(double f, int state) {

	// update w in (10)
	_w = 1.0 / (1.0 + 1.5 * std::exp(-2.6 * f));

	// update C1 and C2 in (11)-(12)
	const double delta1 = 0.05 * (1.0 + Random::get(0., 1.));
	const double delta2 = 0.05 * (1.0 + Random::get(0., 1.));
	switch (state) {
	case 1: {
		_c1 += delta1;
		_c2 += delta2;
		break;
	}
	case 2: {
		_c1 += 0.5 * delta1;
		_c2 -= 0.5 * delta2;
		break;
	}
	case 3: {
		_c1 += 0.5 * delta1;
		_c2 += 0.5 * delta2;
		break;
	}
	default: {
		_c1 -= 0.5 * delta1;
		_c2 -= 0.5 * delta2;
		break;
	}
	}
	_c1 = std::max(1.5, std::min(_c1, 2.5));
	_c2 = std::max(1.5, std::min(_c2, 2.5));
	if (_c1 + _c1 > 4.) {
		const double sum = _c1 + _c2;
		_c1 *= 4. / sum;
		_c2 *= 4. / sum;
	}
}

double PsoSearch::getF() {

	// calculate the distances between the particles (7)
	double dmin = 0.;
	double dmax = std::numeric_limits<double>::infinity();
	int ibest = 0;
	for (int i = 0; i < _np; i++) {
		_ws[i] = 0.;
		for (int j = 0; j < _np; j++) {
			if (j != i) {
				double dij = 0.;
				auto &xi = _swarm[i]->_x;
				auto &xj = _swarm[j]->_x;
				for (int k = 0; k < _n; k++) {
					const double dist = xi[k] - xj[k];
					dij += dist * dist;
				}
				dij = std::sqrt(dij);
				_ws[i] += dij;
			}
		}
		_ws[i] /= (_np - 1.);

		// update the least and greatest distance
		if (_ws[i] > dmax) {
			dmax = _ws[i];
		}
		if (_ws[i] < dmin) {
			dmin = _ws[i];
		}

		// search for the best point in swarm
		if (_swarm[i]->_f < _swarm[ibest]->_f) {
			ibest = i;
		}
	}

	// compute the evolutionary factory in (8)
	return (_ws[ibest] - dmin) / std::max(dmax - dmin, 1e-8);
}

int PsoSearch::nextState(double f) {

	// compute the decision regions for the next state
	const double m1 = mu(f, 1);
	const double m2 = mu(f, 2);
	const double m3 = mu(f, 3);
	const double m4 = mu(f, 4);
	_wmu[0] = m1;
	_wmu[1] = m2;
	_wmu[2] = m3;
	_wmu[3] = m4;
	switch (_state) {
	case 0: {
		return 1 + (std::max_element(_wmu.begin(), _wmu.end()) - _wmu.begin());
	}
	case 1: {
		if (m1 > 0) {
			return 1;
		} else if (m2 > 0) {
			return 2;
		} else if (m4 > 0) {
			return 4;
		} else {
			return 3;
		}
	}
	case 2: {
		if (m2 > 0) {
			return 2;
		} else if (m3 > 0) {
			return 3;
		} else if (m1 > 0) {
			return 1;
		} else {
			return 4;
		}
	}
	case 3: {
		if (m3 > 0) {
			return 3;
		} else if (m4 > 0) {
			return 4;
		} else if (m2 > 0) {
			return 2;
		} else {
			return 1;
		}
	}
	default: {
		if (m4 > 0) {
			return 4;
		} else if (m1 > 0) {
			return 1;
		} else if (m2 > 0) {
			return 2;
		} else {
			return 3;
		}
	}
	}
}

double PsoSearch::mu(double f, int i) {
	switch (i) {
	case 1: {
		if (f >= 0.0 && f <= 0.4) {
			return 0.0;
		} else if (f > 0.4 && f <= 0.6) {
			return 5.0 * f - 2.0;
		} else if (f > 0.6 && f <= 0.7) {
			return 1.0;
		} else if (f > 0.7 && f <= 0.8) {
			return -10.0 * f + 8.0;
		} else {
			return 0.0;
		}
		break;
	}
	case 2: {
		if (f >= 0.0 && f <= 0.2) {
			return 0.0;
		} else if (f > 0.2 && f <= 0.3) {
			return 10.0 * f - 2.0;
		} else if (f > 0.3 && f <= 0.4) {
			return 1.0;
		} else if (f > 0.4 && f <= 0.6) {
			return -5.0 * f + 3.0;
		} else {
			return 0.0;
		}
		break;
	}
	case 3: {
		if (f >= 0.0 && f <= 0.1) {
			return 1.0;
		} else if (f > 0.1 && f <= 0.3) {
			return -5.0 * f + 1.5;
		} else {
			return 0.0;
		}
		break;
	}
	case 4: {
		if (f >= 0.0 && f <= 0.7) {
			return 0.0;
		} else if (f > 0.7 && f <= 0.9) {
			return 5.0 * f - 3.5;
		} else {
			return 1.0;
		}
		break;
	}
	default: {
		throw std::invalid_argument("Fatal error.");
	}
	}
}
