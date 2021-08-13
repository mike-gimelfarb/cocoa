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

 [1] Cheng, Ran & Jin, Yaochu. (2014). A Competitive Swarm Optimizer for Large
 Scale Optimization. IEEE transactions on cybernetics. 45.
 10.1109/TCYB.2014.2322602.

 [2] Kennedy, James & Mendes, Rui. (2002). Population structure and particle
 swarm performance. Proc IEEE Congr Evol Comput. 2. 1671 - 1676.
 10.1109/CEC.2002.1004493.
 */

#include <numeric>

#include "../../random.hpp"

#include "cso.h"

using Random = effolkronium::random_static;

CSOSearch::CSOSearch(int mfev, double tol, double sigmatol, int np, // @suppress("Class members should be properly initialized")
		bool ring, bool correct) {
	_tol = tol;
	_sigma_tol = sigmatol;
	_np = (np % 2 == 0) ? np : np + 1;
	_mfev = mfev;
	_ring = ring;
	_correct = correct;
}

void CSOSearch::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {

	// initialize function
	_f = f;
	_n = n;
	_lower = std::vector<double>(lower, lower + _n);
	_upper = std::vector<double>(upper, upper + _n);

	// initialize swarm
	_phi = computePhi(_np);
	_swarm.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		std::vector<double> v(_n);
		std::vector<double> mean(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
			v[j] = 0.;
		}
		const auto &part = cso_particle { x, v, mean, _f(&x[0]) };
		_swarm.push_back(std::move(part));
	}
	_fev = _np;

	// initialize topology to ring or dense topology
	if (_ring) {
		for (int i = 0; i < _np; i++) {
			auto &part = _swarm[i];
			const int il = i == 0 ? _np - 1 : i - 1;
			const int ir = i == _np - 1 ? 0 : i + 1;
			part._left = &_swarm[il];
			part._right = &_swarm[ir];
			for (int j = 0; j < _n; j++) {
				part._mean[j] = (part._left->_x[j] + part._x[j]
						+ part._right->_x[j]) / 3;
			}
		}
	} else {
		_mean = std::vector<double>(_n);
		for (const auto &p : _swarm) {
			for (int i = 0; i < _n; i++) {
				_mean[i] += p._x[i] / _np;
			}
		}
		for (auto &p : _swarm) {
			p._mean = _mean;
		}
	}
}

void CSOSearch::iterate() {

	// split m particles in the swarm into pairs:
	// shuffle the swarm and assign element i to m/2 + i
	Random::shuffle(_swarm.begin(), _swarm.end());

	// now go through eaching and perform fitness selection
	const int halfm = _np / 2;
	for (int i = 0; i < halfm; i++) {
		const int j = i + halfm;
		compete(_swarm[i], _swarm[j]);
	}

	// update means based on neighbors topology
	if (_ring) {
		for (auto &p : _swarm) {
			for (int i = 0; i < _n; i++) {
				p._mean[i] = (p._left->_x[i] + p._x[i] + p._right->_x[i]) / 3.;
			}
		}
	} else {
		std::fill(_mean.begin(), _mean.end(), 0.);
		for (const auto &p : _swarm) {
			for (int i = 0; i < _n; i++) {
				_mean[i] += p._x[i] / _np;
			}
		}
	}

	// find the best and worst points
	_best = _swarm[0];
	_worst = _swarm[0];
	for (const auto &p : _swarm) {
		if (p._f <= _best._f) {
			_best = p;
		}
		if (p._f >= _worst._f) {
			_worst = p;
		}
	}
}

multivariate_solution CSOSearch::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {

	// initialize parameters
	init(f, n, guess, lower, upper);

	// main iteration loop over generations
	bool converged = false;
	while (_fev < _mfev) {

		// perform a single generation
		iterate();

		// converge when distance in fitness between best and worst points
		// is below the given tolerance
		const double df = std::fabs(_best._f - _worst._f);
		if (df <= _tol) {

			// compute standard deviation of swarm radiuses
			int count = 0;
			double mean = 0., m2 = 0.;
			for (const auto &pt : _swarm) {
				const double x = std::sqrt(
						std::inner_product(pt._x.begin(), pt._x.end(),
								pt._x.begin(), 0.));
				count++;
				const double delta = x - mean;
				mean += delta / count;
				const double delta2 = x - mean;
				m2 += delta * delta2;
			}

			// test convergence in standard deviation
			if (m2 <= (_np - 1) * _sigma_tol * _sigma_tol) {
				converged = true;
				break;
			}
		}
	}
	return {_best._x, _fev, converged};
}

double CSOSearch::computePhi(int m) {

	// this is equation (25)
	if (m <= 100) {
		return 0.0;
	}

	// this is equations (25) and (26): take midpoint between hi and lo
	double phimin, phimax;
	if (m <= 200) {
		phimin = 0.0;
		phimax = 0.1;
	} else if (m <= 400) {
		phimin = 0.1;
		phimax = 0.2;
	} else if (m <= 600) {
		phimin = 0.1;
		phimax = 0.2;
	} else {
		phimin = 0.1;
		phimax = 0.3;
	}
	return (phimin + phimax) / 2;
}

void CSOSearch::compete(cso_particle &first, cso_particle &second) {

	// find the loser
	cso_particle &loser = (first._f > second._f) ? first : second;
	cso_particle &winner = (first._f > second._f) ? second : first;

	// update velocity and position of the loser: equations (6) and (7)
	for (int i = 0; i < _n; i++) {

		// velocity update (6)
		const double r1 = Random::get(0., 1.);
		const double r2 = Random::get(0., 1.);
		const double r3 = Random::get(0., 1.);
		loser._v[i] = r1 * loser._v[i] + r2 * (winner._x[i] - loser._x[i])
				+ _phi * r3 * (loser._mean[i] - loser._x[i]);

		// clip velocity
		const double range = _upper[i] - _lower[i];
		const double maxv = 0.2 * range;
		loser._v[i] = std::max(-maxv, std::min(loser._v[i], maxv));

		// position update: equation (7)
		loser._x[i] += loser._v[i];
	}

	// correct if out of box
	if (_correct) {
		for (int i = 0; i < _n; i++) {
			loser._x[i] = std::max(_lower[i], std::min(loser._x[i], _upper[i]));
		}
	}

	// update the fitness of the loser
	loser._f = _f(&loser._x[0]);
	_fev++;
}
