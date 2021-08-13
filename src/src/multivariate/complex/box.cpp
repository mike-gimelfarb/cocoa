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

 [1] Box, M. J. "A new method of constrained optimization and a comparison
 with other methods." The Computer Journal 8.1 (1965): 42-52.

 [2] Guin, J. A. "Modification of the complex method of constrained
 optimization." The Computer Journal 10.4 (1968): 416-417.

 [3] Krus P., Andersson J., Optimizing Optimization for Design Optimization,
 in Proceedings of ASME Design Automation Conference, Chicago, USA, September 2-6, 2003
 */

#include <algorithm>

#include "../../random.hpp"

#include "box.h"

using Random = effolkronium::random_static;

BoxComplex::BoxComplex(int mfev, double ftol, double xtol, double alpha, // @suppress("Class members should be properly initialized")
		double rfac, double rforget, int nbox, bool movetobest) {
	_ftol = ftol;
	_xtol = xtol;
	_alpha = alpha;
	_rfac = rfac;
	_gamma = rforget;
	_nbox = nbox;
	_mfev = mfev;
	_movetobest = movetobest;
	_adaptalpha = alpha <= 0.;
}

void BoxComplex::init(multivariate f, constraints c, const int n, double *guess,
		double *lower, double *upper) {

	// initialize problem
	_f = f;
	_c = c;
	_n = n;
	_lower = std::vector<double>(lower, lower + n);
	_upper = std::vector<double>(upper, upper + n);
	_fev = _cev = 0;

	// initialize adaptive parameters
	if (_nbox <= 0) {
		_nbox = 2 * _n;
	}
	if (_adaptalpha) {
		_alpha = 1. + 1. / _n;
	}

	// implements the forgetting principle
	_kf = 1. - std::pow(_alpha / 2., (_gamma / _nbox));

	// perform a monte carlo search for a feasible point
	// if the feasible region occupies proportion p of the area in
	// the hypercube [lower, upper], then on average we will need
	// 1 / p constraint evaluations to find a feasible point, e.g.
	// if p = 0.05 then we need around 20 evaluations
	std::vector<double> start(guess, guess + n);
	while (_cev < 100) {
		_cev++;
		if (c(&start[0])) {
			break;
		}
		const double r = Random::get(0., 1.);
		for (int j = 0; j < _n; j++) {
			start[j] = _lower[j] + r * (_upper[j] - _lower[j]);
		}
	}

	// set initial starting point as guess
	_box.clear();
	const auto &pt0 = box_point { start, _f(&start[0]) };
	_box.push_back(std::move(pt0));
	_fev++;
	_center = std::vector<double>(start);
	_center0 = std::vector<double>(_n, 0.);
	_xref = std::vector<double>(_n, 0.);
	_min = std::vector<double>(_n);
	_max = std::vector<double>(_n);

	// add the remaining points
	for (int i = 1; i < _nbox; i++) {

		// sample an initial point within the bounds
		std::vector<double> x(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
		}

		// perform bisection until the point is feasible
		while (!_c(&x[0])) {
			_cev++;
			for (int j = 0; j < _n; j++) {
				x[j] = (x[j] + _center[j]) / 2.;
			}
		}

		// update the center
		for (int j = 0; j < _n; j++) {
			_center[j] += (x[j] - _center[j]) / (i + 1.);
		}

		// add the point
		const auto &pti = box_point { x, _f(&x[0]) };
		_box.push_back(std::move(pti));
		_fev++;
	}
}

void BoxComplex::iterate() {

	// update ranges
	std::fill(_min.begin(), _min.end(),
			std::numeric_limits<double>::infinity());
	std::fill(_max.begin(), _max.end(),
			-std::numeric_limits<double>::infinity());
	_fmin = std::numeric_limits<double>::infinity();
	_fmax = -std::numeric_limits<double>::infinity();
	for (const auto &pt : _box) {
		for (int j = 0; j < _n; j++) {
			_min[j] = std::min(_min[j], pt._x[j]);
			_max[j] = std::max(_max[j], pt._x[j]);
		}
		_fmin = std::min(_fmin, pt._f);
		_fmax = std::max(_fmax, pt._f);
	}

	// forgetting principle
	if (_kf > 0. && (_fmax != _fmin)) {
		for (auto &pt : _box) {
			pt._f = pt._f + (_fmax - _fmin) * _kf;
		}
	}

	// find the point with the highest value and the center of the remaining
	// points in the complex
	auto &ptmax = *std::max_element(_box.begin(), _box.end(),
			box_point::compare_fitness);
	for (int j = 0; j < _n; j++) {
		_center0[j] = _center[j] + (_center[j] - ptmax._x[j]) / (_nbox - 1.);
	}

	// find the reflection of the highest point
	for (int j = 0; j < _n; j++) {
		_xref[j] = _center0[j] + _alpha * (_center0[j] - ptmax._x[j]);
	}

	// enforce the bound constraints for the new point
	for (int j = 0; j < _n; j++) {
		if (_xref[j] < _lower[j]) {
			_xref[j] = _lower[j] + 0.01 * (_upper[j] - _lower[j]);
		}
		if (_xref[j] > _upper[j]) {
			_xref[j] = _upper[j] - 0.01 * (_upper[j] - _lower[j]);
		}
	}

	// while the new point is infeasible, move the new point closer to the center
	while (!_c(&_xref[0])) {
		_cev++;
		for (int j = 0; j < _n; j++) {
			_xref[j] = (_xref[j] + _center0[j]) / 2.;
		}
		if (_cev >= _mfev) {
			return;
		}
	}

	// evaluate at reflection point
	double fref = _f(&_xref[0]);
	_fev++;

	// compute factor for random noise
	double rcoeff = 0.;
	for (int j = 0; j < _n; j++) {
		const double delta = _max[j] - _min[j];
		rcoeff = std::max(rcoeff, delta / (_upper[j] - _lower[j]));
	}

	// while the fitness of the new point is worse than the worst in the
	// box, move the new point closer to the center
	const auto &ptmin = *std::min_element(_box.begin(), _box.end(),
			box_point::compare_fitness);
	int kf = 0;
	while (fref > ptmax._f) {
		kf++;

		// move the new point
		if (_movetobest) {
			const double a = 1. - std::exp(-kf / 4.);
			for (int j = 0; j < _n; j++) {
				const double temp = a * ptmin._x[j] + (1. - a) * _center0[j];
				_xref[j] = (temp + _xref[j]) / 2.;
			}
		} else {
			for (int j = 0; j < _n; j++) {
				_xref[j] = (_center0[j] + _xref[j]) / 2.;
			}
		}

		// add random noise
		if (_rfac > 0.) {
			for (int j = 0; j < _n; j++) {
				_xref[j] += _rfac * rcoeff * (_upper[j] - _lower[j])
						* (Random::get(0., 1.) - 0.5);
			}

			// while the new point is infeasible, move the new point closer to the center
			while (!_c(&_xref[0])) {
				_cev++;
				for (int j = 0; j < _n; j++) {
					_xref[j] = (_xref[j] + _center0[j]) / 2.;
				}
				if (_cev >= _mfev) {
					return;
				}
			}
		}

		// evaluate at the new point
		fref = _f(&_xref[0]);
		_fev++;
		if (_fev >= _mfev) {
			return;
		}
	}

	// replace worst point in complex by the new point
	for (int j = 0; j < _n; j++) {
		_center[j] += (_xref[j] - ptmax._x[j]) / _nbox;
	}
	std::copy(_xref.begin(), _xref.end(), (ptmax._x).begin());
	ptmax._f = fref;
}

constrained_solution BoxComplex::optimize(multivariate f, constraints c,
		const int n, double *guess, double *lower, double *upper) {
	init(f, c, n, guess, lower, upper);
	bool conv = false;
	while (true) {
		if (_fev >= _mfev || _cev >= _mfev) {
			break;
		}
		iterate();
		if (converged()) {
			conv = true;
			break;
		}
	}
	const auto &pt = *std::min_element(_box.begin(), _box.end(),
			box_point::compare_fitness);
	return {pt._x, _fev, _cev, conv};
}

bool BoxComplex::converged() {

	// check bounds on values
	if (_fmax - _fmin > _ftol) {
		return false;
	}

	// check bounds on complex spread
	for (int j = 0; j < _n; j++) {
		if (_max[j] - _min[j] > _xtol) {
			return false;
		}
	}
	return true;
}
