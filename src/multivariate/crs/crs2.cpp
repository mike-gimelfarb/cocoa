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

 [1] Kaelo, P., and M. M. Ali. "Some variants of the controlled random search
 algorithm for global optimization." Journal of optimization theory and
 applications 130.2 (2006): 253-264.
 */

#include <algorithm>
#include <iostream>

#include "../../blas.h"
#include "../../random.hpp"

#include "crs2.h"

using Random = effolkronium::random_static;

Crs2Search::Crs2Search(int mfev, int np, double tol) {
	_mfev = mfev;
	_np = np;
	_tol = tol;
}

void Crs2Search::init(const multivariate_problem &f, const double *guess) {

	// define problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [CRS]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// define memory
	_work = std::vector<double>(_n);
	_indices = std::vector<int>(_np);
	for (int i = 0; i < _np; i++) {
		_indices[i] = i;
	}

	// define pool of points
	_fev = 0;
	_points.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		for (int d = 0; d < _n; d++) {
			x[d] = Random::get(_lower[d], _upper[d]);
		}
		const double fx = _f._f(&x[0]);
		const point pt { x, fx };
		_points.push_back(std::move(pt));
		_fev++;
	}
	std::sort(_points.begin(), _points.end(), point::compare_fitness);
}

void Crs2Search::iterate() {
	crs2iterate();
}

multivariate_solution Crs2Search::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	bool converged = false;
	while (_fev < _mfev) {
		iterate();
		if (stop()) {
			converged = true;
			break;
		}
	}
	return {_points[0]._x, _fev, converged};
}

/* =============================================================
 *
 * 				UPDATING POINT SET SUBROUTINES
 *
 * =============================================================
 */
int Crs2Search::crs2iterate() {

	// select point generation procedure
	while (true) {

		// compute the new trial point
		Random::shuffle(_indices.begin() + 1, _indices.end());
		for (int d = 0; d < _n; d++) {
			_work[d] = _points[0]._x[d] / _n;
		}
		for (int i = 1; i < _n; i++) {
			for (int d = 0; d < _n; d++) {
				_work[d] += _points[_indices[i]]._x[d] / _n;
			}
		}
		for (int d = 0; d < _n; d++) {
			_work[d] = 2. * _work[d] - _points[_indices[_n]]._x[d];
		}

		// check if the trial point is within the bounds
		bool bounded = inBounds(&_work[0]);
		if (!bounded) {
			continue;
		}

		// replace worst member in S with x-tilde if better
		double ftrial = _f._f(&_work[0]);
		_fev++;
		if (ftrial < _points[_np - 1]._f) {
			return replace(_np - 1, &_work[0], ftrial);
		}

		// try to mutate x-tilde through reflection
		for (int d = 0; d < _n; d++) {
			const double w = Random::get(0., 1.);
			_work[d] = (1. + w) * _points[0]._x[d] - w * _work[d];
		}

		// replace worst member in S with y-tilde if better
		ftrial = _f._f(&_work[0]);
		_fev++;
		if (ftrial < _points[_np - 1]._f) {
			return replace(_np - 1, &_work[0], ftrial);
		}
	}
}

int Crs2Search::replace(int iold, double *x, double fx) {
	auto &old = _points[iold];
	std::copy(x, x + _n, old._x.begin());
	old._f = fx;
	const int inew = std::lower_bound(_points.begin(), _points.end(), old,
			point::compare_fitness) - _points.begin();
	if (iold > inew) {
		std::rotate(_points.rend() - iold - 1, _points.rend() - iold,
				_points.rend() - inew);
	} else {
		std::rotate(_points.begin() + iold, _points.begin() + iold + 1,
				_points.begin() + inew + 1);
	}
	return inew;
}

/* =============================================================
 *
 * 			CONVERGENCE AND BOUNDS CHECKING SUBROUTINES
 *
 * =============================================================
 */
bool Crs2Search::stop() {
	double fl = _points[0]._f;
	double fh = _points[_np - 1]._f;
	return std::fabs(fl - fh) < _tol;
}

bool Crs2Search::inBounds(double *p) {
	for (int d = 0; d < _n; d++) {
		if (p[d] < _lower[d] || p[d] > _upper[d]) {
			return false;
		}
	}
	return true;
}
