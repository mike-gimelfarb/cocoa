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

 [1] Rinnooy Kan, A. H., and G. T. Timmer. "Stochastic global optimization methods
 part II: Multi level methods." Mathematical Programming: Series A and B 39.1 (1987):
 57-78.

 [2] Larson, Jeffrey, and Stefan M. Wild. "A batch, derivative-free algorithm for
 finding multiple local minima." Optimization and Engineering 17.1 (2016): 205-228.
 */

#include <cmath>

#include "../../random.hpp"

using Random = effolkronium::random_static;

template<typename T> MLSL<T>::MLSL(UnivariateOptimizer<T> *local, int n,
		int mfev, double sep, double sigma, double mu, double nu) {
	_local = local;
	_n = n;
	_mfev = mfev;
	_sigma = T(sigma);
	_mu = T(mu);
	_nu = T(nu);
	_sep = T(sep);
}

template<typename T> solutions<T> MLSL<T>::optimize(const univariate<T> &f, T a,
		T b) {

	// define problem
	_f = f;
	_a = a;
	_b = b;

	// memory
	_minima = std::vector<T>();
	_S = std::vector<std::shared_ptr<mlsl_point<T>>>();
	_ns = 0;
	_fev = 0;

	// main loop
	bool conv = false;
	while (_fev < _mfev) {
		uniformSampling();
		const int istart = localStart();
		if (istart < 0) {
			conv = true;
			break;
		}
		_S[istart]->_startlocal = true;
		const auto &sol = optimizeLocal(istart);
		if (sol._converged) {
			addMinimum(sol._sol);
			_fev += sol._fev;
		}
	}
	return {_minima, _fev, conv};
}

template<typename T> void MLSL<T>::uniformSampling() {
	for (int i = 0; i < _n; i++) {
		const double z = Random::get(_a, _b);
		const T &Tz = T(z);
		const auto &item = std::make_shared < mlsl_point < T >> (mlsl_point<T> {
				Tz, _f(Tz), false });
		const auto &where = std::upper_bound(_S.begin(), _S.end(), item,
				mlsl_point < T > ::compare_x);
		_S.insert(where, item); // note this is O(|S|) and can be improved
	}
	_fev += _n;
	_ns += _n;
}

template<typename T> solution<T> MLSL<T>::optimizeLocal(const int istart) {

	// find where the bounds should be
	const auto &p = _S[istart];
	T a, b;
	if (_minima.empty()) {
		a = _a;
		b = _b;
	} else {
		const auto &where = std::upper_bound(_minima.begin(), _minima.end(),
				p->_x);
		if (where == _minima.begin()) {
			a = _a;
			b = *(where);
		} else if (where == _minima.end()) {
			a = *(where - 1);
			b = _b;
		} else {
			a = *(where - 1);
			b = *(where);
		}
	}

	// optimize
	return _local->optimize(_f, p->_x, a, b);
}

template<typename T> void MLSL<T>::addMinimum(const T &min) {
	if (_minima.empty()) {
		_minima.push_back(min);
	} else {
		const auto &where = std::upper_bound(_minima.begin(), _minima.end(),
				min);
		if ((where == _minima.end() || std::fabs(*where - min) > _sep)
				&& ((where == _minima.begin()
						|| std::fabs(*(where - 1) - min) > _sep))) {
			_minima.insert(where, min);
		}
	}
}

template<typename T> int MLSL<T>::localStart() {
	const T rk = (_b - _a) / 2. * T(_sigma) * std::log(T(1. * _ns))
			/ T(1. * _ns);
	for (int i = 0; i < _ns; i++) {
		if (validStart(i, rk)) {
			return i;
		}
	}
	return -1;
}

template<typename T> bool MLSL<T>::validStart(const int i, const T rk) {

	// S3: has not started a local optimization run
	const auto &p = _S[i];
	if (p->_startlocal) {
		return false;
	}

	// S4: distance from domain boundary
	if (std::fabs(p->_x - _a) < _mu || std::fabs(p->_x - _b) < _mu) {
		return false;
	}

	// S5: distance from local known min in O(log |S|)
	if (!_minima.empty()) {
		const T &closest = nearestKnownMin(p->_x);
		if (std::fabs(p->_x - closest) < _nu) {
			return false;
		}
	}

	// S2: distance to neighbors
	for (int left = i - 1; left >= 0; left--) {
		if (std::fabs(p->_x - _S[left]->_x) > rk) {
			break;
		} else if (_S[left]->_f < p->_f) {
			return false;
		}
	}
	for (int right = i + 1; right < _ns; right++) {
		if (std::fabs(p->_x - _S[right]->_x) > rk) {
			break;
		} else if (_S[right]->_f < p->_f) {
			return false;
		}
	}

	return true;
}

template<typename T> T MLSL<T>::nearestKnownMin(const T &p) {
	const auto &iter_geq = std::lower_bound(_minima.begin(), _minima.end(), p);
	if (iter_geq == _minima.begin()) {
		return _minima[0];
	} else {
		int res = iter_geq - _minima.begin();
		if (std::fabs(p - *(iter_geq - 1)) < std::fabs(p - *(iter_geq))) {
			res--;
		}
		return _minima[res];
	}
}
