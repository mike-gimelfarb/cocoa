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

 [1] Askarzadeh, Alireza & Rashedi, Esmat. (2017). Harmony Search Algorithm.
 10.4018/978-1-5225-2322-2.ch001.

 [2] Chakraborty, P., Roy, G. G., Das, S., Jain, D., & Abraham, A. (2009). An improved
 harmony search algorithm with differential mutation operator. Fundamenta Informaticae,
 95, 1–26

 [3] Geem, Z. W., & Sim, K. B. (2010). Parameter-setting-free harmony search algorithm.
 Applied Mathematics and Computation, 217(8), 3881–3889. doi:10.1016/j.amc.2010.09.049

 [4] Mahdavi, M., Fesanghary, M., & Damangir, E. (2007). An improved harmony search
 algorithm for solving optimization problems. Applied Mathematics and Computation, 188(2),
 1567–1579. doi:10.1016/j.amc.2006.11.033

 [5] Wang, C. M., & Huang, Y. F. (2010). Self-adaptive harmony search algorithm for
 optimization. Expert Systems with Applications, 37(4), 2826–2837. doi:10.1016/j.
 eswa.2009.09.008

 [6] Woo Z, Hoon J, Loganathan GV. A New Heuristic Optimization Algorithm: Harmony Search.
 SIMULATION. 2001;76(2):60-68. doi:10.1177/003754970107600201
 */

#include <algorithm>
#include <limits>
#include <numeric>
#include <iostream>

#include "../../random.hpp"

#include "harmony.h"

using Random = effolkronium::random_static;

HarmonySearch::HarmonySearch(int mfev, int hms, int hpi, HMCR harmony,
		PAR pitch, PAStrategy pstrat) {
	_hms = hms;
	_hpi = hpi;
	_mfev = mfev;
	_harmony = harmony;
	_pitch = pitch;
	_pstrat = pstrat;
}

void HarmonySearch::init(const multivariate_problem &f, const double *guess) {

	// problem initialization
	if (f._hasc || f._hasbbc) {
		std::cerr
				<< "Warning [HarmonySearch]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// harmony memory
	_hm.clear();
	for (int i = 0; i < _hms; i++) {
		std::vector<double> x(_n);
		std::vector<harmony_op> op(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
			op[j] = random;
		}
		const harmony har { x, _f._f(&x[0]), op };
		_hm.push_back(std::move(har));
	}
	_best = &*std::min_element(_hm.begin(), _hm.end(),
			harmony::compare_fitness);

	// temporary
	std::vector<double> x(_n);
	std::vector<harmony_op> op(_n);
	_temp = harmony { x, 0., op };

	// other memory
	_fev = _hms;
	_it = 0;
	_mit = _mfev / _hpi;

	// initialize parameters to default
	_hmcr = std::vector<double>(_n, _harmony._hmcrinit);
	_par = std::vector<double>(_n, _pitch._parinit);
	_bw = _pstrat._bwinit;
}

void HarmonySearch::iterate() {

	// adapt parameters
	adaptParams();

	// improvisation
	for (int k = 0; k < _hpi; k++) {
		improvise();
		replace();
	}

	// get the best element
	_best = &*std::min_element(_hm.begin(), _hm.end(),
			harmony::compare_fitness);

	_it++;
}

multivariate_solution HarmonySearch::optimize(const multivariate_problem &f,
		const double *guess) {
	init(f, guess);
	while (_fev < _mfev) {
		iterate();
	}
	return {_best->_x, _fev, false};
}

void HarmonySearch::improvise() {

	// for DE, pick two members
	int i1 = -1, i2 = -1;
	if (_pstrat._name == "de") {
		i1 = Random::get(0, _hms - 1);
		i2 = Random::get(0, _hms - 1);
		while (i2 == i1) {
			i2 = Random::get(0, _hms - 1);
		}
	}

	// improvisation
	for (int j = 0; j < _n; j++) {

		// for self-adaptive strategy, compute min and max
		double min_hm = std::numeric_limits<double>::infinity();
		double max_hm = -std::numeric_limits<double>::infinity();
		if (_pstrat._name == "sa") {
			for (const auto &p : _hm) {
				min_hm = std::min(min_hm, p._x[j]);
				max_hm = std::max(max_hm, p._x[j]);
			}
		}

		// proceed with improvisation
		if (Random::get(0., 1.) <= _hmcr[j]) {

			// memory consideration
			const int k = Random::get(0, _hms - 1);
			_temp._x[j] = _hm[k]._x[j];
			_temp._op[j] = memory;

			// pitch adjustment
			if (Random::get(0., 1.) <= _par[j]) {
				double v = 0.;

				// strategy selection
				if (_pstrat._name == "fixed" || _pstrat._name == "improved") {

					// original formula
					v = _bw * (2. * Random::get(0., 1.) - 1.)
							* (_upper[j] - _lower[j]);
				} else if (_pstrat._name == "de") {

					// de/rand/bin/1
					v = _pstrat._cr * (_hm[i1]._x[j] - _hm[i2]._x[j]);
				} else if (_pstrat._name == "sa") {

					// self-adaptive formula
					if (Random::get(0., 1.) < 0.5) {
						v = (max_hm - _temp._x[j]) * Random::get(0., 1.);
					} else {
						v = (min_hm - _temp._x[j]) * Random::get(0., 1.);
					}
				}

				// update harmony
				_temp._x[j] += v;
				_temp._x[j] = std::max(_lower[j],
						std::min(_temp._x[j], _upper[j]));
				_temp._op[j] = pitch;
			}
		} else {

			// random search
			_temp._x[j] = Random::get(_lower[j], _upper[j]);
			_temp._op[j] = random;
		}

	}

	// fitness evaluation
	_temp._f = _f._f(&_temp._x[0]);
	_fev++;
}

void HarmonySearch::replace() {

	// find which harmony is the worst
	auto &worst = *std::max_element(_hm.begin(), _hm.end(),
			harmony::compare_fitness);

	// replacement
	if (_temp._f < worst._f) {
		std::copy(_temp._x.begin(), _temp._x.end(), (worst._x).begin());
		worst._f = _temp._f;
		std::copy(_temp._op.begin(), _temp._op.end(), (worst._op).begin());
	}
}

void HarmonySearch::adaptParams() {

	// adapt HMCR
	if (_harmony._name == "fixed") {

		// use default values
	} else if (_harmony._name == "none") {

		// after the warm-up period, use history to select HMCR
		if (_fev > _harmony._warm * _hms) {
			for (int j = 0; j < _n; j++) {
				_hmcr[j] = 0.;
				for (const auto &h : _hm) {
					if (h._op[j] == memory) {
						_hmcr[j] += 1.;
					}
				}
				_hmcr[j] = std::max(_harmony._hmcrmin,
						std::min(_hmcr[j] / _hms, _harmony._hmcrmax));
			}
		}

		// averaged version
		if (!_harmony._local) {
			const double hmcr = std::accumulate(_hmcr.begin(), _hmcr.end(), 0.)
					/ _n;
			std::fill(_hmcr.begin(), _hmcr.end(), hmcr);
		}
	}

	// adapt PAR
	if (_pitch._name == "fixed") {

		// use default values
	} else if (_pitch._name == "none") {

		// after the warm-up period, use history to select PAR
		if (_fev > _pitch._warm * _hms) {
			for (int j = 0; j < _n; j++) {
				_par[j] = 0.;
				for (const auto &h : _hm) {
					if (h._op[j] == pitch) {
						_par[j] += 1.;
					}
				}
				_par[j] = std::max(_pitch._parmin,
						std::min(_par[j] / _hms, _pitch._parmax));
			}
		}

		// averaged version
		if (!_pitch._local) {
			const double par = std::accumulate(_par.begin(), _par.end(), 0.)
					/ _n;
			std::fill(_par.begin(), _par.end(), par);
		}
	} else if (_pitch._name == "improved") {

		// anneal PAR linearly
		const double par = _pitch._parmin
				+ ((_pitch._parmax - _pitch._parmin) / _mit) * _it;
		std::fill(_par.begin(), _par.end(), par);
	}

	// adapt bandwidth if needed
	if (_pstrat._name == "improved") {

		// anneal BW nonlinearly
		_bw = _pstrat._bwmax
				* std::pow(_pstrat._bwmin / _pstrat._bwmax, (1. * _it) / _mit);
	}
}
