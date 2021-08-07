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

 [1] Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan.
 "Differential evolution algorithm with strategy adaptation for global
 numerical optimization." IEEE transactions on Evolutionary Computation 13.2
 (2009): 398-417.

 [2] Huang, Vicky Ling, A. Kai Qin, and Ponnuthurai N. Suganthan.
 "Self-adaptive differential evolution algorithm for constrained
 real-parameter optimization." Evolutionary Computation, 2006. CEC 2006. IEEE
 Congress on. IEEE, 2006. Yang, Zhenyu, Ke Tang, and Xin Yao.

 [3] "Self-adaptive differential evolution with neighborhood search."
 Evolutionary Computation, 2008. CEC 2008.(IEEE World Congress on
 Computational Intelligence). IEEE Congress on. IEEE, 2008.
 */

#include <stdexcept>

#include "../../random.hpp"
#include "../sade/sade.h"

using Random = effolkronium::random_static;

SadeSearch::SadeSearch(int mfev, double tol, double stol, int np, int lp, // @suppress("Class members should be properly initialized")
		int cp) {
	_tol = tol;
	_stol = stol;
	_np = np;
	_lp = lp;
	_cp = cp;
	_mfev = mfev;
}

void SadeSearch::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {
	_f = f;
	_n = n;
	_lower = std::vector<double>(lower, lower + _n);
	_upper = std::vector<double>(upper, upper + _n);
	_gen = _fev = _ihist = 0;
	_k = 5;
	_scr = 0.1;
	_sf = 0.3;
	_mu = 0.5;

	// initialize members
	_xtrii = std::vector<double>(_n);
	_pool.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
		}
		auto pParticle = std::make_shared<sade_particle>(
				sade_particle { x, _f(&x[0]) });
		_pool.push_back(std::move(pParticle));
	}
	_fev = _np;

	// compute the rankings
	_ibw = std::vector<int>(4);
	rank();

	// initialize learning parameters
	_crm = 0.5;
	_ns = std::vector<int>(_k);
	_nf = std::vector<int>(_k);
	_fns0 = _fnf0 = _fns1 = _fnf1 = 0;
	_p = std::vector<double>(_k);
	_fp = 0.5;
	std::fill(_p.begin(), _p.end(), 1. / _k);
	_cr = std::vector<double>(_np);
	_crrec = std::vector<double>(_np * _lp);
	_dfit = std::vector<double>(_np * _lp);
}

void SadeSearch::iterate() {

	// learning update
	if (_gen >= _lp && _gen % _lp == 0) {

		// update strategy probabilities
		bool update = true;
		for (int k = 0; k < _k; k++) {
			if (_ns[k] == 0) {
				update = false;
				break;
			}
		}
		for (int k = 0; k < _k; k++) {
			if (update) {
				_p[k] = (1. * _ns[k]) / (_ns[k] + _nf[k]);
			}
			_ns[k] = _nf[k] = 0;
		}

		// update crossover CRm parameter
		double denom = 0.;
		double numer = 0.;
		for (int i = 0; i < _ihist; i++) {
			numer += _crrec[i] * _dfit[i];
			denom += _dfit[i];
			_dfit[i] = _crrec[i] = 0.;
		}
		if (denom > 0.0) {
			_crm = numer / denom;
		}
		_ihist = 0;

		// update local search F parameter Fp
		denom = _fns1 * (_fns0 + _fnf0) + _fns0 * (_fns1 + _fnf1);
		if (denom > 0.0) {
			_fp = _fns0 * (_fns1 + _fnf1) / denom;
		}
		_fns0 = _fnf0 = _fns1 = _fnf1 = 0;
	}

	// update population
	for (int i = 0; i < _np; i++) {

		// compute crossover constant F
		const double u = Random::get(0., 1.);
		const bool usegauss = u < _fp;
		double F = 0.;
		while (F <= 0.) {
			if (usegauss) {
				F = Random::get(_Z) * _sf + _mu;
			} else {
				F = std::tan(M_PI * (Random::get(0., 1.) - 0.5));
			}
		}

		// compute CR constant if needed
		if (_gen > 0 && _gen % _cp == 0) {
			double CRi = Random::get(_Z) * _scr + _crm;
			while (CRi <= 0. || CRi >= 1.) {
				CRi = Random::get(_Z) * _scr + _crm;
			}
			_cr[i] = CRi;
		}

		// generate a strategy to use for the current member
		const int ki = roulette();

		// trial vector generation and fitness
		trial(i, ki, F, _cr[i], _ibw[0]);
		const double newy = _f(&_xtrii[0]);

		// update all counters and data for learning
		auto &part = _pool[i];
		if (newy < part->_f) {
			_crrec[_ihist] = _cr[i];
			_dfit[_ihist] = part->_f - newy;
			_ihist++;
			_ns[ki]++;
			if (usegauss) {
				_fns0++;
			} else {
				_fns1++;
			}

			// update population if the new vector is improvement
			std::copy(_xtrii.begin(), _xtrii.end(), part->_x.begin());
			part->_f = newy;
		} else {
			_nf[ki]++;
			if (usegauss) {
				_fnf0++;
			} else {
				_fnf1++;
			}
		}
	}

	// update indices
	_fev += _np;
	_gen++;

	// find the new ranking
	rank();
}

multivariate_solution SadeSearch::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {

	// initialize parameters
	init(f, n, guess, lower, upper);

	// main loop of SADE
	bool converged = false;
	while (_fev < _mfev) {

		// learning and solution update
		iterate();

		// test convergence in function values
		const double y0 = _pool[_ibw[0]]->_f;
		const double y3 = _pool[_ibw[3]]->_f;
		if (std::fabs(y0 - y3) <= _tol) {

			// compute standard deviation of swarm radiuses
			int count = 0;
			double mean = 0.;
			double m2 = 0.;
			for (auto &pt : _pool) {
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
	return {_pool[_ibw[0]]->_x, _fev, converged};
}

void SadeSearch::trial(int i, int ki, double f, double cr, int ib) {

	// randomly select five distinct agents from population
	int a, b, c, d, e;
	do {
		a = Random::get(0, _np - 1);
	} while (a == i);
	do {
		b = Random::get(0, _np - 1);
	} while (b == i || b == a);
	do {
		c = Random::get(0, _np - 1);
	} while (c == i || c == a || c == b);
	do {
		d = Random::get(0, _np - 1);
	} while (d == i || d == a || d == b || d == c);
	do {
		e = Random::get(0, _np - 1);
	} while (e == i || e == a || e == b || e == c || e == d);

	// retrieve their data
	std::vector<double> &p1 = _pool[a]->_x;
	std::vector<double> &p2 = _pool[b]->_x;
	std::vector<double> &p3 = _pool[c]->_x;
	std::vector<double> &p4 = _pool[d]->_x;
	std::vector<double> &p5 = _pool[e]->_x;
	std::vector<double> &x = _pool[i]->_x;
	std::vector<double> &bb = _pool[ib]->_x;

	// use them to generate a mutated vector from the original
	const int jrnd = Random::get(0, _n - 1);
	switch (ki) {
	case 0:

		// DE/rand/1
		for (int j = 0; j < _n; j++) {
			if (j == jrnd || Random::get(0., 1.) <= cr) {
				_xtrii[j] = p1[j] + f * (p2[j] - p3[j]);
			} else {
				_xtrii[j] = x[j];
			}
		}
		break;
	case 1:

		// DE/best/1
		for (int j = 0; j < _n; j++) {
			if (j == jrnd || Random::get(0., 1.) <= cr) {
				_xtrii[j] = bb[j] + f * (p1[j] - p2[j]);
			} else {
				_xtrii[j] = x[j];
			}
		}
		break;
	case 2:

		// DE/current-to-best/1
		for (int j = 0; j < _n; j++) {
			if (j == jrnd || Random::get(0., 1.) <= cr) {
				_xtrii[j] = x[j] + f * (bb[j] - x[j]) + f * (p1[j] - p2[j]);
			} else {
				_xtrii[j] = x[j];
			}
		}
		break;
	case 3:

		// DE/best/2:
		for (int j = 0; j < _n; j++) {
			if (j == jrnd || Random::get(0., 1.) <= cr) {
				_xtrii[j] = bb[j] + f * (p1[j] - p2[j]) + f * (p3[j] - p4[j]);
			} else {
				_xtrii[j] = x[j];
			}
		}
		break;
	case 4:

		// DE/rand/2:
		for (int j = 0; j < _n; j++) {
			if (j == jrnd || Random::get(0., 1.) <= cr) {
				_xtrii[j] = p1[j] + f * (p2[j] - p3[j]) + f * (p4[j] - p5[j]);
			} else {
				_xtrii[j] = x[j];
			}
		}
		break;
	default:

		throw std::invalid_argument(
				"Fatal error when adapting sub-population sizes. Please report this issue on GitHub.");
	}

	// ensure that the new trial vector lies in [lower, upper]
	// if not, then randomize it within this range
	for (int j = 0; j < _n; j++) {
		if (_xtrii[j] > _upper[j] || _xtrii[j] < _lower[j]) {
			_xtrii[j] = Random::get(_lower[j], _upper[j]);
		}
	}
}

void SadeSearch::rank() {
	double y0 = std::numeric_limits<double>::infinity();
	double y1 = y0;
	double y2 = -std::numeric_limits<double>::infinity();
	double y3 = y2;
	_ibw[0] = _ibw[1] = _ibw[2] = _ibw[3] = 0;
	for (int i = 0; i < _np; i++) {
		const double yi = _pool[i]->_f;
		if (yi > y3) {
			y2 = y3;
			y3 = yi;
			_ibw[2] = _ibw[3];
			_ibw[3] = i;
		} else if (yi > y2) {
			y2 = yi;
			_ibw[2] = i;
		}
		if (yi < y0) {
			y1 = y0;
			y0 = yi;
			_ibw[1] = _ibw[0];
			_ibw[0] = i;
		} else if (yi < y1) {
			y1 = yi;
			_ibw[1] = i;
		}
	}
}

int SadeSearch::roulette() {
	const double s = std::accumulate(_p.begin(), _p.end(), 0.);
	double U = Random::get(0., s);
	for (int k = 0; k < _k; k++) {
		U -= _p[k];
		if (U <= 0.) {
			return k;
		}
	}
	return _k - 1;
}
