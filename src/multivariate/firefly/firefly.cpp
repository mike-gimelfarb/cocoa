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

 [1] Wang, Hui, et al. "Randomly attracted firefly algorithm with neighborhood
 search and dynamic parameter adjustment mechanism." Soft Computing 21.18
 (2017): 5325-5339.

 [2] I. Fister Jr., X.-S. Yang, I. Fister, J. Brest, Memetic firefly algorithm
 for combinatorial optimization, in Bioinspired Optimization Methods and their
 Applications (BIOMA 2012), B. Filipic and J.Silc, Eds. Jozef Stefan
 Institute, Ljubljana, Slovenia, 2012.

 [3] Yu S, Zhu S, Ma Y, Mao D. Enhancing firefly algorithm using generalized
 opposition-based learning. Computing. 2015 :97(7) 741–754.

 [4] Shakarami MR, Sedaghati R. A new approach for network reconfiguration
 problem in order to deviation bus voltage minimization with regard to
 probabilistic load model and DGs. International Journal of Electrical,
 Computer, Energetic, Electronic and Communication Engineering.
 2014;8(2):430–5.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <numeric>

#include "../../random.hpp"

#include "firefly.h"

using Random = effolkronium::random_static;

FireflySearch::FireflySearch(int mfev, int np, double gamma, double alpha0,
		double decay, double bmin, double bmax, search_strategy strategy,
		noise_type noise, bool nsearch, int ns, bool osearch, double wbprob) {
	_np = np;
	_bmin = bmin;
	_bmax = bmax;
	_gamma = gamma;
	_noise = noise;
	_strategy = strategy;
	_alpha0 = alpha0;
	_decay = decay;
	_nsearch = nsearch;
	_k = std::min(ns, (_np - 1) / 2);
	_osearch = osearch;
	_wbprob = wbprob;
	_maxit = mfev / _np;
	_mfev = mfev;
}

void FireflySearch::init(const multivariate_problem &f, const double *guess) {

	// initialize problem
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [Firefly]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);

	// initialize fireflies
	_swarm.clear();
	for (int ip = 0; ip < _np; ip++) {
		std::vector<double> x(_n);
		std::vector<double> xb(_n);
		for (int i = 0; i < _n; i++) {
			x[i] = xb[i] = Random::get(_lower[i], _upper[i]);
		}
		const firefly part { x, xb, _f._f(&x[0]), NAN };
		_swarm.push_back(std::move(part));
	}

	// initialize memory
	_fev = 0;
	_it = 0;
	_tmpd = std::vector<double>(_n);
	_tmpk = std::vector<int>(2);
	_tmp4 = std::vector<double>(4);
	_tmpx.clear();
	_tmpx.resize(3, std::vector<double>(_n, 0.));

	updateStats();
}

void FireflySearch::iterate() {
	updateFireflies();
	updateStats();
	_it++;
}

multivariate_solution FireflySearch::optimize(const multivariate_problem &f,
		const double *guess) {

	// initialization
	init(f, guess);

	// main loop
	while (true) {
		iterate();

		// check max number of evaluations
		if (_fev >= _mfev) {
			break;
		}

		// check max number of generations
		if (_it >= _maxit) {
			break;
		}
	}
	return {_best->_x, _fev, false};
}

void FireflySearch::updateFireflies() {

	// main loop of firefly update
	int i = 0;
	for (auto &pt : _swarm) {

		// choose a random index not equal to j
		const int random_j = sample1FromSwarm(i);

		// random attraction model
		if (pt._f > _swarm[random_j]._f) {

			// compute the distance between fireflies i and j
			auto &ffi = pt._x;
			auto &ffj = _swarm[random_j]._x;
			const double r = computeDistance(&ffi[0], &ffj[0]);

			// update the step size alpha
			const double alpha = applyStrategy(i);

			// compute the attraction coefficient beta
			double beta = (_bmax - _bmin) * std::exp(-_gamma * r * r) + _bmin;
			beta = (beta * _fev) / _mfev;

			// move the firefly i closer to j
			std::copy(ffi.begin(), ffi.end(), (pt._xb).begin());
			for (int k = 0; k < _n; k++) {
				const double scale_k = _upper[k] - _lower[k];
				const double noise = alpha * sampleNoise() * scale_k;
				ffi[k] = ffi[k] * (1. - beta) + ffj[k] * beta + noise;
			}

			// correct the coordinates of firefly i if out of bounds
			rectifyBounds(&(pt._x)[0]);

			// update the fitness of firefly i
			pt._f = _f._f(&(pt._x)[0]);
			_fev++;

			// update counters and check for termination
			if (_fev >= _mfev) {
				break;
			}
		} else {

			// neighborhood search
			if (_nsearch) {

				// generate local trial solution
				sample2FromNeighborhood(i);
				sample3Uniform();
				auto &X = pt._x;
				auto &pbest = pt._xb;
				auto &Xi1 = _swarm[_tmpk[0]]._x;
				auto &Xi2 = _swarm[_tmpk[1]]._x;
				for (int j = 0; j < _n; j++) {
					_tmpx[0][j] = _tmp4[0] * X[j] + _tmp4[1] * pbest[j]
							+ _tmp4[2] * (Xi1[j] - Xi2[j]);
				}
				rectifyBounds(&_tmpx[0][0]);

				// generate global trial solution
				sample2FromSwarm(i);
				sample3Uniform();
				auto &gbest = _best->_x;
				auto &Xi3 = _swarm[_tmpk[0]]._x;
				auto &Xi4 = _swarm[_tmpk[1]]._x;
				for (int j = 0; j < _n; j++) {
					_tmpx[1][j] = _tmp4[0] * X[j] + _tmp4[1] * gbest[j]
							+ _tmp4[2] * (Xi3[j] - Xi4[j]);
				}
				rectifyBounds(&_tmpx[1][0]);

				// generate Cauchy trial solution
				for (int j = 0; j < _n; j++) {
					_tmpx[2][j] = X[j] + sampleCauchy();
				}
				rectifyBounds(&_tmpx[2][0]);

				// calculate the fitness values of X1, X2 and X3
				_tmp4[0] = pt._f;
				_tmp4[1] = _f._f(&_tmpx[0][0]);
				_tmp4[2] = _f._f(&_tmpx[1][0]);
				_tmp4[3] = _f._f(&_tmpx[2][0]);
				_fev += 3;

				// select the best solution among X, X1, X2, X3 as the new X
				const int i_min = std::min_element(_tmp4.begin(), _tmp4.end())
						- _tmp4.begin();
				if (i_min >= 1) {
					std::copy((pt._x).begin(), (pt._x).end(), (pt._xb).begin());
					std::copy(_tmpx[i_min - 1].begin(), _tmpx[i_min - 1].end(),
							(pt._x).begin());
					pt._f = _tmp4[i_min];
				}
			}
		}
		i++;
	}

	// compute the dimmest and strongest firefly
	firefly *worst, *best;
	bool first = true;
	for (auto &fly : _swarm) {
		if (first) {
			worst = &fly;
			best = &fly;
			first = false;
		}
		if (fly._f > worst->_f) {
			worst = &fly;
		}
		if (fly._f < best->_f) {
			best = &fly;
		}
	}

	// opposition-based update of the dimmest firefly
	if (_osearch) {
		if (Random::get(0., 1.) < _wbprob) {
			std::copy((best->_x).begin(), (best->_x).end(),
					(worst->_x).begin());
			std::copy((best->_xb).begin(), (best->_xb).end(),
					(worst->_xb).begin());
			worst->_alpha = best->_alpha;
			worst->_f = best->_f;
		} else {
			std::copy((worst->_xb).begin(), (worst->_xb).end(),
					(worst->_x).begin());
			for (int j = 0; j < _n; j++) {
				worst->_x[j] = _lower[j] + _upper[j] - worst->_x[j];
			}
			worst->_alpha = NAN;
			worst->_f = _f._f(&(worst->_x)[0]);
			_fev++;
		}
	}
}

double FireflySearch::applyStrategy(int i) {
	switch (_strategy) {
	case geometric: {
		auto &fly = _swarm[i];
		double alpha;
		if (fly._alpha != fly._alpha) {
			alpha = _alpha0;
		} else {
			alpha = fly._alpha * _decay;
		}
		fly._alpha = alpha;
		return alpha;
	}
	case sh2014: {
		auto &fly = _swarm[i];
		const double decay = std::pow(.5 / _maxit, 1. / _maxit);
		double alpha;
		if (fly._alpha != fly._alpha) {
			alpha = _alpha0;
		} else {
			alpha = fly._alpha * decay;
		}
		fly._alpha = alpha;
		return alpha;
	}
	default: {
		auto &fly = _swarm[i];
		double alpha;
		if (fly._alpha != fly._alpha) {
			alpha = _alpha0;
		} else {
			alpha = fly._alpha * std::pow(_decay, 1. / _maxit);
		}
		fly._alpha = alpha;
		return alpha;
	}
	}
}

double FireflySearch::computeDistance(double *x, double *y) {
	for (int i = 0; i < _n; i++) {
		_tmpd[i] = x[i] - y[i];
	}
	return std::sqrt(
			std::inner_product(_tmpd.begin(), _tmpd.end(), _tmpd.begin(), 0.));
}

void FireflySearch::rectifyBounds(double *x) {
	for (int i = 0; i < _n; i++) {
		x[i] = std::max(_lower[i], std::min(x[i], _upper[i]));
	}
}

void FireflySearch::updateStats() {
	_best = nullptr;
	_mina = 1.;
	_maxa = 0.;
	for (auto &fly : _swarm) {
		if (_best == nullptr) {
			_best = &fly;
		} else {
			if (fly._f < _best->_f) {
				_best = &fly;
			}
		}
		const double alpha = fly._alpha;
		if (alpha != alpha) {
			_mina = std::min(_mina, alpha);
			_maxa = std::max(_maxa, alpha);
		}
	}
}

void FireflySearch::sample2FromNeighborhood(int i) {
	for (int idx = 0; idx < 2; ++idx) {
		int iidx = i;
		while ((idx == 0 && iidx == i)
				|| (idx == 1 && (iidx == i || iidx == _tmpk[0]))) {
			const int r = Random::get(0, _k) + 1;
			if (Random::get(0., 1.) <= 0.5) {
				iidx = (i + r) % _np;
			} else {
				iidx = (i - r + _np) % _np;
			}
		}
		_tmpk[idx] = iidx;
	}
}

int FireflySearch::sample1FromSwarm(int i) {
	int j = i;
	while (j == i) {
		j = Random::get(0, _np - 1);
	}
	return j;
}

void FireflySearch::sample2FromSwarm(int i) {
	for (int idx = 0; idx <= 1; idx++) {
		int iidx = i;
		while ((idx == 0 && iidx == i)
				|| (idx == 1 && (iidx == i || iidx == _tmpk[0]))) {
			iidx = Random::get(0, _np - 1);
		}
		_tmpk[idx] = iidx;
	}
}

void FireflySearch::sample3Uniform() {
	const double r1 = Random::get(0., 1.);
	const double r2 = Random::get(0., 1.);
	const double r3 = Random::get(0., 1.);
	const double sum = r1 + r2 + r3;
	_tmp4[0] = r1 / sum;
	_tmp4[1] = r2 / sum;
	_tmp4[2] = r3 / sum;
}

double FireflySearch::sampleNoise() {
	switch (_noise) {
	case uniform:
		return Random::get(0., 1.) - 0.5;
	case gauss:
		return Random::get(_Z);
	case cauchy:
		return sampleCauchy();
	case none:
		return 0.;
	default:
		return 0.;
	}
}

double FireflySearch::sampleCauchy() {
	return std::tan(M_PI * (Random::get(0., 1.) - 0.5));
}
