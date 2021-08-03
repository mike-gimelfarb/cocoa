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

 [1] Rao, R. "Jaya: A simple and new optimization algorithm for solving
 constrained and unconstrained optimization problems." International Journal
 of Industrial Engineering Computations 7.1 (2016): 19-34.

 [2] Yu, Jiang-Tao, et al. "Jaya algorithm with self-adaptive multi-population
 and Lévy flights for solving economic load dispatch problems." IEEE Access 7
 (2019): 21372-21384.
 */

#define _USE_MATH_DEFINES
#include <algorithm>
#include <math.h>
#include <limits>
#include <stdexcept>

#include "jaya.h"
#include "../../random.hpp"

using Random = effolkronium::random_static;

JayaSearch::JayaSearch(int mfev, int np, int npmin, int k0,  // @suppress("Class members should be properly initialized")
		double scale, double beta, bool adapt) {
	_mfev = mfev;
	_np = np;
	_k0 = k0;
	_npmin = npmin;
	_adapt_k = adapt;
	_scale = 0.01;
	_beta = 1.5;
}

void JayaSearch::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {
	_n = n;
	_f = f;
	_k = _k0;
	_lower = std::vector<double>(lower, lower + _n);
	_upper = std::vector<double>(upper, upper + _n);
	_it = 0;
	_pbest = std::numeric_limits<double>::infinity();
	_best = std::numeric_limits<double>::infinity();
	_pool.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<double> x(_n);
		for (int j = 0; j < _n; j++) {
			x[j] = Random::get(_lower[j], _upper[j]);
		}
		auto pParticle = std::make_shared<jaya_particle>(
				jaya_particle { x, f(&x[0]) });
		_pool.push_back(std::move(pParticle));
	}
	_fev = _np;
	_tmp = std::vector<double>(_n);
	_len = std::vector<int>(_np);
	_r1 = std::vector<double>(_n);
	_r2 = std::vector<double>(_n);
	_sigmau = pow(
			(tgamma(1. + _beta) * sin(_beta * M_PI / 2.))
					/ (tgamma((1. + _beta) / 2.) * _beta
							* pow(2., (_beta - 1.) / 2.)), 1. / _beta);
}

void JayaSearch::iterate() {

	// divide population into K sub-populations
	divideSubpopulation();

	// prepare for the current iteration
	_best = std::numeric_limits<double>::infinity();
	randomizeR();

	// apply the evolution algorithm to each sub-population
	int idx_next_subpop = 0;
	for (int q = 0; q < _k; q++) {

		// get best and worst members
		auto best = _pool[idx_next_subpop];
		auto worst = _pool[idx_next_subpop];
		for (int i = idx_next_subpop; i < idx_next_subpop + _len[q]; i++) {
			if (_pool[i]->_f < best->_f) {
				best = _pool[i];
			}
			if (_pool[i]->_f > worst->_f) {
				worst = _pool[i];
			}
		}

		// perform update
		for (int i = idx_next_subpop; i < idx_next_subpop + _len[q]; i++) {
			evolve(i, *best, *worst);
		}
		idx_next_subpop += _len[q];
	}

	// adapt the number of sub-populations
	adaptK();

	// increment counters
	_pbest = _best;
	_it++;
}

multivariate_solution JayaSearch::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {

	// initialization
	init(f, n, guess, lower, upper);

	// main loop
	while (true) {
		iterate();

		// check max number of evaluations
		if (_fev >= _mfev) {
			break;
		}
	}
	return {_bestX._x, _fev, false};
}

void JayaSearch::divideSubpopulation() {

	// allocate N_subpop = NP // K elements at random to each sub-population
	Random::shuffle(_pool.begin(), _pool.end());
	const int base_len = _np / _k;
	std::fill(_len.begin(), _len.end(), base_len);

	// since the total number of elements must be NP, we fill the remaining
	// sub-populations randomly
	if (base_len * _k < _np) {
		for (int i = 0; i < _np - base_len * _k; i++) {
			const int idx = Random::get(0, _k - 1);
			_len[idx]++;
		}
	}

	// check that the total number of elements is equal to NP among all K
	// sub-population
	const int sum_len = std::accumulate(_len.begin(), _len.begin() + _k, 0);
	if (sum_len != _np) {
		throw std::invalid_argument(
				"Fatal error when adapting sub-population sizes. Please report this issue on GitHub.");
	}
}

void JayaSearch::adaptK() {
	if (_adapt_k) {
		if (_best < _pbest) {
			if (_np >= _npmin * (_k + 1)) {
				_k++;
			}
		} else if (_best > _pbest) {
			if (_k > 1) {
				_k--;
			}
		}
	}
}

void JayaSearch::evolve(int i, jaya_particle &best, jaya_particle &worst) {

	// evolve the position
	auto part = _pool[i];
	for (int j = 0; j < _n; j++) {
		const double step = sampleLevy();
		const double stepSize = _scale * step * (part->_x[j] - best._x[j]);
		const double levy_ij = part->_x[j] + stepSize * Random::get(0., 1.);
		_tmp[j] = levy_ij + _r1[j] * (best._x[j] - std::fabs(part->_x[j]))
				- _r2[j] * (worst._x[j] - std::fabs(part->_x[j]));
		_tmp[j] = std::max(_lower[j], std::min(_tmp[j], _upper[j]));
	}

	// evaluate fitness of new position
	const double ftmp = _f(&_tmp[0]);
	_fev++;

	// copy the particle back to the swarm if it is an improvement
	if (ftmp < part->_f) {
		std::copy(_tmp.begin(), _tmp.end(), part->_x.begin());
		part->_f = ftmp;
	}

	// update the global best for the current iteration
	if (part->_f < _best) {
		_best = part->_f;
		_bestX = *part;
	}
}

void JayaSearch::randomizeR() {
	for (int i = 0; i < _n; i++) {
		_r1[i] = Random::get(0., 1.);
		_r2[i] = Random::get(0., 1.);
	}
}

double JayaSearch::sampleLevy() {

	// Mantegna's algorithm
	const double sigma_v = 1.;
	const double u = Random::get(_Z) * _sigmau;
	const double v = Random::get(_Z) * sigma_v;
	return u / (std::pow(std::fabs(v), 1. / _beta));
}
