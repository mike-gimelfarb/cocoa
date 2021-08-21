/*
 Copyright (c) 2012, Pinar Civicioglu
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <numeric>
#include <iostream>

#include "../../random.hpp"

#include "ds.h"

using Random = effolkronium::random_static;

DifferentialSearch::DifferentialSearch(int mfev, double tol, double stol,
		int np) {
	_tol = tol;
	_stol = stol;
	_np = np;
	_mfev = mfev;
}

void DifferentialSearch::init(const multivariate_problem &f,
		const double *guess) {
	if (f._hasc || f._hasbbc) {
		std::cerr << "Warning [DS]: problem constraints will be ignored."
				<< std::endl;
	}
	_f = f;
	_n = f._n;
	_lower = std::vector<double>(f._lower, f._lower + _n);
	_upper = std::vector<double>(f._upper, f._upper + _n);
	_fev = 0;
	_jind = std::vector<int>(_np);
	_methods = std::vector<int>(4);
	_methods[0] = 1;
	_methods[1] = 2;
	_methods[2] = 3;
	_methods[3] = 4;

	// generate initial individuals, clans and superorganism.
	_swarm.clear();
	for (int i = 0; i < _np; i++) {
		std::vector<int> map(_n);
		std::vector<double> x(_n);
		std::vector<double> so(_n);
		const ds_particle part { map, x, so, nullptr, 0., 0. };
		_swarm.push_back(std::move(part));
	}
	genPop();
}

void DifferentialSearch::iterate() {

	// SETTING OF ALGORITHMIC CONTROL PARAMETERS
	// Trial-pattern generation strategy for morphogenesis;
	// 'one-or-more morphogenesis'. (DEFAULT)
	const double p1 = Random::get(0., 0.3);
	const double p2 = Random::get(0., 0.3);

	// sample a method
	const int imethd = Random::get(0, 3);
	const int methd = _methods[imethd];

	// search direction
	genDir(methd);

	// mutation strategy matrix
	genMap(p1, p2);

	// Recommended Methods for generation of Scale-Factor; R
	// R=4*randn; % brownian walk
	// R=4*randg; % brownian walk
	// R=lognrnd(rand,5*rand); % brownian walk
	// R=1/normrnd(0,5); % pseudo-stable walk
	// we use pseudo-stable walk
	const double R = 1. / (-2. * std::log(Random::get(0., 1.)));

	// bio-interaction (morphogenesis)
	for (auto &p : _swarm) {
		for (int j = 0; j < _n; j++) {
			p._so[j] = p._x[j] + R * p._map[j] * ((*(p._dir))[j] - p._x[j]);
		}
	}

	// Boundary Control
	update();

	// Selection-II
	for (auto &p : _swarm) {
		p._fso = _f._f(&(p._so)[0]);
		if (p._fso < p._f) {
			p._f = p._fso;
			std::copy((p._so).begin(), (p._so).end(), (p._x).begin());
		}
	}
	_fev += _np;
}

multivariate_solution DifferentialSearch::optimize(
		const multivariate_problem &f, const double *guess) {

	// initialization of swarm
	init(f, guess);

	// main loop
	bool converged = false;
	while (_fev < _mfev) {

		// perform iteration
		iterate();

		// converge when distance in fitness between best and worst points
		// is below the given tolerance
		const int idxmin = std::min_element(_swarm.begin(), _swarm.end(),
				ds_particle::compare_fitness) - _swarm.begin();
		const int idxmax = std::max_element(_swarm.begin(), _swarm.end(),
				ds_particle::compare_fitness) - _swarm.begin();
		const double best = _swarm[idxmin]._f;
		const double worst = _swarm[idxmax]._f;
		const double dy = std::fabs(best - worst);
		if (dy <= _tol) {

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
			if (m2 <= (_np - 1) * _stol * _stol) {
				converged = true;
				break;
			}
		}
	}

	// update results
	const int imin = std::min_element(_swarm.begin(), _swarm.end(),
			ds_particle::compare_fitness) - _swarm.begin();
	return {_swarm[imin]._x, _fev, converged};
}

void DifferentialSearch::genDir(int method) {
	switch (method) {
	case 1: {

		// BIJECTIVE DSA (B-DSA) (i.e., go-to-rnd DSA)
		// philosophy: evolve the superorganism (i.e.,population)
		// towards to "permuted-superorganism (i.e., random directions)
		for (int i = 0; i < _np; i++) {
			_jind[i] = i;
		}
		Random::shuffle(_jind.begin(), _jind.end());
		for (int i = 0; i < _np; ++i) {
			const int j = _jind[i];
			_swarm[i]._dir = &(_swarm[j]._x);
		}
		break;
	}
	case 2: {

		// SURJECTIVE DSA (S-DSA) (i.e., go-to-good DSA)
		// philosophy: evolve the superorganism (i.e.,population)
		// towards to "some of the random top-best" solutions
		for (int i = 0; i < _np; i++) {
			_jind[i] = i;
		}
		std::sort(_jind.begin(), _jind.end(),
				[this](const int &a, const int &b) {
					return _swarm[a]._f < _swarm[b]._f;
				});
		const int nums = static_cast<int>(std::ceil(Random::get(0., 1.) * _np));
		for (int i = 0; i < _np; i++) {
			const int j = Random::get(0, nums - 1);
			_swarm[i]._dir = &(_swarm[j]._x);
		}
		break;
	}
	case 3: {

		// ELITIST DSA #1 (E1-DSA) (i.e., go-to-best DSA)
		// philosophy: evolve the superorganism (i.e.,population)
		// towards to "one of the random top-best" solution
		for (int i = 0; i < _np; i++) {
			_jind[i] = i;
		}
		std::sort(_jind.begin(), _jind.end(),
				[this](const int &a, const int &b) {
					return _swarm[a]._f < _swarm[b]._f;
				});
		const int nums1 = std::min(
				static_cast<int>(std::ceil(Random::get(0., 1.) * _np)),
				_np - 1);
		const int ibest = _jind[nums1];
		for (int i = 0; i < _np; i++) {
			_swarm[i]._dir = &(_swarm[ibest]._x);
		}
		break;
	}
	case 4: {

		// ELITIST DSA #2 (E2-DSA) (i.e., go-to-best DSA)
		// philosophy: evolve the superorganism (i.e.,population)
		// towards to "the best" solution
		const int imin = std::min_element(_swarm.begin(), _swarm.end(),
				ds_particle::compare_fitness) - _swarm.begin();
		for (int i = 0; i < _np; i++) {
			_swarm[i]._dir = &(_swarm[imin]._x);
		}
		break;
	}
	}
}

void DifferentialSearch::genPop() {
	for (auto &pt : _swarm) {
		for (int j = 0; j < _n; j++) {
			pt._x[j] = Random::get(_lower[j], _upper[j]);
		}
		pt._f = _f._f(&(pt._x)[0]);
	}
}

void DifferentialSearch::genMap(double p1, double p2) {

	// strategy-selection of active/passive individuals
	if (Random::get(0, 1) == 0) {
		if (Random::get(0., 1.) < p1) {

			// Random-mutation #1 strategy
			for (auto &p : _swarm) {
				for (int j = 0; j < _n; j++) {
					if (Random::get(0, 1) == 0) {
						p._map[j] = 1;
					} else {
						p._map[j] = 0;
					}
				}
			}
		} else {

			// Differential-mutation strategy
			for (auto &p : _swarm) {
				const int j = Random::get(0, _n - 1);
				std::fill((p._map).begin(), (p._map).end(), 0);
				p._map[j] = 1;
			}
		}
	} else {

		// Random-mutation #2 strategy
		const int mapmax = static_cast<int>(std::ceil(p2 * _n));
		for (auto &p : _swarm) {
			std::fill((p._map).begin(), (p._map).end(), 0);
			for (int k = 0; k < mapmax; k++) {
				const int j = Random::get(0, _n - 1);
				p._map[j] = 1;
			}
		}
	}
}

void DifferentialSearch::update() {
	for (auto &p : _swarm) {
		for (int j = 0; j < _n; j++) {

			// first (standard)-method
			if (p._so[j] < _lower[j]) {
				if (Random::get(0, 1) == 0) {
					p._so[j] = Random::get(_lower[j], _upper[j]);
				} else {
					p._so[j] = _lower[j];
				}
			}
			if (p._so[j] > _upper[j]) {
				if (Random::get(0, 1) == 0) {
					p._so[j] = Random::get(_lower[j], _upper[j]);
				} else {
					p._so[j] = _upper[j];
				}
			}
		}
	}
}
