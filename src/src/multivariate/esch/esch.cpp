/*
 Copyright (c) 2008-2013 Carlos Henrique da Silva Santos

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 Translated to C++11 by Mike Gimelfarb
 */

#define _USE_MATH_DEFINES
#include <algorithm>
#include <math.h>

#include "../../random.hpp"

#include "esch.h"

using Random = effolkronium::random_static;

EschSearch::EschSearch(int mfev, int np, int no) { // @suppress("Class members should be properly initialized")
	_mfev = mfev;
	_np = np;
	_no = no;
}

void EschSearch::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {

	// set parameters
	_n = n;
	_f = f;
	_lower = std::vector<double>(lower, lower + _n);
	_upper = std::vector<double>(upper, upper + _n);
	_fev = 0;

	/*********************************
	 * controlling the population size
	 *********************************/
	_parents = std::vector<std::shared_ptr<esch_particle>>();
	_offspring = std::vector<std::shared_ptr<esch_particle>>();
	_total = std::vector<std::shared_ptr<esch_particle>>();
	for (int id = 0; id < _np; id++) {
		std::vector<double> _xpa(_n);
		_parents.push_back(std::make_shared<esch_particle>(esch_particle { _xpa,
				0. }));
		_total.push_back(std::move(std::make_shared<esch_particle>()));
	}
	for (int id = 0; id < _no; id++) {
		std::vector<double> _xof(_n);
		_offspring.push_back(std::make_shared<esch_particle>(esch_particle {
				_xof, 0. }));
		_total.push_back(std::move(std::make_shared<esch_particle>()));
	}

	// From here the population is initialized
	// main vector of parameters to randcauchy
	_v0 = 4;
	_v3 = 0;
	_v4 = 1;
	_v5 = 10;
	_v6 = 1;
	_v7 = 0;

	/**************************************
	 * Initializing parents population
	 **************************************/
	for (int id = 0; id < _np; id++) {
		for (int i = 0; i < _n; i++) {
			_v1 = _lower[i];
			_v2 = _upper[i];
			_v7 += 1;
			_parents[id]->_x[i] = sampleCauchy();
		}
	}
	std::copy(guess, guess + n, _parents[0]->_x.begin());

	/**************************************
	 * Initializing offsprings population
	 **************************************/
	for (int id = 0; id < _no; id++) {
		for (int i = 0; i < _n; i++) {
			_v1 = _lower[i];
			_v2 = _upper[i];
			_v7 += 1;
			_offspring[id]->_x[i] = sampleCauchy();
		}
	}

	/**************************************
	 * Parents fitness evaluation
	 **************************************/
	for (int id = 0; id < _np; id++) {
		_parents[id]->_f = _f(&(_parents[id]->_x[0]));
		_total[id]->_f = _parents[id]->_f;
	}
	_fev = _np;
}

void EschSearch::iterate() {

	/**************************************
	 * Crossover
	 **************************************/
	for (int id = 0; id < _no; id++) {
		const int parent1 = Random::get(0, _np - 1);
		const int parent2 = Random::get(0, _np - 1);
		const int crosspoint = Random::get(0, _n - 1);
		for (int i = 0; i < crosspoint; i++) {
			_offspring[id]->_x[i] = _parents[parent1]->_x[i];
		}
		for (int i = crosspoint; i < _n; i++) {
			_offspring[id]->_x[i] = _parents[parent2]->_x[i];
		}
	}

	/**************************************
	 * Gaussian Mutation
	 **************************************/
	int totalmutation = (int) (_no * _n * 0.1);
	if (totalmutation < 1) {
		totalmutation = 1;
	}
	for (int contmutation = 0; contmutation < totalmutation; contmutation++) {
		const int idoffmutation = Random::get(0, _no - 1);
		const int paramoffmutation = Random::get(0, _n - 1);
		_v1 = _lower[paramoffmutation];
		_v2 = _upper[paramoffmutation];
		_v7 += contmutation;
		_offspring[idoffmutation]->_x[paramoffmutation] = sampleCauchy();
	}

	/**************************************
	 * Offsprings fitness evaluation
	 **************************************/
	for (int id = 0; id < _no; id++) {
		_offspring[id]->_f = _f(&(_offspring[id]->_x[0]));
		_total[id + _np]->_f = _offspring[id]->_f;
	}
	_fev += _no;

	/**************************************
	 * Individual selection
	 **************************************/
	// all the individuals are copied to one vector to easily identify best
	// solutions
	std::copy(_parents.begin(), _parents.end(), _total.begin());
	std::copy(_offspring.begin(), _offspring.end(), _total.begin() + _np);
	std::sort(_total.begin(), _total.end(), esch_particle::compare_fitness);

	// copy after sorting:
	std::copy(_total.begin(), _total.begin() + _np, _parents.begin());
	std::copy(_total.begin() + _np, _total.end(), _offspring.begin());
}

multivariate_solution EschSearch::optimize(multivariate f, const int n, double *guess,
		double *lower, double *upper) {

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
	return {_parents[0]->_x, _fev, false};
}

double EschSearch::sampleCauchy() {
	double mi = _v3;
	double band = _v5;
	double limit_inf = mi - band / 2;
	double limit_sup = mi + band / 2;
	double cauchy_mit;
	do {
		double na_unif = Random::get(0., 1.);
		cauchy_mit = _v4 * std::tan((na_unif - 0.5) * M_PI) + mi;
	} while (cauchy_mit < limit_inf || cauchy_mit > limit_sup);

	if (cauchy_mit < 0.) {
		cauchy_mit = -cauchy_mit;
	} else {
		cauchy_mit += band / 2;
	}
	double valor = cauchy_mit / band;
	valor = _v1 + (_v2 - _v1) * valor;
	return valor;
}
