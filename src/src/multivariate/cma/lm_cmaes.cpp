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

 [1] Loshchilov, Ilya. "LM-CMA: An alternative to L-BFGS for large-scale black
 box optimization." Evolutionary computation 25.1 (2017): 143-171.

 [2] Loshchilov, Ilya. "A computationally efficient limited memory CMA-ES for
 large scale optimization." Proceedings of the 2014 Annual Conference on
 Genetic and Evolutionary Computation. ACM, 2014.
 */

#include <numeric>

#include "../../blas.h"
#include "../../random.hpp"

#include "lm_cmaes.h"

using Random = effolkronium::random_static;

void LmCmaes::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {

	// call super method
	BaseCmaes::init(f, n, guess, lower, upper);

	// adjust the learning parameters for LM-CMA-ES in Loshchilov (2015)
	if (_adaptmemory) {
		_memsize = large_size(_n);
	}
	_memlen = 0;
	if (_new) {
		_nsteps = _n;
		_t = std::max(1, (int) std::log(1. * _n));
		_cc = 0.5 / std::sqrt(1. * _n);
	} else {
		_nsteps = _memsize;
		_t = 1;
		_cc = 1. / _memsize;
	}
	_cs = 0.3;
	_c1 = 0.1 / std::log(_n + 1.);
	_cmu = NAN;
	_ccc = std::sqrt(_cc * (2. - _cc) * _mueff);
	_damps = 1.;
	_sqrt1mc1 = std::sqrt(1. - _c1);
	_zstar = 0.25;
	_s = 0.;

	// initialize additional memory
	_jarr = std::vector<int>(_memsize);
	_larr = std::vector<int>(_memsize);
	std::fill(_jarr.begin(), _jarr.end(), 0);
	std::fill(_larr.begin(), _larr.end(), 0);
	_b = std::vector<double>(_memsize, 0.);
	_d = std::vector<double>(_memsize, 0.);
	_az = std::vector<double>(_n, 0.);
	_fp = std::vector<double>(_lambda, 0.);
	_pcmat.resize(_memsize, std::vector<double>(_n, 0.));
	_vmat.resize(_memsize, std::vector<double>(_n, 0.));

	// initialize pooled ranking
	_mixed.clear();
	for (int n = 0; n < (_lambda << 1); n++) {
		const auto &index = cmaes_index { 0, 0. };
		_mixed.push_back(std::move(index));
	}
}

void LmCmaes::samplePopulation() {
	int sign = 1;
	for (int n = 0; n < _lambda; n++) {
		if (sign == 1) {

			// sample a candidate vector
			if (_samplemode == 0) {

				// sample from a Gaussian distribution
				for (int i = 0; i < _n; i++) {
					_artmp[i] = _az[i] = Random::get(_Z);
				}
			} else {

				// sample from a Rademacher distribution
				for (int i = 0; i < _n; i++) {
					_artmp[i] = _az[i] = Random::get(0., 1.) < 0.5 ? 1. : -1.;
				}
			}

			// perform Cholesky factor vector updates using algorithm 3 in Loshchilov (2015)
			int i0;
			if (_new) {
				i0 = selectSubset(_memlen, n);
			} else {
				i0 = 0;
			}
			for (int i = i0; i < _memlen; i++) {
				const int j = _jarr[i];
				const double dot = _b[j]
						* std::inner_product(_vmat[j].begin(), _vmat[j].end(),
								_artmp.begin(), 0.);
				dscalm(_n, _sqrt1mc1, &_az[0], 1);
				daxpym(_n, dot, &(_pcmat[j])[0], 1, &_az[0], 1);
			}
		}
		daxpy1(_n, sign * _sigma, &_az[0], 1, &_xmean[0], 1, &(_arx[n])[0], 1);
		sign = -sign;
	}
}

void LmCmaes::updateDistribution() {

	// compute weighted mean into xmean
	std::copy(_xmean.begin(), _xmean.end(), _xold.begin());
	std::fill(_xmean.begin(), _xmean.end(), 0.);
	for (int n = 0; n < _mu; n++) {
		const int i = _fitness[n]._index;
		daxpym(_n, _weights[n], &(_arx[i])[0], 1, &_xmean[0], 1);
	}

	// Cumulation: Update evolution paths
	for (int i = 0; i < _n; i++) {
		_pc[i] = (1. - _cc) * _pc[i] + _ccc * (_xmean[i] - _xold[i]) / _sigma;
	}

	if (_it % _t == 0) {

		// select the direction vectors
		const int imin = updateSet();
		if (_memlen < _memsize) {
			_memlen++;
		}

		// copy cumulation path vector into matrix
		int jcur = _jarr[_memlen - 1];
		std::copy(_pc.begin(), _pc.end(), _pcmat[jcur].begin());

		// recompute v vectors
		for (int i = imin; i < _memlen; i++) {

			// this part is adapted from the code by Loshchilov
			jcur = _jarr[i];
			std::copy(_pcmat[jcur].begin(), _pcmat[jcur].end(), _artmp.begin());
			ainvz(i);
			std::copy(_artmp.begin(), _artmp.end(), _vmat[jcur].begin());

			// compute b and d vectors
			const double vnrm2 = std::inner_product(_artmp.begin(),
					_artmp.end(), _artmp.begin(), 0.);
			const double sqrtc1 = std::sqrt(1. + (_c1 / (1. - _c1)) * vnrm2);
			_b[jcur] = (_sqrt1mc1 / vnrm2) * (sqrtc1 - 1.);
			_d[jcur] = (1. / (_sqrt1mc1 * vnrm2)) * (1. - 1. / sqrtc1);
		}
	}

	// update sigma parameters
	updateSigma();
}

bool LmCmaes::converged() {

	// MaxIter
	if (_it >= _mit) {
		return true;
	}

	// SigmaTooSmall
	if (_sigma < _stolmin) {
		return true;
	}

	// TolHistFun
	if (_it >= _hlen && _fworst - _fbest < _tol) {
		return true;
	}

	// EqualFunVals
	if (_best._len >= _n && _kth._len >= _n) {
		int countEq = 0;
		for (int i = 0; i < _n; i++) {
			if (_best.get(i) == _kth.get(i)) {
				countEq++;
				if (3 * countEq >= _n) {
					return true;
				}
			}
		}
	}
	return false;
}

void LmCmaes::evaluateAndSortPopulation() {

	// cache the previous fitness
	if (_it > 0) {
		for (int n = 0; n < _lambda; n++) {
			_fp[n] = _fitness[n]._value;
		}
	}

	// now perform fitness evaluation
	BaseCmaes::evaluateAndSortPopulation();
}

void LmCmaes::updateSigma() {
	if (_it == 0) {
		return;
	}

	// combine the members from the current and previous populations and sort
	for (int n = 0; n < _lambda; n++) {
		_mixed[n]._index = n;
		_mixed[n]._value = _fp[n];
		_mixed[n + _lambda]._index = n + _lambda;
		_mixed[n + _lambda]._value = _fitness[n]._value;
	}
	std::sort(_mixed.begin(), _mixed.end(), cmaes_index::compare_fitness);

	// compute normalized success measure
	double zpsr = 0.;
	for (int n = 0; n < (_lambda << 1); n++) {
		const double f = (1. * n) / _lambda;
		if (_mixed[n]._index < _lambda) {
			zpsr += f;
		} else {
			zpsr -= f;
		}
	}
	zpsr /= (1. * _lambda);
	zpsr -= _zstar;

	// update sigma
	_s = (1. - _cs) * _s + _cs * zpsr;
	_sigma *= std::exp(_s / _damps);
}

void LmCmaes::ainvz(int jlen) {

	// this is algorithm 4 in Loshchilov (2015)
	const double c = 1. / _sqrt1mc1;
	for (int i = 0; i < jlen; i++) {
		const int idx = _jarr[i];
		const double dot = _d[idx]
				* std::inner_product(_vmat[idx].begin(), _vmat[idx].end(),
						_artmp.begin(), 0.);
		dscalm(_n, c, &_artmp[0], 1);
		daxpym(_n, -dot, &(_vmat[idx])[0], 1, &_artmp[0], 1);
	}
}

int LmCmaes::updateSet() {

	// this is algorithm 5 in Loshchilov (2015)
	const int it = _it / _t;
	int imin = 1;
	if (it < _memsize) {
		_jarr[it] = it;
	} else if (_memsize > 1) {
		int iminval = _larr[_jarr[1]] - _larr[_jarr[0]];
		for (int i = 1; i < _memsize - 1; i++) {
			const int val = _larr[_jarr[i + 1]] - _larr[_jarr[i]];
			if (val < iminval) {
				iminval = val;
				imin = i + 1;
			}
		}
		if (iminval >= _nsteps) {
			imin = 0;
		}
		const int jtmp = _jarr[imin];
		for (int i = imin; i < _memsize - 1; i++) {
			_jarr[i] = _jarr[i + 1];
		}
		_jarr[_memsize - 1] = jtmp;
	}
	const int jcur = _jarr[std::min(it, _memsize - 1)];
	_larr[jcur] = _it;
	return imin == 1 ? 0 : imin;
}

int LmCmaes::selectSubset(int m, int k) {

	// this is algorithm 6 in Loshchilov (2015)
	if (m <= 1) {
		return 0;
	}
	int msigma = 4;
	if (k == 0) {
		msigma *= 10;
	}
	int mstar = (int) (msigma * std::abs(Random::get(_Z)));
	mstar = std::min(mstar, m);
	mstar = m - mstar;
	return mstar;
}
