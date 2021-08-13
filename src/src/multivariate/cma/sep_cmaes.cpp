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

 [1] Ros, Raymond, and Nikolaus Hansen. "A simple modification in CMA-ES
 achieving linear time and space complexity." International Conference on
 Parallel Problem Solving from Nature. Springer, Berlin, Heidelberg, 2008.
 */

#include <cmath>
#include <numeric>

#include "../../random.hpp"

#include "sep_cmaes.h"

using Random = effolkronium::random_static;

void SepCmaes::init(multivariate f, const int n, double *guess, double *lower,
		double *upper) {
	BaseCmaes::init(f, n, guess, lower, upper);

	// Strategy parameter setting: Adaptation
	// we slightly modify the parameters given in other implementations of CMAES
	_cc = 4. / (_n + 4.);
	_cs = (_mueff + 2.) / (3. + _n + _mueff);
	_damps = 1. + _cs
			+ 2. * std::max(0., std::sqrt((_mueff - 1.) / (_n + 1.)) - 1.);

	// additional parameter ccov
	_ccov = 2. / ((_n + std::sqrt(2.)) * (_n + std::sqrt(2.)) * _mueff);
	_ccov += std::min(1., (2. * _mueff - 1.) / ((_n + 2.) * (_n + 2.) + _mueff))
			* (1. - 1. / _mueff);

	// apply the empirical learning rate adjustment for separable functions
	// as in Hansen et. al (2008)
	if (_adjustlr) {
		_ccov *= ((_n + 2.) / 3.);
	}

	// Initialize dynamic (internal) strategy parameters and constants
	_diagd = std::vector<double>(_n, 1.);
	_c = std::vector<double>(_n, 1.);

	// initialize convergence parameters
	_flag = 0;
}

void SepCmaes::samplePopulation() {
	for (int n = 0; n < _lambda; n++) {
		for (int i = 0; i < _n; i++) {
			_arx[n][i] = _xmean[i] + _sigma * _diagd[i] * Random::get(_Z);
			if (_bound) {
				_arx[n][i] = std::max(_lower[i],
						std::min(_arx[n][i], _upper[i]));
			}
		}
	}
}

void SepCmaes::updateDistribution() {

	// compute weighted mean into xmean
	std::copy(_xmean.begin(), _xmean.end(), _xold.begin());
	for (int i = 0; i < _n; i++) {
		double sum = 0.;
		for (int n = 0; n < _mu; n++) {
			const int j = _fitness[n]._index;
			sum += _weights[n] * _arx[j][i];
		}
		_xmean[i] = sum;
	}

	// Cumulation: Update evolution paths
	const double csc = std::sqrt(_cs * (2. - _cs) * _mueff);
	for (int i = 0; i < _n; i++) {
		_ps[i] *= (1. - _cs);
		_ps[i] += csc * _c[i] * (_xmean[i] - _xold[i]) / _sigma;
	}

	// compute hsig
	const double pslen = std::sqrt(
			std::inner_product(_ps.begin(), _ps.end(), _ps.begin(), 0.));
	const double denom = 1. - std::pow(1. - _cs, 2. * _fev / _lambda);
	const int hsig =
			pslen / std::sqrt(denom) / _chi < 1.4 + 2. / (_n + 1.) ? 1 : 0;

	// update pc
	const double ccc = std::sqrt(_cc * (2. - _cc) * _mueff);
	for (int i = 0; i < _n; i++) {
		_pc[i] = (1. - _cc) * _pc[i]
				+ hsig * ccc * (_xmean[i] - _xold[i]) / _sigma;
	}

	// Adapt covariance matrix C
	for (int i = 0; i < _n; i++) {

		// old matrix plus rank-one update
		double sum = (1. - _ccov) * _c[i] + (_ccov / _mueff) * _pc[i] * _pc[i];

		// rank mu update
		for (int k = 0; k < _mu; k++) {
			const int m = _fitness[k]._index;
			const double di = (_arx[m][i] - _xold[i]) / _sigma;
			sum += _ccov * (1. - 1. / _mueff) * _weights[k] * di * di;
		}
		_c[i] = sum;
		_diagd[i] = std::sqrt(_c[i]);
	}

	// Adapt step size sigma
	updateSigma();
}

bool SepCmaes::converged() {

	// we use the original Hansen convergence test but modified for the
	// diagonal representation of the covariance matrix
	// MaxIter
	if (_it >= _mit) {
		_flag = 1;
		return true;
	}

	// TolHistFun
	if (_it >= _hlen && _fworst - _fbest < _tol) {
		_flag = 2;
		return true;
	}

	// EqualFunVals
	if (_best._len >= _n && _kth._len >= _n) {
		int countEq = 0;
		for (int i = 0; i < _n; i++) {
			if (_best.get(i) == _kth.get(i)) {
				countEq++;
				if (3 * countEq >= _n) {
					_flag = 3;
					return true;
				}
			}
		}
	}

	// TolX
	bool converged = true;
	for (int i = 0; i < _n; i++) {
		if (std::max(_pc[i], _diagd[i]) * _sigma / _sigma0 >= _tol) {
			converged = false;
			break;
		}
	}
	if (converged) {
		_flag = 4;
		return true;
	}

	// TolUpSigma
	if (_sigma / _sigma0 > 1.0e20 * _diagd[_n - 1]) {
		_flag = 5;
		return true;
	}

	// ConditionCov
	if (_diagd[_n - 1] > 1.0e7 * _diagd[0]) {
		_flag = 7;
		return true;
	}

	// NoEffectAxis
	const int iaxis = _n - 1 - ((_it - 1) % _n);
	if (_xmean[iaxis] == _xmean[iaxis] + 0.1 * _sigma * _diagd[iaxis]) {
		_flag = 8;
		return true;
	}

	// NoEffectCoor
	for (int i = 0; i < _n; i++) {
		if (_xmean[i] == _xmean[i] + 0.2 * _sigma * _diagd[i]) {
			_flag = 9;
			return true;
		}
	}
	return false;
}
