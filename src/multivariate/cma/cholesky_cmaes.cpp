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

 [1] Krause, Oswin, Dídac Rodríguez Arbonès, and Christian Igel. "CMA-ES with
 optimal covariance update and storage complexity." Advances in Neural
 Information Processing Systems. 2016.
 */

#include <cmath>
#include <numeric>

#include "cholesky_cmaes.h"
#include "../../random.hpp"

using Random = effolkronium::random_static;

void CholeskyCmaes::init(multivariate f, const int n, double *guess,
		double *lower, double *upper) {
	BaseCmaes::init(f, n, guess, lower, upper);

	// Initialize dynamic (internal) strategy parameters and constants
	_dmean = std::vector<double>(_n, 0.);
	_a.clear();
	_a.resize(_n, std::vector<double>(_n, 0.));
	_mattmp.clear();
	_mattmp.resize(_n, std::vector<double>(_n, 0.));
	for (int d = 0; d < _n; d++) {
		_a[d][d] = 1.;
	}
}

void CholeskyCmaes::updateDistribution() {

	// cache the old xmean
	std::copy(_xmean.begin(), _xmean.end(), _xold.begin());

	// compute weighted mean into xmean
	for (int i = 0; i < _n; i++) {
		double sum = 0.;
		for (int n = 0; n < _mu; n++) {
			const int j = _fitness[n]->_index;
			sum += _weights[n] * _arx[j][i];
		}
		_xmean[i] = sum;
		_dmean[i] = (_xmean[i] - _xold[i]) / _sigma;
	}

	// update pc
	const double ccc = std::sqrt(_cc * (2. - _cc) * _mueff);
	for (int i = 0; i < _n; i++) {
		_pc[i] = (1. - _cc) * _pc[i] + ccc * _dmean[i];
	}

	// apply formula (2) to A(t)
	const double acoeff = std::sqrt(1. - _c1 - _cmu);
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {
			_mattmp[i][j] = acoeff * _a[i][j];
		}
	}

	// perform the rank 1 updates to A(t+1)
	std::copy(_pc.begin(), _pc.end(), _artmp.begin());
	rank1Update (_c1);
	for (int i = 0; i < _mu; i++) {
		for (int j = 0; j < _n; j++) {
			_artmp[j] = (_arx[i][j] - _xmean[j]) / _sigma;
		}
		rank1Update(_cmu * _weights[i]);
	}

	// do back-substitution to compute A^-1 * (m(t+1)-m(t))
	std::copy(_dmean.begin(), _dmean.end(), _artmp.begin());
	for (int i = 0; i < _n; i++) {
		_artmp[i] -= std::inner_product(_a[i].begin(), _a[i].begin() + i,
				_artmp.begin(), 0.);
		_artmp[i] /= _a[i][i];
	}

	// compute the vector pc
	const double csc = std::sqrt(_cs * (2. - _cs) * _mueff);
	for (int i = 0; i < _n; i++) {
		_ps[i] = (1. - _cs) * _ps[i] + csc * _artmp[i];
	}

	// update A(t) to A(t+1)
	for (int i = 0; i < _n; i++) {
		std::copy(_mattmp[i].begin(), _mattmp[i].end(), _a[i].begin());
	}

	// update the step size
	updateSigma();
}

void CholeskyCmaes::samplePopulation() {
	for (int n = 0; n < _lambda; n++) {
		for (int i = 0; i < _n; i++) {
			_artmp[i] = Random::get(_Z);
		}
		for (int i = 0; i < _n; i++) {
			const double sum = std::inner_product(_a[i].begin(), _a[i].end(),
					_artmp.begin(), 0.);
			_arx[n][i] = _xmean[i] + _sigma * sum;
			if (_bound) {
				_arx[n][i] = std::max(_lower[i],
						std::min(_arx[n][i], _upper[i]));
			}
		}
	}
}

bool CholeskyCmaes::converged() {

	// check convergence in fitness difference between best and worst points
	const double y0 = _ybw[0];
	const double y3 = _ybw[3];
	if (std::abs(y0 - y3) > _tol) {
		return false;
	}

	// compute standard deviation of swarm radiuses
	int count = 0;
	double mean = 0.;
	double m2 = 0.;
	for (auto &pt : _arx) {
		const double x = std::sqrt(
				std::inner_product(pt.begin(), pt.end(), pt.begin(), 0.));
		count++;
		const double delta = x - mean;
		mean += delta / count;
		const double delta2 = x - mean;
		m2 += delta * delta2;
	}

	// test convergence in standard deviation
	return m2 <= (_lambda - 1) * _stol * _stol;
}

void CholeskyCmaes::rank1Update(double beta) {
	double b = 1.;
	for (int j = 0; j < _n; j++) {
		const double ajj = _mattmp[j][j];
		const double alfaj = _artmp[j];
		const double gam = ajj * ajj * b + beta * alfaj * alfaj;
		const double a1jj = _mattmp[j][j] = std::sqrt(gam / b);
		for (int k = j + 1; k < _n; k++) {
			_artmp[k] -= (alfaj / ajj) * _mattmp[k][j];
			_mattmp[k][j] = (a1jj / ajj) * _mattmp[k][j]
					+ (a1jj * beta * alfaj) / gam * _artmp[k];
		}
		b += beta * (alfaj / ajj) * (alfaj / ajj);
	}
}
