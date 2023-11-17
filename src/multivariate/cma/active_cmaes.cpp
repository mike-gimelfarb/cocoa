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

 [1] Hansen, Nikolaus, and Raymond Ros. "Benchmarking a weighted negative
 covariance matrix update on the BBOB-2010 noiseless testbed." Proceedings of
 the 12th annual conference companion on Genetic and evolutionary computation.
 ACM, 2010.

 [2] Jastrebski, Grahame A., and Dirk V. Arnold. "Improving evolution
 strategies through active covariance matrix adaptation." Evolutionary
 Computation, 2006. CEC 2006. IEEE Congress on. IEEE, 2006.
 */

#include <numeric>
#include <iostream>

#include "../../blas.h"

#include "active_cmaes.h"

void ActiveCmaes::init(const multivariate_problem &f, const double *guess) {
	Cmaes::init(f, guess);

	// re-define other parameters
	// note that some parameters may be re-defined in future implementations
	// as described in Hansen et al. (2010)
	_cm = 1.;
	_alphaold = 0.5;
	_cc = (4. + 0. * _mueff / _n) / (_n + 4. + 0. * 2. * _mueff / _n);
	_cs = (_mueff + 2.) / (3. + _n + _mueff);
	_c1 = _alphacov * std::min(1., _lambda / 6.)
			/ ((_n + 1.3) * (_n + 1.3) + _mueff);
	_cmu = 1. - _c1;
	_cmu = std::min(_cmu,
			_alphacov * (_mueff - 2. + 1. / _mueff)
					/ ((2. + _n) * (2. + _n) + _alphacov * _mueff / 2.));
	_cneg = (1. - _cmu) * (_alphacov / 8.) * _mueff
			/ (std::pow(_n + 2., 1.5) + 2. * _mueff);
	_damps = 1. + _cs
			+ 2. * std::max(0., std::sqrt((_mueff - 1.) / (_n + 1.)) - 1.);

	// we perform an eigenvalue decomposition every O(d) iterations
	_eigenfreq = _eigenrate * (1. / (_c1 + _cmu + _cneg)) / _n;
	_eigenlastev = 0;

	// other new storage
	_ycoeff = std::vector<double>(_mu, 0.);
}

void ActiveCmaes::updateDistribution() {

	// compute weighted mean into xmean
	std::copy(_xmean.begin(), _xmean.end(), _xold.begin());
	for (int i = 0; i < _n; i++) {
		double sum = 0.;
		for (int n = 0; n < _mu; n++) {
			const int j = _fitness[n]._index;
			sum += _weights[n] * _arx[j][i];
		}
		_xmean[i] = _xold[i] * (1. - _cm) + sum * _cm;
		if (_bound) {
			_xmean[i] = std::max(_lower[i], std::min(_xmean[i], _upper[i]));
		}
	}

	// Cumulation: Update evolution paths
	const double csc = std::sqrt(_cs * (2. - _cs) * _mueff);
	for (int i = 0; i < _n; i++) {
		_ps[i] *= (1. - _cs);
		for (int j = 0; j < _n; j++) {
			_ps[i] += csc * _invsqrtc[i][j] * (_xmean[j] - _xold[j])
					/ (_cm * _sigma);
		}
	}

	// compute hsig
	const double pslen = dnrm2(_n, &_ps[0]);
	const double denom = 1. - std::pow(1. - _cs, 2. * _fev / _lambda);
	int hsig;
	if (pslen / std::sqrt(denom) / _chi < 1.4 + 2. / (_n + 1.)) {
		hsig = 1;
	} else {
		hsig = 0;
	}

	// update pc
	const double ccc = std::sqrt(_cc * (2. - _cc) * _mueff);
	for (int i = 0; i < _n; i++) {
		_pc[i] = (1. - _cc) * _pc[i]
				+ hsig * ccc * (_xmean[i] - _xold[i]) / (_cm * _sigma);
	}

	// compute the coefficients for the vectors for the negative update
	for (int i = 0; i < _mu; i++) {
		const int mtop = _fitness[_lambda - _mu + 1 + i - 1]._index;
		const int mbot = _fitness[_lambda - i - 1]._index;
		double ssqtop = 0.;
		double ssqbot = 0.;
		for (int j = 0; j < _n; j++) {
			double termtop = 0.;
			double termbot = 0.;
			for (int l = 0; l < _n; l++) {
				termtop += _invsqrtc[j][l] * (_arx[mtop][l] - _xold[l]);
				termbot += _invsqrtc[j][l] * (_arx[mbot][l] - _xold[l]);
			}
			ssqtop += termtop * termtop;
			ssqbot += termbot * termbot;
		}
		ssqbot = std::max(ssqbot, 1e-8);
		_ycoeff[i] = ssqtop / ssqbot;
	}

	// Adapt covariance matrix C
	const double c2 = (1. - hsig) * _cc * (2. - _cc);
	const double cmu1 = _cmu + _cneg * (1. - _alphaold);
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {

			// old matrix plus rank-one update
			double sum = (1. - _c1 - _cmu + _cneg * _alphaold) * _c[i][j]
					+ _c1 * (_pc[i] * _pc[j] + c2 * _c[i][j]);

			// rank mu update
			for (int k = 0; k < _mu; k++) {
				const int m = _fitness[k]._index;
				const double di = (_arx[m][i] - _xold[i]) / _sigma;
				const double dj = (_arx[m][j] - _xold[j]) / _sigma;
				sum += cmu1 * _weights[k] * di * dj;
			}

			// active update: this is the main modification in active CMA-ES
			for (int k = 0; k < _mu; k++) {
				const int m = _fitness[_lambda - k - 1]._index;
				const double di = (_arx[m][i] - _xold[i]) / _sigma;
				const double dj = (_arx[m][j] - _xold[j]) / _sigma;
				sum -= _cneg * _weights[k] * _ycoeff[k] * di * dj;
			}
			_c[i][j] = sum;
		}
	}

	// update sigma
	updateSigma();

	// Decomposition of C into B*diag(D.^2)*B' (diagonalization)
	eigenDecomposition();
}

