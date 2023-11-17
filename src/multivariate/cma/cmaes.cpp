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

 [1] Hansen, Nikolaus, and Andreas Ostermeier. "Completely derandomized
 self-adaptation in evolution strategies." Evolutionary computation 9.2
 (2001): 159-195.

 [2] Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv
 preprint arXiv:1604.00772 (2016).
 */

#include <iostream>
#include <numeric>

#include "../../blas.h"
#include "../../math_utils.h"
#include "../../random.hpp"

#include "cmaes.h"

using Random = effolkronium::random_static;

void Cmaes::init(const multivariate_problem &f, const double *guess) {
	BaseCmaes::init(f, guess);

	// we perform an eigenvalue decomposition every O(d) iterations
	_eigenfreq = _eigenrate * _lambda / (_c1 + _cmu) / _n;
	_eigenlastev = 0;

	// Initialize dynamic (internal) strategy parameters and constants
	_diagd = std::vector<double>(_n, 1.);
	_b.resize(_n, std::vector<double>(_n, 0.));
	_c.resize(_n, std::vector<double>(_n, 0.));
	_invsqrtc.clear();
	_invsqrtc.resize(_n, std::vector<double>(_n, 0.));
	for (int d = 0; d < _n; d++) {
		_c[d][d] = _invsqrtc[d][d] = _b[d][d] = 1.;
	}

	// Initialize convergence parameters
	_flag = 0;
}

void Cmaes::samplePopulation() {
	for (int n = 0; n < _lambda; n++) {
		for (int i = 0; i < _n; i++) {
			_artmp[i] = _diagd[i] * Random::get(_Z);
		}
		for (int i = 0; i < _n; i++) {
			const double s = std::inner_product(_b[i].begin(), _b[i].end(),
					_artmp.begin(), 0.);
			_arx[n][i] = _xmean[i] + _sigma * s;
			if (_bound) {
				_arx[n][i] = std::max(_lower[i],
						std::min(_arx[n][i], _upper[i]));
			}
		}
	}
}

void Cmaes::updateDistribution() {

	// compute weighted mean into xmean
	std::copy(_xmean.begin(), _xmean.end(), _xold.begin());
	for (int i = 0; i < _n; i++) {
		double sum = 0.;
		for (int n = 0; n < _mu; n++) {
			const int j = _fitness[n]._index;
			sum += _weights[n] * _arx[j][i];
		}
		_xmean[i] = sum;
		if (_bound) {
			_xmean[i] = std::max(_lower[i], std::min(_xmean[i], _upper[i]));
		}
	}

	// Cumulation: Update evolution paths
	const double csc = std::sqrt(_cs * (2. - _cs) * _mueff);
	for (int i = 0; i < _n; i++) {
		_ps[i] *= (1. - _cs);
		for (int j = 0; j < _n; j++) {
			_ps[i] += csc * _invsqrtc[i][j] * (_xmean[j] - _xold[j]) / _sigma;
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
				+ hsig * ccc * (_xmean[i] - _xold[i]) / _sigma;
	}

	// Adapt covariance matrix C
	const double c2 = (1. - hsig) * _cc * (2. - _cc);
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {

			// old matrix plus rank-one update
			double sum = (1. - _c1 - _cmu) * _c[i][j]
					+ _c1 * (_pc[i] * _pc[j] + c2 * _c[i][j]);

			// rank mu update
			for (int k = 0; k < _mu; k++) {
				const int m = _fitness[k]._index;
				const double di = (_arx[m][i] - _xold[i]) / _sigma;
				const double dj = (_arx[m][j] - _xold[j]) / _sigma;
				sum += _cmu * _weights[k] * di * dj;
			}
			_c[i][j] = sum;
		}
	}

	// update sigma parameters
	updateSigma();

	// Decomposition of C into B*diag(D.^2)*B' (diagonalization)
	eigenDecomposition();
}

bool Cmaes::converged() {

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
		if (std::max(_pc[i], std::sqrt(_c[i][i])) * _sigma / _sigma0 >= _tol) {
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
	converged = true;
	for (int i = 0; i < _n; i++) {
		if (_xmean[i]
				!= _xmean[i] + 0.1 * _sigma * _diagd[iaxis] * _b[iaxis][i]) {
			converged = false;
			break;
		}
	}
	if (converged) {
		_flag = 8;
		return true;
	}

	// NoEffectCoor
	for (int i = 0; i < _n; i++) {
		if (_xmean[i] == _xmean[i] + 0.2 * _sigma * std::sqrt(_c[i][i])) {
			_flag = 9;
			return true;
		}
	}
	return false;
}

void Cmaes::eigenDecomposition() {

	// skip the eigenvalue-decomposition O(D^3) until condition is reached
	// this is done once every O(D) iterations making the algorithm O(D^2)
	if (_fev - _eigenlastev <= _eigenfreq) {
		return;
	}

	// enforce symmetry
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {
			_b[i][j] = _b[j][i] = _c[i][j];
		}
	}

	// eigenvalue decomposition, B==normalized eigenvectors
	_eigenlastev = _fev;
	tred2();
	tql2();

	// limit condition number of covariance matrix
	if (_diagd[0] <= 0.) {
		for (int i = 0; i < _n; i++) {
			_diagd[i] = std::max(_diagd[i], 0.);
		}
		const double shift = _diagd[_n - 1] / 1e14;
		for (int i = 0; i < _n; i++) {
			_c[i][i] += shift;
			_diagd[i] += shift;
		}
	}
	if (_diagd[_n - 1] > 1e14 * _diagd[0]) {
		const double shift = _diagd[_n - 1] / 1e14 - _diagd[0];
		for (int i = 0; i < _n; i++) {
			_c[i][i] += shift;
			_diagd[i] += shift;
		}
	}

	// take square root of eigenvalues
	for (int i = 0; i < _n; i++) {
		_diagd[i] = std::sqrt(_diagd[i]);
	}

	// invsqrtC = B * diag(D^-1) * B^T
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j <= i; j++) {
			double sum = 0.;
			for (int k = 0; k < _n; k++) {
				sum += _b[i][k] / _diagd[k] * _b[j][k];
			}
			_invsqrtc[i][j] = _invsqrtc[j][i] = sum;
		}
	}
}

void Cmaes::tred2() {

	// This is derived from the Algol procedures tred2 by
	// Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	// Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	// Fortran subroutine in EISPACK.
	std::copy(_b[_n - 1].begin(), _b[_n - 1].end(), _diagd.begin());

	// Householder reduction to tridiagonal form.
	for (int i = _n - 1; i > 0; i--) {

		// Scale to avoid under/overflow.
		double scale = 0.;
		double h = 0.;
		for (int k = 0; k < i; k++) {
			scale += std::abs(_diagd[k]);
		}
		if (scale == 0.) {
			_artmp[i] = _diagd[i - 1];
			for (int j = 0; j < i; j++) {
				_diagd[j] = _b[i - 1][j];
				_b[i][j] = _b[j][i] = 0.;
			}
		} else {

			// Generate Householder vector.
			for (int k = 0; k < i; k++) {
				_diagd[k] /= scale;
				h += _diagd[k] * _diagd[k];
			}
			double f = _diagd[i - 1];
			double g = std::sqrt(h);
			if (f > 0) {
				g = -g;
			}
			_artmp[i] = scale * g;
			h = h - f * g;
			_diagd[i - 1] = f - g;
			std::fill(_artmp.begin(), _artmp.begin() + i, 0.);

			// Apply similarity transformation to remaining columns.
			for (int j = 0; j < i; j++) {
				f = _diagd[j];
				_b[j][i] = f;
				g = _artmp[j] + _b[j][j] * f;
				for (int k = j + 1; k <= i - 1; k++) {
					g += _b[k][j] * _diagd[k];
					_artmp[k] += _b[k][j] * f;
				}
				_artmp[j] = g;
			}
			dscalm(i, 1. / h, &_artmp[0], 1);
			f = std::inner_product(_artmp.begin(), _artmp.begin() + i,
					_diagd.begin(), 0.);
			const double hh = f / (h + h);
			daxpym(i, -hh, &_diagd[0], 1, &_artmp[0], 1);
			for (int j = 0; j < i; j++) {
				f = _diagd[j];
				g = _artmp[j];
				for (int k = j; k <= i - 1; k++) {
					_b[k][j] -= (f * _artmp[k] + g * _diagd[k]);
				}
				_diagd[j] = _b[i - 1][j];
				_b[i][j] = 0.;
			}
		}
		_diagd[i] = h;
	}

	// Accumulate transformations.
	for (int i = 0; i < _n - 1; i++) {
		_b[_n - 1][i] = _b[i][i];
		_b[i][i] = 1.;
		const double h = _diagd[i + 1];
		if (h != 0.) {
			for (int k = 0; k <= i; k++) {
				_diagd[k] = _b[k][i + 1] / h;
			}
			for (int j = 0; j <= i; j++) {
				double g = 0.;
				for (int k = 0; k <= i; k++) {
					g += _b[k][i + 1] * _b[k][j];
				}
				for (int k = 0; k <= i; k++) {
					_b[k][j] -= g * _diagd[k];
				}
			}
		}
		for (int k = 0; k <= i; k++) {
			_b[k][i + 1] = 0.;
		}
	}
	std::copy(_b[_n - 1].begin(), _b[_n - 1].end(), _diagd.begin());
	std::fill(_b[_n - 1].begin(), _b[_n - 1].end(), 0.);
	_b[_n - 1][_n - 1] = 1.;
	_artmp[0] = 0.;
}

void Cmaes::tql2() {
	for (int i = 1; i < _n; i++) {
		_artmp[i - 1] = _artmp[i];
	}
	_artmp[_n - 1] = 0.;
	double f = 0., tst1 = 0.;
	double eps = std::pow(2., -52.);
	for (int l = 0; l < _n; l++) {

		// Find small subdiagonal element
		tst1 = std::max(tst1, std::abs(_diagd[l]) + std::abs(_artmp[l]));
		int m = l;
		for (m = l; m < _n; m++) {
			if (std::abs(_artmp[m]) <= eps * tst1) {
				break;
			}
		}
		if (m >= _n) {
			break;
		}

		// If m == l, d[l] is an eigenvalue, otherwise, iterate.
		if (m > l) {
			do {

				// Compute implicit shift
				double g = _diagd[l];
				double p = (_diagd[l + 1] - g) / (2. * _artmp[l]);
				double r = hypot(p, 1.);
				r = sign(r, p);
				_diagd[l] = _artmp[l] / (p + r);
				_diagd[l + 1] = _artmp[l] * (p + r);
				double dl1 = _diagd[l + 1];
				double h = g - _diagd[l];
				for (int i = l + 2; i < _n; i++) {
					_diagd[i] -= h;
				}
				f += h;

				// Implicit QL transformation.
				p = _diagd[m];
				double c = 1., c2 = c, c3 = c;
				double el1 = _artmp[l + 1];
				double s = 0., s2 = 0.;
				for (int i = m - 1; i >= l; i--) {
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * _artmp[i];
					h = c * p;
					r = hypot(p, _artmp[i]);
					_artmp[i + 1] = s * r;
					s = _artmp[i] / r;
					c = p / r;
					p = c * _diagd[i] - s * g;
					_diagd[i + 1] = h + s * (c * g + s * _diagd[i]);

					// Accumulate transformation.
					for (int k = 0; k < _n; k++) {
						h = _b[k][i + 1];
						_b[k][i + 1] = s * _b[k][i] + c * h;
						_b[k][i] = c * _b[k][i] - s * h;
					}
				}
				p = -s * s2 * c3 * el1 * _artmp[l] / dl1;
				_artmp[l] = s * p;
				_diagd[l] = c * p;

				// Check for convergence.
			} while (std::abs(_artmp[l]) > eps * tst1);
		}
		_diagd[l] += f;
		_artmp[l] = 0.;
	}

	// Sort eigenvalues and corresponding vectors.
	for (int i = 0; i < _n - 1; i++) {
		int k = i;
		double p = _diagd[i];
		for (int j = i + 1; j < _n; j++) {
			if (_diagd[j] < p) {
				k = j;
				p = _diagd[j];
			}
		}
		if (k != i) {
			_diagd[k] = _diagd[i];
			_diagd[i] = p;
			for (int j = 0; j < _n; j++) {
				p = _b[j][i];
				_b[j][i] = _b[j][k];
				_b[j][k] = p;
			}
		}
	}
}
