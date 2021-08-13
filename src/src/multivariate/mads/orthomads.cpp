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

 [1] Abramson, Mark A., et al. "OrthoMADS: A deterministic MADS instance with
 orthogonal directions." SIAM Journal on Optimization 20.2 (2009): 948-966.
 */

#include <algorithm>
#include <cmath>
#include <numeric>

#include "orthomads.h"

void OrthoMADS::init(multivariate f, constraints c, const int n, double *guess,
		double *lower, double *upper) {
	LTMADS::init(f, c, n, guess, lower, upper);

	// additional parameters
	_primes = std::vector<long>(_n);
	_t0 = fillPrimes();
	_tk = _t0;
	_nhalton = std::vector<long>(_n, 0L);
	_dhalton = std::vector<long>(_n, 1L);
	_q = std::vector<long>(_n);
	_uhalton = std::vector<double>(_n);
	_tkmax = _tk;
	_lmax = _deltampow;

	// generate initial Halton sequence to p_n
	for (int it = 1; it <= _tk; it++) {
		nextHalton();
	}
}

void OrthoMADS::generateBasis() {

	// update t_k
	const int tkold = _tk;
	if (_deltampow >= _lmax) {
		_lmax = _deltampow;
		_tk = _lmax + _t0;
	} else {
		_tk = 1 + _tkmax;
	}
	_tkmax = std::max(_tkmax, _tk);

	// generate Halton sequence
	for (int it = 1; it <= _tk - tkold; it++) {
		nextHalton();
	}

	// compute alpha by solving the optimization problem
	// 				max_alpha || q_t(alpha) ||
	//				s.t. || q_t(alpha) || <= 2 ^ (|l| / 2)
	computeAlpha();

	// compute ||q||^2, which is an integer
	long qnorm2 = 0L;
	for (int i = 0; i < _n; i++) {
		qnorm2 += _q[i] * _q[i];
	}

	// compute D = [H -H]
	for (int i = 0; i < _n; i++) {
		for (int j = 0; j < _n; j++) {
			if (i == j) {
				_D[i][j] = qnorm2 - 2L * _q[i] * _q[j];
			} else {
				_D[i][j] = -2L * _q[i] * _q[j];
			}
			_D[i + _n][j] = -_D[i][j];
		}
	}
}

void OrthoMADS::computeAlpha() {

	// prepare u first
	double umin = INF;
	double umax = -INF;
	for (int i = 0; i < _n; i++) {
		_temp[i] = 2. * _uhalton[i] - 1.;
		umin = std::min(umin, std::fabs(_temp[i]));
		umax = std::max(umax, std::fabs(_temp[i]));
	}
	const double unorm = std::sqrt(
			std::inner_product(_temp.begin(), _temp.end(), _temp.begin(), 0.));

	// compute minimum value of alpha according to Lemma 3.2
	const double ubl = std::pow(2., std::fabs(_deltampow) / 2.);
	const double beta = ubl / std::sqrt(1. * _n) - 0.5;

	// compute conservative starting index j
	const int jmin = std::max(0, (int) (beta * umin / unorm - 0.5));
	double bestqnorm = -INF;
	double bestalpha = unorm / (2. * umax);

	// loop over i and j, where (i, j) is a plateau on which ||q_t(alpha)|| is constant
	for (int j = jmin; j < 999; j++) {
		bool jfeas = false;
		for (int i = 0; i < _n; i++) {

			// alpha lies on a plateau
			const double alpha = (j + 0.5) * unorm / std::fabs(_temp[i]);

			// check whether alpha satisfies the minimum value according to Lemma 3.2.
			if (alpha >= beta) {

				// compute ||q_t(alpha)||
				double qnorm = 0.;
				for (int k = 0; k < _n; k++) {
					const int qk = (int) std::round(alpha * _temp[k] / unorm);
					qnorm += qk * qk;
				}
				qnorm = std::sqrt(qnorm);

				// check if alpha improves the objective, and if this alpha is feasible
				if (qnorm <= ubl) {
					if (qnorm > bestqnorm) {
						bestqnorm = qnorm;
						bestalpha = alpha;
					}
					jfeas = true;
				}
			}
		}

		// if ||q_t(alpha)|| > ubl for all i = 1, 2... n for the current j
		// then this condition is true also for j + 1, since ||q_t(alpha)|| is monotone
		// non-decreasing in alpha, and alpha is increasing in j
		if (!jfeas) {
			break;
		}
	}

	// compute q vector
	for (int i = 0; i < _n; i++) {
		_q[i] = (long) std::round(bestalpha * _temp[i] / unorm);
	}
}

void OrthoMADS::nextHalton() {
	for (int i = 0; i < _n; i++) {
		const long x = _dhalton[i] - _nhalton[i];
		if (x == 1L) {
			_nhalton[i] = 1L;
			_dhalton[i] *= _primes[i];
		} else {
			long y = _dhalton[i] / _primes[i];
			while (y >= x) {
				y /= _primes[i];
			}
			_nhalton[i] = (_primes[i] + 1L) * y - x;
		}
		_uhalton[i] = (1. * _nhalton[i]) / _dhalton[i];
	}
}

int OrthoMADS::fillPrimes() {
	int m = 2;
	int nprime = 0;
	_primes.clear();
	while (true) {
		if (isPrime(m)) {
			nprime++;
			_primes.push_back(m);
			if (nprime == _n) {
				return m;
			}
		}
		m++;
	}
}

bool OrthoMADS::isPrime(const int n) {
	if (n % 2 == 0) {
		return n == 2;
	}
	if (n % 3 == 0) {
		return n == 3;
	}
	int s = 4;
	int m = (int) std::sqrt(1. * n) + 1;
	for (int i = 5; i < m; s = 6 - s, i += s) {
		if (n % i == 0) {
			return false;
		}
	}
	return true;
}
