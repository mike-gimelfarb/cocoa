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

 [1] Audet, Charles & Dennis, J.. (2006). Mesh Adaptive Direct Search Algorithms
 for Constrained Optimization. SIAM Journal on Optimization. 17. 188-217.
 10.1137/060671267.
 */

#include <cmath>

#include "../../blas.h"
#include "../../math_utils.h"
#include "../../random.hpp"

#include "ltmads.h"

using Random = effolkronium::random_static;

LTMADS::LTMADS(int mfev, double tol, bool maxbasis, bool search) { // @suppress("Class members should be properly initialized")
	_mfev = mfev;
	_tol = tol;
	_maximal = maxbasis;
	_search = search;
}

void LTMADS::init(multivariate f, constraints c, const int n, double *guess,
		double *lower, double *upper) {

	// define problem
	_f = f;
	_omega = c;
	_n = n;
	_guess = std::vector<double>(guess, guess + n);
	_lower = std::vector<double>(lower, lower + n);
	_upper = std::vector<double>(upper, upper + n);

	// define parameters
	_fev = 0;
	_cev = 0;
	_deltampow = 0;
	_lc = 0;
	_x = std::vector<double>(_guess);
	_fx = evaluateBarrier(&_x[0]);
	_xold = std::vector<double>(_guess);
	_fxold = _fx;
	_deltapoll = 1.;

	// working memory
	_temp = std::vector<double>(_n);
	_L.clear();
	_L.resize(_n - 1, std::vector<long>(_n - 1));
	_B.clear();
	_B.resize(_n, std::vector<long>(_n));
	_D.clear();
	if (_maximal) {
		_fmesh = std::vector<double>(2 * _n);
		_D.resize(2 * _n, std::vector<long>(_n));
	} else {
		_fmesh = std::vector<double>(_n + 1);
		_D.resize(_n + 1, std::vector<long>(_n));
	}
	_N = std::vector<int>(_n);
	for (int i = 0; i < _n; i++) {
		_N[i] = i;
	}
	_Nm1 = std::vector<int>(_n - 1);
	_bl = std::vector<long>(_n);
}

void LTMADS::iterate() {

	// search
	bool improved = false;
	if (_search) {
		improved = search();
	}

	// poll
	if (!improved) {
		poll();
	}
}

constrained_solution LTMADS::optimize(multivariate f, constraints c,
		const int n, double *guess, double *lower, double *upper) {
	init(f, c, n, guess, lower, upper);
	bool converged = false;
	while (true) {
		iterate();

		// reached budget
		if (_fev >= _mfev) {
			converged = false;
			break;
		}

		// minimal frame with small poll size parameter
		if (_minframe && _deltapoll < _tol) {
			converged = true;
			break;
		}

		// mesh size too small
		const double deltam = std::pow(4., -_deltampow);
		if (deltam < ulp<double>() || _deltapoll < _tol) {
			converged = true;
			break;
		}
	}
	return {_x, _fev, _cev, converged};
}

bool LTMADS::search() {

	// simulate dynamic ordering according to the paper
	// when the previous iteration succeeds in finding an improved mesh point
	if (_fx < _fxold) {

		// try the point in the direction of the last descent direction
		for (int i = 0; i < _n; i++) {
			const double dx = _x[i] - _xold[i];
			_temp[i] = _xold[i] + 4. * dx;
		}

		// evaluate fitness
		const double fxse = evaluateBarrier(&_temp[0]);
		if (fxse < _fx) {

			// an improved mesh point is found
			std::copy(_x.begin(), _x.end(), _xold.begin());
			_fxold = _fx;
			std::copy(_temp.begin(), _temp.end(), _x.begin());
			_fx = fxse;

			// update the mesh parameters
			_deltampow--;
			return true;
		}
	}

	// otherwise, continue with poll
	return false;
}

void LTMADS::poll() {

	// generate the basis
	generateBasis();

	// evaluate f_omega on the frame P_k
	// the frame computed by generateBasis() is guaranteed to satisfy
	// the assumptions of Definition 2.2
	int np;
	if (_maximal) {
		np = 2 * _n;
	} else {
		np = _n + 1;
	}
	const double deltam = std::min(1., std::pow(4., -_deltampow));
	for (int i = 0; i < np; i++) {
		for (int j = 0; j < _n; j++) {
			_temp[j] = _x[j] + deltam * _D[i][j];
		}
		_fmesh[i] = evaluateBarrier(&_temp[0]);
	}

	// check if the poll step generates an improved mesh point
	const int imin = std::min_element(_fmesh.begin(), _fmesh.end())
			- _fmesh.begin();
	if (_fmesh[imin] < _fx) {

		// an improved mesh point is found
		std::copy(_x.begin(), _x.end(), _xold.begin());
		_fxold = _fx;
		for (int j = 0; j < _n; j++) {
			_x[j] += deltam * _D[imin][j];
		}
		_fx = _fmesh[imin];

		// update the mesh parameters
		_deltampow--;
		_deltapoll = std::pow(2., -_deltampow);
		_minframe = false;
	} else {

		// an improved mesh point is not found
		std::copy(_x.begin(), _x.end(), _xold.begin());
		_fxold = _fx;

		// update the mesh parameters
		_deltampow++;
		_deltapoll = std::pow(2., -_deltampow) * _n;
		_minframe = true;
	}
}

void LTMADS::generateBasis() {

	// construct the direction b(l) and index ihat
	const int l = std::max(0, _deltampow);
	generateBl(l);

	// basis construction in R^(n-1)
	const long twopowl = 1L << l;
	for (int i = 0; i < _n - 1; i++) {
		if (Random::get(0., 1.) < 0.5) {
			_L[i][i] = twopowl;
		} else {
			_L[i][i] = -twopowl;
		}
		for (int j = i + 1; j < _n - 1; j++) {
			_L[i][j] = Random::get(-twopowl + 1L, twopowl - 1L);
		}
	}

	// permutation of the lines of L, and completion to a basis in R^n
	int k = 0;
	for (int i = 0; i < _n; i++) {
		if (i != _ihat) {
			_Nm1[k] = i;
			k++;
		}
	}
	Random::shuffle(_Nm1);
	for (int i = 0; i < _n - 1; i++) {
		for (int j = 0; j < _n - 1; j++) {
			_B[j][_Nm1[i]] = _L[j][i];
		}
	}
	for (int j = 0; j < _n - 1; j++) {
		_B[j][_ihat] = 0;
	}
	std::copy(_bl.begin(), _bl.end(), _B[_n - 1].begin());

	// permutation of the columns of B
	Random::shuffle(_B.begin(), _B.end());

	// completion to a positive basis
	if (_maximal) {
		for (int i = 0; i < _n; i++) {
			std::copy(_B[i].begin(), _B[i].end(), _D[i].begin());
			for (int j = 0; j < _n; j++) {
				_D[i + _n][j] = -_B[i][j];
			}
		}
	} else {
		for (int i = 0; i < _n; i++) {
			std::copy(_B[i].begin(), _B[i].end(), _D[i].begin());
			_D[_n][i] = 0;
			for (int j = 0; j < _n; j++) {
				_D[_n][i] -= _B[j][i];
			}
		}
	}
}

void LTMADS::generateBl(const int l) {

	// if lc > l, then exit this procedure with the existing b(l)
	if (_lc > l) {
		return;
	}

	// otherwise continue onto the next step
	_lc++;
	const int idx = Random::get(0, _n - 1);
	_ihat = _N[idx];
	const long twopowl = 1L << l;
	for (int i = 0; i < _n; i++) {
		if (i == _ihat) {
			if (Random::get(0., 1.) < 0.5) {
				_bl[i] = twopowl;
			} else {
				_bl[i] = -twopowl;
			}
		} else {
			_bl[i] = Random::get(-twopowl + 1L, twopowl - 1L);
		}
	}
}

double LTMADS::evaluateBarrier(const double *x) {
	_cev++;
	if (_omega(x)) {
		_fev++;
		return _f(x);
	} else {
		return INF;
	}
}

