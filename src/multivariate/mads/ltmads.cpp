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
#include <iostream>
#include <stdexcept>

#include "../../blas.h"
#include "../../math_utils.h"
#include "../../random.hpp"

#include "ltmads.h"

using Random = effolkronium::random_static;

LTMADSMesh::LTMADSMesh(bool maximal) {
	_maximal = maximal;
}

void LTMADSMesh::init(MADS *parent) {
	const int n = parent->_n;
	_L.clear();
	_L.resize(n - 1, std::vector<long long int>(n - 1, 0L));
	_B.clear();
	_B.resize(n, std::vector<long long int>(n, 0L));
	_D.clear();
	if (_maximal) {
		_D.resize(2 * n, std::vector<long long int>(n, 0L));
	} else {
		_D.resize(n + 1, std::vector<long long int>(n, 0L));
	}
	_Nm1 = std::vector<int>(n - 1);
	_bl = std::vector<long long int>(n);
	_lc = 0;
	_ihat = -1;

	// initialize mesh size parameters
	_lk = 0;
	_deltam = _deltap = 1.;
}

void LTMADSMesh::update(MADS *parent) {

	// construct the direction b(l) and index ihat
	const int l = std::max(0, _lk);
	generatebl(parent, l);

	// basis construction in R^(n-1)
	const int n = parent->_n;
	const long long int twopowl = 1L << l;
	for (int i = 0; i < n - 1; i++) {
		if (Random::get(0., 1.) < 0.5) {
			_L[i][i] = twopowl;
		} else {
			_L[i][i] = -twopowl;
		}
		for (int j = i + 1; j < n - 1; j++) {
			_L[i][j] = Random::get(-twopowl + 1L, twopowl - 1L);
		}
	}

	// permutation of the lines of L, and completion to a basis in R^n
	if (_ihat < 0) {
		throw std::invalid_argument(
				"Error [LTMADS]: b(l) vector is not set correctly.");
	}
	int k = 0;
	for (int i = 0; i < n; i++) {
		if (i != _ihat) {
			_Nm1[k] = i;
			k++;
		}
	}
	Random::shuffle(_Nm1);
	for (int i = 0; i < n - 1; i++) {
		for (int j = 0; j < n - 1; j++) {
			_B[j][_Nm1[i]] = _L[j][i];
		}
	}
	for (int j = 0; j < n - 1; j++) {
		_B[j][_ihat] = 0L;
	}
	std::copy(_bl.begin(), _bl.end(), _B[n - 1].begin());

	// permutation of the columns of B
	Random::shuffle(_B.begin(), _B.end());

	// completion to a positive basis
	if (_maximal) {
		for (int i = 0; i < n; i++) {
			std::copy(_B[i].begin(), _B[i].end(), _D[i].begin());
			for (int j = 0; j < n; j++) {
				_D[i + n][j] = -_B[i][j];
			}
		}
	} else {
		for (int i = 0; i < n; i++) {
			std::copy(_B[i].begin(), _B[i].end(), _D[i].begin());
			_D[n][i] = 0L;
			for (int j = 0; j < n; j++) {
				_D[n][i] -= _B[j][i];
			}
		}
	}
}

void LTMADSMesh::updateParameters(MADS *parent) {
	if (parent->_searchsuccess || !parent->_minframe) {
		_lk--;
	} else {
		_lk++;
	}
	_deltam = std::min(1., std::pow(4., -_lk));
	_deltap = std::pow(2., -_lk);
}

void LTMADSMesh::computeTrial(MADS *parent, int idx, double *x0, double *out) {
	const auto &d = _D[idx];
	for (int i = 0; i < parent->_n; i++) {
		out[i] = x0[i] + _deltam * d[i];
	}
}

bool LTMADSMesh::converged(MADS *parent) {

	// minimal frame with small poll size parameter
	if (parent->_minframe && _deltap < parent->_tol) {
		return true;
	}

	// mesh size too small
	if (_deltam < parent->_tol) {
		return true;
	}

	// check for overflow
	if (_lk > 63) {
		std::cerr
				<< "Warning [LTMADS]: small l value detected. Algorithm may overflow."
				<< std::endl;
	}

	return false;
}

void LTMADSMesh::generatebl(MADS *parent, const int l) {

	// VERIFICATION IF b(l) WAS ALREADY CREATED
	if (_lc > l) {
		return;
	}
	_lc++;

	// INDEX OF ENTRY WITH LARGEST COMPONENT
	_ihat = Random::get(0, parent->_n - 1);

	// CONSTRUCTION OF b(l)
	const long long int twopowl = 1L << l;
	for (int i = 0; i < parent->_n; i++) {
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
