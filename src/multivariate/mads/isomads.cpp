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

 [1] Audet, Charles, Sébastien Le Digabel, and Christophe Tribes. "Dynamic scaling
 in the mesh adaptive direct search algorithm for blackbox optimization."
 Optimization and Engineering 17.2 (2016): 333-358.
 */

#include <algorithm>
#include <cmath>

#include "../../blas.h"
#include "../../random.hpp"

#include "isomads.h"

using Random = effolkronium::random_static;

IsoMADSMesh::IsoMADSMesh(double scale, double beta, bool aniso) {
	_scale = scale;
	_beta0 = beta;
	_aniso = aniso;
}

void IsoMADSMesh::init(MADS *parent) {

	// define some constants
	const int n = parent->_n;
	_beta = std::vector<double>(n, _beta0);

	// basis working memory
	_u = std::vector<double>(n);
	_D.clear();
	_D.resize(2 * n, std::vector<long long int>(n));

	// initialize mesh and poll size parameters by dynamic scaling
	_r = std::vector<int>(n, 0);
	_deltap0 = std::vector<double>(n);
	_deltam0 = std::vector<double>(n);
	for (int i = 0; i < n; i++) {
		_deltap0[i] = _scale * (parent->_upper[i] - parent->_lower[i]);
		_deltam0[i] = _deltap0[i] / std::sqrt(1. * n);
	}
	_deltap = std::vector<double>(_deltap0);
	_deltam = std::vector<double>(_deltam0);
}

void IsoMADSMesh::update(MADS *parent) {

	// generate point in the unit sphere
	const int n = parent->_n;
	while (true) {
		for (int i = 0; i < n; i++) {
			_u[i] = Random::get(_Z);
		}
		const double unorm2 = std::sqrt(
				std::inner_product(_u.begin(), _u.end(), _u.begin(), 0.));
		dscalm(n, 1. / unorm2, &_u[0], 1);
		if (unorm2 > 1e-8) {
			break;
		}
	}

	// build the poll set
	for (int l = 0; l < n; l++) {
		for (int j = 0; j < n; j++) {
			double h;
			if (l == j) {
				h = 1. - 2. * _u[l] * _u[j];
			} else {
				h = -2. * _u[l] * _u[j];
			}
			const double arg = _deltap[j] * h / _deltam[j];
			_D[l][j] = static_cast<long long int>(std::round(arg));
			_D[l + n][j] = -_D[l][j];
		}
	}
}

void IsoMADSMesh::updateParameters(MADS *parent) {

	// update r_j
	const int rbar = *std::max_element(_r.begin(), _r.end());
	if (!parent->_searchsuccess && parent->_minframe) {

		// unsuccessful iteration
		for (int i = 0; i < parent->_n; i++) {
			_r[i]--;
		}
	} else {

		// successful iteration
		if (_aniso) {
			const auto &d = parent->_succdir;
			double dinf = 0.;
			for (int i = 0; i < parent->_n; i++) {
				dinf = std::max(dinf, std::fabs(d[i]));
			}
			for (int i = 0; i < parent->_n; i++) {
				if (_r[i] >= -2 || std::fabs(d[i]) > dinf / parent->_n) {
					_r[i]++;
				}
			}
		} else {
			for (int i = 0; i < parent->_n; i++) {
				_r[i]++;
			}
		}
	}

	// correct r_{j+1} that are too small
	for (int i = 0; i < parent->_n; i++) {
		if (_r[i] < -2 && _r[i] < 2 * rbar) {
			_r[i]++;
		}
	}

	// update parameters
	for (int i = 0; i < parent->_n; i++) {
		_deltap[i] = _deltap0[i] * std::pow(_beta[i], _r[i]);
		_deltam[i] = _deltam0[i] * std::pow(_beta[i], _r[i] - std::fabs(_r[i]));
	}
}

void IsoMADSMesh::computeTrial(MADS *parent, int idx, double *x0, double *out) {
	const auto &d = _D[idx];
	for (int i = 0; i < parent->_n; i++) {
		out[i] = x0[i] + _deltam[i] * d[i];
	}
}

bool IsoMADSMesh::converged(MADS *parent) {
	const double deltammin = *std::min_element(_deltam.begin(), _deltam.end());
	return (deltammin < parent->_tol);
}
