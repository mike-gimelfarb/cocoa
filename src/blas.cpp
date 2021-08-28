/*
 Copyright (c) 2020

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
 */

#include <algorithm>
#include <cmath>
#include <numeric>

#include "blas.h"

void dxpym(int n, double *dx, int idx, double *dy, int idy) {
	if (n <= 0) {
		return;
	}

	// CODE FOR BOTH INCREMENTS EQUAL TO 1
	const int m = n % 4;
	if (m != 0) {
		for (int i = 0; i < m; i++) {
			dy[i + idy - 1] += dx[i + idx - 1];
		}
		if (n < 4) {
			return;
		}
	}
	for (int i = m; i < n; i += 4) {
		dy[i + idy - 1] += dx[i + idx - 1];
		dy[i + 1 + idy - 1] += dx[i + 1 + idx - 1];
		dy[i + 2 + idy - 1] += dx[i + 2 + idx - 1];
		dy[i + 3 + idy - 1] += dx[i + 3 + idx - 1];
	}
}

void daxpym(int n, double da, double *dx, int idx, double *dy, int idy) {
	if (n <= 0 || da == 0.) {
		return;
	}

	// CODE FOR BOTH INCREMENTS EQUAL TO 1
	const int m = n % 4;
	if (m != 0) {
		for (int i = 0; i < m; i++) {
			dy[i + idy - 1] += da * dx[i + idx - 1];
		}
		if (n < 4) {
			return;
		}
	}
	for (int i = m; i < n; i += 4) {
		dy[i + 0 + idy - 1] += da * dx[i + 0 + idx - 1];
		dy[i + 1 + idy - 1] += da * dx[i + 1 + idx - 1];
		dy[i + 2 + idy - 1] += da * dx[i + 2 + idx - 1];
		dy[i + 3 + idy - 1] += da * dx[i + 3 + idx - 1];
	}
}

void daxpy1(int n, double da, double *dx, int idx, double *dy, int idy,
		double *dz, int idz) {
	if (n <= 0) {
		return;
	}

	if (da == 0.) {

		// COPY Y INTO Z
		std::copy(dy + idy - 1, dy + idy - 1 + n, dz + idz - 1);
		return;
	}

	// CODE FOR BOTH INCREMENTS EQUAL TO 1
	const int m = n % 4;
	if (m != 0) {
		for (int i = 0; i < m; i++) {
			dz[i + idz - 1] = dy[i + idy - 1] + da * dx[i + idx - 1];
		}
		if (n < 4) {
			return;
		}
	}
	for (int i = m; i < n; i += 4) {
		dz[i + 0 + idz - 1] = dy[i + 0 + idy - 1] + da * dx[i + 0 + idx - 1];
		dz[i + 1 + idz - 1] = dy[i + 1 + idy - 1] + da * dx[i + 1 + idx - 1];
		dz[i + 2 + idz - 1] = dy[i + 2 + idy - 1] + da * dx[i + 2 + idx - 1];
		dz[i + 3 + idz - 1] = dy[i + 3 + idy - 1] + da * dx[i + 3 + idx - 1];
	}
}

void dscalm(int n, double da, double *dx, int idx) {
	if (n <= 0) {
		return;
	}

	// code for increment equal to 1
	const int m = n % 5;
	if (m != 0) {
		for (int i = 0; i < m; i++) {
			dx[i + idx - 1] *= da;
		}
		if (n < 5) {
			return;
		}
	}
	for (int i = m; i < n; i += 5) {
		dx[i + 0 + idx - 1] *= da;
		dx[i + 1 + idx - 1] *= da;
		dx[i + 2 + idx - 1] *= da;
		dx[i + 3 + idx - 1] *= da;
		dx[i + 4 + idx - 1] *= da;
	}
}

void dscal1(int n, double da, double *dx, int idx, double *dy, int idy) {
	if (n <= 0) {
		return;
	}

	// code for increment equal to 1
	const int m = n % 5;
	if (m != 0) {
		for (int i = 0; i < m; i++) {
			dy[i + idy - 1] = dx[i + idx - 1] * da;
		}
		if (n < 5) {
			return;
		}
	}
	for (int i = m; i < n; i += 5) {
		dy[i + 0 + idy - 1] = dx[i + 0 + idx - 1] * da;
		dy[i + 1 + idy - 1] = dx[i + 1 + idx - 1] * da;
		dy[i + 2 + idy - 1] = dx[i + 2 + idx - 1] * da;
		dy[i + 3 + idy - 1] = dx[i + 3 + idx - 1] * da;
		dy[i + 4 + idy - 1] = dx[i + 4 + idx - 1] * da;
	}
}

double dnrm2(int n, double *x) {
	double absxi;
	int i, ix;
	double norm, scale, ssq;
	if (n < 1) {
		norm = 0.;
	} else if (n == 1) {
		norm = std::fabs(x[0]);
	} else {
		scale = 0.;
		ssq = 1.;
		ix = 0;
		for (i = 0; i < n; i++) {
			if (x[ix] != 0.) {
				absxi = std::fabs(x[ix]);
				if (scale < absxi) {
					ssq = 1. + ssq * (scale / absxi) * (scale / absxi);
					scale = absxi;
				} else {
					ssq = ssq + (absxi / scale) * (absxi / scale);
				}
			}
			ix++;
		}
		norm = scale * std::sqrt(ssq);
	}
	return norm;
}

double ddot(int n, double *dx, int idx, double *dy, int idy) {
	return std::inner_product(dx + idx - 1, dx + idx - 1 + n, dy + idy - 1, 0.);
}
