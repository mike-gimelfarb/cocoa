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
 */

#include <cmath>

template<typename T> GoldenSectionSearch<T>::GoldenSectionSearch(int mfev,
		double atol, double rtol) {
	_mfev = mfev;
	_rtol = rtol;
	_atol = atol;
}

template<typename T> solution<T> GoldenSectionSearch<T>::optimize(
		const univariate<T> &f, T guess, T a, T b) {

	// initialization
	const T gold = (sqrt(T(5.)) + 1.) / 2.;
	T a1 = a;
	T b1 = b;
	T c = b1 - (b1 - a1) / gold;
	T d = a1 + (b1 - a1) / gold;
	int fev = 0;
	bool converged = false;

	while (fev < _mfev) {

		// bisect
		const T mid = (c + d) / 2.;

		// check convergence
		const T tol = T(_rtol) * fabs(mid) + T(_atol);
		if (fabs(c - d) <= tol) {
			converged = true;
			break;
		}

		// evaluate at new points
		const T fc = f(c);
		const T fd = f(d);
		fev += 2;

		// update interval
		if (fc < fd) {
			b1 = d;
		} else {
			a1 = c;
		}
		const T del = (b1 - a1) / gold;
		c = b1 - del;
		d = a1 + del;
	}

	const T min_x = (c + d) / 2.;
	return {min_x, fev, converged};
}
