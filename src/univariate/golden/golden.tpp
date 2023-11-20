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
	const T tau = (sqrt(T(5.)) - 1.) / 2.;
	T x1 = a + (1. - tau) * (b - a);
	T x2 = a + tau * (b - a);
	T f1 = f(x1);
	T f2 = f(x2);
	int fev = 2;
	bool converged = false;

	while (fev < _mfev) {

		// check convergence
		const T mid = (a + b) / 2.;
		const T tol = T(_rtol) * fabs(mid) + T(_atol);
		if (fabs(b - a) <= tol) {
			converged = true;
			break;
		}

		// locate the minimum
		if (f1 > f2){
			a = x1;
			x1 = x2;
			f1 = f2;
			x2 = a + tau * (b - a);
			f2 = f(x2);
		} else {
			b = x2;
			x2 = x1;
			f2 = f1;
			x1 = a + (1. - tau) * (b - a);
			f1 = f(x1);
		}
		fev += 1;
	}

	const T min_x = (a + b) / 2.;
	return {min_x, fev, converged};
}
