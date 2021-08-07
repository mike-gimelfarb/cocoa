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

#ifndef UNIVARIATE_DERIV_SECANT_TPP_
#define UNIVARIATE_DERIV_SECANT_TPP_

#include "../../math_utils.h"

template<typename T> SecantSearch<T>::SecantSearch(double rtol, double atol,
		int mfev) {
	_mfev = mfev;
	_rtol = rtol;
	_atol = atol;
}

template<typename T> solution_for_differentiable<T> SecantSearch<T>::optimize(
		univariate<T> f, univariate<T> df, T a, T b) {

	// generate two points
	double dfb = df(b);
	double x0 = a + (b - a) / 3;
	double df0 = df(x0);
	double x1 = a + T(2.0) * (b - a) / 3;
	double df1 = df(x1);
	int dfev = 3;
	bool secant = false;

	// main loop of secant method for f'
	while (true) {

		// bisect the interval
		const T mid = a + (b - a) / 2;
		if (fabs(df1) <= ulp<T>()) {

			// first-order condition is satisfied
			return {x1, 0, dfev, true};
		}

		// estimate the second derivative
		T x2;
		bool secant1;
		const T d2f = (df1 - df0) / (x1 - x0);
		if (fabs(d2f) <= ulp<T>()) {

			// f" estimate is zero, use bisection instead
			x2 = mid;
			secant1 = false;
		} else {

			// attempt a secant step
			x2 = x1 - df1 / d2f;

			// accept or reject secant step
			if (x2 <= a || x2 >= b) {
				x2 = mid;
				secant1 = false;
			} else {
				secant1 = true;
			}
		}

		// test for budget
		if (dfev >= _mfev) {
			return {x1, 0, dfev, false};
		}

		// test sufficient reduction in the size of the bracket
		T xtol = T(_atol) + T(_rtol) * f(mid);
		const T df2 = df(x2);
		dfev++;
		if (fabs(b - a) <= xtol) {
			return {x2, 0, dfev, true};
		}

		// test assuming secant method was used
		if (secant1 && secant) {
			xtol = T(_atol) + T(_rtol) * fabs(x2);
			if (fabs(x2 - x1) <= xtol && fabs(df2) <= T(_atol)) {
				return {x2, 0, dfev, true};
			}
		}

		// update the bracket
		x0 = x1;
		x1 = x2;
		df0 = df1;
		df1 = df2;
		secant = secant1;
		if (df1 * dfb < T(0)) {
			a = x1;
		} else {
			b = x1;
			dfb = df1;
		}
	}
}

#endif /* UNIVARIATE_DERIV_SECANT_TPP_ */
