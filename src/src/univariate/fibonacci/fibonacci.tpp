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

#include "../../math_utils.h"

template<typename T> FibonacciSearch<T>::FibonacciSearch(double rtol,
		double atol, int mfev) {
	_mfev = mfev;
	_rtol = rtol;
	_atol = atol;
}

template<typename T> T FibonacciSearch<T>::determine_fibonacci_n(T a, T b) {
	const T one = T(1.);
	const T adjtol = T(_atol) / (b - a);
	T fib1 = one;
	T fib2 = one;
	int n = 2;
	while (one / fib2 >= adjtol) {
		const T fib3 = fib1 + fib2;
		fib1 = fib2;
		fib2 = fib3;
		n++;
	}
	return n;
}

template<typename T> solution<T> FibonacciSearch<T>::optimize(univariate<T> f,
		T guess, T a, T b) {

	// find the smallest n such that 1/F(n) < tolerance / (b - a)
	const int n = determine_fibonacci_n(a, b);

	// evaluate initial constants
	const T alpha0 = T(0.01);
	const T sqrt5 = sqrt(T(5.));
	const T c = (sqrt5 - 1.) / 2.;
	const T s = (1. - sqrt5) / (1. + sqrt5);
	T p1 = pow(s, n);
	T p2 = p1 * s;
	T alpha = c * (1. - p1) / (1. - p2);
	T x1 = a;
	T x4 = b;
	T x3 = alpha * x4 + (1. - alpha) * x1;
	T f3 = f(x3);
	int fev = 1;
	bool converged = false;

	// main loop
	for (int i = 1; i <= n - 1; ++i) {

		// introduce new point X2
		T x2;
		if (i == n - 1) {
			x2 = alpha0 * x1 + (1. - alpha0) * x3;
		} else {
			x2 = alpha * x1 + (1. - alpha) * x4;
		}
		const T f2 = f(x2);
		fev++;

		// update the interval
		if (f2 < f3) {
			x4 = x3;
			x3 = x2;
			f3 = f2;
		} else {
			x1 = x4;
			x4 = x2;
		}

		// update the step size
		const int d = n - i;
		p1 = pow(s, d);
		p2 = p1 * s;
		alpha = c * (1. - p1) / (1. - p2);

		// check tolerance
		const T mid = (x1 + x4) / 2.;
		const T tol = T(_rtol) * fabs(mid) + T(_atol);
		if (fabs(x4 - x1) <= tol) {
			converged = true;
			break;
		}

		// check budget
		if (fev >= _mfev) {
			converged = false;
			break;
		}
	}

	const T min_x = (x1 + x4) / 2.;
	return {min_x, fev, converged};
}
