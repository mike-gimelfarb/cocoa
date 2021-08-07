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

 [1] Hager, W. W. "A derivative-based bracketing scheme for univariate
 minimization and the conjugate gradient method." Computers & Mathematics with
 Applications 18.9 (1989): 779-795.
 */

#ifndef UNIVARIATE_DERIV_CUBIC_TPP_
#define UNIVARIATE_DERIV_CUBIC_TPP_

#include "../../math_utils.h"

template<typename T> CubicSearch<T>::CubicSearch(double rtol, double atol,
		int mfev) {
	_mfev = mfev;
	_rtol = rtol;
	_atol = atol;
}

template<typename T> T CubicSearch<T>::step(T a, T b, T c, T tau) {
	const T y = std::min(a, b);
	const T z = std::max(a, b);
	if (y + tau <= c && c <= z - tau) {
		return c;
	} else if (c > (a + b) / 2) {
		return z - tau;
	} else {
		return y + tau;
	}
}

template<typename T> T CubicSearch<T>::cubic(T a, T fa, T da, T b, T fb, T db) {
	const T delta = b - a;
	if (std::fabs(delta) <= ulp<T>()) {
		return a - 1;
	}
	const T v = da + db - 3 * (fb - fa) / delta;
	T w;
	if (v * v - da * db > T(0)) {
		w = sgn(delta) * std::sqrt(v * v - da * db);
	} else {
		w = T(0.0);
	}
	return b - delta * (db + w - v) / (db - da + 2 * w);
}

template<typename T> void CubicSearch<T>::update(T a, T fa, T dfa, T b, T c,
		T fc, T dfc, T &o1, T &o2) {
	T a1, b1;
	if (fc > fa) {
		a1 = a;
		b1 = c;
	} else if (fc < fa) {
		if (dfc * (a - c) <= T(0)) {
			a1 = c;
			b1 = a;
		} else {
			a1 = c;
			b1 = b;
		}
	} else if (dfc * (a - c) < T(0)) {
		a1 = c;
		b1 = a;
	} else if (dfa * (b - a) < T(0)) {
		a1 = a;
		b1 = c;
	} else {
		a1 = c;
		b1 = b;
	}
	o1 = a1;
	o2 = b1;
}

template<typename T> solution_for_differentiable<T> CubicSearch<T>::optimize(
		univariate<T> f, univariate<T> df, T a, T b) {

	// first convert the guess to a bracket
	T el, c, fc, dfc;
	T fa = f(a);
	T dfa = df(a);
	int fev = 1;

	// main loop
	while (true) {

		// step 1 check convergence
		T mid = (a + b) / 2;
		T xtol = T(_atol) + T(_rtol) * fabs(mid);
		if (fabs(b - a) <= xtol) {
			return {mid, fev, fev, true};
		}

		// check the budget
		if (fev >= _mfev) {
			return {mid, fev, fev, false};
		}

		// interpolation step
		el = 2 * fabs(b - a);
		T gamma = cubic(a, fa, dfa, b, f(b), df(b));
		fev++;
		c = step(a, b, gamma, T(_atol));
		fc = f(c);
		dfc = f(c);
		fev++;
		update(a, fa, dfa, b, c, fc, dfc, a, b);
		fa = f(a);
		dfa = df(a);
		fev++;

		// begin inner optimization loop
		while (true) {

			// step 2 check convergence
			mid = (a + b) / 2;
			xtol = T(_atol) + T(_rtol) * fabs(mid);
			if (fabs(b - a) <= xtol) {
				return {mid, fev, fev, true};
			}

			// check budget
			if (fev >= _mfev) {
				return {mid, fev, fev, false};
			}

			el /= 2;
			if (fabs(c - a) > el || (dfc - dfa) / (c - a) <= T(0)) {
				break;
			} else {

				// step 4
				gamma = cubic(c, fc, dfc, a, fa, dfa);
				if (gamma < a || gamma > b) {
					break;
				} else {

					// interpolation step
					c = step(a, b, gamma, T(_atol));
					fc = f(c);
					dfc = df(c);
					fev++;
					update(a, fa, dfa, b, c, fc, dfc, a, b);
					fa = f(a);
					dfa = df(a);
					fev++;
				}
			}
		}

		// step 5 bisection step
		c = (a + b) / 2;
		fc = f(c);
		dfc = df(c);
		fev++;
		update(a, fa, dfa, b, c, fc, dfc, a, b);
		fa = f(a);
		dfa = df(a);
		fev++;
	}
}

#endif /* UNIVARIATE_DERIV_CUBIC_TPP_ */
