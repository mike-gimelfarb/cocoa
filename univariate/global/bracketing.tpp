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

#ifndef BRACKETING_TPP_
#define BRACKETING_TPP_

#include <algorithm>
#include <stack>

template<typename T> BracketingSearch<T>::BracketingSearch(
		UnivariateOptimizer<T> *local, double tol, int mfev, double factor) {
	_local = local;
	_tol = tol;
	_factor = factor;
	_mfev = mfev;
}

template<typename T> solutions<T> BracketingSearch<T>::optimize_all(
		univariate<T> f, T a, T b) {

	// store found minima
	std::vector<double> minima;

	// add the parent interval onto the stack
	std::stack<interval<T>> stack;
	stack.push( { a, b });

	// some constants
	const T resolution = T(_tol);
	const T fac = T(_factor);

	// main loop
	int fev = 0;
	bool converged = true;
	while (!stack.empty()) {

		interval<T> interval = stack.top();
		stack.pop();

		// interval too small
		if (fabs(interval._right - interval._left) < resolution) {
			continue;
		}

		// perform bracketing within the interval
		const T a = interval._left;
		const T b = interval._right;
		const T mid = (a + b) / 2;
		const T x0 = std::max(a, mid - resolution);
		const T x1 = std::min(b, mid + resolution);
		const bracket<T> bra = bracketSolution(f, x0, x1, _mfev - fev, fac, a,
				b);
		fev += bra._fev;

		// note that the behavior of the algorithm changes depending on whether
		// a root is successfully bracketed
		if (bra._converged) {

			// a minimum can be bracketed: now invoke the local minimizer
			const solution<T> sol = _local->optimize(f, bra._left, bra._right);
			const T x = sol._sol;
			if (sol._converged) {
				minima.push_back(x);
			}
			fev += sol._fev;

			// bisect the interval around the solution and recurse
			stack.push( { a, x - resolution });
			stack.push( { x + resolution, b });
		} else {

			// a minimum could not be bracketed: consider the two intervals outside
			if (bra._right > mid) {
				stack.push( { a, x0 });
				stack.push( { bra._right, b });
			} else {
				stack.push( { a, bra._left });
				stack.push( { x1, b });
			}
		}

		// reached the budget
		if (fev >= _mfev) {
			converged = false;
			break;
		}
	}

	// sort the locations of the minimum in ascending order
	std::sort(minima.begin(), minima.end());
	return {minima, fev, converged};
}

#endif /* BRACKETING_TPP_ */
