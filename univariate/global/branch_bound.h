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

 [1] Aaid, Djamel, Amel Noui, and Mohand Ouanes. "New technique for solving
 univariate global optimization." Archivum Mathematicum 53.1 (2017): 19-33.

 [2] Le Thi, Hoai An, and Mohand Ouanes. "Convex quadratic underestimation
 and Branch and Bound for univariate global optimization with one nonconvex
 constraint." Rairo-Operations Research 40.3 (2006): 285-302.
 */

#ifndef BRANCH_BOUND_H_
#define BRANCH_BOUND_H_

#include "../univariate/univariate.h"

template<typename T> struct interval {

	T _a, _b, _f_a, _f_b, _lb, _ub;

	static bool compare_lb(const interval<T> &x, const interval<T> &y) {
		return x._lb < y._lb;
	}

	static bool compare_ub(const interval<T> &x, const interval<T> &y) {
		return x._ub < y._ub;
	}
};

template<typename T> class BranchBoundSearch: public UnivariateOptimizer<T> {

protected:
	int _mfev, _n;
	double _tol, _K;

public:
	BranchBoundSearch(double tol, int mfev, double K, int n);

	solution<T> optimize(univariate<T> f, T a, T b);
};

#include "global/branch_bound.tpp"

#endif /* BRANCH_BOUND_H_ */
