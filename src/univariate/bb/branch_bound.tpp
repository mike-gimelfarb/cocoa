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

#ifndef UNIVARIATE_GLOBAL_BRANCH_BOUND_TPP_
#define UNIVARIATE_GLOBAL_BRANCH_BOUND_TPP_

#include <algorithm>
#include <limits>
#include <vector>

#include "../../math_utils.h"

template<typename T> BranchBoundSearch<T>::BranchBoundSearch(double tol,
		int mfev, double K, int n) {
	_mfev = mfev;
	_tol = tol;
	_K = K;
	_n = n;
}

template<typename T> solution<T> BranchBoundSearch<T>::optimize(
		univariate<T> f, T a, T b) {

	// initialization
	const T K = T(_K);
	const T tol = T(_tol);
	std::vector < interval < T >> M;
	T xp = a;
	T fp = f(xp);
	T h = (b - a) / _n;
	int fev = 1;
	for (int i = 1; i <= _n; i++) {
		const T xi = a + h * i;
		const T fi = f(xi);
		const T mid = (xp + xi) / 2;
		const T slope = (fi - fp) / (xi - xp);
		const T xstar = std::max(xp, std::min(mid - slope / K, xi));
		T fxstar;
		if (xstar <= xp) {
			fxstar = fp;
		} else if (xstar >= xi) {
			fxstar = fi;
		} else {
			fxstar = f(xstar);
			fev++;
		}
		const T ubi = std::min(fxstar, fi);
		const T lbi = K * xstar * xstar / 2 + (slope - K * mid) * xstar
				+ K * xi * xp / 2 + (fp * xi - fi * xp) / (xi - xp);
		M.push_back( { xp, xi, fp, fi, lbi, ubi });
		xp = xi;
		fp = fi;
	}
	fev += _n;

	// compute LB
	int i_lb = std::min_element(M.begin(), M.end(), interval < T > ::compare_lb)
			- M.begin();
	T lb = M[i_lb]._lb;

	// compute UB
	int i_ub = std::min_element(M.begin(), M.end(), interval < T > ::compare_ub)
			- M.begin();
	T ub = M[i_ub]._ub;

	// iteration
	bool converged = true;
	while (ub - lb > tol && M.size() > 0) {

		// perform subdivision
		a = M[i_lb]._a;
		b = M[i_lb]._b;
		xp = a;
		fp = M[i_lb]._f_a;
		h = (b - a) / _n;
		M.erase(M.begin() + i_lb);
		for (int i = 1; i <= _n; i++) {
			const T xi = a + h * i;
			const T fi = f(xi);
			const T mid = (xp + xi) / 2;
			const T slope = (fi - fp) / (xi - xp);
			const T xstar = std::max(xp, std::min(mid - slope / K, xi));
			T fxstar;
			if (xstar <= xp) {
				fxstar = fp;
			} else if (xstar >= xi) {
				fxstar = fi;
			} else {
				fxstar = f(xstar);
				fev++;
			}
			const T ubi = std::min(fxstar, fi);
			const T lbi = K * xstar * xstar / 2 + (slope - K * mid) * xstar
					+ K * xi * xp / 2 + (fp * xi - fi * xp) / (xi - xp);
			M.push_back( { xp, xi, fp, fi, lbi, ubi });
			xp = xi;
			fp = fi;
		}
		fev += _n;

		// update UB
		i_ub = std::min_element(M.begin(), M.end(), interval < T > ::compare_ub)
				- M.begin();
		ub = M[i_ub]._ub;

		// elimination
		const int m = M.size();
		for (int i = m - 1; i >= 0; i--) {
			if (ub - M[i]._lb < tol) {
				M.erase(M.begin() + i);
			}
		}

		// selection
		i_lb = std::min_element(M.begin(), M.end(), interval < T > ::compare_lb)
				- M.begin();
		lb = M[i_lb]._lb;

		// reached budget
		if (fev > _mfev) {
			converged = false;
			break;
		}

		// bad under-estimator
		if (ub < lb && M.size() > 0) {
			converged = false;
			break;
		}
	}

	// did not converge
	if (fabs(ub - lb) > tol) {
		converged = false;
	}

	// prepare solution
	i_ub = std::min_element(M.begin(), M.end(), interval < T > ::compare_ub)
			- M.begin();
	T xopt;
	if (M[i_ub]._f_a < M[i_ub]._f_b) {
		xopt = M[i_ub]._a;
	} else {
		xopt = M[i_ub]._b;
	}
	return {xopt, fev, converged};
}

#endif /* UNIVARIATE_GLOBAL_BRANCH_BOUND_TPP_ */
