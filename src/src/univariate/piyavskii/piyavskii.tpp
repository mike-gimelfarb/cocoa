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

 [1] Piyavskii, S. A. "An algorithm for finding the absolute extremum of a
 function." USSR Computational Mathematics and Mathematical Physics 12.4
 (1972): 57-67.

 [2] Lera, Daniela, and Yaroslav D. Sergeyev. "Acceleration of univariate
 global optimization algorithms working with Lipschitz functions and Lipschitz
 first derivatives." SIAM Journal on Optimization 23.1 (2013): 508-529.
 */
#include <algorithm>
#include <limits>
#include <vector>

#include "../../math_utils.h"

template<typename T> PiyavskiiSearch<T>::PiyavskiiSearch(double tol, int mfev,
		double r, double xi) {
	_mfev = mfev;
	_tol = tol;
	_r = r;
	_xi = xi;
}

template<typename T> solution<T> PiyavskiiSearch<T>::optimize(univariate<T> f,
		T guess, T a, T b) {

	// first two trials
	std::vector<T> xlist, zlist, llist;
	xlist.push_back(a);
	xlist.push_back(b);
	zlist.push_back(f(a));
	zlist.push_back(f(b));
	llist.assign(_mfev, T(0.));

	// main loop of adaptive Piyavskii algorithm
	int k = 2;
	bool converged = false;
	while (k < _mfev) {

		// compute the lipschitz constants
		T xmax = T(0.);
		T hmax = T(0.);
		for (int i = 0; i <= k - 2; i++) {
			const T xnew = xlist[i + 1] - xlist[i];
			const T zdif = zlist[i + 1] - zlist[i];
			const T hnew = fabs(zdif) / xnew;
			xmax = std::max(xmax, xnew);
			hmax = std::max(hmax, hnew);
		}
		for (int i = 0; i <= k - 2; i++) {
			const int ilo = std::max(i - 1, 0);
			const int ihi = std::min(i + 1, k - 2);
			T lambda = T(0.);
			for (int j = ilo; j <= ihi; j++) {
				const T zdif = zlist[j + 1] - zlist[j];
				const T xdif = xlist[j + 1] - xlist[j];
				const T hnew = fabs(zdif) / xdif;
				lambda = std::max(lambda, hnew);
			}
			const T xdif = xlist[i + 1] - xlist[i];
			const T gamma = hmax * xdif / xmax;
			llist[i] = _r * std::max(_xi, std::max(lambda, gamma));
		}

		// find an interval where the next trial will be executed
		T rmin = std::numeric_limits < T > ::max();
		int t = -1;
		for (int i = 0; i <= k - 2; i++) {
			const T zmid = (zlist[i + 1] + zlist[i]) / 2.;
			const T xdif = (xlist[i + 1] - xlist[i]) / 2.;
			const T rnew = zmid - llist[i] * xdif;
			if (rnew < rmin) {
				t = i;
				rmin = rnew;
			}
		}
		const T xleft = xlist[t];
		const T xright = xlist[t + 1];
		const T zleft = zlist[t];
		const T zright = zlist[t + 1];
		const T lip = llist[t];

		// execute the next trial point
		if (xright - xleft > T(_tol)) {

			// compute the next trial point
			const T xmid = (xright + xleft) / 2.;
			const T zdif = (zleft - zright) / 2.;
			const T xtry = xmid + (zdif / lip);
			const T ztry = f(xtry);

			// insert the next trial point into memory
			auto x_ins = std::upper_bound(xlist.begin(), xlist.end(), xtry);
			auto z_ins = zlist.begin() + std::distance(xlist.begin(), x_ins);
			xlist.insert(x_ins, xtry);
			zlist.insert(z_ins, ztry);
			k++;
		} else {

			// we have converged
			converged = true;
			break;
		}
	}

	// return result
	auto min_it = std::min_element(zlist.begin(), zlist.end());
	const T min_x = xlist[std::distance(zlist.begin(), min_it)];
	return {min_x, k, converged};
}

