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

 [1] Calvin, James. "An adaptive univariate global optimization algorithm and
 its convergence rate under the Wiener measure." Informatica 22.4 (2011):
 471-488.
 */
#include <algorithm>
#include <limits>
#include <vector>

#include "../../math_utils.h"

template<typename T> CalvinSearch<T>::CalvinSearch(double tol, int mfev,
		double lam) {
	_tol = tol;
	_mfev = mfev;
	_lam = lam;
}

template<typename T> solution<T> CalvinSearch<T>::optimize(univariate<T> f,
		T guess, T a, T b) {

	// initialize the mesh
	const double pguess = (guess - a) / (b - a);
	std::vector<T> tarr, farr;
	tarr.push_back(T(0.));
	tarr.push_back(T(pguess));
	tarr.push_back(T(1.));

	// initialize the function evaluations at the end points
	farr.push_back(f(rescale(T(0.), a, b)));
	farr.push_back(f(rescale(T(pguess), a, b)));
	farr.push_back(f(rescale(T(1.), a, b)));

	// initialize the tracking parameters
	T tau = T(0.5);
	T gtau = sqrt(-T(_lam) * tau * log(tau));
	T vmin = std::min(std::min(farr[0], farr[1]), farr[2]);

	// main loop
	int fev = 3;
	bool converged = false;
	for (int n = 2; n < _mfev; ++n) {

		// find out which interval to split
		T rhomax = std::numeric_limits < T > ::lowest();
		int imax = -1;
		for (int i = 1; i <= n; ++i) {
			const T num = tarr[i] - tarr[i - 1];
			const T den1 = farr[i - 1] - vmin + gtau;
			const T den2 = farr[i] - vmin + gtau;
			const T rho = num / (den1 * den2);
			if (rho > rhomax) {
				rhomax = rho;
				imax = i;
			}
		}

		// split the interval at i_min
		const T left = tarr[imax - 1];
		const T rght = tarr[imax];
		const T tmid = (left + rght) / 2.;
		const T fmid = f(rescale(tmid, a, b));
		tarr.insert(tarr.begin() + imax, tmid);
		farr.insert(farr.begin() + imax, fmid);
		fev++;

		// update tracking parameters
		tau = std::min(tau, tmid - left);
		tau = std::min(tau, rght - tmid);
		gtau = sqrt(-T(_lam) * tau * log(tau));
		vmin = std::min(vmin, fmid);

		// check convergence
		if (tau <= _tol) {

			// we have converged
			converged = true;
			break;
		}
	}

	// assemble final result
	auto min_it = std::min_element(farr.begin(), farr.end());
	const T min_x = rescale(tarr[std::distance(farr.begin(), min_it)], a, b);
	return {min_x, fev, converged};
}
