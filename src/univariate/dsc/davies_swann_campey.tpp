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

 [1] Antoniou, Andreas, and Wu-Sheng Lu. Practical optimization: algorithms
 and engineering applications. Springer Science & Business Media, 2007.
 */

template<typename T> DaviesSwannCampey<T>::DaviesSwannCampey(int mfev,
		double tol, double decay) {
	_mfev = mfev;
	_tol = tol;
	_decay = decay;
}

template<typename T> solution<T> DaviesSwannCampey<T>::optimize(univariate<T> f,
		T a, T b) {

	// initialize constants
	const T one = T(1.0);
	const T delta1 = (b - a) / 2;
	const T guess = (a + b) / 2;

	// step 1: initialization
	T x0 = guess;
	T delta = delta1;
	int fev = 0;

	// main loop of DSC algorithm
	while (true) {

		// step 2
		const T xm1 = x0 - delta;
		const T xp1 = x0 + delta;
		const T f0 = f(x0);
		const T fp1 = f(xp1);
		fev += 2;

		// step 3
		T p;
		if (f0 > fp1) {
			p = one;
		} else {
			const T fm1 = f(xm1);
			fev++;
			if (fm1 < f0) {
				p = -one;
			} else {

				// step 7: update the position of the minimum
				const T num = delta * (fm1 - fp1);
				const T den = 2 * (fm1 - 2 * f0 + fp1);
				x0 += (num / den);
				x0 = std::max(x0, a);
				x0 = std::min(x0, b);
				if (delta <= T(_tol)) {
					return {x0, fev, true};
				} else {
					delta *= _decay;
					continue;
				}
			}
		}

		// step 4
		T twonm1 = one;
		T fnm2 = f(xm1);
		T xnm1 = x0;
		T fnm1 = f0;
		T xn, fn;
		fev++;
		while (true) {
			xn = xnm1 + twonm1 * p * delta;
			fn = f(xn);
			fev++;
			if (fn > fnm1) {
				break;
			} else {
				fnm2 = fnm1;
				xnm1 = xn;
				fnm1 = fn;
				twonm1 *= 2;
			}
		}

		// step 5
		const T twonm2 = twonm1 / 2;
		const T xm = xnm1 + twonm2 * p * delta;
		const T fm = f(xm);
		fev++;

		// step 6: update the position of the minimum
		if (fm >= fnm1) {
			const T num = twonm2 * p * delta * (fnm2 - fm);
			const T den = 2 * (fnm2 - 2 * fnm1 + fm);
			x0 = xnm1 + (num / den);
		} else {
			const T num = twonm2 * p * delta * (fnm1 - fn);
			const T den = 2 * (fnm1 - 2 * fm + fn);
			x0 = xm + (num / den);
		}
		x0 = std::max(x0, a);
		x0 = std::min(x0, b);

		// convergence test
		if (twonm2 * delta <= T(_tol)) {
			return {x0, fev, true};
		}
		if (fev >= _mfev) {
			return {x0, fev, false};
		}

		// adjust the step size
		delta *= _decay;
	}
}
