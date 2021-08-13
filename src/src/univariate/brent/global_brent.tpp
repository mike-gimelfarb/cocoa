/*
 Original FORTRAN77 version by Richard Brent.
 FORTRAN90 version by John Burkardt.
 C++ version by Michael Gimelfarb.

 This code is distributed under the GNU LGPL license.

 ================================================================
 REFERENCES:

 [1] Brent, Richard P. Algorithms for minimization without derivatives.
 Courier Corporation, 2013.
 */

#include "../../math_utils.h"

template<typename T> GlobalBrentSearch<T>::GlobalBrentSearch(double tol,
		int mfev, double bound_on_hessian) {
	_mfev = mfev;
	_tol = tol;
	_boundhess = bound_on_hessian;
}

template<typename T> solution<T> GlobalBrentSearch<T>::optimize(univariate<T> f,
		T guess, T a, T b) {

	// constants
	const T eps = ulp<T>();
	const T M = T(_boundhess);
	const T tol = T(_tol);
	const T m2 = (16. * eps + 1.) / 2. * M;
	const T decay = T(0.9);

	// initialization
	T a0 = b, a2 = a, a3, c = b, d0 { }, d1 { }, d2 { }, p { }, q { }, qs { },
			r { }, s { }, sc = T(a - 1.), x = a0, y0 = f(b), y1 { }, y2 = f(a),
			y = y2, y3 { }, yb = y0, z0 { }, z1 { }, z2 { };
	T h = T(9.) / T(11.);
	int fev = 2;

	// first step
	int k = 3;
	if (y0 < y) {
		y = y0;
	} else {
		x = a;
	}
	if (M <= T(0.) || a >= b) {
		return {x, fev, false};
	}
	if (sc <= a || sc >= b) {
		sc = guess;
	} else {
		sc = c;
	}
	y1 = f(sc);
	fev++;
	d0 = a2 - sc;
	if (y1 < y) {
		x = sc;
		y = y1;
	}

	// main loop
	while (true) {
		d1 = a2 - a0;
		d2 = sc - a0;
		z2 = b - a2;
		z0 = y2 - y1;
		z1 = y2 - y0;
		r = d1 * d1 * z0 - d0 * d0 * z1;
		p = r;
		qs = 2. * (d0 * z1 - d1 * z0);
		q = qs;
		bool goto40 = k <= 1000000 || y >= y2;
		do {
			if (goto40) {
				const T right = z2 * m2 * r * (z2 * q - r);
				if (q * (r * (yb - y2) + z2 * q * (y2 - y + tol)) < right) {
					a3 = a2 + r / q;
					y3 = f(a3);
					fev++;
					if (y3 < y) {
						x = a3;
						y = y3;
					}
				}
			}
			k = (1611 * k) % 1048576;
			q = T(1.);
			r = (b - a) * T(1e-5) * k;
			goto40 = true;
			if (fev >= _mfev) {
				return {x, fev, false};
			}
		} while (r < z2);

		r = m2 * d0 * d1 * d2;
		s = sqrt((y2 - y + tol) / m2);
		h = (h + 1) / 2.;
		p = h * (p + 2. * r * s);
		q += (qs / 2);
		r = -(d0 + (z0 + T(2.01) * tol) / (d0 * m2)) / 2.;
		if (r >= s && d0 >= T(0.)) {
			r += a2;
		} else {
			r = a2 + s;
		}
		if (p * q <= T(0.)) {
			a3 = r;
		} else {
			a3 = a2 + p / q;
		}

		while (true) {
			if (a3 < r) {
				a3 = r;
			}
			if (a3 < b) {
				y3 = f(a3);
				fev++;
			} else {
				a3 = b;
				y3 = yb;
			}
			if (y3 < y) {
				x = a3;
				y = y3;
			}
			d0 = a3 - a2;
			if (a3 <= r) {
				break;
			} else {
				p = 2. * (y2 - y3) / (M * d0);
				const T right = (y2 - y) + (y3 - y) + 2. * tol;
				if (fabs(p) >= (1. + 9. * eps) * d0
						|| m2 * (d0 * d0 + p * p) / 2. <= right) {
					break;
				} else {
					a3 = (a2 + a3) / 2.;
					h *= decay;
					if (fev >= _mfev) {
						return {x, fev, false};
					}
				}
			}
		}
		if (a3 >= b) {
			return {x, fev, true};
		} else {
			a0 = sc;
			sc = a2;
			a2 = a3;
			y0 = y1;
			y1 = y2;
			y2 = y3;
			if (fev >= _mfev) {
				return {x, fev, false};
			}
		}
	}
}

