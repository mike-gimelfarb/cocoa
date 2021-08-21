/*
 Original FORTRAN77 version by Richard Brent.
 FORTRAN90 version by John Burkardt.
 C++ version by Michael Gimelfarb.

 This code is distributed under the GNU LGPL license.

 ================================================================
 REFERENCES:

 [1] Brent, Richard P. Algorithms for minimization without derivatives.
 Courier Corporation, 2013.

 [2] Kahaner, David, Cleve Moler, and Stephen Nash. "Numerical methods and
 software." Englewood Cliffs: Prentice Hall, 1989 (1989).
 */

#include <iostream>

#include "../../math_utils.h"

template<typename T> BrentSearch<T>::BrentSearch(int mfev, double atol,
		double rtol) {
	_mfev = mfev;
	_rtol = rtol;
	_atol = atol;
}

template<typename T> solution<T> BrentSearch<T>::optimize(
		const univariate<T> &f, T guess, T a, T b) {
	bool converged = false;
	int status = 0, fev = 0;
	T aa = a, bb = b, arg = guess, value = f(arg), c { }, d { }, e { }, fu { },
			fv { }, fw { }, fx { }, midpoint { }, p { }, q { }, r { }, tol1 { },
			tol2 { }, u { }, v { }, w { }, x { };

	// main loop 
	while (true) {
		iterate(aa, bb, arg, status, value, c, d, e, fu, fv, fw, fx, midpoint,
				p, q, r, tol1, tol2, u, v, w, x);
		if (status == 0) {
			converged = true;
			break;
		} else if (fev >= _mfev) {
			break;
		} else {
			value = f(arg);
			fev++;
		}
	}
	return {arg, fev, converged};
}

template<typename T> void BrentSearch<T>::iterate(T &a, T &b, T &arg,
		int &status, T value, T &c, T &d, T &e, T &fu, T &fv, T &fw, T &fx,
		T &midpoint, T &p, T &q, T &r, T &tol1, T &tol2, T &u, T &v, T &w,
		T &x) {

	if (status == 0) {

		// STATUS (INPUT) = 0, startup.
		if (b <= a) {
			std::cerr << "Warning [Brent]: bounding interval is ill defined."
					<< std::endl;
			status = -1;
			return;
		}
		c = (T(3.) - sqrt(T(5.))) / 2.;
		v = a + c * (b - a);
		w = x = v;
		e = T(0.);
		status = 1;
		arg = x;
		return;
	} else if (status == 1) {

		// STATUS (INPUT) = 1, return with initial function value of FX.
		fx = value;
		fv = fw = fx;
	} else if (2 <= status) {

		// STATUS (INPUT) = 2 or more, update the data.
		fu = value;
		if (fu <= fx) {
			if (x <= u) {
				a = x;
			} else {
				b = x;
			}
			v = w;
			fv = fw;
			w = x;
			fw = fx;
			x = u;
			fx = fu;
		} else {
			if (u < x) {
				a = u;
			} else {
				b = u;
			}
			if (fu <= fw || w == x) {
				v = w;
				fv = fw;
				w = u;
				fw = fu;
			} else if (fu <= fv || v == x || v == w) {
				v = u;
				fv = fu;
			}
		}
	}

	// Take the next step.
	midpoint = (a + b) / 2.;
	tol1 = T(_rtol) * fabs(x) + T(_atol) / 3.;
	tol2 = tol1 * 2.;

	// If the stopping criterion is satisfied, we can exit.
	if (fabs(x - midpoint) <= (tol2 - (b - a) / 2.)) {
		status = 0;
		return;
	}

	// Is golden-section necessary?
	if (fabs(e) <= tol1) {
		if (midpoint <= x) {
			e = a - x;
		} else {
			e = b - x;
		}
		d = c * e;
	} else {

		// Consider fitting a parabola.
		r = (x - w) * (fx - fv);
		q = (x - v) * (fx - fw);
		p = (x - v) * q - (x - w) * r;
		q = 2. * (q - r);
		if (q > T(0.)) {
			p = -p;
		}
		q = fabs(q);
		r = e;
		e = d;

		// Choose a golden-section step if the parabola is not advised.
		if (fabs(q * r / 2.) <= fabs(p) || p <= q * (a - x)
				|| q * (b - x) <= p) {
			if (midpoint <= x) {
				e = a - x;
			} else {
				e = b - x;
			}
			d = c * e;
		} else {

			// Choose a parabolic interpolation step.
			d = p / q;
			u = x + d;
			if (u - a < tol2) {
				d = sign(tol1, midpoint - x);
			}
			if (b - u < tol2) {
				d = sign(tol1, midpoint - x);
			}
		}
	}

	// F must not be evaluated too close to X.
	if (tol1 <= fabs(d)) {
		u = x + d;
	}
	if (fabs(d) < tol1) {
		u = x + sign(tol1, d);
	}

	// Request value of F(U).
	arg = u;
	status++;
}
