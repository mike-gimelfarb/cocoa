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

#ifndef BRENT_H_
#define BRENT_H_

#include "../univariate/univariate.h"

template<typename T> class BrentSearch: public UnivariateOptimizer<T> {

protected:
	int _mfev;
	double _rtol, _atol;

public:
	BrentSearch(double rtol, double atol, int mfev);

	void iterate(T &a, T &b, T &arg, int &status, T value, T &c, T &d, T &e,
			T &fu, T &fv, T &fw, T &fx, T &midpoint, T &p, T &q, T &r, T &tol1,
			T &tol2, T &u, T &v, T &w, T &x);

	solution<T> optimize(univariate<T> f, T a, T b);
};

#include "local/brent.tpp"

#endif /* BRENT_H_ */
