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

#ifndef GLOBAL_BRENT_H_
#define GLOBAL_BRENT_H_

#include "../univariate/univariate.h"

template<typename T> class GlobalBrentSearch: public UnivariateOptimizer<T> {

protected:
	int _mfev;
	double _tol, _boundhess;

public:
	GlobalBrentSearch(double tol, int mfev, double bound_on_hessian);

	solution<T> optimize(univariate<T> f, T a, T b);
};

#include "global/global_brent.tpp"

#endif /* GLOBAL_BRENT_H_ */
