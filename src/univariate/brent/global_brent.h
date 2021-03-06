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

#ifndef UNIVARIATE_GLOBAL_BRENT_H_
#define UNIVARIATE_GLOBAL_BRENT_H_

#include "../univariate.h"

template<typename T> class GlobalBrentSearch: public UnivariateOptimizer<T> {

protected:
	int _mfev;
	double _tol, _boundhess;

public:
	GlobalBrentSearch(int mfev, double tol, double bound_on_hessian);

	solution<T> optimize(const univariate<T> &f, T guess, T a, T b);
};

#include "global_brent.tpp"

#endif /* UNIVARIATE_GLOBAL_BRENT_H_ */
