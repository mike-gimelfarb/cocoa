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

#ifndef UNIVARIATE_GLOBAL_CALVIN_H_
#define UNIVARIATE_GLOBAL_CALVIN_H_

#include "../univariate.h"

template<typename T> class CalvinSearch: public UnivariateOptimizer<T> {

protected:
	int _mfev;
	double _tol, _lam;

public:
	CalvinSearch(double tol, int mfev, double lam=16.);

	solution<T> optimize(univariate<T> f, T guess, T a, T b);
};

#include "calvin.tpp"

#endif /* UNIVARIATE_GLOBAL_CALVIN_H_ */
