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

#ifndef UNIVARIATE_GLOBAL_PIYAVSKII_H_
#define UNIVARIATE_GLOBAL_PIYAVSKII_H_

#include "../univariate.h"

template<typename T> class PiyavskiiSearch: public UnivariateOptimizer<T> {

protected:
	int _mfev;
	double _tol, _r, _xi;

public:
	PiyavskiiSearch(double tol, int mfev, double r=1.4, double xi=1e-6);

	solution<T> optimize(univariate<T> f, T a, T b);
};

#include "piyavskii.tpp"

#endif /* UNIVARIATE_GLOBAL_PIYAVSKII_H_ */
