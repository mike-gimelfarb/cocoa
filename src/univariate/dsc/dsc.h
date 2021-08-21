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

#ifndef UNIVARIATE_DAVIES_SWANN_CAMPEY_H_
#define UNIVARIATE_DAVIES_SWANN_CAMPEY_H_

#include "../univariate.h"

template<typename T> class DaviesSwannCampey: public UnivariateOptimizer<T> {

protected:
	int _mfev;
	double _tol, _decay;

public:
	DaviesSwannCampey(int mfev, double tol, double decay=0.1);

	solution<T> optimize(const univariate<T> &f, T guess, T a, T b);
};

#include "dsc.tpp"

#endif /* UNIVARIATE_DAVIES_SWANN_CAMPEY_H_ */
