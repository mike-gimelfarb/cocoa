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
 */

#ifndef UNIVARIATE_DERIV_UNIVARIATE_DIFFERENTIABLE_H_
#define UNIVARIATE_DERIV_UNIVARIATE_DIFFERENTIABLE_H_

#include "../univariate.h"

template<typename T> struct solution_for_differentiable {

	T _sol;
	int _fev, _dfev;
	bool _converged;

	std::string toString() {
		std::string ss = toStringFull(_sol);
		return "x*: " + ss + "\n" + "calls to f: " + std::to_string(_fev) + "\n"
				+ "calls to df: " + std::to_string(_dfev) + "\n" + "converged: "
				+ std::to_string(_converged);
	}
};

template<typename T> class UnivariateDifferentiableOptimizer {

public:
	virtual ~UnivariateDifferentiableOptimizer() {
	}

	virtual solution_for_differentiable<T> optimize(univariate<T> f,
			univariate<T> df, T a, T b) = 0;
};

#endif /* UNIVARIATE_DERIV_UNIVARIATE_DIFFERENTIABLE_H_ */
