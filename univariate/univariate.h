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

#ifndef UNIVARIATE_H_
#define UNIVARIATE_H_

#include "string_utils.h"

template<typename T> using univariate = T (*)(const T);

template<typename T> struct solution {

	T _sol;
	int _fev;
	bool _converged;

	std::string toString() {
		std::string ss = toStringFull(_sol);
		return "x*: " + ss + "\n" + "calls to f: " + std::to_string(_fev) + "\n"
				+ "converged: " + std::to_string(_converged);
	}
};

template<typename T> struct bracket {

	T _left, _right;
	int _fev;
	bool _converged;
};

template<typename T> static bracket<T> bracketSolution(univariate<T> f, T x0,
		const int mfev, const T factor, const T leftBound, const T rightBound);

template<typename T> class UnivariateOptimizer {

public:
	virtual ~UnivariateOptimizer() {
	}

	virtual solution<T> optimize(univariate<T> f, T a, T b) = 0;
};

#include "univariate.tpp"

#endif /* UNIVARIATE_H_ */
