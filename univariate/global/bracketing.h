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

#ifndef BRACKETING_H_
#define BRACKETING_H_

#include <vector>
#include "../univariate/univariate.h"

template<typename T> struct interval {

	T _left, _right;
};

template<typename T> struct solutions {

	std::vector<double> _sols;
	int _fev;
	bool _converged;

	std::string toString() {
		std::string ss = "";
		for (auto i = _sols.begin(); i != _sols.end(); ++i) {
			ss += toStringFull(*i) + "\n";
		}
		return "x*: " + ss + "calls to f: " + std::to_string(_fev) + "\n"
				+ "converged: " + std::to_string(_converged);
	}
};

template<typename T> class BracketingSearch {

protected:
	int _mfev;
	double _tol, _factor;
	UnivariateOptimizer<T> *_local;

public:
	BracketingSearch(UnivariateOptimizer<T> *local, double tol, int mfev,
			double factor=1.68);

	solutions<T> optimize_all(univariate<T> f, T a, T b);
};

#include "global/bracketing.tpp"

#endif /* BRACKETING_H_ */
