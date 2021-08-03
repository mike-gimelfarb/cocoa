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

#ifndef MULTIVARIATE_MULTIVARIATE_H_
#define MULTIVARIATE_MULTIVARIATE_H_

#include <vector>

#include "../string_utils.h"

using multivariate = double (*)(const double*);

struct multivariate_solution {

	std::vector<double> _sol;
	int _fev;
	bool _converged;

	std::string toString() {
		std::string ss = "";
		for (auto i = _sol.begin(); i != _sol.end(); i++) {
			ss += std::to_string(*i) + " ";
		}
		return "x*: " + ss + "\n" + "calls to f: " + std::to_string(_fev) + "\n"
				+ "converged: " + std::to_string(_converged);
	}
};

struct constrained_solution {

	std::vector<double> _sol;
	int _fev, _gev;
	bool _converged;

	std::string toString() {
		std::string ss = "";
		for (auto i = _sol.begin(); i != _sol.end(); i++) {
			ss += std::to_string(*i) + " ";
		}
		return "x*: " + ss + "\n" + "calls to f: " + std::to_string(_fev) + "\n"
				+ "calls to g: " + std::to_string(_gev) + "\n" + +"converged: "
				+ std::to_string(_converged);
	}
};

class MultivariateOptimizer {

public:
	virtual ~MultivariateOptimizer() {
	}

	virtual void init(multivariate f, const int n, double *guess, double *lower,
			double *upper) = 0;

	virtual void iterate() = 0;

	virtual multivariate_solution optimize(multivariate f, const int n,
			double *guess, double *lower, double *upper) = 0;
};

#endif /* MULTIVARIATE_MULTIVARIATE_H_ */
