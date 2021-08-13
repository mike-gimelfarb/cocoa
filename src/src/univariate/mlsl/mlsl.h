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

 [1] Rinnooy Kan, A. H., and G. T. Timmer. "Stochastic global optimization methods
 part II: Multi level methods." Mathematical Programming: Series A and B 39.1 (1987):
 57-78.

 [2] Larson, Jeffrey, and Stefan M. Wild. "A batch, derivative-free algorithm for
 finding multiple local minima." Optimization and Engineering 17.1 (2016): 205-228.
 */

#ifndef MLSL_H_
#define MLSL_H_

#include <memory>
#include <vector>

#include "../univariate.h"

template<typename T> struct solutions {

	std::vector<T> _sols;
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

template<typename T> struct mlsl_point {

	T _x, _f;
	bool _startlocal;

	static bool compare_x(const std::shared_ptr<mlsl_point<T>> &x,
			const std::shared_ptr<mlsl_point<T>> &y) {
		return x->_x < y->_x;
	}
};

template<typename T> class MLSL {

protected:
	int _n, _ns, _fev, _mfev;
	T _sigma, _mu, _nu, _a, _b, _sep;
	univariate<T> _f;
	UnivariateOptimizer<T> *_local;
	std::vector<T> _minima;
	std::vector<std::shared_ptr<mlsl_point<T>>> _S;

public:
	MLSL(UnivariateOptimizer<T> *local, int n, int mfev, double sep = 1e-4,
			double sigma = 4., double mu = 1e-6, double nu = 1e-6);

	solutions<T> optimize(univariate<T> f, T a, T b);

private:
	void uniformSampling();

	solution<T> optimizeLocal(const int istart);

	void addMinimum(const T &min);

	int localStart();

	bool validStart(const int i, const T rk);

	T nearestKnownMin(const T &x);
};

#include "mlsl.tpp"

#endif /* MLSL_H_ */
