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

 [1] Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan.
 "Differential evolution algorithm with strategy adaptation for global
 numerical optimization." IEEE transactions on Evolutionary Computation 13.2
 (2009): 398-417.

 [2] Huang, Vicky Ling, A. Kai Qin, and Ponnuthurai N. Suganthan.
 "Self-adaptive differential evolution algorithm for constrained
 real-parameter optimization." Evolutionary Computation, 2006. CEC 2006. IEEE
 Congress on. IEEE, 2006. Yang, Zhenyu, Ke Tang, and Xin Yao.

 [3] "Self-adaptive differential evolution with neighborhood search."
 Evolutionary Computation, 2008. CEC 2008.(IEEE World Congress on
 Computational Intelligence). IEEE Congress on. IEEE, 2008.
 */

#ifndef MULTIVARIATE_SADE_H_
#define MULTIVARIATE_SADE_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class SadeSearch: public MultivariateOptimizer {

protected:
	int _n, _fev, _gen, _ihist, _fns0, _fnf0, _fns1, _fnf1;
	double _crm, _fp;
	multivariate_problem _f;
	std::vector<int> _ns, _nf, _ibw;
	std::vector<double> _p, _cr, _crrec, _dfit, _xtrii, _lower, _upper;
	std::vector<point> _pool;

	int _np, _lp, _cp, _k, _mfev;
	double _tol, _stol, _scr, _sf, _mu;

public:
	std::normal_distribution<> _Z { 0., 1. };

	SadeSearch(int mfev, double tol, double stol, int np, int lp = 25, int cp =
			5);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void trial(int i, int ki, double f, double cr, int ib);

	void rank();

	int roulette();
};

#endif /* MULTIVARIATE_SADE_H_ */
