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

 [1] Rao, R. "Jaya: A simple and new optimization algorithm for solving
 constrained and unconstrained optimization problems." International Journal
 of Industrial Engineering Computations 7.1 (2016): 19-34.

 [2] Yu, Jiang-Tao, et al. "Jaya algorithm with self-adaptive multi-population
 and Lévy flights for solving economic load dispatch problems." IEEE Access 7
 (2019): 21372-21384.
 */

#ifndef JAYA_H_
#define JAYA_H_

#include <memory>
#include <random>
#include "../multivariate.h"

struct jaya_particle {

	std::vector<double> _x;
	double _f;
};

class JayaSearch: public MultivariateOptimizer {

private:
	int _n, _it, _fev, _k;
	double _pbest, _best, _sigmau;
	multivariate _f;
	jaya_particle _bestX;
	std::vector<int> _len;
	std::vector<double> _lower, _upper, _tmp, _r1, _r2;
	std::vector<std::shared_ptr<jaya_particle>> _pool;

	void divideSubpopulation();

	void adaptK();

	void evolve(int i, jaya_particle &best, jaya_particle &worst);

	void randomizeR();

	double sampleLevy();

protected:
	bool _adapt_k;
	int _np, _npmin, _mfev, _k0;
	double _scale, _beta;

public:
	std::normal_distribution<> _Z { 0., 1. };

	JayaSearch(int mfev, int np, int npmin, int k0 = 2, double scale = 0.01,
			double beta = 1.5, bool adapt = true);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);
};

#endif /* JAYA_H_ */
