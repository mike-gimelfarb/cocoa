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

 [1] Loshchilov, Ilya. "LM-CMA: An alternative to L-BFGS for large-scale black
 box optimization." Evolutionary computation 25.1 (2017): 143-171.

 [2] Loshchilov, Ilya. "A computationally efficient limited memory CMA-ES for
 large scale optimization." Proceedings of the 2014 Annual Conference on
 Genetic and Evolutionary Computation. ACM, 2014.
 */

#ifndef MULTIVARIATE_CMAES_LM_CMAES_H_
#define MULTIVARIATE_CMAES_LM_CMAES_H_

#include <cmath>
#include <random>

#include "base_cmaes.h"

static constexpr int small_size(int x) {
	return 4 + (int) std::log(3. * x);
}

static constexpr int large_size(int x) {
	return (int) (2. * std::sqrt(1. * x));
}

class LmCmaes: public BaseCmaes {

protected:
	bool _new, _adaptmemory;
	int _samplemode, _memlen, _memsize, _nsteps, _t;
	double _stolmin, _sqrt1mc1, _zstar, _s, _ccc;
	std::vector<int> _jarr, _larr;
	std::vector<double> _b, _d, _az, _fp;
	std::vector<std::vector<double>> _pcmat, _vmat;
	std::vector<std::shared_ptr<cmaes_index>> _mixed;

public:
	std::normal_distribution<> _Z { 0., 1. };

	LmCmaes(int mfev, double tol, int np, int memory = 0, double sigma0 = 2., // @suppress("Class members should be properly initialized")
			bool rademacher = true, bool usenew = true) :
			BaseCmaes(mfev, tol, np, sigma0) {
		_samplemode = rademacher ? 1 : 0;
		_new = usenew;
		_stolmin = 1e-16;
		_memsize = memory;
		_adaptmemory = memory <= 0;
	}

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void samplePopulation();

	void updateDistribution();

	bool converged();

	void evaluateAndSortPopulation();

	void updateSigma();

	void ainvz(int jlen);

	int updateSet();

	int selectSubset(int m, int k);
};

#endif /* MULTIVARIATE_CMAES_LM_CMAES_H_ */
