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

 [1] Hansen, Nikolaus, and Raymond Ros. "Benchmarking a weighted negative
 covariance matrix update on the BBOB-2010 noiseless testbed." Proceedings of
 the 12th annual conference companion on Genetic and evolutionary computation.
 ACM, 2010.

 [2] Jastrebski, Grahame A., and Dirk V. Arnold. "Improving evolution
 strategies through active covariance matrix adaptation." Evolutionary
 Computation, 2006. CEC 2006. IEEE Congress on. IEEE, 2006.
 */

#ifndef MULTIVARIATE_ACTIVE_CMAES_H_
#define MULTIVARIATE_ACTIVE_CMAES_H_

#include "cmaes.h"

class ActiveCmaes: public Cmaes {

protected:
	double _alphacov, _cm, _cneg, _alphaold;
	std::vector<double> _ycoeff;

public:
	ActiveCmaes(int mfev, double tol, int np, double sigma0 = 2., bool bound =
			false, double alphacov = 2., double eigenrate = 0.25) :
			Cmaes(mfev, tol, np, sigma0, bound, eigenrate) {
		_alphacov = alphacov;
	}

	void init(const multivariate_problem &f, const double *guess);

	void updateDistribution();
};

#endif /* MULTIVARIATE_ACTIVE_CMAES_H_ */
