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

 [1] Krause, Oswin, Dídac Rodríguez Arbonès, and Christian Igel. "CMA-ES with
 optimal covariance update and storage complexity." Advances in Neural
 Information Processing Systems. 2016.
 */

#ifndef CHOLESKY_CMAES_H_
#define CHOLESKY_CMAES_H_

#include <random>
#include "base_cmaes.h"

class CholeskyCmaes: public BaseCmaes {

protected:
	double _stol;
	std::vector<double> _dmean;
	std::vector<std::vector<double>> _a, _mattmp;

public:
	std::normal_distribution<> _Z { 0., 1. };

	CholeskyCmaes(int mfev, double tol, double stol, int np, double sigma0 = 2.) : // @suppress("Class members should be properly initialized")
			BaseCmaes(mfev, tol, np, sigma0) {
		_stol = stol;
	}

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void updateDistribution();

	void samplePopulation();

	bool converged();

	void rank1Update(double beta);
};

#endif /* CHOLESKY_CMAES_H_ */
