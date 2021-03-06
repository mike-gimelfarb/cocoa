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

 [1] Hansen, Nikolaus, and Andreas Ostermeier. "Completely derandomized
 self-adaptation in evolution strategies." Evolutionary computation 9.2
 (2001): 159-195.

 [2] Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv
 preprint arXiv:1604.00772 (2016).
 */

#ifndef MULTIVARIATE_CMAES_CMAES_H_
#define MULTIVARIATE_CMAES_CMAES_H_

#include <random>

#include "base_cmaes.h"

class Cmaes: public BaseCmaes {

protected:
	int _flag, _eigenlastev;
	double _eigenfreq, _eigenrate;
	std::vector<double> _diagd;
	std::vector<std::vector<double>> _b, _c, _invsqrtc;

public:
	std::normal_distribution<> _Z { 0., 1. };

	Cmaes(int mfev, double tol, int np, double sigma0 = 2., bool bound = false,
			double eigenrate = 0.25) :
			BaseCmaes(mfev, tol, np, sigma0, bound) {
		_eigenrate = eigenrate;
	}

	void init(const multivariate_problem &f, const double *guess);

	void samplePopulation();

	void updateDistribution();

	bool converged();

	void eigenDecomposition();

	void tred2();

	void tql2();
};

#endif /* MULTIVARIATE_CMAES_CMAES_H_ */
