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

 [1] Abramson, Mark A., et al. "OrthoMADS: A deterministic MADS instance with
 orthogonal directions." SIAM Journal on Optimization 20.2 (2009): 948-966.
 */

#ifndef ORTHOMADS_H_
#define ORTHOMADS_H_

#include "ltmads.h"

class OrthoMADS: public LTMADS {

protected:
	int _t0, _tk, _tkmax, _lmax;
	std::vector<long> _primes, _nhalton, _dhalton, _q;
	std::vector<double> _uhalton;

public:
	OrthoMADS(int mfev, double tol, bool search = true) : // @suppress("Class members should be properly initialized")
			LTMADS(mfev, tol, true, search) {
	}

	void init(multivariate f, constraints c, const int n, double *guess,
			double *lower, double *upper);

private:
	void generateBasis();

	void computeAlpha();

	void nextHalton();

	int fillPrimes();

	bool isPrime(const int n);
};

#endif /* ORTHOMADS_H_ */
