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

 [2] Audet, Charles, et al. "Reducing the number of function evaluations in mesh
 adaptive direct search algorithms." SIAM Journal on Optimization 24.2 (2014):
 621-642.
 */

#ifndef MULTIVARIATE_ORTHOMADS_H_
#define MULTIVARIATE_ORTHOMADS_H_

#include "mads.h"

class OrthoMADSMesh: public MADSMesh {

protected:
	bool _reduced;
	int _lk, _t0, _tk, _tkmax;
	double _deltam, _deltap, _deltapmin;
	std::vector<long long int> _primes, _nhalton, _dhalton, _q;
	std::vector<double> _uhalton, _uhat;

public:
	OrthoMADSMesh(bool reduced = true);

	void init(MADS *parent);

	void update(MADS *parent);

	void updateParameters(MADS *parent);

	bool converged(MADS *parent);

	void computeTrial(MADS *parent, int idx, double *x0, double *out);

private:
	void computeAlpha(const int n, const int l);

	void nextHalton(const int n);

	int fillPrimes(const int n);

	bool isPrime(const int n);
};

#endif /* MULTIVARIATE_ORTHOMADS_H_ */
