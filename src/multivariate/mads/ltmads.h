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

 [1] Audet, Charles & Dennis, J.. (2006). Mesh Adaptive Direct Search Algorithms
 for Constrained Optimization. SIAM Journal on Optimization. 17. 188-217.
 10.1137/060671267.
 */

#ifndef MULTIVARIATE_LTMADS_H_
#define MULTIVARIATE_LTMADS_H_

#include "mads.h"

class LTMADSMesh: public MADSMesh {

protected:
	bool _maximal;
	int _lk, _lc, _ihat;
	double _deltam, _deltap;
	std::vector<int> _N, _Nm1;
	std::vector<long long int> _bl;
	std::vector<std::vector<long long int>> _L, _B;

public:
	LTMADSMesh(bool maximal = true);

	void init(MADS *parent);

	void update(MADS *parent);

	void updateParameters(MADS *parent);

	bool converged(MADS *parent);

	void computeTrial(MADS *parent, int idx, double *x0, double *out);

private:
	void generatebl(MADS *parent, const int l);
};

#endif
