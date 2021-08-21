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

 [1] Audet, Charles, Sébastien Le Digabel, and Christophe Tribes. "Dynamic scaling
 in the mesh adaptive direct search algorithm for blackbox optimization."
 Optimization and Engineering 17.2 (2016): 333-358.
 */

#ifndef MULTIVARIATE_ISOMADS_H_
#define MULTIVARIATE_ISOMADS_H_

#include <random>

#include "mads.h"

class IsoMADSMesh: public MADSMesh {

	bool _aniso;
	double _beta0, _scale;
	std::vector<int> _r;
	std::vector<double> _deltap0, _deltam0, _deltap, _deltam, _beta, _u;

public:
	std::normal_distribution<> _Z { 0., 1. };

	IsoMADSMesh(double scale = 0.1, double beta = 2., bool aniso = true);

	void init(MADS *parent);

	void update(MADS *parent);

	void updateParameters(MADS *parent);

	bool converged(MADS *parent);

	void computeTrial(MADS *parent, int idx, double *x0, double *out);
};

#endif /* MULTIVARIATE_ISOMADS_H_ */
