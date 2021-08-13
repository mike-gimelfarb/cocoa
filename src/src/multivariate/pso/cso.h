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

 [1] Cheng, Ran & Jin, Yaochu. (2014). A Competitive Swarm Optimizer for Large
 Scale Optimization. IEEE transactions on cybernetics. 45.
 10.1109/TCYB.2014.2322602.

 [2] Kennedy, James & Mendes, Rui. (2002). Population structure and particle
 swarm performance. Proc IEEE Congr Evol Comput. 2. 1671 - 1676.
 10.1109/CEC.2002.1004493.
 */

#ifndef MULTIVARIATE_EVOL_CSO_H_
#define MULTIVARIATE_EVOL_CSO_H_

#include <memory>
#include <random>

#include "../multivariate.h"

struct cso_particle {

	std::vector<double> _x, _v;
	std::vector<double> _mean;
	double _f;

	cso_particle *_left, *_right;
};

class CSOSearch: public MultivariateOptimizer {

private:
	int _n, _fev;
	multivariate _f;
	cso_particle _best, _worst;
	std::vector<double> _lower, _upper;

	double computePhi(int m);

	void compete(cso_particle &first, cso_particle &second);

protected:
	bool _ring, _correct;
	int _np, _mfev;
	double _phi, _tol, _sigma_tol;
	std::vector<double> _mean;
	std::vector<cso_particle> _swarm;

public:
	CSOSearch(int mfev, double tol, double sigmatol, int np, bool ring=false,
			bool correct=true);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);
};

#endif /* MULTIVARIATE_EVOL_CSO_H_ */
