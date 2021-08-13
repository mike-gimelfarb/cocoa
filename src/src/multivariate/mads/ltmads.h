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

#ifndef LTMADS_H_
#define LTMADS_H_

#include "../constrained.h"

class LTMADS: public ConstrainedOptimizer {

protected:
	static constexpr double INF = std::numeric_limits<double>::infinity();

	bool _maximal, _search, _minframe;
	int _n, _fev, _cev, _mfev, _deltampow, _lc, _ihat;
	double _tol, _fx, _fxold, _deltapoll;
	multivariate _f;
	constraints _omega;
	std::vector<int> _N, _Nm1;
	std::vector<long> _bl;
	std::vector<double> _guess, _lower, _upper, _xold, _x, _fmesh, _temp;
	std::vector<std::vector<long>> _L, _B, _D;

public:
	LTMADS(int mfev, double tol, bool maxbasis = true, bool search = true);

	virtual ~LTMADS() {
	}

	void init(multivariate f, constraints c, const int n, double *guess,
			double *lower, double *upper);

	void iterate();

	constrained_solution optimize(multivariate f, constraints c, const int n,
			double *guess, double *lower, double *upper);

private:
	bool search();

	void poll();

	virtual void generateBasis();

	void generateBl(const int l);

	double evaluateBarrier(const double *x);
};

#endif /* LTMADS_H_ */
