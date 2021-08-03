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

 [1] Li, Xiaodong, and Xin Yao. "Cooperatively coevolving particle swarms for
 large scale optimization." IEEE Transactions on Evolutionary Computation 16.2
 (2012): 210-224.
 */

#ifndef MULTIVARIATE_EVOL_CCPSO_H_
#define MULTIVARIATE_EVOL_CCPSO_H_

#include <memory>
#include <random>
#include <vector>

#include "../multivariate.h"

struct ccpso_particle {

	std::vector<double> _x, _xpb;
	double *_xlb;
};

class CcPsoSearch: public MultivariateOptimizer {

private:
	int _n, _fev;
	multivariate _f;
	std::vector<double> _lower, _upper;

	void randomizeComponents();

	void randomizeSwarm();

	void updateSwarm(int i);

	void updatePosition(int i);

	void updateParameters();

	double evaluate(int i, double *z);

	double sampleCauchy();

protected:
	bool _correct;
	int _np, _npps, _mfev, _update, _nswarm, _gen, _cpswarm, _is;
	double _tol, _stol;
	int *_pps;

	// temporary storage for the swarm topology
	double _fbest;
	std::vector<int> _topidx;
	std::vector<double> _topfit, _sbest, _xbest, _work;
	std::vector<std::vector<bool>> _cauchy;
	std::vector<std::vector<int>> _k;
	std::vector<std::vector<double>> _pbfit;
	std::vector<std::shared_ptr<ccpso_particle>> _swarm;

	// data for adaptive parameter updates
	int _fupdate, _csucc, _cfail, _gsucc, _gfail;
	double _fx;

public:
	std::normal_distribution<> _Z { 0., 1. };

	CcPsoSearch(int mfev, double tol, double sigmatol, int np, int *pps, int npps,
			bool correct=true, int update=30);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);
};

#endif /* MULTIVARIATE_EVOL_CCPSO_H_ */
