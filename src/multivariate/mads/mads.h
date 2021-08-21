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

 [2] Le Digabel, Sébastien. "Algorithm 909: NOMAD: Nonlinear optimization with
 the MADS algorithm." ACM Transactions on Mathematical Software (TOMS) 37.4 (2011):
 1-15.
 */

#ifndef MULTIVARIATE_MADS_H_
#define MULTIVARIATE_MADS_H_

#include <numeric>

#include "../multivariate.h"

// mesh structure
class MADS;

class MADSMesh {

public:
	std::vector<std::vector<long long int>> _D;

	virtual ~MADSMesh() {
	}

	virtual void init(MADS *parent) = 0;

	virtual void update(MADS *parent) = 0;

	virtual void updateParameters(MADS *parent) = 0;

	virtual bool converged(MADS *parent) = 0;

	virtual void computeTrial(MADS *parent, int idx, double *x0,
			double *out) = 0;

	int size() {
		return _D.size();
	}
};

// search strategy
class MADSSearch {

public:
	virtual ~MADSSearch() {
	}

	virtual void init(MADS *parent) = 0;

	virtual void search(MADS *parent) = 0;
};

// surrogate models
class MADSSurrogateModel {

public:
	virtual ~MADSSurrogateModel() {
	}

	virtual void init(MADS *parent) = 0;

	virtual void updateModel(const double *x, double fx) = 0;

	virtual double evaluate(const double *x) = 0;
};

// algorithm base
class MADS: public MultivariateOptimizer {

public:
	static double constexpr INF = std::numeric_limits<double>::infinity();

	bool _searchsuccess, _minframe;
	int _n, _fev, _bbev, _mfev;
	double _tol, _fx;
	multivariate_problem _f;
	MADSMesh *_mesh;
	MADSSearch *_search;
	MADSSurrogateModel *_model;
	std::vector<int> _ranks;
	std::vector<double> _guess, _lower, _upper, _x, _trial, _pbest, _auxf,
			_succdir;

	MADS(MADSMesh *mesh, MADSSearch *search, int mfev, double tol,
			MADSSurrogateModel *model = nullptr);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

	double evaluateBarrier(const double *x);

protected:
	void poll();
};

// simple one-point line search
class MeshLineSearch: public MADSSearch {

protected:
	double _fxold;
	std::vector<double> _trial, _xold, _descent;

public:
	void init(MADS *parent);

	void search(MADS *parent);
};

// simple surrogate
class UserDefinedMADSSurrogate: public MADSSurrogateModel {

protected:
	multivariate _h;

public:
	UserDefinedMADSSurrogate(const multivariate &h) :
			_h(h) {
	}

	void init(MADS *parent) {
	}

	void updateModel(const double *x, double fx) {
	}

	double evaluate(const double *x) {
		return _h(x);
	}
};

#endif /* MULTIVARIATE_MADS_H_ */
