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

 [1] Zhan, Zhang, and Chung. Adaptive particle swarm optimization, IEEE
 Transactions on Systems, Man, and Cybernetics, Part B: CyberneticsVolume 39,
 Issue 6, 2009, Pages 1362-1381 (2009)
 */

#ifndef PSO_H_
#define PSO_H_

#include <memory>
#include <random>
#include "../multivariate.h"

struct pso_particle {

	std::vector<double> _x, _v, _xb;
	double _f, _fb;
};

class PsoSearch: public MultivariateOptimizer {

private:
	int _n, _it, _fev, _maxit, _state, _worst;
	double _smin, _smax, _w, _c1, _c2, _fbest;
	multivariate _f;
	std::vector<double> _wp, _ws, _wmu, _lower, _upper, _xbest;
	std::vector<std::shared_ptr<pso_particle>> _swarm;

	void updateParticle(pso_particle &particle);

	void updateBest();

	void updateSwarm();

	void updateParams(double f, int state);

	double getF();

	int nextState(double f);

	double mu(double f, int i);

protected:
	bool _correct;
	int _np, _mfev;
	double _tol, _stol;

public:
	std::normal_distribution<> _Z { 0., 1. };

	PsoSearch(int mfev, double tol, double stol, int np, bool correct=true);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);
};

#endif /* PSO_H_ */
