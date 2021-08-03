/*
 Please note, even though the main code for AMALGAM can be available under MIT
 licensed, the dchdcm subroutine is a derivative of LINPACK code that is licensed
 under the 3-Clause BSD license. The other subroutines:

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

 [1] Bosman, Peter AN, J�rn Grahl, and Dirk Thierens. "AMaLGaM IDEAs in
 noiseless black-box optimization benchmarking." Proceedings of the 11th
 Annual Conference Companion on Genetic and Evolutionary Computation
 Conference: Late Breaking Papers. ACM, 2009.
 */

#ifndef AMALGAM_H_
#define AMALGAM_H_

#include <memory>
#include <random>
#include "../multivariate.h"

struct amalgam_solution {

	std::vector<double> _x;
	double _f;

	static bool compare_fitness(const std::shared_ptr<amalgam_solution> &x,
			const std::shared_ptr<amalgam_solution> &y) {
		return x->_f < y->_f;
	}
};

class AmalgamIdea: public MultivariateOptimizer {

private:
	void runParallel();

	bool converged();

	double computeSDR();

	void updateDistribution();

	int samplePopulation();

	int dchdcm();

protected:
	bool _iamalgam, _noparam, _print;
	int _n, _fev, _mfev, _s, _runs, _nbase, _budget, _nelite, _nams, _nis,
			_nismax, _np, _ss, _t;
	double _tol, _stol, _mincmult, _tau, _alphaams, _deltaams, _etasigma,
			_etashift, _etadec, _etainc, _cmult, _thetasdr, _fbest, _fbestrun,
			_fbestrunold;
	multivariate _f;
	std::vector<double> _lower, _upper, _mu, _muold, _mushift, _mushiftold,
			_best, _tmp, _xavg;
	std::vector<std::vector<double>> _cov, _chol;
	std::vector<std::shared_ptr<amalgam_solution>> _sols;

public:
	std::normal_distribution<> _Z { 0., 1. };

	AmalgamIdea(int mfev, double tol, double stol, int np = 0, bool iamalgam =
			false, bool noparam = true, bool print = true);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);
};

#endif /* AMALGAM_H_ */
