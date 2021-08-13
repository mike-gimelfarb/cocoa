/*
 Copyright (c) 2012, Pinar Civicioglu
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MULTIVARIATE_EVOL_DS_H_
#define MULTIVARIATE_EVOL_DS_H_

#include <memory>

#include "../multivariate.h"

struct ds_particle {

	std::vector<int> _map;
	std::vector<double> _x, _so;
	std::vector<double> *_dir;
	double _f, _fso;

	static bool compare_fitness(const ds_particle &x, const ds_particle &y) {
		return x._f < y._f;
	}
};

class DifferentialSearch: public MultivariateOptimizer {

private:
	int _n, _fev;
	multivariate _f;
	std::vector<int> _methods, _jind;
	std::vector<double> _lower, _upper;
	std::vector<ds_particle> _swarm;

	void genDir(int method);

	void genPop();

	void genMap(double p1, double p2);

	void update();

protected:
	int _np, _mfev;
	double _tol, _stol;

public:
	DifferentialSearch(int mfev, double tol, double stol, int np);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);
};

#endif /* MULTIVARIATE_EVOL_DS_H_ */
