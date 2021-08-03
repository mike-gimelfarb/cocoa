/*
 Copyright (c) 2008-2013 Carlos Henrique da Silva Santos

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 Translated to C++11 by Michael Gimelfarb in 2021
 */

#ifndef MULTIVARIATE_EVOL_ESCH_H_
#define MULTIVARIATE_EVOL_ESCH_H_

#include <memory>

#include "../multivariate.h"

struct esch_particle {

	std::vector<double> _x;
	double _f;

	static bool compare_fitness(const std::shared_ptr<esch_particle> &x,
			const std::shared_ptr<esch_particle> &y) {
		return x->_f < y->_f;
	}
};

class EschSearch: public MultivariateOptimizer {

private:
	int _n, _fev;
	double _v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7;
	multivariate _f;
	std::vector<double> _lower, _upper;
	std::vector<std::shared_ptr<esch_particle>> _parents, _offspring, _total;

	double sampleCauchy();

protected:
	int _np, _no, _mfev;

public:
	EschSearch(int mfev, int np, int no);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);
};

#endif /* MULTIVARIATE_EVOL_ESCH_H_ */
