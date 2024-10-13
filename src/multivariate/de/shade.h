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
 */

#ifndef MULTIVARIATE_DE_SHADE_H_
#define MULTIVARIATE_DE_SHADE_H_

#include <memory>
#include <random>

#include "../multivariate.h"

class ShadeSearch: public MultivariateOptimizer {

protected:
	bool _archive;
	int _n, _np, _fev, _mfev, _h, _k;
	double _tol;
	multivariate_problem _f;
	std::vector<double> _lower, _upper, _MCR, _MF, _work, _SCR, _SF, _w;
	std::vector<std::vector<double>> _arch;
	std::vector<point> _swarm;

public:
	std::normal_distribution<> _Z { 0., 1. };

	ShadeSearch(int mfev, int np, double tol, bool archive = true, int h = 100);

	void init(const multivariate_problem &f, const double *guess);

	void iterate();

	multivariate_solution optimize(const multivariate_problem &f,
			const double *guess);

private:
	void mutate(double *x, double *best, double *xr1, double *xr2, double *out,
			int n, double F, double CR);

	double sampleCauchy();
};

#endif /* MULTIVARIATE_DE_SHADE_H_ */
