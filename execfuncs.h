/*
 * execfuncs.h
 *
 *  Created on: Jul. 26, 2021
 *      Author: mgime
 */

#ifndef EXECFUNCS_H_
#define EXECFUNCS_H_

#include <cmath>

int _n = 10;

static double sphere(const double *x) {
	double f = 0.;
	for (int i = 0; i < _n; i++) {
		f += x[i] * x[i];
	}
	return f;
}

static void dsphere(const double *x, double *g){
	for (int i = 0; i < _n; i++){
		g[i] = 2. * x[i];
	}
}

static double rosenbrock(const double *x) {
	double f = 0.;
	for (int i = 0; i < _n - 1; i++) {
		f += 100 * std::pow(x[i + 1] - x[i] * x[i], 2.);
		f += std::pow(1. - x[i] * x[i], 2.);
	}
	return f;
}

static void drosenbrock(const double *x, double *g) {
	g[0] = 100 * 2 * (x[1] - x[0] * x[0]) * (-2 * x[0])
			+ 2 * (1. - x[0] * x[0]) * (-2 * x[0]);
	for (int i = 1; i < _n - 1; i++) {
		g[i] = 100 * 2 * (x[i] - x[i - 1] * x[i - 1])
				+ 100 * 2 * (x[i + 1] - x[i] * x[i]) * (-2 * x[i])
				+ 2 * (1. - x[i] * x[i]) * (-2 * x[i]);
	}
	g[_n - 1] = 100 * (x[_n - 1] - x[_n - 2] * x[_n - 2]) * 2;
}

static double ackley(const double *x) {
	double f = 0.;
	double g = 0.;
	for (int i = 0; i < _n; i++) {
		f += x[i] * x[i];
		g += std::cos(2. * M_PI * x[i]);
	}
	f = -20. * std::exp(-0.2 * std::sqrt(f / _n));
	g = std::exp(g / _n);
	return f - g + 20. + std::exp(1.);
}

static double rastrigin(const double *x) {
	double f = 10. * _n;
	for (int i = 0; i < _n; i++) {
		double y = x[i] - 1.232;
		f += y * y - 10. * std::cos(2. * M_PI * y);
	}
	return f;
}

static double griewank(const double *x) {
	double f = 0.;
	double f2 = 1.;
	for (int i = 0; i < _n; i++) {
		f += std::pow(x[i] / 20., 2.);
		f2 *= std::cos(x[i] / std::sqrt(1. * (i + 1)));
	}
	return f - f2 + 1.;
}

#endif /* EXECFUNCS_H_ */
