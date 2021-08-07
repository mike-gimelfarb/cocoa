/*
 * ex1d.cpp
 *
 *  Created on: Jul. 5, 2021
 *      Author: mgime
 */
#include <ctime>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>

#include "../src/blas.h"
#include "../src/multivariate/algencan/algencan.h"
#include "../src/random.hpp"

#include "../src/multivariate/sade/sade.h"

#include "../src/univariate/piyavskii/piyavskii.h"

using Random = effolkronium::random_static;
typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

static double f1(const double x) {
	return std::sin(x) + std::sin((10. / 3.) * x);
}

const int _n = 30;

static double rosenbrock(const double *x) {
	double f = 0.;
	for (int i = 0; i < _n - 1; i++) {
		f += 100 * std::pow(x[i + 1] - x[i] * x[i], 2.);
		f += std::pow(1. - x[i] * x[i], 2.);
	}
	return f;
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

int main() {
	TimeVar t1 = timeNow();

	// 1D example
	std::cout << "1D test" << std::endl;
	PiyavskiiSearch<double> o1d(1e-6, 10000);
	auto sol1d = o1d.optimize(f1, -2.7, 7.5);
	std::cout << sol1d.toString() << "\n" << std::endl;

	// ND example
	std::cout << _n << "D test" << std::endl;
	SadeSearch ond(100000, 1e-6, 1e-6, 30);
	std::vector<double> lower(_n, 0.);
	std::vector<double> upper(_n, +500.);
	std::vector<double> guess(_n);
	for (int i = 0; i < _n; i++) {
		guess[i] = Random::get(lower[i], upper[i]);
	}
	auto solnd = ond.optimize(rosenbrock, _n, &guess[0], &lower[0], &upper[0]);
	std::cout << solnd.toString() << std::endl;

	// timing
	double dur = duration(timeNow()-t1);
	std::cout << std::to_string(dur / 1000000.) << "ms";
	return 0;
}
