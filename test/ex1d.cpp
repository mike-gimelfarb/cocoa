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

#include "../src/multivariate/simplex/nelder_mead.h"
#include "../src/univariate/piyavskii/piyavskii.h"
#include "../src/random.hpp"

using Random = effolkronium::random_static;
typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

static double f1(const double x) {
	return std::sin(x) + std::sin((10. / 3.) * x);
}

static double f20(const double *x) {
	double f = 0.;
	for (int i = 0; i < 20 - 1; i++) {
		f += 100 * std::pow(x[i + 1] - x[i] * x[i], 2.);
		f += std::pow(1. - x[i] * x[i], 2.);
	}
	return f;
}

int main() {
	TimeVar t1 = timeNow();

	// 1D example
	PiyavskiiSearch<double> o1(1e-6, 10000);
	auto sol1 = o1.optimize(f1, -2.7, 7.5);
	std::cout << "1D test" << std::endl;
	std::cout << sol1.toString() << "\n" << std::endl;

	// 20D example
	NelderMead o20(100000, 1e-6, 5.0);
	std::vector<double> lower(20, -10.);
	std::vector<double> upper(20, +10.);
	std::vector<double> guess(20);
	for (int i = 0; i < 20; i++) {
		guess[i] = Random::get(lower[i], upper[i]);
	}
	auto sol20 = o20.optimize(f20, 20, &guess[0], &lower[0], &upper[0]);
	std::cout << "20D test" << std::endl;
	std::cout << sol20.toString() << std::endl;

	// timing
	double dur = duration(timeNow()-t1);
	std::cout << std::to_string(dur / 1000000.) << "ms";
	return 0;
}
