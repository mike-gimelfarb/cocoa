/*
 * exec.cpp
 *
 *  Created on: Jul. 5, 2021
 *      Author: mgime
 */
#include <ctime>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>
#include "execfuncs.h"
#include "multivariate/direct/nelder_mead.h"
#include "random.hpp"

using Random = effolkronium::random_static;

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

int main() {
	NelderMead src { 300000, 1e-8, 5., 20 };
	multivariate f = rosenbrock;
	std::vector<double> lower(_n, -8);
	std::vector<double> upper(_n, +5);
	std::vector<double> guess(_n);
	for (int i = 0; i < _n; i++) {
		guess[i] = Random::get(lower[i], upper[i]);
	}

	TimeVar t1 = timeNow();
	auto vec = src.optimize(f, _n, &guess[0], &lower[0], &upper[0]);

	std::cout << vec.toString() << std::endl;
	std::cout << f(&vec._sol[0]) << std::endl;
	double dur = duration(timeNow()-t1);
	std::cout << std::to_string(dur / 1000000.) << "ms";
	return 0;
}
