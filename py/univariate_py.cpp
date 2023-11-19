#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "../src/univariate/univariate.h"
#include "../src/univariate/bb/branch_bound.h"
#include "../src/univariate/brent/brent.h"
#include "../src/univariate/brent/global_brent.h"
#include "../src/univariate/calvin/calvin.h"
#include "../src/univariate/dsc/dsc.h"
#include "../src/univariate/fibonacci/fibonacci.h"
#include "../src/univariate/golden/golden.h"
#include "../src/univariate/piyavskii/piyavskii.h"

namespace py = pybind11;
using namespace pybind11::literals;

void build_bb(py::module_ &m) {
	py::class_<BranchBoundSearch<double>, UnivariateOptimizer<double>> solver(
			m, "BranchAndBound");
	solver.def(py::init<int, double, double, int>(), "mfev"_a, "tol"_a, "K"_a, "n"_a = 16);
}

void build_brent(py::module_ &m) {
	py::class_<BrentSearch<double>, UnivariateOptimizer<double>> solver(m,
			"Brent");
	solver.def(py::init<int, double, double>(), "mfev"_a, "atol"_a, "rtol"_a = 1e-15);
}

void build_global_brent(py::module_ &m) {
	py::class_<GlobalBrentSearch<double>, UnivariateOptimizer<double>> solver(
			m, "GlobalBrent");
	solver.def(py::init<int, double, double>(), "mfev"_a, "tol"_a, "bound_on_hessian"_a);
}

void build_calvin(py::module_ &m) {
	py::class_<CalvinSearch<double>, UnivariateOptimizer<double>> solver(m,
			"Calvin");
	solver.def(py::init<int, double, double>(), "mfev"_a, "tol"_a, "lam"_a = 16.);
}

void build_dsc(py::module_ &m) {
	py::class_<DaviesSwannCampey<double>, UnivariateOptimizer<double>> solver(m,
			"DSC");
	solver.def(py::init<int, double, double>(), "mfev"_a, "tol"_a, "decay"_a =
			0.1);
}

void build_fibonacci(py::module_ &m) {
	py::class_<FibonacciSearch<double>, UnivariateOptimizer<double>> solver(m,
			"Fibonacci");
	solver.def(py::init<int, double, double>(), "mfev"_a, "atol"_a, "rtol"_a =
			1e-15);
}

void build_golden(py::module_ &m) {
	py::class_<GoldenSectionSearch<double>, UnivariateOptimizer<double>> solver(
			m, "GoldenSection");
	solver.def(py::init<int, double, double>(), "mfev"_a, "atol"_a, "rtol"_a =
			1e-15);
}

void build_piyavskii(py::module_ &m) {
	py::class_<PiyavskiiSearch<double>, UnivariateOptimizer<double>> solver(
			m, "Piyavskii");
	solver.def(py::init<int, double, double, double>(), "mfev"_a, "tol"_a,
			"r"_a = 1.4, "xi"_a = 1e-6);
}

void build_univariate(py::module_ &m) {

	// build the base class for univariate solution
	py::class_<solution<double> > sol(m, "UnivariateSolution");
	sol.def("__str__", &solution<double>::toString);
	sol.def_property_readonly("x", [](solution<double> &self) {
		return self._sol;
	});
	sol.def_property_readonly("converged", [](solution<double> &self) {
		return self._converged;
	});
	sol.def_property_readonly("n_evals", [](solution<double> &self) {
		return self._fev;
	});

	// build the base class for univariate optimizer
	py::class_<UnivariateOptimizer<double>> solver(m, "UnivariateSearch");
	solver.def("optimize", &UnivariateOptimizer<double>::optimize,
			"f"_a, "guess"_a, "lower"_a, "upper"_b);

	// register all methods
	build_bb(m);
	build_brent(m);
	build_global_brent(m);
	build_calvin(m);
	build_dsc(m);
	build_fibonacci(m);
	build_golden(m);
	build_piyavskii(m);
}
