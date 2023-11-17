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
#include "../src/univariate/mlsl/mlsl.h"
#include "../src/univariate/piyavskii/piyavskii.h"

namespace py = pybind11;
using namespace pybind11::literals;

// define module
void build_univariate(py::module_ &m) {

	// build the base class for univariate solution

	py::class_<solution<double> > sol(m, "Solution");
	sol.def("toString", &solution<double>::toString);
	sol.def_property_readonly("sol", [](solution<double> &self) {
		return self._sol;
	});
	sol.def_property_readonly("converged", [](solution<double> &self) {
		return self._converged;
	});
	sol.def_property_readonly("fev", [](solution<double> &self) {
		return self._fev;
	});

	// build the base class for univariate optimizer
	py::class_<UnivariateOptimizer<double>> univaropt(m, "UnivariateOptimizer");
	univaropt.def("optimize", &UnivariateOptimizer<double>::optimize);

	// specific implementations without derivatives
	py::class_<BranchBoundSearch<double>, UnivariateOptimizer<double>> branchbound(
			m, "BranchBoundSearch");
	branchbound.def(py::init<int, double, double, int>(), "mfev"_a, "tol"_a,
			"K"_a, "n"_a = 16);

	py::class_<BrentSearch<double>, UnivariateOptimizer<double>> brent(m,
			"BrentSearch");
	brent.def(py::init<int, double, double>(), "mfev"_a, "atol"_a, "rtol"_a =
			1e-15);

	py::class_<GlobalBrentSearch<double>, UnivariateOptimizer<double>> globrent(
			m, "GlobalBrentSearch");
	globrent.def(py::init<int, double, double>(), "mfev"_a, "tol"_a,
			"bound_on_hessian"_a);

	py::class_<CalvinSearch<double>, UnivariateOptimizer<double>> calvin(m,
			"CalvinSearch");
	calvin.def(py::init<int, double, double>(), "mfev"_a, "tol"_a, "lam"_a =
			16.);

	py::class_<DaviesSwannCampey<double>, UnivariateOptimizer<double>> dsc(m,
			"DaviesSwannCampey");
	dsc.def(py::init<int, double, double>(), "mfev"_a, "tol"_a, "decay"_a =
			0.1);

	py::class_<FibonacciSearch<double>, UnivariateOptimizer<double>> fib(m,
			"FibonacciSearch");
	fib.def(py::init<int, double, double>(), "mfev"_a, "atol"_a, "rtol"_a =
			1e-15);

	py::class_<GoldenSectionSearch<double>, UnivariateOptimizer<double>> golden(
			m, "GoldenSectionSearch");
	golden.def(py::init<int, double, double>(), "mfev"_a, "atol"_a, "rtol"_a =
			1e-15);

	py::class_<PiyavskiiSearch<double>, UnivariateOptimizer<double>> piyavskii(
			m, "PiyavskiiSearch");
	piyavskii.def(py::init<int, double, double, double>(), "mfev"_a, "tol"_a,
			"r"_a = 1.4, "xi"_a = 1e-6);
}
