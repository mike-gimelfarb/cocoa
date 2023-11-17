#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "pybind11/numpy.h"
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "../src/multivariate/multivariate.h"
#include "../src/multivariate/constrained.h"
#include "../src/multivariate/amp/cttm.h"
#include "../src/multivariate/complex/box.h"
#include "../src/multivariate/mads/mads.h"
#include "../src/multivariate/mads/ltmads.h"
#include "../src/multivariate/mads/orthomads.h"
#include "../src/multivariate/algencan/algencan.h"

namespace py = pybind11;
using namespace pybind11::literals;

// generic constrained solvers
void build_constrained_amp(py::module_ &m) {
	py::class_<ConstrainedAMPTTM, ConstrainedOptimizer> cttm(m,
			"ConstrainedAMPTTM");

	// enumerations
	py::enum_<ConstrainedAMPTTM::tabu_removal_strategy> tabu_removal(cttm,
			"tabu_removal_strategy");
	tabu_removal.value("oldest",
			ConstrainedAMPTTM::tabu_removal_strategy::oldest);
	tabu_removal.value("farthest",
			ConstrainedAMPTTM::tabu_removal_strategy::farthest);
	tabu_removal.export_values();

	cttm.def(
			py::init<ConstrainedOptimizer*, int, bool, double, double, int, int,
					unsigned int, ConstrainedAMPTTM::tabu_removal_strategy>(),
			"local"_a, "mfev"_a, "print"_a = false, "eps1"_a = 0.02, "eps2"_a =
					0.1, "totaliter"_a = 9999, "maxiter"_a = 5, "tabutenure"_a =
					5, "remove"_a =
					ConstrainedAMPTTM::tabu_removal_strategy::farthest);
}

void build_box(py::module_ &m) {
	py::class_<BoxComplex, ConstrainedOptimizer> box(m, "BoxComplex");
	box.def(py::init<int, double, double, double, double, double, int, bool>(),
			"mfev"_a, "ftol"_a, "xtol"_a, "alpha"_a = 0., "rfac"_a = 0.,
			"rforget"_a = 0.3, "nbox"_a = 0, "movetobest"_a = true);
}

void build_mads(py::module_ &m) {
	py::class_ < MADSMesh > (m, "MADSMesh");
	py::class_ < MADSSearch > (m, "MADSSearch");
	py::class_<MADS, ConstrainedOptimizer> mads(m, "MADS");
	mads.def(py::init<MADSMesh*, MADSSearch*, int, double>(), "mesh"_a,
			"search"_a, "mfev"_a, "tol"_a);

	py::class_<LTMADSMesh, MADSMesh> ltmads(m, "LTMADSMesh");
	ltmads.def(py::init<bool>(), "maximal"_a = true);
	py::class_<LTMADSLineSearch, MADSSearch> orthomads(m, "LTMADSLineSearch");
	orthomads.def(py::init<>());
	py::class_<OrthoMADSMesh, MADSMesh>(m, "OrthoMADSMesh").def(py::init<>());
}

// constrained solvers with structure
void build_algencan(py::module_ &m) {
	py::class_<Algencan, StructuredConstrainedOptimizer> algen(m, "Algencan");
	algen.def(
			py::init<MultivariateOptimizer*, int, double, bool, double, double,
					double, double>(), "local"_a, "mit"_a, "tol"_a, "print"_a =
					false, "tau"_a = 0.5, "gamma"_a = 10., "lambda0"_a = 0.,
			"mu0"_a = 0.);
}

// wrapper for constraints
typedef std::function<double(const py::array_t<double>&)> multivariate_wrapper;
typedef std::function<bool(const py::array_t<double>&)> constraints_wrapper;
typedef std::function<py::array_t<double>(const py::array_t<double>&)> structured_constraints_wrapper;

// build base classes
void build_constrained(py::module_ &m) {

	// build object to hold solution
	py::class_<constrained_solution> sol(m, "ConstrainedSolution");
	sol.def("toString", &constrained_solution::toString);
	sol.def_property_readonly("sol", [](constrained_solution &self) {
		const auto &vec = self._sol;
		return py::array_t<double>(vec.size(), &vec[0]);
	});
	sol.def_property_readonly("converged", [](constrained_solution &self) {
		return self._converged;
	});
	sol.def_property_readonly("fev", [](constrained_solution &self) {
		return self._fev;
	});
	sol.def_property_readonly("gev", [](constrained_solution &self) {
		return self._gev;
	});

	// build base optimizer objects
	py::class_<ConstrainedOptimizer> optimizer(m, "ConstrainedOptimizer");
	optimizer.def("optimize",
			[](ConstrainedOptimizer &self, multivariate_wrapper f,
					constraints_wrapper g, py::array_t<double> &guess,
					py::array_t<double> &lower, py::array_t<double> &upper) {

				// cast numpy array -> double *
				const py::buffer_info &guess_info = guess.request();
				double *guess_ptr = static_cast<double*>(guess_info.ptr);
				const py::buffer_info &lower_info = lower.request();
				double *lower_ptr = static_cast<double*>(lower_info.ptr);
				const py::buffer_info &upper_info = upper.request();
				double *upper_ptr = static_cast<double*>(upper_info.ptr);
				const int n = guess.size();

				// cast [f : numpy array -> double] -> [f : double * -> double]
				const multivariate &fc = [&f, &n](const double *x) -> double {
					const auto &arr = py::array_t<double>(n, x);
					return f(arr);
				};

				// cast [g : numpy array -> bool] -> [g : double * -> bool]
				const constraints &gc = [&g, &n](const double *x) -> double {
					const auto &arr = py::array_t<double>(n, x);
					return g(arr);
				};

				// dispatch to C++ routine
				return self.optimize(fc, gc, n, guess_ptr, lower_ptr, upper_ptr);
			}
			, py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());
	optimizer.def("initialize",
			[](ConstrainedOptimizer &self, multivariate_wrapper f,
					constraints_wrapper g, py::array_t<double> &guess,
					py::array_t<double> &lower, py::array_t<double> &upper) {

				// cast numpy array -> double *
				const py::buffer_info &guess_info = guess.request();
				double *guess_ptr = static_cast<double*>(guess_info.ptr);
				const py::buffer_info &lower_info = lower.request();
				double *lower_ptr = static_cast<double*>(lower_info.ptr);
				const py::buffer_info &upper_info = upper.request();
				double *upper_ptr = static_cast<double*>(upper_info.ptr);
				const int n = guess.size();

				// cast [f : numpy array -> double] -> [f : double * -> double]
				const multivariate &fc = [&f, &n](const double *x) -> double {
					const auto &arr = py::array_t<double>(n, x);
					return f(arr);
				};

				// cast [g : numpy array -> bool] -> [g : double * -> bool]
				const constraints &gc = [&g, &n](const double *x) -> double {
					const auto &arr = py::array_t<double>(n, x);
					return g(arr);
				};

				// dispatch to C++ routine
				return self.init(fc, gc, n, guess_ptr, lower_ptr, upper_ptr);
			}
			, py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());
	optimizer.def("iterate", &ConstrainedOptimizer::iterate,
			py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());

	// put algorithm-specific bindings here
	build_constrained_amp(m);
	build_box(m);
	build_mads(m);
}

void build_structured_constrained(py::module_ &m) {

	// constrained optimizer with structured constraints
	py::class_<StructuredConstrainedOptimizer> optimizer(m,
			"StructuredConstrainedOptimizer");
	optimizer.def("optimize",
			[](StructuredConstrainedOptimizer &self, multivariate_wrapper f,
					structured_constraints_wrapper g, int neq, int nineq,
					py::array_t<double> &guess, py::array_t<double> &lower,
					py::array_t<double> &upper) {

				// cast numpy array -> double *
				const py::buffer_info &guess_info = guess.request();
				double *guess_ptr = static_cast<double*>(guess_info.ptr);
				const py::buffer_info &lower_info = lower.request();
				double *lower_ptr = static_cast<double*>(lower_info.ptr);
				const py::buffer_info &upper_info = upper.request();
				double *upper_ptr = static_cast<double*>(upper_info.ptr);
				const int n = guess.size();

				// cast [f : numpy array -> double] -> [f : double * -> double]
				const multivariate &fc = [&f, &n](const double *x) -> double {
					const auto &arr = py::array_t<double>(n, x);
					return f(arr);
				};

				// cast [g : numpy array -> void] -> [g : double * -> void]
				const structured_constraints &gc = [&g, &n](const double *x,
						double *o) -> void {
					const auto &arr = py::array_t<double>(n, x);
					const auto &res = g(arr);
					double *res_ptr = static_cast<double*>(res.request().ptr);
					std::copy(res_ptr, res_ptr + res.size(), o);
				};

				// dispatch to C++ routine
				return self.optimize(fc, gc, n, neq, nineq, guess_ptr,
						lower_ptr, upper_ptr);
			}
			, py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());
	optimizer.def("initialize",
			[](StructuredConstrainedOptimizer &self, multivariate_wrapper f,
					structured_constraints_wrapper g, int neq, int nineq,
					py::array_t<double> &guess, py::array_t<double> &lower,
					py::array_t<double> &upper) {

				// cast numpy array -> double *
				const py::buffer_info &guess_info = guess.request();
				double *guess_ptr = static_cast<double*>(guess_info.ptr);
				const py::buffer_info &lower_info = lower.request();
				double *lower_ptr = static_cast<double*>(lower_info.ptr);
				const py::buffer_info &upper_info = upper.request();
				double *upper_ptr = static_cast<double*>(upper_info.ptr);
				const int n = guess.size();

				// cast [f : numpy array -> double] -> [f : double * -> double]
				const multivariate &fc = [&f, &n](const double *x) -> double {
					const auto &arr = py::array_t<double>(n, x);
					return f(arr);
				};

				// cast [g : numpy array -> numpy array] -> [g : double * -> void]
				const structured_constraints &gc = [&g, &n](const double *x,
						double *o) -> void {
					const auto &arr = py::array_t<double>(n, x);
					const auto &res = g(arr);
					double *res_ptr = static_cast<double*>(res.request().ptr);
					std::copy(res_ptr, res_ptr + res.size(), o);
				};

				// dispatch to C++ routine
				return self.init(fc, gc, n, neq, nineq, guess_ptr, lower_ptr,
						upper_ptr);
			}
			, py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());
	optimizer.def("iterate", &StructuredConstrainedOptimizer::iterate,
			py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());

	// put algorithm-specific bindings here
	build_algencan(m);
}
