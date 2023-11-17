#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "pybind11/numpy.h"
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "../src/multivariate/multivariate.h"
#include "../src/multivariate/amalgam/amalgam.h"
#include "../src/multivariate/amp/ttm.h"
#include "../src/multivariate/basin/basinhopping.h"
#include "../src/multivariate/cma/base_cmaes.h"
#include "../src/multivariate/cma/cmaes.h"
#include "../src/multivariate/cma/active_cmaes.h"
#include "../src/multivariate/cma/cholesky_cmaes.h"
#include "../src/multivariate/cma/lm_cmaes.h"
#include "../src/multivariate/cma/sep_cmaes.h"
#include "../src/multivariate/cma/ipop_cmaes.h"
#include "../src/multivariate/cma/bipop_cmaes.h"
#include "../src/multivariate/direct/directl.h"
#include "../src/multivariate/esch/esch.h"
#include "../src/multivariate/firefly/firefly.h"
#include "../src/multivariate/harmony/harmony.h"
#include "../src/multivariate/hees/hees.h"
#include "../src/multivariate/jaya/jaya.h"
#include "../src/multivariate/mayfly/mayfly.h"
#include "../src/multivariate/simplex/nelder_mead.h"
#include "../src/multivariate/pikaia/pikaia.h"
#include "../src/multivariate/powell/bobyqa.h"
#include "../src/multivariate/powell/newuoa.h"
#include "../src/multivariate/praxis/praxis.h"
#include "../src/multivariate/pso/ccpso.h"
#include "../src/multivariate/pso/cso.h"
#include "../src/multivariate/pso/ds.h"
#include "../src/multivariate/pso/pso.h"
#include "../src/multivariate/rosenbrock/rosenbrock.h"
#include "../src/multivariate/sade/sade.h"

namespace py = pybind11;
using namespace pybind11::literals;

void build_amalgam(py::module_ &m) {
	py::class_<Amalgam, MultivariateOptimizer> amalgam(m, "Amalgam");
	amalgam.def(py::init<int, double, double, int, bool, bool, bool>(),
			"mfev"_a, "tol"_a, "stol"_a, "np"_a = 0, "iamalgam"_a = true,
			"noparam"_a = true, "print"_a = true);
}

void build_amp(py::module_ &m) {
	py::class_<AMPTTM, MultivariateOptimizer> ampttm(m, "AMPTTM");

	py::enum_<AMPTTM::tabu_removal_strategy> tabu_removal(ampttm,
			"tabu_removal_strategy");
	tabu_removal.value("oldest", AMPTTM::tabu_removal_strategy::oldest);
	tabu_removal.value("farthest", AMPTTM::tabu_removal_strategy::farthest);
	tabu_removal.export_values();

	ampttm.def(
			py::init<MultivariateOptimizer*, int, bool, double, double, int,
					int, unsigned int, AMPTTM::tabu_removal_strategy>(),
			"local"_a, "mfev"_a, "print"_a = false, "eps1"_a = 0.02, "eps2"_a =
					0.1, "totaliter"_a = 9999, "maxiter"_a = 5, "tabutenure"_a =
					5, "remove"_a = AMPTTM::tabu_removal_strategy::farthest);
}

void build_basin_hopping(py::module_ &m) {
	py::class_<StepsizeStrategy> step(m, "StepsizeStrategy");
	step.def(py::init<double>(), "stepsize"_a);

	py::class_<AdaptiveStepsizeStrategy, StepsizeStrategy> adaptivestep(m,
			"AdaptiveStepsizeStrategy");
	adaptivestep.def(py::init<double, double, int, double>(), "stepsize"_a = 1.,
			"accept_rate"_a = 0.5, "interval"_a = 5, "factor"_a = 0.9);

	py::class_<BasinHopping, MultivariateOptimizer> basin(m, "BasinHopping");
	basin.def(
			py::init<MultivariateOptimizer*, StepsizeStrategy*, bool, int,
					double>(), "minimizer"_a, "stepstrat"_a, "print"_a = true,
			"mit"_a = 99, "temp"_a = 1.);
}

void build_cmaes(py::module_ &m) {
	py::class_<BaseCmaes, MultivariateOptimizer> basecma(m, "BaseCmaes");

	py::class_<Cmaes, BaseCmaes> cma(m, "Cmaes");
	cma.def(py::init<int, double, int, double, double>(), "mfev"_a, "tol"_a,
			"np"_a, "sigma0"_a = 2., "eigenrate"_a = 0.25);

	py::class_<ActiveCmaes, Cmaes> acma(m, "ActiveCmaes");
	acma.def(py::init<int, double, int, double, double, double>(), "mfev"_a,
			"tol"_a, "np"_a, "sigma0"_a = 2., "alphacov"_a = 2., "eigenrate"_a =
					0.25);

	py::class_<CholeskyCmaes, BaseCmaes> cholcma(m, "CholeskyCmaes");
	cholcma.def(py::init<int, double, double, int, double>(), "mfev"_a, "tol"_a,
			"stol"_a, "np"_a, "sigma0"_a = 2.);

	py::class_<LmCmaes, BaseCmaes> lmcma(m, "LmCmaes");
	lmcma.def(py::init<int, double, int, int, double, bool, bool>(), "mfev"_a,
			"tol"_a, "np"_a, "memory"_a = 0, "sigma0"_a = 2., "rademacher"_a =
					true, "usenew"_a = true);

	py::class_<SepCmaes, BaseCmaes> sepcma(m, "SepCmaes");
	sepcma.def(py::init<int, double, int, double, bool>(), "mfev"_a, "tol"_a,
			"np"_a, "sigma0"_a = 2., "adjustlr"_a = true);

	py::class_<IPopCmaes, MultivariateOptimizer> ipopcma(m, "IPopCmaes");
	ipopcma.def(py::init<BaseCmaes*, int, bool, double, bool, double, bool>(),
			"base"_a, "mfev"_a, "print"_a = false, "sigma0"_a = 2., "nipop"_a =
					true, "ksigmadec"_a = 1.6, "boundlambda"_a = true);

	py::class_<BiPopCmaes, MultivariateOptimizer> bipopcma(m, "BiPopCmaes");
	bipopcma.def(
			py::init<BaseCmaes*, int, bool, double, int, bool, double, double>(),
			"base"_a, "mfev"_a, "print"_a = false, "sigma0"_a = 2.,
			"maxlargeruns"_a = 9, "nbipop"_a = true, "ksigmadec"_a = 1.6,
			"kbudget"_a = 2.);
}

void build_directl(py::module_ &m) {
	py::class_<Directl, MultivariateOptimizer> direct(m, "Directl");
	direct.def(py::init<int, double, double, double, int>(), "mfev"_a,
			"volper"_a, "sigmaper"_a, "eps"_a = -1., "method"_a = 0);
}

void build_esch(py::module_ &m) {
	py::class_<EschSearch, MultivariateOptimizer>(m, "EschSearch").def(
			py::init<int, int, int>(), "mfev"_a, "np"_a, "no"_a);
}

void build_firefly(py::module_ &m) {
	py::class_<FireflySearch, MultivariateOptimizer> firefly(m,
			"FireflySearch");

	py::enum_<FireflySearch::noise_type> noise(firefly, "noise_type");
	noise.value("uniform", FireflySearch::noise_type::uniform);
	noise.value("gauss", FireflySearch::noise_type::gauss);
	noise.value("cauchy", FireflySearch::noise_type::cauchy);
	noise.value("none", FireflySearch::noise_type::none);
	noise.export_values();

	py::enum_<FireflySearch::search_strategy> search(firefly,
			"search_strategy");
	search.value("geometric", FireflySearch::search_strategy::geometric);
	search.value("sh2014", FireflySearch::search_strategy::sh2014);
	search.value("memetic", FireflySearch::search_strategy::memetic);
	search.export_values();

	firefly.def(
			py::init<int, int, double, double, double, double, double,
					FireflySearch::search_strategy, FireflySearch::noise_type,
					bool, int, bool, double>(), "mfev"_a, "np"_a, "gamma"_a,
			"alpha0"_a, "decay"_a = 0.97, "bmin"_a = 0.1, "bmax"_a = 0.9,
			"strategy"_a = FireflySearch::search_strategy::sh2014, "noise"_a =
					FireflySearch::noise_type::uniform, "nsearch"_a = true,
			"ns"_a = 2, "osearch"_a = true, "wbprob"_a = 0.25);
}

void build_harmony(py::module_ &m) {

	// harmony memory
	py::class_ < HMCR > (m, "HMCR");
	py::class_<HS_HMCR, HMCR>(m, "HS_HMCR").def(py::init<double>(), "hmcr"_a);
	py::class_<PSFHS_HMCR, HMCR> psfhs_hmcr(m, "PSFHS_HMCR");
	psfhs_hmcr.def(py::init<double, double, double, int, bool>(), "hmcrinit"_a =
			0.5, "hmcrmin"_a = 0.01, "hmcrmax"_a = 0.99, "warm"_a = 10,
			"local"_a = false);

	// pitch adjustment
	py::class_ < PAR > (m, "PAR");
	py::class_<HS_PAR, PAR>(m, "HS_PAR").def(py::init<double>(), "par"_a);
	py::class_<PSFHS_PAR, PAR> psfhs_par(m, "PSFHS_PAR");
	psfhs_par.def(py::init<double, double, double, int, bool>(), "parinit"_a =
			0.5, "parmin"_a = 0.01, "parmax"_a = 0.99, "warm"_a = 10,
			"local"_a = false);
	py::class_<IHS_PAR, PAR> ihs_par(m, "IHS_PAR");
	ihs_par.def(py::init<double, double>(), "parmin"_a, "parmax"_a);

	// pitch evolution
	py::class_ < PAStrategy > (m, "PAStrategy");
	py::class_<HS_PA, PAStrategy> hs_pa(m, "HS_PA");
	hs_pa.def(py::init<double>(), "bw"_a = 0.2);
	py::class_<IHS_PA, PAStrategy> ihs_pa(m, "IHS_PA");
	ihs_pa.def(py::init<double, double>(), "bwmin"_a = 0.01, "bwmax"_a = 0.99);
	py::class_<SHS_PA, PAStrategy>(m, "SHS_PA").def(py::init<>());
	py::class_<DHS_PA, PAStrategy>(m, "DHS_PA").def(py::init<double>(), "cr"_a);

	// harmony search class
	py::class_<HarmonySearch, MultivariateOptimizer> hs(m, "HarmonySearch");
	hs.def(py::init<int, int, int, HMCR, PAR, PAStrategy>(), "mfev"_a, "hms"_a,
			"hpi"_a = 5, "harmony"_a = PSFHS_HMCR(), "pitch"_a = PSFHS_PAR(),
			"pstrat"_a = IHS_PA());
}

void build_hees(py::module_ &m) {
	py::class_<Hees, MultivariateOptimizer> hees(m, "Hees");
	hees.def(py::init<int, double, int, bool, int, double>(), "mfev"_a, "tol"_a,
			"mres"_a = 1, "print"_a = false, "np"_a = 0, "sigma0"_a = 2.);
}

void build_jaya(py::module_ &m) {
	py::class_<JayaSearch, MultivariateOptimizer> jaya(m, "JayaSearch");
	jaya.def(py::init<int, int, int, int, double, double, bool>(), "mfev"_a,
			"np"_a, "npmin"_a, "k0"_a = 2, "scale"_a = 0.01, "beta"_a = 1.5,
			"adapt"_a = true);
}

void build_mayfly(py::module_ &m) {
	py::class_<MayflySearch, MultivariateOptimizer> mayfly(m, "MayflySearch");
	mayfly.def(
			py::init<int, int, double, double, double, double, double, double,
					double, double, double, double, double, double, double,
					double, bool>(), "np"_a, "mfev"_a, "a1"_a = 1., "a2"_a =
					1.5, "a3"_a = 1.5, "beta"_a = 2., "dance"_a = 5.,
			"ddamp"_a = 0.9, "fl"_a = 1., "fldamp"_a = 0.99, "gmin"_a = 0.8,
			"gmax"_a = 0.8, "vdamp"_a = 0.1, "pmutdim"_a = 0.01, "pmutnp"_a =
					0.05, "l"_a = 0.5, "pgb"_a = true);
}

void build_simplex(py::module_ &m) {
	py::class_<NelderMead, MultivariateOptimizer> neldermead(m, "NelderMead");

	py::enum_<NelderMead::simplex_initializer> simplex_init(neldermead,
			"simplex_initializer");
	simplex_init.value("original", NelderMead::simplex_initializer::original);
	simplex_init.value("spendley", NelderMead::simplex_initializer::spendley);
	simplex_init.value("pfeffer", NelderMead::simplex_initializer::pfeffer);
	simplex_init.value("random", NelderMead::simplex_initializer::random);
	simplex_init.export_values();

	neldermead.def(
			py::init<int, double, double, int, NelderMead::simplex_initializer,
					bool>(), "mfev"_a, "tol"_a, "rad0"_a, "checkev"_a = 10,
			"minit"_a = NelderMead::simplex_initializer::spendley, "adapt"_a =
					true);
}

void build_pikaia(py::module_ &m) {
	py::class_<PikaiaSearch, MultivariateOptimizer> pikaia(m, "PikaiaSearch");
	pikaia.def(
			py::init<int, int, int, double, int, double, double, double, double,
					int, int>(), "np"_a, "ngen"_a, "nd"_a, "pcross"_a = 0.85,
			"imut"_a = 2, "pmut"_a = 0.005, "pmutmn"_a = 0.0005, "pmutmx"_a =
					0.25, "fdif"_a = 1., "irep"_a = 1, "ielite"_a = 0);
}

void build_powell(py::module_ &m) {
	py::class_<Bobyqa, MultivariateOptimizer> bobyqa(m, "Bobyqa");
	bobyqa.def(py::init<int, int, double, double>(), "mfev"_a, "np"_a, "rho"_a,
			"tol"_a);

	py::class_<Newuoa, MultivariateOptimizer> newuoa(m, "Newuoa");
	newuoa.def(py::init<int, int, double, double>(), "mfev"_a, "np"_a, "rho"_a,
			"tol"_a);
}

void build_praxis(py::module_ &m) {
	py::class_<Praxis, MultivariateOptimizer> praxis(m, "Praxis");
	praxis.def(py::init<double, double>(), "tol"_a, "mstep"_a);
}

CcPsoSearch init_ccpso(int mfev, double tol, double sigmatol, int np,
		std::vector<int> &pps, bool correct, int update) {
	return CcPsoSearch(mfev, tol, sigmatol, np, &pps[0], pps.size(), correct,
			update);
}

void build_pso(py::module_ &m) {
	py::class_<CSOSearch, MultivariateOptimizer> cso(m, "CSOSearch");
	cso.def(py::init<int, double, double, int, bool, bool>(), "mfev"_a, "tol"_a,
			"sigmatol"_a, "np"_a, "ring"_a = false, "correct"_a = true);

	py::class_<CcPsoSearch, MultivariateOptimizer> ccpso(m, "CcPsoSearch");
	ccpso.def(py::init(&init_ccpso), "mfev"_a, "tol"_a, "sigmatol"_a, "np"_a,
			"pps"_a, "correct"_a = true, "update"_a = 30);

	py::class_<DifferentialSearch, MultivariateOptimizer> ds(m,
			"DifferentialSearch");
	ds.def(py::init<int, double, double, int>(), "mfev"_a, "tol"_a, "stol"_a,
			"np"_a);

	py::class_<PsoSearch, MultivariateOptimizer> pso(m, "PsoSearch");
	pso.def(py::init<int, double, double, int, bool>(), "mfev"_a, "tol"_a,
			"stol"_a, "np"_a, "correct"_a = true);
}

void build_rosenbrock(py::module_ &m) {
	py::class_<Rosenbrock, MultivariateOptimizer> rosen(m, "Rosenbrock");
	rosen.def(py::init<int, double, double, double>(), "mfev"_a, "tol"_a,
			"step0"_a, "decf"_a = 0.1);
}

void build_sade(py::module_ &m) {
	py::class_<SadeSearch, MultivariateOptimizer> sade(m, "SadeSearch");
	sade.def(py::init<int, double, double, int, int, int>(), "mfev"_a, "tol"_a,
			"stol"_a, "np"_a, "lp"_a = 25, "cp"_a = 5);
}

// wrapper for multivariate
typedef std::function<double(const py::array_t<double>&)> multivariate_wrapper;

// build base classes
void build_multivariate(py::module_ &m) {

	// build the base class for multivariate solution
	py::class_<multivariate_solution> sol(m, "MultivariateSolution");
	sol.def("toString", &multivariate_solution::toString);
	sol.def_property_readonly("sol", [](multivariate_solution &self) {
		const auto &vec = self._sol;
		return py::array_t<double>(vec.size(), &vec[0]);
	});
	sol.def_property_readonly("converged", [](multivariate_solution &self) {
		return self._converged;
	});
	sol.def_property_readonly("fev", [](multivariate_solution &self) {
		return self._fev;
	});

	// build the base class for multivariate optimizer
	py::class_<MultivariateOptimizer> optimizer(m, "MultivariateOptimizer");
	optimizer.def("optimize",
			[](MultivariateOptimizer &self, multivariate_wrapper f,
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

				// dispatch to C++ routine
				return self.optimize(fc, n, guess_ptr, lower_ptr, upper_ptr);
			}
			, py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());
	optimizer.def("initialize",
			[](MultivariateOptimizer &self, multivariate_wrapper f,
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

				// dispatch to C++ routine
				return self.init(fc, n, guess_ptr, lower_ptr, upper_ptr);
			}
			, py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());
	optimizer.def("iterate", &MultivariateOptimizer::iterate,
			py::call_guard<py::scoped_ostream_redirect,
					py::scoped_estream_redirect>());

	// put algorithm-specific bindings here
	build_amalgam(m);
	build_amp(m);
	build_basin_hopping(m);
	build_cmaes(m);
	build_directl(m);
	build_esch(m);
	build_firefly(m);
	build_harmony(m);
	build_hees(m);
	build_jaya(m);
	build_mayfly(m);
	build_pikaia(m);
	build_powell(m);
	build_praxis(m);
	build_pso(m);
	build_rosenbrock(m);
	build_sade(m);
	build_simplex(m);
}
