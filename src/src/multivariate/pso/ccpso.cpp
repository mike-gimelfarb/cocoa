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

 [1] Li, Xiaodong, and Xin Yao. "Cooperatively coevolving particle swarms for
 large scale optimization." IEEE Transactions on Evolutionary Computation 16.2
 (2012): 210-224.
 */

#include "ccpso.h"
#include "../../random.hpp"

using Random = effolkronium::random_static;

CcPsoSearch::CcPsoSearch(int mfev, double tol, double sigmatol, int np, // @suppress("Class members should be properly initialized")
		int *pps, int npps, bool correct, int update) {
	_tol = tol;
	_stol = sigmatol;
	_np = np;
	_pps = pps;
	_npps = npps;
	_mfev = mfev;
	_correct = correct;
	_update = update;
}

void CcPsoSearch::init(multivariate f, const int n, double *guess,
		double *lower, double *upper) {

	// initialize domain
	_f = f;
	_n = n;
	_lower = std::vector<double>(lower, lower + _n);
	_upper = std::vector<double>(upper, upper + _n);

	// uses a ring topology
	_topidx = std::vector<int>(3);
	_topfit = std::vector<double>(3);

	// initialize adaptive params
	_csucc = _cfail = _gsucc = _gfail = 0;
	_fx = 0.5;

	// initialize algorithm primitives and initialize swarms' components
	_fev = _gen = 0;
	randomizeComponents();

	// initialize the swarms
	_swarm.clear();
	for (int ip = 0; ip < _np; ip++) {

		// create particle
		std::vector<double> x(_n);
		std::vector<double> xpb(_n);
		auto pParticle = std::make_shared<ccpso_particle>(ccpso_particle { x,
				xpb, nullptr });
		_swarm.push_back(std::move(pParticle));
	}
	_sbest = std::vector<double>(_n);
	_xbest = std::vector<double>(_n);
	_work = std::vector<double>(_n);
	randomizeSwarm();
}

void CcPsoSearch::iterate() {

	// save the old best fitness value of swarms to track improvement
	const double oldf = _fbest;

	// update each swarm's personal bests
	for (int is = 0; is < _nswarm; is++) {
		updateSwarm(is);
	}

	// update each swarm's particle positions
	for (int is = 0; is < _nswarm; is++) {
		updatePosition(is);
	}

	// check if a randomization of the components is required
	if (_gen > 0 && _fbest == oldf) {
		randomizeComponents();
	}

	// check if we need to reset the counters for exploration
	updateParameters();
}

multivariate_solution CcPsoSearch::optimize(multivariate f, const int n,
		double *guess, double *lower, double *upper) {

	// initialization
	init(f, n, guess, lower, upper);

	// main loop
	bool converged = false;
	for (_gen = 0; _gen < 9999999; _gen++) {
		iterate();

		// check max number of evaluations
		if (_fev >= _mfev) {
			break;
		}

		// converge when distance in fitness between best and worst points
		// is below the given tolerance
		double bestFit = std::numeric_limits<double>::infinity();
		double worstFit = -std::numeric_limits<double>::infinity();
		for (int is = 0; is < _nswarm; is++) {
			for (int ip = 0; ip < _np; ip++) {
				const double fit = _pbfit[is][ip];
				bestFit = std::min(bestFit, fit);
				worstFit = std::max(worstFit, fit);
			}
		}
		const double dy = std::fabs(bestFit - worstFit);
		if (dy <= _tol) {

			// compute standard deviation of swarm radiuses
			int count = 0;
			double mean = 0., m2 = 0.;
			for (auto &pt : _swarm) {
				const double x = std::sqrt(
						std::inner_product(pt->_x.begin(), pt->_x.end(),
								pt->_x.begin(), 0.));
				count++;
				const double delta = x - mean;
				mean += delta / count;
				const double delta2 = x - mean;
				m2 += delta * delta2;
			}

			// test convergence in standard deviation
			if (m2 <= (_np - 1) * _stol * _stol) {
				converged = true;
				break;
			}
		}
	}
	return {_xbest, _fev, converged};
}

void CcPsoSearch::randomizeComponents() {

	// sample an s at random, the number of components per swarm
	_is = Random::get(0, _npps - 1);
	_cpswarm = _pps[_is];
	_nswarm = _n / _cpswarm;

	// re-dimension all arrays
	_k.clear();
	_k.resize(_nswarm, std::vector<int>(_cpswarm, 0));
	_pbfit.clear();
	_pbfit.resize(_nswarm, std::vector<double>(_np, 0.));
	_cauchy.clear();
	_cauchy.resize(_nswarm, std::vector<bool>(_np, false));

	// initialize the component indices for each swarm
	std::vector<int> range(_n);
	for (int i = 0; i < _n; i++) {
		range[i] = i;
	}
	Random::shuffle(range.begin(), range.end());
	int i = 0;
	for (auto &k : _k) {
		for (int j = 0; j < _cpswarm; j++) {
			k[j] = range[i];
			i++;
		}
	}
}

void CcPsoSearch::randomizeSwarm() {

	// initialize the particles in all swarms in range [lb, ub]
	for (auto &p : _swarm) {
		for (int i = 0; i < _n; i++) {
			p->_x[i] = Random::get(_lower[i], _upper[i]);
		}
		std::copy((p->_x).begin(), (p->_x).end(), (p->_xpb).begin());
	}

	// compute the fitness of all particles and get the global best particle
	_fbest = std::numeric_limits<double>::infinity();
	int bestip = -1;
	int ip = 0;
	for (auto &p : _swarm) {
		const double fit = _f(&(p->_x)[0]);
		_fev++;
		if (fit < _fbest) {
			_fbest = fit;
			bestip = ip;
		}
		ip++;
	}

	// set the swarm's best positions and global best position
	std::copy((_swarm[bestip]->_x).begin(), (_swarm[bestip]->_x).end(),
			_sbest.begin());
	std::copy((_swarm[bestip]->_x).begin(), (_swarm[bestip]->_x).end(),
			_xbest.begin());
}

void CcPsoSearch::updateSwarm(int i) {

	// update particle personal bests
	double fPyhat = evaluate(i, &_sbest[0]);
	int ip = 0;
	for (auto &p : _swarm) {

		// compute fitness
		const double fPx = evaluate(i, &(p->_x)[0]);
		_pbfit[i][ip] = evaluate(i, &(p->_xpb)[0]);

		// perform update of the personal best
		if (fPx < _pbfit[i][ip]) {
			for (int k : _k[i]) {
				p->_xpb[k] = p->_x[k];
			}
			_pbfit[i][ip] = fPx;

			// update strategy probabilities for Cauchy/Gaussian sampling
			if (_cauchy[i][ip]) {
				_csucc++;
			} else {
				_gsucc++;
			}
		} else {

			// update strategy probabilities for Cauchy/Gaussian sampling
			if (_cauchy[i][ip]) {
				_cfail++;
			} else {
				_gfail++;
			}
		}

		// perform update of the swarm best
		if (_pbfit[i][ip] < fPyhat) {
			for (int k : _k[i]) {
				_sbest[k] = p->_xpb[k];
			}
			fPyhat = _pbfit[i][ip];
		}
		ip++;
	}

	// update particle local best positions
	ip = 0;
	for (auto &p : _swarm) {

		// get the fitness values in neighborhood of particle ip
		_topidx[0] = (ip - 1 + _np) % _np;
		_topidx[1] = ip;
		_topidx[2] = (ip + 1) % _np;
		for (int k = 0; k < 3; k++) {
			_topfit[k] = _pbfit[i][_topidx[k]];
		}

		// get the best local particle among neighbors
		const int imin = std::min_element(_topfit.begin(), _topfit.end())
				- _topfit.begin();
		p->_xlb = &(_swarm[_topidx[imin]]->_xpb)[0];
		ip++;
	}

	// update global best vector position and fitness
	if (fPyhat < _fbest) {
		for (int k : _k[i]) {
			_xbest[k] = _sbest[k];
		}
		_fbest = _f(&_xbest[0]);
		_fev++;
	}
}

void CcPsoSearch::updatePosition(int i) {
	int ip = 0;
	for (auto &p : _swarm) {

		// decide whether the next sample will come from a Cauchy or Gaussian
		// distribution
		const double rand = Random::get(0., 1.);
		const bool cauchy = rand <= _fx;

		// evolve the particle
		if (cauchy) {
			for (int k : _k[i]) {
				const double c = sampleCauchy();
				const double dist = p->_xpb[k] - p->_xlb[k];
				p->_x[k] = p->_xpb[k] + c * std::fabs(dist);
			}
		} else {
			for (int k : _k[i]) {
				const double c = Random::get(_Z);
				const double dist = p->_xpb[k] - p->_xlb[k];
				p->_x[k] = p->_xlb[k] + c * std::fabs(dist);
			}
		}
		_cauchy[i][ip] = cauchy;

		// apply bounds constraint
		if (_correct) {
			for (int k : _k[i]) {
				if (p->_x[k] < _lower[k] || p->_x[k] > _upper[k]) {
					p->_x[k] = Random::get(_lower[k], _upper[k]);
				}
			}
		}
		ip++;
	}
}

void CcPsoSearch::updateParameters() {
	if (_gen >= _fupdate && _gen % _fupdate == 0) {

		// update F
		if (_csucc + _cfail > 0 && _gsucc + _gfail > 0 && _csucc > 0
				&& _gsucc > 0) {
			const double p1 = (1. * _csucc) / (_csucc + _cfail);
			const double p2 = (1. * _gsucc) / (_gsucc + _gfail);
			_fx = p1 / (p1 + p2);
			_fx = std::max(0.05, std::min(_fx, 0.95));
		}

		// reset counters
		_csucc = _cfail = _gsucc = _gfail = 0;
	}
}

double CcPsoSearch::evaluate(int i, double *z) {

	// cache the swarm best position currently for swarm is
	// then change the component values for swarm is to z
	for (const int i : _k[i]) {
		_work[i] = _sbest[i];
		_sbest[i] = z[i];
	}

	// evaluate function at the modified vector
	const double fit = _f(&_sbest[0]);
	_fev++;

	// restore the swarm best position
	for (const int i : _k[i]) {
		_sbest[i] = _work[i];
	}
	return fit;
}

double CcPsoSearch::sampleCauchy() {
	return std::tan(M_PI * (Random::get(0., 1.) - 0.5));
}
