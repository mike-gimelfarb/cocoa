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

 [1] Askarzadeh, Alireza & Rashedi, Esmat. (2017). Harmony Search Algorithm.
 10.4018/978-1-5225-2322-2.ch001.

 [2] Chakraborty, P., Roy, G. G., Das, S., Jain, D., & Abraham, A. (2009). An improved
 harmony search algorithm with differential mutation operator. Fundamenta Informaticae,
 95, 1–26

 [3] Geem, Z. W., & Sim, K. B. (2010). Parameter-setting-free harmony search algorithm.
 Applied Mathematics and Computation, 217(8), 3881–3889. doi:10.1016/j.amc.2010.09.049

 [4] Mahdavi, M., Fesanghary, M., & Damangir, E. (2007). An improved harmony search
 algorithm for solving optimization problems. Applied Mathematics and Computation, 188(2),
 1567–1579. doi:10.1016/j.amc.2006.11.033

 [5] Wang, C. M., & Huang, Y. F. (2010). Self-adaptive harmony search algorithm for
 optimization. Expert Systems with Applications, 37(4), 2826–2837. doi:10.1016/j.
 eswa.2009.09.008

 [6] Woo Z, Hoon J, Loganathan GV. A New Heuristic Optimization Algorithm: Harmony Search.
 SIMULATION. 2001;76(2):60-68. doi:10.1177/003754970107600201
 */

#ifndef MULTIVARIATE_EVOL_HARMONY_H_
#define MULTIVARIATE_EVOL_HARMONY_H_

#include <memory>

#include "../multivariate.h"

// harmony memory parameter HMCR strategy
struct HMCR {
	bool _local;
	int _warm;
	double _hmcrinit, _hmcrmin, _hmcrmax;
	std::string _name = "";

	virtual ~HMCR() {
	}
};

struct HS_HMCR: HMCR {
	HS_HMCR(double hmcr) {
		_hmcrinit = hmcr;
		_name = "fixed";
	}
};

struct PSFHS_HMCR: HMCR {
	PSFHS_HMCR(double hmcrinit = 0.5, double hmcrmin = 0.01, double hmcrmax =
			0.99, int warm = 10, bool local = false) {
		_hmcrinit = hmcrinit;
		_hmcrmin = hmcrmin;
		_hmcrmax = hmcrmax;
		_warm = warm;
		_local = local;
		_name = "none";
	}
};

// pitch adjustment parameter PAR strategy
struct PAR {
	bool _local;
	int _warm;
	double _parinit, _parmin, _parmax;
	std::string _name = "";

	virtual ~PAR() {
	}
};

struct HS_PAR: PAR {
	HS_PAR(double par) {
		_parinit = par;
		_name = "fixed";
	}
};

struct PSFHS_PAR: PAR {
	PSFHS_PAR(double parinit = 0.5, double parmin = 0.01, double parmax = 0.99,
			int warm = 10, bool local = false) {
		_parinit = parinit;
		_parmin = parmin;
		_parmax = parmax;
		_warm = warm;
		_local = local;
		_name = "none";
	}
};

struct IHS_PAR: PAR {
	IHS_PAR(double parmin, double parmax) {
		_parmin = parmin;
		_parmax = parmax;
		_name = "improved";
	}
};

// strategy for evolving new harmony
struct PAStrategy {
	double _bwinit, _bwmin, _bwmax, _cr;
	std::string _name = "";

	virtual ~PAStrategy() {
	}
};

struct HS_PA: PAStrategy {
	HS_PA(double bw) {
		_bwinit = bw;
		_name = "fixed";
	}
};

struct IHS_PA: PAStrategy {
	IHS_PA(double bwmin = 0.01, double bwmax = 0.99) {
		_bwmin = bwmin;
		_bwmax = bwmax;
		_name = "improved";
	}
};

struct SHS_PA: PAStrategy {
	SHS_PA() {
		_name = "sa";
	}
};

struct DHS_PA: PAStrategy {
	DHS_PA(double cr) {
		_cr = cr;
		_name = "de";
	}
};

enum harmony_op {
	random, pitch, memory
};

struct harmony {

	std::vector<double> _x;
	double _f;
	std::vector<harmony_op> _op;

	static bool compare_fitness(const std::shared_ptr<harmony> &x,
			const std::shared_ptr<harmony> &y) {
		return x->_f < y->_f;
	}
};

class HarmonySearch: public MultivariateOptimizer {

protected:
	int _hms, _hpi, _mfev, _n, _fev, _it, _mit;
	double _bw;
	multivariate _f;
	HMCR _harmony;
	PAR _pitch;
	PAStrategy _pstrat;
	std::shared_ptr<harmony> _best, _temp;
	std::vector<double> _lower, _upper, _hmcr, _par;
	std::vector<std::shared_ptr<harmony>> _hm;

public:
	HarmonySearch(int mfev, int hms, int hpi, HMCR harmony, PAR pitch,
			PAStrategy pstrat);

	void init(multivariate f, const int n, double *guess, double *lower,
			double *upper);

	void iterate();

	multivariate_solution optimize(multivariate f, const int n, double *guess,
			double *lower, double *upper);

private:
	void improvise();

	void replace();

	void adaptParams();
};

#endif /* MULTIVARIATE_EVOL_HARMONY_H_ */
