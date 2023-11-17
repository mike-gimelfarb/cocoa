/*
 * linear_blas.cpp
 *
 *  Created on: Aug. 26, 2021
 *      Author: mgime
 */

#include <algorithm>
#include <cmath>

#include "blas.h"

#include "linear_blas.h"

double* r8mat_copy_new(int m, int n, double a1[]) {
	double *a2;
	a2 = new double[m * n];
	std::copy(a1, a1 + m * n, a2);
	return a2;
}

double r8mat_amax(int m, int n, double a[]) {
	int i, j;
	double value;
	value = std::fabs(a[0]);
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			value = std::fmax(value, std::fabs(a[i + j * m]));
		}
	}
	return value;
}

void dswap(int n, double x[], int incx, double y[], int incy) {
	int i, ix, iy, m;
	double temp;

	if (n <= 0) {
	} else if (incx == 1 && incy == 1) {
		m = n % 3;
		for (i = 0; i < m; i++) {
			temp = x[i];
			x[i] = y[i];
			y[i] = temp;
		}
		for (i = m; i < n; i = i + 3) {
			temp = x[i];
			x[i] = y[i];
			y[i] = temp;
			temp = x[i + 1];
			x[i + 1] = y[i + 1];
			y[i + 1] = temp;
			temp = x[i + 2];
			x[i + 2] = y[i + 2];
			y[i + 2] = temp;
		}
	} else {
		if (0 <= incx) {
			ix = 0;
		} else {
			ix = (-n + 1) * incx;
		}
		if (0 <= incy) {
			iy = 0;
		} else {
			iy = (-n + 1) * incy;
		}
		for (i = 0; i < n; i++) {
			temp = x[ix];
			x[ix] = y[iy];
			y[iy] = temp;
			ix = ix + incx;
			iy = iy + incy;
		}
	}
}

void dqrdc(double a[], int lda, int n, int p, double qraux[], int jpvt[],
		double work[], int job) {
	int j, jp, l, lup, maxj;
	double maxnrm, nrmxl;
	int pl, pu;
	bool swapj;
	double t, tt;
	pl = 1;
	pu = 0;

	//  If pivoting is requested, rearrange the columns.
	if (job != 0) {
		for (j = 1; j <= p; j++) {
			swapj = (0 < jpvt[j - 1]);
			if (jpvt[j - 1] < 0) {
				jpvt[j - 1] = -j;
			} else {
				jpvt[j - 1] = j;
			}
			if (swapj) {
				if (j != pl) {
					dswap(n, a + (pl - 1) * lda, 1, a + (j - 1), 1);
				}
				jpvt[j - 1] = jpvt[pl - 1];
				jpvt[pl - 1] = j;
				pl = pl + 1;
			}
		}
		pu = p;
		for (j = p; 1 <= j; j--) {
			if (jpvt[j - 1] < 0) {
				jpvt[j - 1] = -jpvt[j - 1];
				if (j != pu) {
					dswap(n, a + (pu - 1) * lda, 1, a + (j - 1) * lda, 1);
					jp = jpvt[pu - 1];
					jpvt[pu - 1] = jpvt[j - 1];
					jpvt[j - 1] = jp;
				}
				pu = pu - 1;
			}
		}
	}

	//  Compute the norms of the free columns.
	for (j = pl; j <= pu; j++) {
		qraux[j - 1] = dnrm2(n, a + (j - 1) * lda);
	}
	for (j = pl; j <= pu; j++) {
		work[j - 1] = qraux[j - 1];
	}

	//  Perform the Householder reduction of A.
	lup = std::min(n, p);
	for (l = 1; l <= lup; l++) {

		//  Bring the column of largest norm into the pivot position.
		if (pl <= l && l < pu) {
			maxnrm = 0.;
			maxj = l;
			for (j = l; j <= pu; j++) {
				if (maxnrm < qraux[j - 1]) {
					maxnrm = qraux[j - 1];
					maxj = j;
				}
			}
			if (maxj != l) {
				dswap(n, a + (l - 1) * lda, 1, a + (maxj - 1) * lda, 1);
				qraux[maxj - 1] = qraux[l - 1];
				work[maxj - 1] = work[l - 1];
				jp = jpvt[maxj - 1];
				jpvt[maxj - 1] = jpvt[l - 1];
				jpvt[l - 1] = jp;
			}
		}

		//  Compute the Householder transformation for column L.
		qraux[l - 1] = 0.;
		if (l != n) {
			nrmxl = dnrm2(n - l + 1, a + l - 1 + (l - 1) * lda);
			if (nrmxl != 0.) {
				if (a[l - 1 + (l - 1) * lda] != 0.) {
					if (a[l - 1 + (l - 1) * lda] < 0) {
						nrmxl *= -1;
					}
				}
				dscalm(n - l + 1, 1. / nrmxl, a + l - 1 + (l - 1) * lda, 1);
				a[l - 1 + (l - 1) * lda] = 1. + a[l - 1 + (l - 1) * lda];

				//  Apply the transformation to the remaining columns, updating the norms.
				for (j = l + 1; j <= p; j++) {
					t = -ddot(n - l + 1, a + l - 1 + (l - 1) * lda, 1,
							a + l - 1 + (j - 1) * lda, 1)
							/ a[l - 1 + (l - 1) * lda];
					daxpym(n - l + 1, t, a + l - 1 + (l - 1) * lda, 1,
							a + l - 1 + (j - 1) * lda, 1);
					if (pl <= j && j <= pu) {
						if (qraux[j - 1] != 0.) {
							tt = 1.
									- std::pow(
											std::fabs(a[l - 1 + (j - 1) * lda])
													/ qraux[j - 1], 2);
							tt = std::fmax(tt, 0.);
							t = tt;
							tt = 1.
									+ 0.05 * tt
											* std::pow(
													qraux[j - 1] / work[j - 1],
													2);
							if (tt != 1.) {
								qraux[j - 1] = qraux[j - 1] * std::sqrt(t);
							} else {
								qraux[j - 1] = dnrm2(n - l,
										a + l + (j - 1) * lda);
								work[j - 1] = qraux[j - 1];
							}
						}
					}
				}

				//  Save the transformation.
				qraux[l - 1] = a[l - 1 + (l - 1) * lda];
				a[l - 1 + (l - 1) * lda] = -nrmxl;
			}
		}
	}
}

void dqrank(double a[], int lda, int m, int n, double tol, int &kr, int jpvt[],
		double qraux[], double work[]) {
	int j, job, k;
	std::fill(jpvt, jpvt + n, 0);
	job = 1;
	dqrdc(a, lda, m, n, qraux, jpvt, work, job);
	kr = 0;
	k = std::min(m, n);
	for (j = 0; j < k; j++) {
		if (std::fabs(a[j + j * lda]) <= tol * std::fabs(a[0])) {
			return;
		}
		kr = j + 1;
	}
	return;
}

int dqrsl(double a[], int lda, int n, int k, double qraux[], double y[],
		double qy[], double qty[], double b[], double rsd[], double ab[],
		int job) {
	bool cab, cb, cqty, cqy, cr;
	int info, j, jj, ju;
	double t, temp;

	//  set info flag.
	info = 0;

	//  Determine what is to be computed.
	cqy = (job / 10000 != 0);
	cqty = ((job % 10000) != 0);
	cb = ((job % 1000) / 100 != 0);
	cr = ((job % 100) / 10 != 0);
	cab = ((job % 10) != 0);
	ju = std::min(k, n - 1);

	//  Special action when N = 1.
	if (ju == 0) {
		if (cqy) {
			qy[0] = y[0];
		}
		if (cqty) {
			qty[0] = y[0];
		}
		if (cab) {
			ab[0] = y[0];
		}
		if (cb) {
			if (a[0] == 0.) {
				info = 1;
			} else {
				b[0] = y[0] / a[0];
			}
		}
		if (cr) {
			rsd[0] = 0.;
		}
		return info;
	}

	//  Set up to compute QY or QTY.
	if (cqy) {
		std::copy(y, y + n, qy);
	}
	if (cqty) {
		std::copy(y, y + n, qty);
	}

	//  Compute QY.
	if (cqy) {
		for (jj = 1; jj <= ju; jj++) {
			j = ju - jj + 1;
			if (qraux[j - 1] != 0.) {
				temp = a[j - 1 + (j - 1) * lda];
				a[j - 1 + (j - 1) * lda] = qraux[j - 1];
				t = -ddot(n - j + 1, a + j - 1 + (j - 1) * lda, 1, qy + j - 1,
						1) / a[j - 1 + (j - 1) * lda];
				daxpym(n - j + 1, t, a + j - 1 + (j - 1) * lda, 1, qy + j - 1,
						1);
				a[j - 1 + (j - 1) * lda] = temp;
			}
		}
	}

	//  Compute Q'*Y.
	if (cqty) {
		for (j = 1; j <= ju; j++) {
			if (qraux[j - 1] != 0.) {
				temp = a[j - 1 + (j - 1) * lda];
				a[j - 1 + (j - 1) * lda] = qraux[j - 1];
				t = -ddot(n - j + 1, a + j - 1 + (j - 1) * lda, 1, qty + j - 1,
						1) / a[j - 1 + (j - 1) * lda];
				daxpym(n - j + 1, t, a + j - 1 + (j - 1) * lda, 1, qty + j - 1,
						1);
				a[j - 1 + (j - 1) * lda] = temp;
			}
		}
	}

	//  Set up to compute B, RSD, or AB.
	if (cb) {
		std::copy(qty, qty + k, b);
	}
	if (cab) {
		std::copy(qty, qty + k, ab);
	}
	if (cr && k < n) {
		std::copy(qty + k, qty + n, rsd + k);
	}
	if (cab && k + 1 <= n) {
		std::fill(ab + k, ab + n, 0.);
	}
	if (cr) {
		std::fill(rsd, rsd + k, 0.);
	}

	//  Compute B.
	if (cb) {
		for (jj = 1; jj <= k; jj++) {
			j = k - jj + 1;
			if (a[j - 1 + (j - 1) * lda] == 0.) {
				info = j;
				break;
			}
			b[j - 1] = b[j - 1] / a[j - 1 + (j - 1) * lda];
			if (j != 1) {
				t = -b[j - 1];
				daxpym(j - 1, t, a + (j - 1) * lda, 1, b, 1);
			}
		}
	}

	//  Compute RSD or AB as required.
	if (cr || cab) {
		for (jj = 1; jj <= ju; jj++) {
			j = ju - jj + 1;
			if (qraux[j - 1] != 0.) {
				temp = a[j - 1 + (j - 1) * lda];
				a[j - 1 + (j - 1) * lda] = qraux[j - 1];
				if (cr) {
					t = -ddot(n - j + 1, a + j - 1 + (j - 1) * lda, 1,
							rsd + j - 1, 1) / a[j - 1 + (j - 1) * lda];
					daxpym(n - j + 1, t, a + j - 1 + (j - 1) * lda, 1,
							rsd + j - 1, 1);
				}
				if (cab) {
					t = -ddot(n - j + 1, a + j - 1 + (j - 1) * lda, 1,
							ab + j - 1, 1) / a[j - 1 + (j - 1) * lda];
					daxpym(n - j + 1, t, a + j - 1 + (j - 1) * lda, 1,
							ab + j - 1, 1);
				}
				a[j - 1 + (j - 1) * lda] = temp;
			}
		}
	}
	return info;
}

void dqrlss(double a[], int lda, int m, int n, int kr, double b[], double x[],
		double rsd[], int jpvt[], double qraux[]) {
	int i, j, job, k;
	double t;
	if (kr != 0) {
		job = 110;
		dqrsl(a, lda, m, kr, qraux, b, rsd, rsd, x, rsd, rsd, job);
	}
	for (i = 0; i < n; i++) {
		jpvt[i] = -jpvt[i];
	}
	std::fill(x + kr, x + n, 0.);
	for (j = 1; j <= n; j++) {
		if (jpvt[j - 1] <= 0) {
			k = -jpvt[j - 1];
			jpvt[j - 1] = k;
			while (k != j) {
				t = x[j - 1];
				x[j - 1] = x[k - 1];
				x[k - 1] = t;
				jpvt[k - 1] = -jpvt[k - 1];
				k = jpvt[k - 1];
			}
		}
	}
}

int dqrls(double a[], int lda, int m, int n, double tol, int &kr, double b[],
		double x[], double rsd[], int jpvt[], double qraux[], int itask,
		double work[]) {
	int ind;
	if (lda < m) {
		ind = -1;
		return ind;
	}
	if (n <= 0) {
		ind = -2;
		return ind;
	}
	if (itask < 1) {
		ind = -3;
		return ind;
	}
	ind = 0;
	if (itask == 1) {
		dqrank(a, lda, m, n, tol, kr, jpvt, qraux, work);
	}
	dqrlss(a, lda, m, n, kr, b, x, rsd, jpvt, qraux);
	return ind;
}

double qrsolm(int n, double a[], double b[], double out[], double work[],
		int iwork[]) {

	// work at least n * n + 3 * n, iwork at least n
	int itask;
	int kr, lda;
	double tol;
	std::copy(a, a + n * n, work);
	lda = n;
	tol = 2.220446049250313E-016 / r8mat_amax(n, n, work);
	itask = 1;

	dqrls(work, lda, n, n, tol, kr, b, out, work + n * n, iwork,
			work + n * n + n, itask, work + n * n + 2 * n);

	double err = 0.;
	for (int i = 0; i < n; i++) {
		double rsum = 0.;
		for (int j = 0; j < n; j++) {
			rsum += a[i + n * j] * out[j];
		}
		err = std::max(err, std::fabs(rsum - b[i]));
	}
	return err;
}
