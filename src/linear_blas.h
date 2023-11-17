/*
 * linear_model.h
 */

#ifndef LINEAR_BLAS_H_
#define LINEAR_BLAS_H_

double r8mat_amax(int m, int n, double a[]);

void dswap(int n, double x[], int incx, double y[], int incy);

void dqrdc(double a[], int lda, int n, int p, double qraux[], int jpvt[],
		double work[], int job);

void dqrank(double a[], int lda, int m, int n, double tol, int &kr, int jpvt[],
		double qraux[], double work[]);

int dqrsl(double a[], int lda, int n, int k, double qraux[], double y[],
		double qy[], double qty[], double b[], double rsd[], double ab[],
		int job);

void dqrlss(double a[], int lda, int m, int n, int kr, double b[], double x[],
		double rsd[], int jpvt[], double qraux[]);

int dqrls(double a[], int lda, int m, int n, double tol, int &kr, double b[],
		double x[], double rsd[], int jpvt[], double qraux[], int itask,
		double work[]);

double qrsolm(int n, double a[], double b[], double out[], double work[],
		int iwork[]);

#endif /* LINEAR_BLAS_H_ */
