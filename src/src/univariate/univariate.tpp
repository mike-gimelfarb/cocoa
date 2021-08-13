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
 */

template<typename T> static bracket<T> bracketSolution(univariate<T> f,
		const T x0, const T x1, const int mfev, const T factor,
		const T leftBound, const T rightBound) {

	// initialize second point of bracket
	T a = x0;
	T b = x1;
	T fa = f(a);
	T fb = f(b);
	int fev = 2;

	// initialize descent direction
	T c, fc;
	if (fa < fb) {
		c = a;
		fc = fa;
		a = b;
		b = c;
		fb = fc;
	}

	// move towards descent direction
	c = std::max(leftBound, std::min(b + factor * (b - a), rightBound));
	fc = f(c);
	fev++;

	// continue adjusting the bracket until a minimum is bracketed
	bool converged = true;
	while (fc <= fb) {

		// move towards descent direction
		const T d = std::max(leftBound,
				std::min(c + factor * (c - b), rightBound));

		// out of bounds or reached budget
		if (fev >= mfev || (a == b && b == c && c == d)) {
			converged = false;
			break;
		}

		// update the bracket interval
		const T fd = f(d);
		fev++;
		a = b;
		b = c;
		fb = fc;
		c = d;
		fc = fd;
	}

	// finalize bracket
	const T left = std::min(a, c);
	const T right = std::max(a, c);
	return {left, right, fev, converged};
}

