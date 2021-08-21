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

template<typename T> T rescale(const T x, const T a, const T b) {
	return a + x * (b - a);
}

template<typename T> T sign(const T x, const T y) {
	return y >= T(0) ? std::fabs(x) : -std::fabs(x);
}

template<typename T> T ulp() {
	return std::nextafter(T(0.), T(0.));
}

template<typename T> int sgn(const T val) {
	return (T(0) < val) - (val < T(0));
}

template<typename T> T hypot(const T a, const T b) {
	const T absa = std::abs(a);
	const T absb = std::abs(b);
	T r = T(0.);
	if (absa > absb) {
		r = b / a;
		r = absa * std::sqrt(T(1.) + r * r);
	} else if (b != 0.) {
		r = a / b;
		r = absb * std::sqrt(T(1.) + r * r);
	}
	return r;
}
