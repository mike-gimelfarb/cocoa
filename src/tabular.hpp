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

#ifndef TABULAR_HPP_
#define TABULAR_HPP_

#include <initializer_list>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "string_utils.h"

class Tabular {

private:
	struct tabular_container {
		std::string _str;

		template<typename T> tabular_container(const T &t) {
			_str = toStringFull(t);
		}

		tabular_container(const char *t) {
			_str = t;
		}

		tabular_container(const std::string &t) {
			_str = t;
		}
	};

protected:
	int _nr, _it, _nrsum;
	std::vector<int> _lens;

public:
	void setWidth(std::initializer_list<int> lens) {
		_lens = std::vector<int>(lens);
		_nr = _lens.size();
		_it = 0;
		_nrsum = std::accumulate(_lens.begin(), _lens.end(), 0);
	}

	template<class ... Tps> void printRow(Tps ... vals) {
		const std::vector<tabular_container> vec = { vals... };
		for (int i = 0; i < _nr; i++) {
			std::cout << " | " << std::setw(_lens[i]) << vec[i]._str;
		}
		std::cout << " | " << std::endl;
		if (_it == 0) {
			std::cout << " |" << std::string(_nrsum + 3 * (_nr - 1) + 2, '=')
					<< "| " << std::endl;
		}
		_it++;
	}
};

#endif /* TABULAR_HPP_ */
