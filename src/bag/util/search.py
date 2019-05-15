# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides search related utilities.
"""

from typing import Optional, Callable, Any, Container, Iterable

from collections import namedtuple

MinCostResult = namedtuple('MinCostResult', ['x', 'xmax', 'vmax', 'nfev'])


class BinaryIterator(object):
    """A class that performs binary search over integers.

    This class supports both bounded or unbounded binary search, and
    you can also specify a step size.

    Parameters
    ----------
    low : int
        the lower bound (inclusive).
    high : Optional[int]
        the upper bound (exclusive).  None for unbounded binary search.
    step : int
        the step size.  All return values will be low + N * step
    """

    def __init__(self, low, high, step=1):
        # type: (int, Optional[int], int) -> None

        if not isinstance(low, int) or not isinstance(step, int):
            raise ValueError('low and step must be integers.')

        self._offset = low
        self._step = step
        self._high = None  # type: Optional[int]
        self._low = 0  # type: int
        self._current = 0  # type: int
        self._save_marker = None  # type: Optional[int]

        if high is not None:
            if not isinstance(high, int):
                raise ValueError('high must be None or integer.')

            nmax = (high - low) // step
            if low + step * nmax < high:
                nmax += 1
            self._high = nmax
            self._current = (self._low + self._high) // 2
        else:
            self._high = None
            self._current = self._low

        self._save_marker = None
        self._save_info = None

    def set_current(self, val):
        # type: (int) -> None
        """Set the value of the current marker."""
        if (val - self._offset) % self._step != 0:
            raise ValueError('value %d is not multiple of step size.' % val)
        self._current = (val - self._offset) // self._step

    def has_next(self):
        # type: () -> bool
        """returns True if this iterator is not finished yet."""
        return self._high is None or self._low < self._high

    def get_next(self):
        # type: () -> int
        """Returns the next value to look at."""
        return self._current * self._step + self._offset

    def up(self):
        # type: () -> None
        """Increment this iterator."""
        self._low = self._current + 1

        if self._high is not None:
            self._current = (self._low + self._high) // 2
        else:
            if self._current > 0:
                self._current *= 2
            else:
                self._current = 1

    def down(self):
        # type: () -> None
        """Decrement this iterator."""
        self._high = self._current
        self._current = (self._low + self._high) // 2

    def save(self):
        # type: () -> None
        """Save the current index."""
        self._save_marker = self._current

    def save_info(self, info):
        # type: (Any) -> None
        """Save current information."""
        self.save()
        self._save_info = info

    def get_last_save(self):
        # type: () -> Optional[int]
        """Returns the last saved index."""
        if self._save_marker is None:
            return None
        return self._save_marker * self._step + self._offset

    def get_last_save_info(self):
        # type: () -> Any
        """Return last save information."""
        return self._save_info


class FloatBinaryIterator(object):
    """A class that performs binary search over floating point numbers.

    This class supports both bounded or unbounded binary search, and terminates
    when we can guarantee the given error tolerance.

    Parameters
    ----------
    low : float
        the lower bound.
    high : Optional[float]
        the upper bound.  None for unbounded binary search.
    tol : float
        we will guarantee that the final solution will be within this
        tolerance.
    search_step : float
        for unbounded binary search, this is the initial step size when
        searching for upper bound.
    """

    def __init__(self, low, high, tol=1.0, search_step=1.0):
        # type: (float, Optional[float], float, float) -> None
        self._offset = low
        self._tol = tol
        self._high = None  # type: Optional[float]
        self._low = 0.0  # type: float
        self._search_step = search_step
        self._save_marker = None  # type: Optional[float]

        if high is not None:
            self._high = high - low
            self._current = self._high / 2
        else:
            self._high = None
            self._current = 0

        self._save_marker = None
        self._save_info = None

    def has_next(self):
        # type: () -> bool
        """returns True if this iterator is not finished yet."""
        return self._high is None or self._low + 2 * self._tol < self._high

    def get_next(self):
        # type: () -> float
        """Returns the next value to look at."""
        return self._current + self._offset

    def up(self):
        # type: () -> None
        """Increment this iterator."""
        self._low = self._current

        if self._high is not None:
            self._current = (self._low + self._high) / 2
        else:
            if self._current != 0:
                self._current *= 2
            else:
                self._current = self._search_step

    def down(self):
        # type: () -> None
        """Decrement this iterator."""
        self._high = self._current
        self._current = (self._low + self._high) / 2

    def save(self):
        # type: () -> None
        """Save the current index"""
        self._save_marker = self._current

    def save_info(self, info):
        # type: (Any) -> None
        """Save current information."""
        self.save()
        self._save_info = info

    def get_last_save(self):
        # type: () -> Optional[float]
        """Returns the last saved index."""
        if self._save_marker is None:
            return None
        return self._save_marker + self._offset

    def get_last_save_info(self):
        # type: () -> Any
        """Return last save information."""
        return self._save_info


def _contains(test_name, container_list):
    # type: (str, Iterable[Container[str]]) -> bool
    """Returns true if test_name is in any container."""
    for container in container_list:
        if test_name in container:
            return True
    return False


def get_new_name(base_name, *args):
    # type: (str, *Container[str]) -> str
    """Generate a new unique name.

    This method appends an index to the given basename.  Binary
    search is used to achieve logarithmic run time.

    Parameters
    ----------
    base_name : str
        the base name.
    *args : Container[str]
        a list of containers of used names.

    Returns
    -------
    new_name : str
        the unique name.
    """
    if not _contains(base_name, args):
        return base_name

    bin_iter = BinaryIterator(1, None)
    while bin_iter.has_next():
        new_name = '{}_{:d}'.format(base_name, bin_iter.get_next())
        if _contains(new_name, args):
            bin_iter.up()
        else:
            bin_iter.save_info(new_name)
            bin_iter.down()

    result = bin_iter.get_last_save_info()
    assert result is not None, 'binary search should find a solution'
    return result


def minimize_cost_binary(f,  # type: Callable[[int], float]
                         vmin,  # type: float
                         start=0,  # type: int
                         stop=None,  # type: Optional[int]
                         step=1,  # type: int
                         save=None,  # type: Optional[int]
                         nfev=0,  # type: int
                         ):
    # type: (...) -> MinCostResult
    """Minimize cost given minimum output constraint using binary search.

    Given discrete function f, find the minimum integer x such that f(x) >= vmin using
    binary search.

    This algorithm only works if f is monotonically increasing, or if f monontonically increases
    then monontonically decreases, but stop is given and f(stop) >= vmin.

    Parameters
    ----------
    f : Callable[[int], float]
        a function that takes a single integer and output a scalar value.  Must monotonically
        increase then monotonically decrease.
    vmin : float
        the minimum output value.
    start : int
        the input lower bound.
    stop : Optional[int]
        the input upper bound.  Use None for unbounded binary search.
    step : int
        the input step.  function will only be evaulated at the points start + step * N
    save : Optional[int]
        If not none, this value will be returned if no solution is found.
    nfev : int
        number of function calls already made.

    Returns
    -------
    result : MinCostResult
        the MinCostResult named tuple, with attributes:

        x : Optional[int]
            the minimum integer such that f(x) >= vmin.  If no such x exists, this will be None.
        nfev : int
            total number of function calls made.

    """
    bin_iter = BinaryIterator(start, stop, step=step)
    while bin_iter.has_next():
        x_cur = bin_iter.get_next()
        v_cur = f(x_cur)
        nfev += 1

        if v_cur >= vmin:
            save = x_cur
            bin_iter.down()
        else:
            bin_iter.up()
    return MinCostResult(x=save, xmax=None, vmax=None, nfev=nfev)


def minimize_cost_golden(f, vmin, offset=0, step=1, maxiter=1000):
    # type: (Callable[[int], float], float, int, int, Optional[int]) -> MinCostResult
    """Minimize cost given minimum output constraint using golden section/binary search.

    Given discrete function f that monotonically increases then monotonically decreases,
    find the minimum integer x such that f(x) >= vmin.

    This method uses Fibonacci search to find the upper bound of x.  If the upper bound
    is found, a binary search is performed in the interval to find the solution.  If
    vmin is close to the maximum of f, a golden section search is performed to attempt
    to find x.

    Parameters
    ----------
    f : Callable[[int], float]
        a function that takes a single integer and output a scalar value.  Must monotonically
        increase then monotonically decrease.
    vmin : float
        the minimum output value.
    offset : int
        the input lower bound.  We will for x in the range [offset, infinity).
    step : int
        the input step.  function will only be evaulated at the points offset + step * N
    maxiter : Optional[int]
        maximum number of iterations to perform.  If None, will run indefinitely.

    Returns
    -------
    result : MinCostResult
        the MinCostResult named tuple, with attributes:

        x : Optional[int]
            the minimum integer such that f(x) >= vmin.  If no such x exists, this will be None.
        xmax : Optional[int]
            the value at which f achieves its maximum.  This is set only if x is None
        vmax : Optional[float]
            the maximum value of f.  This is set only if x is None.
        nfev : int
            total number of function calls made.
    """
    fib2 = fib1 = fib0 = 0
    cur_idx = 0
    nfev = 0
    xmax = vmax = v_prev = None
    while maxiter is None or nfev < maxiter:
        v_cur = f(step * fib0 + offset)
        nfev += 1

        if v_cur >= vmin:
            # found upper bound, use binary search to find answer
            stop = step * fib0 + offset
            return minimize_cost_binary(f, vmin, start=step * (fib1 + 1) + offset,
                                        stop=stop, save=stop, step=step, nfev=nfev)
        else:
            if vmax is not None and v_cur <= vmax:
                if cur_idx <= 3:
                    # special case: 0 <= xmax < 3, and we already checked all possibilities, so
                    # we know vmax < vmin.  There is no solution and just return.
                    return MinCostResult(x=None, xmax=step * xmax + offset, vmax=vmax, nfev=nfev)
                else:
                    # we found the bracket that encloses maximum, perform golden section search
                    a, x, b = fib2, fib1, fib0
                    fx = v_prev
                    while x > a + 1 or b > x + 1:
                        u = a + b - x
                        fu = f(step * u + offset)
                        nfev += 1

                        if fu >= fx:
                            if u > x:
                                a, x = x, u
                                fx = fu
                            else:
                                x, b = u, x
                                fx = fu

                            if fx >= vmin:
                                # found upper bound, use binary search to find answer
                                stop = step * x + offset
                                return minimize_cost_binary(f, vmin, start=step * (a + 1) + offset,
                                                            stop=stop, save=stop, step=step,
                                                            nfev=nfev)
                        else:
                            if u > x:
                                b = u
                            else:
                                a = u

                    # golden section search terminated, the maximum is less than vmin
                    return MinCostResult(x=None, xmax=step * x + offset, vmax=fx, nfev=nfev)
            else:
                # still not close to maximum, continue searching
                vmax = v_prev = v_cur
                xmax = fib0
                cur_idx += 1
                if cur_idx <= 3:
                    fib2, fib1, fib0 = fib1, fib0, cur_idx
                else:
                    fib2, fib1, fib0 = fib1, fib0, fib1 + fib0

    raise ValueError('Maximum number of iteration achieved')


def minimize_cost_binary_float(f,  # type: Callable[[float], float]
                               vmin,  # type: float
                               start,  # type: float
                               stop,  # type: float
                               tol=1e-8,  # type: float
                               save=None,  # type: Optional[float]
                               nfev=0,  # type: int
                               ):
    # type: (...) -> MinCostResult
    """Minimize cost given minimum output constraint using binary search.

    Given discrete function f and an interval, find minimum input x such that f(x) >= vmin using
    binary search.

    This algorithm only works if f is monotonically increasing, or if f monontonically increases
    then monontonically decreases, and f(stop) >= vmin.

    Parameters
    ----------
    f : Callable[[int], float]
        a function that takes a single integer and output a scalar value.  Must monotonically
        increase then monotonically decrease.
    vmin : float
        the minimum output value.
    start : float
        the input lower bound.
    stop : float
        the input upper bound.
    tol : float
        output tolerance.
    save : Optional[float]
        If not none, this value will be returned if no solution is found.
    nfev : int
        number of function calls already made.

    Returns
    -------
    result : MinCostResult
        the MinCostResult named tuple, with attributes:

        x : Optional[float]
            the minimum x such that f(x) >= vmin.  If no such x exists, this will be None.
        nfev : int
            total number of function calls made.

    """
    bin_iter = FloatBinaryIterator(start, stop, tol=tol)
    while bin_iter.has_next():
        x_cur = bin_iter.get_next()
        v_cur = f(x_cur)
        nfev += 1

        if v_cur >= vmin:
            save = x_cur
            bin_iter.down()
        else:
            bin_iter.up()
    return MinCostResult(x=save, xmax=None, vmax=None, nfev=nfev)


def minimize_cost_golden_float(f, vmin, start, stop, tol=1e-8, maxiter=1000):
    # type: (Callable[[float], float], float, float, float, float, int) -> MinCostResult
    """Minimize cost given minimum output constraint using golden section/binary search.

    Given discrete function f that monotonically increases then monotonically decreases,
    find the minimum integer x such that f(x) >= vmin.

    This method uses Fibonacci search to find the upper bound of x.  If the upper bound
    is found, a binary search is performed in the interval to find the solution.  If
    vmin is close to the maximum of f, a golden section search is performed to attempt
    to find x.

    Parameters
    ----------
    f : Callable[[int], float]
        a function that takes a single integer and output a scalar value.  Must monotonically
        increase then monotonically decrease.
    vmin : float
        the minimum output value.
    start : float
        the input lower bound.
    stop : float
        the input upper bound.
    tol : float
        the solution tolerance.
    maxiter : int
        maximum number of iterations to perform.

    Returns
    -------
    result : MinCostResult
        the MinCostResult named tuple, with attributes:

        x : Optional[int]
            the minimum integer such that f(x) >= vmin.  If no such x exists, this will be None.
        xmax : Optional[int]
            the value at which f achieves its maximum.  This is set only if x is None
        vmax : Optional[float]
            the maximum value of f.  This is set only if x is None.
        nfev : int
            total number of function calls made.
    """

    fa = f(start)
    if fa >= vmin:
        # solution found at start
        return MinCostResult(x=start, xmax=None, vmax=None, nfev=1)

    fb = f(stop)  # type: Optional[float]
    if fb is None:
        raise TypeError("f(stop) returned None instead of float")
    if fb >= vmin:
        # found upper bound, use binary search to find answer
        return minimize_cost_binary_float(f, vmin, start, stop, tol=tol, save=stop, nfev=2)

    # solution is somewhere in middle
    gr = (5**0.5 + 1) / 2
    delta = (stop - start) / gr
    c = stop - delta
    d = start + delta

    fc = f(c)  # type: Optional[float]
    if fc is None:
        raise TypeError("f(c) returned None instead of float")
    if fc >= vmin:
        # found upper bound, use binary search to find answer
        return minimize_cost_binary_float(f, vmin, start, c, tol=tol, save=stop, nfev=3)

    fd = f(d)  # type: Optional[float]
    if fd is None:
        raise TypeError("f(d) returned None instead of float")
    if fd >= vmin:
        # found upper bound, use binary search to find answer
        return minimize_cost_binary_float(f, vmin, start, c, tol=tol, save=stop, nfev=4)

    if fc > fd:
        a, b, d = start, d, c
        c = b - (b - a) / gr
        fb, fc, fd = fd, None, fc
    else:
        a, b, c = c, stop, d
        d = a + (b - a) / gr
        fa, fc, fd = fc, fd, None

    nfev = 4
    while abs(b - a) > tol and nfev < maxiter:
        if fc is None:
            fc = f(c)
        else:
            fd = f(d)
        assert fc is not None, 'Either fc or fd was None and the above should have set it'
        assert fd is not None, 'Either fc or fd was None and the above should have set it'
        nfev += 1
        if fc > fd:
            if fc >= vmin:
                return minimize_cost_binary_float(f, vmin, a, c, tol=tol, save=stop, nfev=nfev)
            b, d = d, c
            c = b - (b - a) / gr
            fb, fc, fd = fd, None, fc
        else:
            if fd >= vmin:
                return minimize_cost_binary_float(f, vmin, a, d, tol=tol, save=stop, nfev=nfev)
            a, c = c, d
            d = a + (b - a) / gr
            fa, fc, fd = fc, fd, None

    test = (a + b) / 2
    vmax = f(test)
    nfev += 1
    if vmax >= vmin:
        return MinCostResult(x=test, xmax=test, vmax=vmax, nfev=nfev)
    else:
        return MinCostResult(x=None, xmax=test, vmax=vmax, nfev=nfev)
