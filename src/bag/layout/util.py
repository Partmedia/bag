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

"""This module contains utility classes used for layout
"""

# noinspection PyUnresolvedReferences
from pybag.core import BBox, BBoxArray, BBoxCollection


class PortSpec(object):
    """Specification of a port.

    Parameters
    ----------
    ntr : int
        number of tracks the port should occupy
    idc : float
        DC current the port should support, in Amperes.
    """

    def __init__(self, ntr, idc):
        self._ntr = ntr
        self._idc = idc

    @property
    def ntr(self):
        """minimum number of tracks the port should occupy"""
        return self._ntr

    @property
    def idc(self):
        """minimum DC current the port should support, in Amperes"""
        return self._idc

    def __str__(self):
        return repr(self)

    def __repr__(self):
        fmt_str = '%s(%d, %.4g)'
        return fmt_str % (self.__class__.__name__, self._ntr, self._idc)


class Pin(object):
    """A layout pin.

    Multiple pins can share the same terminal name.

    Parameters
    ----------
    pin_name : str
        the pin label.
    term_name : str
        the terminal name.
    layer : str
        the pin layer name.
    bbox : bag.layout.util.BBox
        the pin bounding box.
    """

    def __init__(self, pin_name, term_name, layer, bbox):
        if not bbox.is_physical():
            raise Exception('Non-physical pin bounding box: %s' % bbox)

        self._pin_name = pin_name
        self._term_name = term_name
        self._layer = layer
        self._bbox = bbox

    @property
    def pin_name(self):
        """the pin label."""
        return self._pin_name

    @property
    def term_name(self):
        """the terminal name."""
        return self._term_name

    @property
    def layer(self):
        """the pin layer name"""
        return self._layer

    @property
    def bbox(self):
        """the pin bounding box."""
        return self._bbox

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '%s(%s, %s, %s, %s)' % (self.__class__.__name__, self._pin_name,
                                       self._term_name, self._layer, self._bbox)
