# SPDX-License-Identifier: Apache-2.0
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

"""This module handles high level simulation routines.

This module defines SimAccess, which provides methods to run simulations
and retrieve results.
"""

from typing import Sequence, Tuple, Union, Iterable, List, Dict

import math
from enum import Enum
from dataclasses import dataclass

import numpy as np

from ..util.immutable import ImmutableList


class SweepTypes(Enum):
    LINEAR = 0
    LINEAR_STEP = 1
    LOG = 2
    LOG_DEC = 3
    LIST = 4


@dataclass(eq=True, frozen=True)
class SweepLinear:
    start: float
    stop: float
    num: int

    @property
    def step(self) -> float:
        return (self.stop - self.start) / self.num

    @property
    def stop_include(self) -> float:
        return self.start + (self.num - 1) * self.step


@dataclass(eq=True, frozen=True)
class SweepLinearStep:
    start: float
    stop: float
    step: float

    @property
    def num(self) -> int:
        return int(math.ceil((self.stop - self.start) / self.step))

    @property
    def stop_include(self) -> float:
        return self.start + (self.num - 1) * self.step


@dataclass(eq=True, frozen=True)
class SweepLog:
    start: float
    stop: float
    num: int

    @property
    def start_log(self) -> float:
        return math.log10(self.start)

    @property
    def stop_log(self) -> float:
        return math.log10(self.stop)

    @property
    def step_log(self) -> float:
        return (self.stop_log - self.start_log) / self.num

    @property
    def stop_include(self) -> float:
        return 10.0**(self.start_log + (self.num - 1) * self.step_log)


@dataclass(eq=True, frozen=True)
class SweepLog:
    start: float
    stop: float
    num: int

    @property
    def start_log(self) -> float:
        return math.log10(self.start)

    @property
    def stop_log(self) -> float:
        return math.log10(self.stop)

    @property
    def step_log(self) -> float:
        return (self.stop_log - self.start_log) / self.num

    @property
    def stop_include(self) -> float:
        return 10.0**(self.start_log + (self.num - 1) * self.step_log)


@dataclass(eq=True, frozen=True)
class SweepLogDec:
    start: float
    stop: float
    dec: int


@dataclass(eq=True, frozen=True, init=False)
class SweepList:
    values: ImmutableList[float]

    def __init__(self, values: Sequence[float]) -> None:
        object.__setattr__(self, 'values', ImmutableList(values))

    @property
    def start(self) -> float:
        return self.values[0]


SweepSpec = Union[SweepLinear, SweepLinearStep, SweepLog, SweepLogDec, SweepList]


@dataclass(eq=True, frozen=True)
class MDSweepInfo:
    params: ImmutableList[Tuple[str, SweepSpec]]

    def __init__(self, params: Sequence[Tuple[str, SweepSpec]]) -> None:
        object.__setattr__(self, 'params', ImmutableList(params))

    def default_items(self) -> Iterable[Tuple[str, float]]:
        for name, spec in self.params:
            yield name, spec.start


@dataclass(eq=True, frozen=True)
class SetSweepInfo:
    params: ImmutableList[str]
    values: ImmutableList[Tuple[float, ...]]

    def __init__(self, params: Sequence[str], values: Sequence[Sequence[float]]) -> None:
        object.__setattr__(self, 'params', ImmutableList(params))

        val_list = []
        num_par = len(params)
        for combo in values:
            if len(combo) != num_par:
                raise ValueError('Invalid param set values.')
            val_list.append(tuple(combo))

        object.__setattr__(self, 'values', ImmutableList(val_list))

    def default_items(self) -> Iterable[Tuple[str, float]]:
        for idx, name in enumerate(self.params):
            yield name, self.values[0][idx]


SweepInfo = Union[MDSweepInfo, SetSweepInfo]


class MDArray:
    """A data structure that stores simulation data as a multi-dimensional array."""

    def __init__(self, env_list: List[str],
                 data: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]]) -> None:
        self._corners = ImmutableList(env_list)
        self._corners_arr = np.array(env_list)
        self._master_table = data

        if self._master_table:
            self._cur_key = next(iter(self._master_table.keys()))
            tmp = self._master_table[self._cur_key]
            self._cur_data: Dict[str, np.ndarray] = tmp[0]
            self._cur_swp_params: Dict[str, List[str]] = tmp[1]
        else:
            raise ValueError('Empty simulation data.')

    @property
    def analysis(self) -> str:
        return self._cur_key

    @property
    def env_list(self) -> ImmutableList[str]:
        return self._corners

    def __getitem__(self, item: str) -> np.ndarray:
        if item == 'corner':
            return self._corners_arr
        return self._cur_data[item]

    def get_swp_params(self, item: str) -> ImmutableList[str]:
        return ImmutableList(self._cur_swp_params[item])

    def set_analysis(self, val: str) -> None:
        if val not in self._master_table:
            raise ValueError(f'Analysis {val} not found.')
        self._cur_key = val
        tmp = self._master_table[self._cur_key]
        self._cur_data: Dict[str, np.ndarray] = tmp[0]
        self._cur_swp_params: Dict[str, List[str]] = tmp[1]
