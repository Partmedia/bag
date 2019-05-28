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

from typing import Tuple, Union, Iterable, List, Dict, Any, Optional, TypeVar, Type

import math
from enum import Enum
from dataclasses import dataclass

import numpy as np

from ..util.immutable import ImmutableList, ImmutableSortedDict


###############################################################################
# Sweep specifications
###############################################################################

class SweepType(Enum):
    LIST = 0
    LINEAR = 1
    LOG = 2


@dataclass(eq=True, frozen=True)
class SweepList:
    values: ImmutableList[float]

    @property
    def start(self) -> float:
        return self.values[0]


@dataclass(eq=True, frozen=True)
class SweepLinear:
    """stop is inclusive"""
    start: float
    stop: float
    num: int
    endpoint: bool = True

    @property
    def step(self) -> float:
        den = self.num - 1 if self.endpoint else self.num
        return (self.stop - self.start) / den

    @property
    def stop_inc(self) -> float:
        return self.stop if self.endpoint else self.start + (self.num - 1) * self.step


@dataclass(eq=True, frozen=True)
class SweepLog:
    """stop is inclusive"""
    start: float
    stop: float
    num: int
    endpoint: bool = True

    @property
    def start_log(self) -> float:
        return math.log10(self.start)

    @property
    def stop_log(self) -> float:
        return math.log10(self.stop)

    @property
    def step_log(self) -> float:
        den = self.num - 1 if self.endpoint else self.num
        return (self.stop_log - self.start_log) / den

    @property
    def stop_inc(self) -> float:
        if self.endpoint:
            return self.stop
        return 10.0**(self.start_log + (self.num - 1) * self.step_log)


SweepSpec = Union[SweepLinear, SweepLog, SweepList]


def swp_spec_from_dict(table: Dict[str, Any]) -> SweepSpec:
    swp_type = SweepType[table['type']]
    if swp_type is SweepType.LIST:
        return SweepList(ImmutableList(table['values']))
    elif swp_type is SweepType.LINEAR:
        return SweepLinear(table['start'], table['stop'], table['num'], table.get('endpoint', True))
    elif swp_type is SweepType.LOG:
        return SweepLog(table['start'], table['stop'], table['num'], table.get('endpoint', True))
    else:
        raise ValueError(f'Unsupported sweep type: {swp_type}')


@dataclass(eq=True, frozen=True)
class MDSweepInfo:
    params: ImmutableList[Tuple[str, SweepSpec]]

    @property
    def ndim(self) -> int:
        return len(self.params)

    def default_items(self) -> Iterable[Tuple[str, float]]:
        for name, spec in self.params:
            yield name, spec.start


@dataclass(eq=True, frozen=True)
class SetSweepInfo:
    params: ImmutableList[str]
    values: ImmutableList[ImmutableList[float]]

    def default_items(self) -> Iterable[Tuple[str, float]]:
        for idx, name in enumerate(self.params):
            yield name, self.values[0][idx]


SweepInfo = Union[MDSweepInfo, SetSweepInfo]


def swp_info_from_struct(table: Union[List[Any], Dict[str, Any]]) -> SweepInfo:
    if isinstance(table, dict):
        params = ImmutableList(table['params'])
        values = []
        num_par = len(params)
        for combo in table['values']:
            if len(combo) != num_par:
                raise ValueError('Invalid param set values.')
            values.append(ImmutableList(combo))

        return SetSweepInfo(params, ImmutableList(values))
    else:
        par_list = [(par, swp_spec_from_dict(spec)) for par, spec in table]
        return MDSweepInfo(ImmutableList(par_list))


###############################################################################
# Analyses
###############################################################################

class AnalysisType(Enum):
    DC = 0
    AC = 1
    TRAN = 2
    SP = 3
    NOISE = 4


class SPType(Enum):
    S = 0
    Y = 1
    Z = 2
    YZ = 3


T = TypeVar('T', bound='AnalysisSweep1D')


@dataclass(eq=True, frozen=True)
class AnalysisSweep1D:
    param: str
    sweep: Optional[SweepSpec]
    options: ImmutableSortedDict[str, str]

    @classmethod
    def from_dict(cls: Type[T], table: Dict[str, Any], def_param: str = '') -> T:
        param = table.get('param', def_param)
        sweep = table.get('sweep', None)
        opt = table.get('options', {})
        if not param or sweep is None:
            param = ''
            swp = None
        else:
            swp = swp_spec_from_dict(sweep)

        return cls(param, swp, ImmutableSortedDict(opt))


@dataclass(eq=True, frozen=True)
class AnalysisDC(AnalysisSweep1D):
    @property
    def name(self) -> str:
        return 'dc'


@dataclass(eq=True, frozen=True)
class AnalysisAC(AnalysisSweep1D):
    freq: float

    @property
    def name(self) -> str:
        return 'ac'

    @classmethod
    def from_dict(cls: Type[T], table: Dict[str, Any], def_param: str = '') -> T:
        base = AnalysisSweep1D.from_dict(table, def_param='freq')
        if base.param != 'freq':
            freq_val = table['freq']
        else:
            freq_val = 0.0

        return cls(base.param, base.sweep, base.options, freq_val)


@dataclass(eq=True, frozen=True)
class AnalysisSP(AnalysisAC):
    ports: ImmutableList[str]
    param_type: SPType

    @property
    def name(self) -> str:
        return 'sp'


@dataclass(eq=True, frozen=True)
class AnalysisNoise(AnalysisAC):
    out_probe: str
    in_probe: str

    @property
    def name(self) -> str:
        return 'noise'


@dataclass(eq=True, frozen=True)
class AnalysisTran:
    start: float
    stop: float
    strobe: float
    options: ImmutableSortedDict[str, str]

    @property
    def name(self) -> str:
        return 'tran'


AnalysisInfo = Union[AnalysisDC, AnalysisAC, AnalysisSP, AnalysisNoise, AnalysisTran]


def analysis_from_dict(table: Dict[str, Any]) -> AnalysisInfo:
    ana_type = AnalysisType[table['type']]
    if ana_type is AnalysisType.DC:
        return AnalysisDC.from_dict(table)
    elif ana_type is AnalysisType.AC:
        return AnalysisAC.from_dict(table)
    elif ana_type is AnalysisType.SP:
        base = AnalysisAC.from_dict(table)
        return AnalysisSP(base.param, base.sweep, base.options, base.freq,
                          ImmutableList(table['ports']), SPType[table['param_type']])
    elif ana_type is AnalysisType.NOISE:
        base = AnalysisAC.from_dict(table)
        return AnalysisNoise(base.param, base.sweep, base.options, base.freq,
                             table['out_probe'], table['in_probe'])
    elif ana_type is AnalysisType.TRAN:
        return AnalysisTran(table.get('start', 0.0), table['stop'], table.get('strobe', 0.0),
                            ImmutableSortedDict(table.get('options', {})))
    else:
        raise ValueError(f'Unknown analysis type: {ana_type}')


###############################################################################
# Simulation Netlist Info
###############################################################################

@dataclass(eq=True, frozen=True)
class SimNetlistInfo:
    sim_envs: ImmutableList[str]
    analyses: ImmutableList[AnalysisInfo]
    params: ImmutableSortedDict[str, float]
    env_params: ImmutableSortedDict[str, ImmutableList[float]]
    swp_info: SweepInfo
    outputs: ImmutableSortedDict[str, str]
    options: ImmutableSortedDict[str, Any]


def netlist_info_from_dict(table: Dict[str, Any]) -> SimNetlistInfo:
    sim_envs = table['sim_envs']
    analyses = table['analyses']
    params = table.get('params', {})
    env_params = table.get('env_params', {})
    swp_info = table.get('swp_info', [])
    outputs = table.get('outputs', {})
    options = table.get('options', {})

    if not sim_envs:
        raise ValueError('simulation environments list is empty')

    env_par_dict = {}
    num_env = len(sim_envs)
    for key, val in env_params.items():
        if len(val) != num_env:
            raise ValueError("Invalid env_param value.")
        env_par_dict[key] = ImmutableList(val)

    ana_list = [analysis_from_dict(val) for val in analyses]

    return SimNetlistInfo(ImmutableList(sim_envs), ImmutableList(ana_list),
                          ImmutableSortedDict(params), ImmutableSortedDict(env_par_dict),
                          swp_info_from_struct(swp_info), ImmutableSortedDict(outputs),
                          ImmutableSortedDict(options))


###############################################################################
# Simulation data classes
###############################################################################

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
