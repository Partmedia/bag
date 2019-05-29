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

from typing import (
    Tuple, Union, Iterable, List, Dict, Any, Optional, TypeVar, Type, Sequence, ItemsView
)

import math
from enum import Enum
from dataclasses import dataclass

import numpy as np

from ..util.immutable import ImmutableList, ImmutableSortedDict


###############################################################################
# Sweep specifications
###############################################################################

class SweepSpecType(Enum):
    LIST = 0
    LINEAR = 1
    LOG = 2


class SweepInfoType(Enum):
    MD = 0
    SET = 1


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
    swp_type = SweepSpecType[table['type']]
    if swp_type is SweepSpecType.LIST:
        return SweepList(ImmutableList(table['values']))
    elif swp_type is SweepSpecType.LINEAR:
        return SweepLinear(table['start'], table['stop'], table['num'], table.get('endpoint', True))
    elif swp_type is SweepSpecType.LOG:
        return SweepLog(table['start'], table['stop'], table['num'], table.get('endpoint', True))
    else:
        raise ValueError(f'Unsupported sweep type: {swp_type}')


@dataclass(eq=True, frozen=True)
class MDSweepInfo:
    params: ImmutableList[Tuple[str, SweepSpec]]

    @property
    def ndim(self) -> int:
        return len(self.params)

    @property
    def stype(self) -> SweepInfoType:
        return SweepInfoType.MD

    def default_items(self) -> Iterable[Tuple[str, float]]:
        for name, spec in self.params:
            yield name, spec.start


@dataclass(eq=True, frozen=True)
class SetSweepInfo:
    params: ImmutableList[str]
    values: ImmutableList[ImmutableList[float]]

    @property
    def stype(self) -> SweepInfoType:
        return SweepInfoType.SET

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

    @property
    def param_start(self) -> float:
        if self.param:
            return self.sweep.start
        return 0.0


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
    def param(self) -> str:
        return ''

    @property
    def param_start(self) -> float:
        return 0.0

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
                             table['out_probe'], table.get('in_probe', ''))
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

    @property
    def sweep_type(self) -> SweepInfoType:
        return self.swp_info.stype


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

    def __init__(self, sim_envs: Sequence[str],
                 data: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]]
                 ) -> None:
        self._corners = ImmutableList(sim_envs)
        self._master_table = data

        if self._master_table:
            self._cur_name = next(iter(self._master_table.keys()))
            tmp = self._master_table[self._cur_name]
            self._cur_data: Dict[str, np.ndarray] = tmp[0]
            self._cur_swp_params: Dict[str, List[str]] = tmp[1]
        else:
            raise ValueError('Empty simulation data.')

    @property
    def group(self) -> str:
        return self._cur_name

    @property
    def group_list(self) -> List[str]:
        return list(self._master_table.keys())

    @property
    def sim_envs(self) -> ImmutableList[str]:
        return self._corners

    def __getitem__(self, item: str) -> np.ndarray:
        return self._cur_data[item]

    def __contains__(self, item: str) -> bool:
        return item in self._cur_data

    def items(self) -> ItemsView[str, np.ndarray]:
        return self._cur_data.items()

    def get_swp_params(self, item: str) -> Optional[ImmutableList[str]]:
        tmp = self._cur_swp_params.get(item, None)
        if tmp:
            return ImmutableList(tmp)
        return None

    def open_group(self, val: str) -> None:
        tmp = self._master_table.get(val, None)
        if tmp is None:
            raise ValueError(f'Group {val} not found.')
        self._cur_name = val
        self._cur_data, self._cur_swp_params = tmp

    def open_analysis(self, atype: AnalysisType) -> None:
        self.open_group(atype.name.lower())

    def insert(self, name: str, data: np.ndarray, swp_vars: List[str]) -> None:
        for idx, var in enumerate(swp_vars):
            arr = self._cur_data.get(var, None)
            if arr is None:
                raise ValueError(f'Cannot find sweep variable {var}.')
            if arr.size != data.shape[idx]:
                raise ValueError(f'Sweep variable {var} shape mismatch.')
        self._cur_data[name] = data
        self._cur_swp_params[name] = swp_vars
