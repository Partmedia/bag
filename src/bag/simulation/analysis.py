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
from __future__ import annotations

from typing import Optional, Union, Dict, Any, TypeVar, Type

from enum import Enum
from dataclasses import dataclass

from ..util.immutable import ImmutableList, ImmutableSortedDict
from .data import SweepSpec, swp_spec_from_dict


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
    pass


@dataclass(eq=True, frozen=True)
class AnalysisAC(AnalysisSweep1D):
    freq: float

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


@dataclass(eq=True, frozen=True)
class AnalysisNoise(AnalysisAC):
    out_probe: str
    in_probe: str


@dataclass(eq=True, frozen=True)
class AnalysisTran:
    start: float
    stop: float
    strobe: float
    options: ImmutableSortedDict[str, str]


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
