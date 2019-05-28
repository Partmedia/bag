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

"""This module handles high level simulation routines.

This module defines SimAccess, which provides methods to run simulations
and retrieve results.
"""

from typing import Dict, Optional, Sequence, Any, Tuple, Union, Iterable

import abc
import math
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

from pybag.enum import DesignOutput

from ..data.core import MDArray
from ..concurrent.core import SubProcessManager
from ..util.immutable import ImmutableList
from .base import InterfaceBase

ProcInfo = Tuple[Union[str, Sequence[str]], str, Optional[Dict[str, str]], str]


def get_corner_temp(env_str: str) -> Tuple[str, int]:
    idx = env_str.rfind('_')
    if idx < 0:
        raise ValueError(f'Invalid environment string: {env_str}')
    return env_str[:idx], int(env_str[idx + 1:])


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


class SimAccess(InterfaceBase, abc.ABC):
    """A class that interacts with a simulator.

    Parameters
    ----------
    parent : str
        parent directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, parent: str, sim_config: Dict[str, Any]) -> None:
        InterfaceBase.__init__(self)

        self._config = sim_config
        self._dir_path = Path(parent) / "simulations"

    @property
    @abc.abstractmethod
    def netlist_type(self) -> DesignOutput:
        return DesignOutput.CDL

    @abc.abstractmethod
    def create_netlist(self, output_file: str, sch_netlist: Path,
                       analyses: Dict[str, Dict[str, Any]], sim_envs: Sequence[str],
                       params: Dict[str, float], swp_info: SweepInfo,
                       env_params: Dict[str, Sequence[float]], outputs: Dict[str, str],
                       precision: int = 6, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def load_md_array(self, dir_path: Path, sim_tag: str, precision: int) -> MDArray:
        """Load simulation results.

        Parameters
        ----------
        dir_path : Path
            the working directory path.
        sim_tag : str
            optional simulation name.  Empty for default.
        precision : int
            the floating point number precision.

        Returns
        -------
        data : Dict[str, Any]
            the simulation data dictionary.
        """
        pass

    @abc.abstractmethod
    async def async_run_simulation(self, netlist: str, sim_tag: str) -> None:
        """A coroutine for simulation a testbench.

        Parameters
        ----------
        netlist : str
            the netlist file name.
        sim_tag : str
            optional simulation name.  Empty for default.
        """
        pass

    @property
    def dir_path(self) -> Path:
        """Path: the directory for simulation files."""
        return self._dir_path

    @property
    def config(self) -> Dict[str, Any]:
        """Dict[str, Any]: simulation configurations."""
        return self._config


class SimProcessManager(SimAccess, abc.ABC):
    """An implementation of :class:`SimAccess` using :class:`SubProcessManager`.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir: str, sim_config: Dict[str, Any]) -> None:
        SimAccess.__init__(self, tmp_dir, sim_config)

        cancel_timeout = sim_config.get('cancel_timeout_ms', 10000) / 1e3
        self._manager = SubProcessManager(max_workers=sim_config.get('max_workers', 0),
                                          cancel_timeout=cancel_timeout)

    @abc.abstractmethod
    def setup_sim_process(self, netlist: str, sim_tag: str) -> ProcInfo:
        """This method performs any setup necessary to configure a simulation process.

        Parameters
        ----------
        netlist : str
            the netlist file name.
        sim_tag : str
            optional simulation name.  Empty for default.

        Returns
        -------
        args : Union[str, Sequence[str]]
            command to run, as string or list of string arguments.
        log : str
            log file name.
        env : Optional[Dict[str, str]]
            environment variable dictionary.  None to inherit from parent.
        cwd : str
            working directory for the subprocess.
        """
        return '', '', None, ''

    async def async_run_simulation(self, netlist: str, sim_tag: str) -> None:
        args, log, env, cwd = self.setup_sim_process(netlist, sim_tag)

        await self._manager.async_new_subprocess(args, log, env=env, cwd=cwd)
