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

from typing import Dict, Optional, Sequence, Any, Tuple, Union

import abc
from pathlib import Path

from pybag.enum import DesignOutput

from ..concurrent.core import SubProcessManager
from .data import SimNetlistInfo, MDArray, SweepInfoType

ProcInfo = Tuple[Union[str, Sequence[str]], str, Optional[Dict[str, str]], str]


def get_corner_temp(env_str: str) -> Tuple[str, int]:
    idx = env_str.rfind('_')
    if idx < 0:
        raise ValueError(f'Invalid environment string: {env_str}')
    return env_str[:idx], int(env_str[idx + 1:])


class SimAccess(abc.ABC):
    """A class that interacts with a simulator.

    Parameters
    ----------
    parent : str
        parent directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, parent: str, sim_config: Dict[str, Any]) -> None:
        self._config = sim_config
        self._dir_path = (Path(parent) / "simulations").resolve()

    @property
    @abc.abstractmethod
    def netlist_type(self) -> DesignOutput:
        return DesignOutput.CDL

    @abc.abstractmethod
    def create_netlist(self, output_path: Path, sch_netlist: Path, info: SimNetlistInfo,
                       precision: int = 6) -> None:
        pass

    @abc.abstractmethod
    def load_md_array(self, dir_path: Path, sim_tag: str) -> MDArray:
        """Load simulation results.

        Parameters
        ----------
        dir_path : Path
            the working directory path.
        sim_tag : str
            optional simulation name.  Empty for default.

        Returns
        -------
        data : Dict[str, Any]
            the simulation data dictionary.
        """
        pass

    @abc.abstractmethod
    async def async_run_simulation(self, netlist: Path, sim_tag: str, stype: SweepInfoType) -> None:
        """A coroutine for simulation a testbench.

        Parameters
        ----------
        netlist : Path
            the netlist file name.
        sim_tag : str
            optional simulation name.  Empty for default.
        stype : SweepInfoType
            the parameter sweep type.
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
    def setup_sim_process(self, netlist: Path, sim_tag: str) -> ProcInfo:
        """This method performs any setup necessary to configure a simulation process.

        Parameters
        ----------
        netlist : Path
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
