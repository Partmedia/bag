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

from ..io import make_temp_dir
from ..concurrent.core import SubProcessManager
from .base import InterfaceBase


class SimAccess(InterfaceBase, abc.ABC):
    """A class that interacts with a simulator.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir, sim_config):
        # type: (str, Dict[str, Any]) -> None
        InterfaceBase.__init__(self)

        self.sim_config = sim_config
        self.tmp_dir = make_temp_dir('simTmp', parent_dir=tmp_dir)

    @abc.abstractmethod
    def format_parameter_value(self, param_config, precision):
        # type: (Dict[str, Any], int) -> str
        """Format the given parameter value as a string.

        To support both single value parameter and parameter sweeps, each parameter value is
        represented as a string instead of simple floats.  This method will cast a parameter
        configuration (which can either be a single value or a sweep) to a
        simulator-specific string.

        Parameters
        ----------
        param_config: Dict[str, Any]
            a dictionary that describes this parameter value.

            4 formats are supported.  This is best explained by example.

            single value:
            dict(type='single', value=1.0)

            sweep a given list of values:
            dict(type='list', values=[1.0, 2.0, 3.0])

            linear sweep with inclusive start, inclusive stop, and step size:
            dict(type='linstep', start=1.0, stop=3.0, step=1.0)

            logarithmic sweep with given number of points per decade:
            dict(type='decade', start=1.0, stop=10.0, num=10)

        precision : int
            the parameter value precision.

        Returns
        -------
        param_str : str
            a string representation of param_config
        """
        return ""

    @abc.abstractmethod
    async def async_run_simulation(self, tb_lib, tb_cell, outputs, precision=6, sim_tag=None):
        # type: (str, str, Dict[str, str], int, Optional[str]) -> str
        """A coroutine for simulation a testbench.

        Parameters
        ----------
        tb_lib : str
            testbench library name.
        tb_cell : str
            testbench cell name.
        outputs : Dict[str, str]
            the variable-to-expression dictionary.
        precision : int
            precision of floating point results.
        sim_tag : Optional[str]
            a descriptive tag describing this simulation run.

        Returns
        -------
        value : str
            the save directory path.
        """
        pass

    @abc.abstractmethod
    async def async_load_results(self, lib, cell, hist_name, outputs, precision=6):
        # type: (str, str, str, Dict[str, str], int) -> str
        """A coroutine for loading simulation results.

        Parameters
        ----------
        lib : str
            testbench library name.
        cell : str
            testbench cell name.
        hist_name : str
            simulation history name.
        outputs : Dict[str, str]
            the variable-to-expression dictionary.
        precision : int
            precision of floating point results.

        Returns
        -------
        value : str
            the save directory path.
        """
        pass


ProcInfo = Tuple[Union[str, Sequence[str]], str, Optional[Dict[str, str]], Optional[str], str]


class SimProcessManager(SimAccess, metaclass=abc.ABCMeta):
    """An implementation of :class:`SimAccess` using :class:`SubProcessManager`.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir, sim_config):
        # type: (str, Dict[str, Any]) -> None
        SimAccess.__init__(self, tmp_dir, sim_config)
        cancel_timeout = sim_config.get('cancel_timeout_ms', None)
        if cancel_timeout is not None:
            cancel_timeout /= 1e3
        self._manager = SubProcessManager(max_workers=sim_config.get('max_workers', None),
                                          cancel_timeout=cancel_timeout)

    @abc.abstractmethod
    def setup_sim_process(self, lib, cell, outputs, precision, sim_tag):
        # type: (str, str, Dict[str, str], int, Optional[str]) -> ProcInfo
        """This method performs any setup necessary to configure a simulation process.

        Parameters
        ----------
        lib : str
            testbench library name.
        cell : str
            testbench cell name.
        outputs : Dict[str, str]
            the variable-to-expression dictionary.
        precision : int
            precision of floating point results.
        sim_tag : Optional[str]
            a descriptive tag describing this simulation run.

        Returns
        -------
        args : Union[str, Sequence[str]]
            command to run, as string or list of string arguments.
        log : str
            log file name.
        env : Optional[Dict[str, str]]
            environment variable dictionary.  None to inherit from parent.
        cwd : Optional[str]
            working directory path.  None to inherit from parent.
        save_dir : str
            save directory path.
        """
        return '', '', None, None, ''

    @abc.abstractmethod
    def setup_load_process(self, lib, cell, hist_name, outputs, precision):
        # type: (str, str, str, Dict[str, str], int) -> ProcInfo
        """This method performs any setup necessary to configure a result loading process.

        Parameters
        ----------
        lib : str
            testbench library name.
        cell : str
            testbench cell name.
        hist_name : str
            simulation history name.
        outputs : Dict[str, str]
            the variable-to-expression dictionary.
        precision : int
            precision of floating point results.

        Returns
        -------
        args : Union[str, Sequence[str]]
            command to run, as string or list of string arguments.
        log : str
            log file name.
        env : Optional[Dict[str, str]]
            environment variable dictionary.  None to inherit from parent.
        cwd : Optional[str]
            working directory path.  None to inherit from parent.
        save_dir : str
            save directory path.
        """
        return '', '', None, None, ''

    async def async_run_simulation(self, tb_lib: str, tb_cell: str,
                                   outputs: Dict[str, str],
                                   precision: int = 6,
                                   sim_tag: Optional[str] = None) -> str:
        args, log, env, cwd, save_dir = self.setup_sim_process(tb_lib, tb_cell, outputs, precision,
                                                               sim_tag)

        await self._manager.async_new_subprocess(args, log, env=env, cwd=cwd)
        return save_dir

    async def async_load_results(self, lib: str, cell: str, hist_name: str,
                                 outputs: Dict[str, str],
                                 precision: int = 6) -> str:
        args, log, env, cwd, save_dir = self.setup_load_process(lib, cell, hist_name, outputs,
                                                                precision)

        await self._manager.async_new_subprocess(args, log, env=env, cwd=cwd)
        return save_dir
