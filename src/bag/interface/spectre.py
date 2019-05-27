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

"""This module implements bag's interface with spectre simulator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Sequence, Optional, List, Tuple

from pathlib import Path

from pybag.enum import DesignOutput
from srr_pybind11 import load_md_array

from ..data.core import MDArray
from ..io.file import read_yaml, open_file
from .simulator import SimProcessManager, get_corner_temp

if TYPE_CHECKING:
    from .simulator import ProcInfo


def _write_sim_env(lines: List[str], models: List[Tuple[str, str]], temp: int) -> None:
    for fname, section in models:
        lines.append(f'include "{fname}" section={section}')
    lines.append(f'tempOption options temp={temp}')


class SpectreInterface(SimProcessManager):
    """This class handles interaction with Ocean simulators.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary.
    """

    def __init__(self, tmp_dir: str, sim_config: Dict[str, Any]) -> None:
        SimProcessManager.__init__(self, tmp_dir, sim_config)
        self._model_setup: Dict[str, List[Tuple[str, str]]] = read_yaml(sim_config['env_file'])

    @property
    def netlist_type(self) -> DesignOutput:
        return DesignOutput.SPECTRE

    # TODO: figure out sweep params spec
    def create_netlist(self, output_file: str, sch_netlist: Path,
                       analyses: Dict[str, Dict[str, Any]], sim_envs: Sequence[str],
                       params: Dict[str, float], swp_params: Sequence[Tuple[str, Sequence[float]]],
                       env_params: Dict[str, Sequence[float]], outputs: Dict[str, str],
                       precision: int = 6, **kwargs: Any) -> None:
        if not sim_envs:
            raise ValueError('simulation environments list is empty')
        def_corner, def_temp = get_corner_temp(sim_envs[0])

        with open_file(sch_netlist, 'r') as f:
            lines = f.readlines()

        # write default model statements
        _write_sim_env(lines, self._model_setup[def_corner], def_temp)
        lines.append('')

        # write parameters
        param_fmt = 'parameters {}={:.%dg}' % precision
        env_param_keys = sorted(env_params.keys())

        for par in sorted(params.keys()):
            lines.append(param_fmt.format(par, params[par]))
        for par in sorted(swp_params.keys()):
            lines.append(param_fmt.format(par, swp_params[par][0]))
        for par in env_param_keys:
            lines.append(param_fmt.format(par, env_params[par][0]))

        # write statements for each simulation environment
        for idx, sim_env in enumerate(sim_envs):
            # write altergroup statement
            corner, temp = get_corner_temp(sim_env)
            lines.append(f'{sim_env} altergroup {{')
            _write_sim_env(lines, self._model_setup[corner], temp)
            for par in env_param_keys:
                lines.append(param_fmt.format(par, env_params[par][idx]))
            lines.append('}')

            # write sweep statements

    def load_md_array(self, dir_path: Path, sim_tag: str, precision: int) -> MDArray:
        dir_name = str(dir_path.resolve() / f'{sim_tag}.raw')
        env_list, data = load_md_array(dir_name)
        return MDArray(env_list, data)

    def setup_sim_process(self, netlist: str, sim_tag: str) -> ProcInfo:
        sim_kwargs: Dict[str, Any] = self.config['kwargs']
        cmd_str: str = sim_kwargs.get('command', 'spectre')
        env: Optional[Dict[str, str]] = sim_kwargs.get('env', None)
        cwd: str = sim_kwargs.get('cwd', '')
        run_64: bool = sim_kwargs.get('run_64', True)
        fmt: str = sim_kwargs.get('format', 'psfxl')
        psf_version: str = sim_kwargs.get('psfversion', '1.1')

        sim_cmd = [cmd_str, '-cols', '100', '-format', fmt, '-raw', f'{sim_tag}.raw']

        if fmt == 'psfxl':
            sim_cmd.append('-psfversion')
            sim_cmd.append(psf_version)
        if run_64:
            sim_cmd.append('-64')

        if not cwd:
            # make sure working directory to netlist directory if None
            cwd_path = Path(netlist).resolve().parent
        else:
            cwd_path = Path(cwd)

        # create empty log file to make sure it exists.
        log_path = cwd_path / 'spectre_output.log'
        return sim_cmd, str(log_path), env, str(cwd_path)
