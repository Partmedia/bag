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

import asyncio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from pybag.enum import DesignOutput
from srr_pybind11 import load_md_array

from ..math import float_to_si_string
from ..io.file import read_yaml, open_file
from ..util.immutable import ImmutableList, ImmutableSortedDict
from .data import (
    MDSweepInfo, MDArray, SetSweepInfo, SweepLinear, SweepLog, SweepList, SimNetlistInfo,
    SweepSpec, AnalysisInfo, AnalysisAC, AnalysisSP, AnalysisNoise, AnalysisTran,
    AnalysisSweep1D, SweepInfoType
)
from .base import SimProcessManager, get_corner_temp
from .hdf5 import save_md_array_hdf5, load_md_array_hdf5

if TYPE_CHECKING:
    from .data import SweepInfo
    from .base import ProcInfo


def write_md_array_hdf5(hdf5_path: Path, dir_name: str) -> None:
    env_list, data_dict = load_md_array(dir_name)
    save_md_array_hdf5(env_list, data_dict, hdf5_path)


def _write_sim_env(lines: List[str], models: List[Tuple[str, str]], temp: int) -> None:
    for fname, section in models:
        lines.append(f'include "{fname}" section={section}')
    lines.append(f'tempOption options temp={temp}')


def _write_param_set(lines: List[str], params: Sequence[str],
                     values: Sequence[ImmutableList[float]], precision: int) -> None:
    # get list of lists of strings to print, and compute column widths
    data = [params]
    col_widths = [len(par) for par in params]
    for combo in values:
        str_list = []
        for idx, val in enumerate(combo):
            cur_str = float_to_si_string(val, precision)
            col_widths[idx] = max(col_widths[idx], len(cur_str))
            str_list.append(cur_str)
        data.append(str_list)

    # write the columns
    lines.append('swp_data paramset {')
    for row in data:
        lines.append(' '.join(val.ljust(width) for val, width in zip(row, col_widths)))
    lines.append('}')


def _get_sweep_str(par: str, swp_spec: Optional[SweepSpec], precision: int) -> str:
    if not par or swp_spec is None:
        return ''

    if isinstance(swp_spec, SweepList):
        tmp = ' '.join((float_to_si_string(val, precision) for val in swp_spec.values))
        val_str = f'[{tmp}]'
    elif isinstance(swp_spec, SweepLinear):
        # spectre: stop is inclusive, lin = number of points excluding the last point
        val_str = f'start={swp_spec.start} stop={swp_spec.stop_inc} lin={swp_spec.num - 1}'
    elif isinstance(swp_spec, SweepLog):
        # spectre: stop is inclusive, log = number of points excluding the last point
        val_str = f'start={swp_spec.start} stop={swp_spec.stop_inc} log={swp_spec.num - 1}'
    else:
        raise ValueError('Unknown sweep specification.')

    return f'param={par} {val_str}'


def _get_options_str(options: ImmutableSortedDict[str, str]) -> str:
    return ' '.join((f'{key}={val}' for key, val in options.items()))


def _write_sweep_start(lines: List[str], swp_info: SweepInfo, swp_idx: int, precision: int) -> int:
    if isinstance(swp_info, MDSweepInfo):
        for dim_idx, (par, swp_spec) in enumerate(swp_info.params):
            statement = _get_sweep_str(par, swp_spec, precision)
            lines.append(f'swp{swp_idx}{dim_idx} sweep {statement} {{')
        return swp_info.ndim
    else:
        lines.append(f'swp{swp_idx} sweep paramset=swp_data {{')
        return 1


def _write_analysis(lines: List[str], sim_env: str, ana: AnalysisInfo, precision: int) -> None:
    cur_line = f'\\@{ana.name}\\@{sim_env}\\@ {ana.name}'
    if isinstance(ana, AnalysisTran):
        cur_line += f' start={ana.start} stop={ana.stop} strobeperiod={ana.strobe}'
    elif isinstance(ana, AnalysisSweep1D):
        par = ana.param
        sweep_str = _get_sweep_str(par, ana.sweep, precision)
        cur_line += ' '
        cur_line += sweep_str
        if isinstance(ana, AnalysisAC) and par != 'freq':
            cur_line += f' freq={float_to_si_string(ana.freq, precision)}'

        if isinstance(ana, AnalysisSP):
            cur_line += f' ports=[{" ".join(ana.ports)}] paramtype={ana.param_type.name}'
        elif isinstance(ana, AnalysisNoise):
            cur_line += f' oprobe=[{ana.out_probe}] iprobe={ana.in_probe}'
    else:
        raise ValueError('Unknown analysis specification.')

    opt_str = _get_options_str(ana.options)
    if opt_str:
        cur_line += ' '
        cur_line += opt_str

    lines.append(cur_line)


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

    def create_netlist(self, output_file: str, sch_netlist: Path, info: SimNetlistInfo,
                       precision: int = 6) -> None:
        sim_envs = info.sim_envs
        analyses = info.analyses
        params = info.params
        env_params = info.env_params
        swp_info = info.swp_info

        def_corner, def_temp = get_corner_temp(sim_envs[0])

        with open_file(sch_netlist, 'r') as f:
            lines = f.readlines()

        # write default model statements
        _write_sim_env(lines, self._model_setup[def_corner], def_temp)
        lines.append('')

        # write parameters
        param_fmt = 'parameters {}={}'
        for par, val in params.items():
            lines.append(param_fmt.format(par, float_to_si_string(val, precision)))
        for par, val in swp_info.default_items():
            lines.append(param_fmt.format(par, float_to_si_string(val, precision)))
        for par, val_list in env_params.items():
            lines.append(param_fmt.format(par, float_to_si_string(val_list[0], precision)))

        if isinstance(swp_info, SetSweepInfo):
            # write paramset declaration if needed
            _write_param_set(lines, swp_info.params, swp_info.values, precision)

        # write statements for each simulation environment
        for idx, sim_env in enumerate(sim_envs):
            # write altergroup statement
            corner, temp = get_corner_temp(sim_env)
            lines.append(f'{sim_env} altergroup {{')
            _write_sim_env(lines, self._model_setup[corner], temp)
            for par, val_list in env_params.items():
                lines.append(param_fmt.format(par, val_list[idx]))
            lines.append('}')

            # write sweep statements
            num_brackets = _write_sweep_start(lines, swp_info, idx, precision)

            # write analyses
            for ana in analyses:
                _write_analysis(lines, sim_env, ana, precision)

            # close sweep statements
            for _ in range(num_brackets):
                lines.append('}')

    def load_md_array(self, dir_path: Path, sim_tag: str) -> MDArray:
        hdf5_path = dir_path / f'{sim_tag}.hdf5'
        return load_md_array_hdf5(hdf5_path)

    def setup_sim_process(self, netlist: str, sim_tag: str) -> ProcInfo:
        sim_kwargs: Dict[str, Any] = self.config['kwargs']
        cmd_str: str = sim_kwargs.get('command', 'spectre')
        env: Optional[Dict[str, str]] = sim_kwargs.get('env', None)
        run_64: bool = sim_kwargs.get('run_64', True)
        fmt: str = sim_kwargs.get('format', 'psfxl')
        psf_version: str = sim_kwargs.get('psfversion', '1.1')

        sim_cmd = [cmd_str, '-cols', '100', '-format', fmt, '-raw', f'{sim_tag}.raw']

        if fmt == 'psfxl':
            sim_cmd.append('-psfversion')
            sim_cmd.append(psf_version)
        if run_64:
            sim_cmd.append('-64')

        cwd_path = Path(netlist).parent.resolve()
        # create empty log file to make sure it exists.
        log_path = cwd_path / 'spectre_output.log'
        return sim_cmd, str(log_path), env, str(cwd_path)

    async def async_run_simulation(self, netlist: str, sim_tag: str, stype: SweepInfoType) -> None:
        args, log, env, cwd = self.setup_sim_process(netlist, sim_tag)

        # TODO: change implementation to move HDF5 writing to C++
        await self._manager.async_new_subprocess(args, log, env=env, cwd=cwd)

        cwd_path = Path(cwd)
        raw_path = cwd_path / f'{sim_tag}.raw'
        if not raw_path.exists():
            raise FileNotFoundError(f'Cannot find simulation data directory: {raw_path}')

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as pool:
            if stype:
                hdf5_path = cwd_path / f'{sim_tag}_md.hdf5'
                await loop.run_in_executor(pool, write_md_array_hdf5, (hdf5_path, str(raw_path)))
