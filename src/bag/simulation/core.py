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

from __future__ import annotations
from typing import (
    Optional, Dict, Any, Tuple, List, Iterable, Sequence, Mapping, Type, cast
)

import abc
import importlib
import itertools
from pathlib import Path

from pybag.enum import DesignOutput, get_extension

from ..math import float_to_si_string
from ..io.file import read_yaml, write_yaml
from ..layout.template import TemplateDB
from ..design.database import ModuleDB, ModuleType
from ..concurrent.core import batch_async_task
from ..core import BagProject

from .base import SimAccess
from .data import SimNetlistInfo, MDArray


class TestbenchManager(abc.ABC):
    """A class that creates and setups up a testbench for simulation, then save the result.

    This class is used by MeasurementManager to run simulations.

    Parameters
    ----------
    sim : SimAccess
        the simulator interface object.
    work_dir : Path
        working directory path.
    tb_name : str
        testbench name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        testbench specs.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    precision : int
        numeric precision in simulation netlist generation.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        self._sim = sim
        self._work_dir = work_dir.resolve()
        self._tb_name = tb_name
        self._impl_lib = impl_lib
        self._specs = specs
        self._sim_view_list = sim_view_list
        self._env_list = env_list
        self._precision = precision

        self._work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def specs(self) -> Dict[str, Any]:
        return self._specs

    @abc.abstractmethod
    def get_netlist_info(self) -> SimNetlistInfo:
        """Configure the simulation state of the given testbench.

        No need to call update_testbench(), set_simulation_environments(), and
        set_simulation_view().  These are called for you.

        Returns
        -------
        sim_specs : Dict[str, Any]
            the simulation specification dictionary.
        """
        pass

    async def setup_and_simulate(self, sch_db: ModuleDB, sch_params: Dict[str, Any],
                                 dut_cv_info_list: List[Any], dut_netlist: Path) -> None:
        ext = get_extension(self._sim.netlist_type)
        if sch_params is None:
            tb_netlist_path = self._work_dir / f'{self._tb_name}.{ext}'
        else:
            tb_netlist_path = self._create_tb_schematic(sch_db, sch_params,
                                                        dut_cv_info_list=dut_cv_info_list,
                                                        dut_netlist=dut_netlist)

        netlist_info = self.get_netlist_info()
        sim_netlist_name = f'{self._tb_name}_sim.{ext}'
        output_file = str(tb_netlist_path.with_name(sim_netlist_name))
        self._sim.create_netlist(output_file, tb_netlist_path, netlist_info, self._precision)
        # run simulation and save/return raw result
        print(f'Simulating {self._tb_name}')
        await self._sim.async_run_simulation(output_file, self._tb_name)
        print(f'Finished simulating {self._tb_name}')

    def load_md_array(self) -> MDArray:
        return self._sim.load_md_array(self._work_dir, self._tb_name, self._precision)

    def _create_tb_schematic(self, sch_db: ModuleDB, sch_params: Mapping[str, Any],
                             dut_cv_info_list: List[Any], dut_netlist: Path) -> Path:
        """Helper method to create a testbench schematic.

        Parameters
        ----------
        sch_db : ModuleDB
            the schematic database.
        sch_params : Dict[str, Any]
            the testbench schematic parameters dictionary.
        dut_cv_info_list : List[str]
            cv_info for DUT cell netlist
        dut_netlist : Path
            netlist of DUT cell

        Returns
        -------
        tb_netlist_path : Path
            the simulation Testbench netlist.
        """
        tb_lib: str = self._specs['tb_lib']
        tb_cell: str = self._specs['tb_cell']

        gen_cls = cast(Type[ModuleType], ModuleDB.get_schematic_class(tb_lib, tb_cell))
        sch_master = sch_db.new_master(gen_cls, sch_params)
        print(f'Creating testbench {self._tb_name} schematic master')
        sch_db.batch_schematic([(sch_master, self._tb_name)], cv_info_out=dut_cv_info_list)
        print(f'Testbench {self._tb_name} schematic master done')

        # create netlist for tb schematic
        ext = get_extension(self._sim.netlist_type)
        tb_netlist_path = self._work_dir / f'{self._tb_name}.{ext}'
        print(f'Creating testbench {self._tb_name} netlist')
        sch_db.batch_schematic([(sch_master, self._tb_name)], output=self._sim.netlist_type,
                               top_subckt=False, fname=str(tb_netlist_path),
                               cv_info_list=dut_cv_info_list, cv_netlist=str(dut_netlist))
        print(f'Testbench {self._tb_name} netlisting done')

        return tb_netlist_path


class MeasurementManager(abc.ABC):
    """A class that handles circuit performance measurement.

    This class handles all the steps needed to measure a specific performance
    metric of the device-under-test.  This may involve creating and simulating
    multiple different testbenches, where configuration of successive testbenches
    depends on previous simulation results. This class reduces the potentially
    complex measurement tasks into a few simple abstract methods that designers
    simply have to implement.

    Parameters
    ----------
    sim : SimAccess
        the simulator interface object.
    dir_path : Path
        Simulation data directory.
    meas_name : str
        measurement setup name.
    impl_lib : str
        implementation library name.
    specs : Dict[str, Any]
        the measurement specification dictionary.
    wrapper_lookup : Dict[str, str]
        the DUT wrapper cell name lookup table.
    sim_view_list : Sequence[Tuple[str, str]]
        simulation view list
    env_list : Sequence[str]
        simulation environments list.
    precision : int
        numeric precision in simulation netlist generation.
    """

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        self._sim = sim
        self._dir_path = dir_path.resolve()
        self._meas_name = meas_name
        self._impl_lib = impl_lib
        self._specs = specs
        self._wrapper_lookup = wrapper_lookup
        self._sim_view_list = sim_view_list
        self._env_list = env_list
        self._precision = precision

        self._dir_path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return ''

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Get information about the next testbench.

        Override this method to perform more complex operations.

        Parameters
        ----------
        state : str
            the current FSM state.
        prev_output : Optional[Dict[str, Any]]
            the previous post-processing output.

        Returns
        -------
        tb_name : str
            cell name of the next testbench.  Should incorporate self.meas_name to avoid
            collision with testbench for other designs.
        tb_type : str
            the next testbench type.
        tb_specs : str
            the testbench specification dictionary.
        tb_params : Optional[Dict[str, Any]]
            the next testbench schematic parameters.  If we are reusing an existing
            testbench, this should be None.
        """
        tb_type = state
        tb_name = self.get_testbench_name(tb_type)
        tb_specs = self.get_testbench_specs(tb_type).copy()
        tb_params = self.get_default_tb_sch_params(tb_type)

        return tb_name, tb_type, tb_specs, tb_params

    @abc.abstractmethod
    def process_output(self, state: str, data: MDArray, tb_manager: TestbenchManager
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        """Process simulation output data.

        Parameters
        ----------
        state : str
            the current FSM state
        data : MDArray
            simulation data dictionary.
        tb_manager : TestbenchManager
            the testbench manager object.

        Returns
        -------
        done : bool
            True if this measurement is finished.
        next_state : str
            the next FSM state.
        output : Dict[str, Any]
            a dictionary containing post-processed data.
        """
        return False, '', {}

    def get_testbench_name(self, tb_type: str) -> str:
        """Returns a default testbench name given testbench type."""
        return f'{self._meas_name}_TB_{tb_type}'

    async def async_measure_performance(self, sch_db: ModuleDB, dut_cvi_list: List[Any],
                                        dut_netlist: Path, load_from_file: bool = False
                                        ) -> Dict[str, Any]:
        """A coroutine that performs measurement.

        The measurement is done like a FSM.  On each iteration, depending on the current
        state, it creates a new testbench (or reuse an existing one) and simulate it.
        It then post-process the simulation data to determine the next FSM state, or
        if the measurement is done.

        Parameters
        ----------
        sch_db : ModuleDB
            the schematic database.
        dut_cvi_list : List[str]
            cv_info for DUT cell netlist
        dut_netlist : Path
            netlist of DUT cell
        load_from_file : bool
            If True, then load existing simulation data instead of running actual simulation.

        Returns
        -------
        output : Dict[str, Any]
            the last dictionary returned by process_output().
        """
        cur_state = self.get_initial_state()
        prev_output = None
        done = False

        while not done:
            # create and setup testbench
            tb_name, tb_type, tb_specs, tb_sch_params = self.get_testbench_info(cur_state,
                                                                                prev_output)

            tb_package = tb_specs['tb_package']
            tb_cls_name = tb_specs['tb_class']
            tb_module = importlib.import_module(tb_package)
            tb_cls = getattr(tb_module, tb_cls_name)
            work_dir = self._dir_path / cur_state
            tb_manager: TestbenchManager = tb_cls(self._sim, work_dir, tb_name, self._impl_lib,
                                                  tb_specs, self._sim_view_list, self._env_list,
                                                  precision=self._precision)

            if load_from_file:
                print(f'Measurement {self._meas_name} in state {cur_state}, '
                      'load sim data from file.')
                try:
                    cur_results = tb_manager.load_md_array()
                except FileNotFoundError:
                    print('Cannot find data file, simulating...')
                    await tb_manager.setup_and_simulate(sch_db, tb_sch_params,
                                                        dut_cv_info_list=dut_cvi_list,
                                                        dut_netlist=dut_netlist)
                    cur_results = tb_manager.load_md_array()
            else:
                await tb_manager.setup_and_simulate(sch_db, tb_sch_params,
                                                    dut_cv_info_list=dut_cvi_list,
                                                    dut_netlist=dut_netlist)
                cur_results = tb_manager.load_md_array()

            # process and save simulation data
            print(f'Measurement {self._meas_name} in state {cur_state}, '
                  f'processing data from {tb_name}')
            done, next_state, prev_output = self.process_output(cur_state, cur_results, tb_manager)
            write_yaml(self._dir_path / f'{cur_state}.yaml', prev_output)

            cur_state = next_state

        return prev_output

    def get_state_output(self, state: str) -> Dict[str, Any]:
        """Get the post-processed output of the given state."""
        return read_yaml(self._dir_path / f'{state}.yaml')

    def get_testbench_specs(self, tb_type: str) -> Dict[str, Any]:
        """Helper method to get testbench specifications."""
        return self._specs['testbenches'][tb_type]

    def get_default_tb_sch_params(self, tb_type: str) -> Dict[str, Any]:
        """Helper method to return a default testbench schematic parameters dictionary.

        This method loads default values from specification file, the fill in dut_lib
        and dut_cell for you.

        Parameters
        ----------
        tb_type : str
            the testbench type.

        Returns
        -------
        sch_params : Dict[str, Any]
            the default schematic parameters dictionary.
        """
        tb_specs = self.get_testbench_specs(tb_type)
        wrapper_type = tb_specs['wrapper_type']

        if 'sch_params' in tb_specs:
            tb_params = tb_specs['sch_params'].copy()
        else:
            tb_params = {}

        tb_params['dut_lib'] = self._impl_lib
        tb_params['dut_cell'] = self._wrapper_lookup[wrapper_type]
        return tb_params


class DesignManager:
    """A class that manages instantiating design instances and running simulations.

    This class provides various methods to allow you to sweep design parameters
    and generate multiple instances at once.  It also provides methods for running
    simulations and helps you interface with TestbenchManager instances.

    Parameters
    ----------
    prj : BagProject
        The BagProject instance.
    spec_file : str
        the specification file name or the data directory.
    """

    def __init__(self, prj: BagProject, spec_file: str, sch_db: Optional[ModuleDB] = None,
                 lay_db: Optional[TemplateDB] = None) -> None:
        self._prj = prj

        spec_path = Path(spec_file).resolve()
        if spec_path.is_file():
            self._specs = read_yaml(spec_path)
            self._root_dir: Path = Path(self._specs['root_dir']).resolve()
        elif spec_path.is_dir():
            self._root_dir: Path = spec_path
            self._specs = read_yaml(self._root_dir / 'specs.yaml')
        else:
            raise ValueError(f'{spec_path} is neither data directory or specification file.')

        self._swp_var_list = tuple(sorted(self._specs['sweep_params'].keys()))

        impl_lib = self._specs['impl_lib']
        if sch_db is None:
            self._sch_db = ModuleDB(prj.tech_info, impl_lib, prj=prj)
        else:
            self._sch_db = sch_db

        if lay_db is None:
            self._lay_db = TemplateDB(prj.grid, impl_lib, prj=prj)
        else:
            self._lay_db = lay_db

    @classmethod
    def load_state(cls, prj: BagProject, root_dir: str) -> DesignManager:
        """Create the DesignManager instance corresponding to data in the given directory."""
        return cls(prj, root_dir)

    @classmethod
    def get_measurement_name(cls, dsn_name: str, meas_type: str) -> str:
        """Returns the measurement name.

        Parameters
        ----------
        dsn_name : str
            design cell name.
        meas_type : str
            measurement type.

        Returns
        -------
        meas_name : str
            measurement name
        """
        return f'{dsn_name}_MEAS_{meas_type}'

    @classmethod
    def get_wrapper_name(cls, dut_name: str, wrapper_name: str) -> str:
        """Returns the wrapper cell name corresponding to the given DUT."""
        return f'{dut_name}_WRAPPER_{wrapper_name}'

    @property
    def specs(self) -> Dict[str, Any]:
        """Return the specification dictionary."""
        return self._specs

    @property
    def swp_var_list(self) -> Tuple[str, ...]:
        return self._swp_var_list

    async def extract_design(self, lib_name: str, dsn_name: str,
                             rcx_params: Optional[Dict[str, Any]], netlist: Optional[Path]) -> None:
        """A coroutine that runs LVS/RCX on a given design.

        Parameters
        ----------
        lib_name : str
            library name.
        dsn_name : str
            design cell name.
        rcx_params : Optional[Dict[str, Any]]
            extraction parameters dictionary.
        netlist: Path
            CDL netlist path
        """
        print(f'Running LVS on {dsn_name}')
        lvs_passed, lvs_log = await self._prj.async_run_lvs(lib_name, dsn_name, netlist=netlist)
        if not lvs_passed:
            raise ValueError('LVS failed for %s.  Log file: %s' % (dsn_name, lvs_log))

        print(f'LVS passed on {dsn_name}')
        print(f'Running RCX on {dsn_name}')
        rcx_passed, rcx_log = await self._prj.async_run_rcx(lib_name, dsn_name,
                                                            rcx_params=rcx_params)
        if not rcx_passed:
            raise ValueError(f'RCX failed for {dsn_name}.  Log file: {rcx_log}')
        print(f'RCX passed on {dsn_name}')

    async def verify_design(self, lib_name: str, dsn_name: str,
                            dut_cvi_list: List[Any], dut_netlist: Path,
                            load_from_file: bool = False, ) -> None:
        """Run all measurements on the given design.

        Parameters
        ----------
        lib_name : str
            library name.
        dsn_name : str
            design cell name.
        dut_cvi_list : List[str]
            cv_info for DUT cell netlist
        dut_netlist : Path
            netlist of DUT cell
        load_from_file : bool
            If True, then load existing simulation data instead of running actual simulation.
        """
        meas_list = self.specs['measurements']
        summary_fname = self.specs['summary_fname']
        view_name = self.specs['view_name']
        env_list = self.specs['env_list']
        wrapper_list = self.specs['dut_wrappers']

        wrapper_lookup = {'': dsn_name}
        for wrapper_config in wrapper_list:
            wrapper_type = wrapper_config['name']
            wrapper_lookup[wrapper_type] = self.get_wrapper_name(dsn_name, wrapper_type)

        result_summary = {}
        dsn_data_dir = self._root_dir / dsn_name
        for meas_specs in meas_list:
            meas_type = meas_specs['meas_type']
            meas_package = meas_specs['meas_package']
            meas_cls_name = meas_specs['meas_class']
            out_fname = meas_specs['out_fname']
            meas_name = self.get_measurement_name(dsn_name, meas_type)
            data_dir = self.get_measurement_directory(dsn_name, meas_type)

            meas_module = importlib.import_module(meas_package)
            meas_cls = getattr(meas_module, meas_cls_name)

            meas_manager: MeasurementManager = meas_cls(self._prj.sim_access, data_dir, meas_name,
                                                        lib_name, meas_specs, wrapper_lookup,
                                                        [(dsn_name, view_name)], env_list)
            print(f'Performing measurement {meas_name} on {dsn_name}')
            meas_res = await meas_manager.async_measure_performance(self._sch_db,
                                                                    dut_cvi_list=dut_cvi_list,
                                                                    dut_netlist=dut_netlist,
                                                                    load_from_file=load_from_file)
            print(f'Measurement {meas_name} finished on {dsn_name}')

            write_yaml(data_dir / out_fname, meas_res)
            result_summary[meas_type] = meas_res

        write_yaml(dsn_data_dir / summary_fname, result_summary)

    async def main_task(self, lib_name: str, dsn_name: str, rcx_params: Optional[Dict[str, Any]],
                        dut_cv_info_list: List[str], dut_cdl_netlist: Path, dut_sim_netlist: Path,
                        extract: bool = True, measure: bool = True, load_from_file: bool = False,
                        ) -> None:
        """The main coroutine."""
        if extract:
            await self.extract_design(lib_name, dsn_name, rcx_params, netlist=dut_cdl_netlist)
        if measure:
            await self.verify_design(lib_name, dsn_name, load_from_file=load_from_file,
                                     dut_cvi_list=dut_cv_info_list, dut_netlist=dut_sim_netlist)

    def characterize_designs(self, generate: bool = True, measure: bool = True,
                             load_from_file: bool = False) -> None:
        """Sweep all designs and characterize them.

        Parameters
        ----------
        generate : bool
            If True, create schematic/layout and run LVS/RCX.
        measure : bool
            If True, run all measurements.
        load_from_file : bool
            If True, measurements will load existing simulation data
            instead of running simulations.
        """
        if generate:
            extract = self.specs['view_name'] != 'schematic'
        else:
            extract = False
        dut_cv_info_list, dut_cdl_netlist, dut_sim_netlist = self.create_designs(extract)

        rcx_params = self.specs.get('rcx_params', None)
        impl_lib = self.specs['impl_lib']
        dsn_name_list = [self.get_design_name(combo_list)
                         for combo_list in self.get_combinations_iter()]

        coro_list = [self.main_task(impl_lib, dsn_name, rcx_params, extract=extract,
                                    measure=measure, load_from_file=load_from_file,
                                    dut_cv_info_list=dut_cv_info_list,
                                    dut_cdl_netlist=dut_cdl_netlist,
                                    dut_sim_netlist=dut_sim_netlist)
                     for dsn_name in dsn_name_list]

        results = batch_async_task(coro_list)
        if results is not None:
            for val in results:
                if isinstance(val, Exception):
                    raise val

    def get_result(self, dsn_name: str) -> Dict[str, Any]:
        """Returns the measurement result summary dictionary.

        Parameters
        ----------
        dsn_name : str
            the design name.

        Returns
        -------
        result : Dict[str, Any]
            the result dictionary.
        """
        return read_yaml(self._root_dir / dsn_name / self._specs['summary_fname'])

    def test_layout(self, gen_sch: bool = True) -> None:
        """Create a test schematic and layout for debugging purposes"""

        sweep_params = self.specs['sweep_params']
        dsn_name = self.specs['dsn_basename'] + '_TEST'

        val_list = tuple((sweep_params[key][0] for key in self.swp_var_list))
        lay_params = self.get_layout_params(val_list)

        print('create test layout')
        sch_params_list = self.create_dut_layouts([lay_params], [dsn_name], self._lay_db)

        if gen_sch:
            print('create test schematic')
            self.create_dut_schematics(sch_params_list, [dsn_name], self._sch_db,
                                       gen_wrappers=False)
        print('done')

    def create_designs(self, create_layout: bool) -> [List[str], Path, Path]:
        """Create DUT schematics/layouts.
        """
        # make layouts
        dsn_name_list, lay_params_list, combo_list_list = [], [], []
        for combo_list in self.get_combinations_iter():
            dsn_name = self.get_design_name(combo_list)
            lay_params = self.get_layout_params(combo_list)
            dsn_name_list.append(dsn_name)
            lay_params_list.append(lay_params)
            combo_list_list.append(combo_list)

        if create_layout:
            print('creating all layouts.')
            sch_params_list = self.create_dut_layouts(lay_params_list, dsn_name_list, self._lay_db)
        else:
            print('schematic simulation, skipping layouts.')
            sch_params_list = [self.get_schematic_params(combo_list)
                               for combo_list in combo_list_list]

        print('creating all schematics.')
        dut_cv_info_list, dut_cdl_netlist, dut_sim_netlist = self.create_dut_schematics(
            sch_params_list, dsn_name_list, self._sch_db, gen_wrappers=True)

        print('design generation done.')
        return dut_cv_info_list, dut_cdl_netlist, dut_sim_netlist

    def get_swp_var_values(self, var: str) -> List[Any]:
        """Returns a list of valid sweep variable values.

        Parameter
        ---------
        var : str
            the sweep variable name.

        Returns
        -------
        val_list : List[Any]
            the sweep values of the given variable.
        """
        return self.specs['sweep_params'][var]

    def get_combinations_iter(self) -> Iterable[Tuple[Any, ...]]:
        """Returns an iterator of schematic parameter combinations we sweep over.

        Returns
        -------
        combo_iter : Iterable[Tuple[Any, ...]]
            an iterator of tuples of schematic parameters values that we sweep over.
        """

        swp_par_dict = self.specs['sweep_params']
        return itertools.product(*(swp_par_dict[var] for var in self.swp_var_list))

    def get_dsn_name_iter(self) -> Iterable[str]:
        """Returns an iterator over design names.

        Returns
        -------
        dsn_name_iter : Iterable[str]
            an iterator of design names.
        """
        return (self.get_design_name(combo_list) for combo_list in self.get_combinations_iter())

    def get_measurement_directory(self, dsn_name: str, meas_type: str) -> Path:
        meas_name = self.get_measurement_name(dsn_name, meas_type)
        return self._root_dir / dsn_name / meas_name

    def get_layout_params(self, val_list: Tuple[Any, ...]) -> Dict[str, Any]:
        """Returns the layout dictionary from the given sweep parameter values."""
        lay_params = self.specs['layout_params'].copy()
        for var, val in zip(self.swp_var_list, val_list):
            lay_params[var] = val

        return lay_params

    def get_schematic_params(self, val_list: Tuple[Any, ...]) -> Dict[str, Any]:
        """Returns the layout dictionary from the given sweep parameter values."""
        lay_params = self.specs['schematic_params'].copy()
        for var, val in zip(self.swp_var_list, val_list):
            lay_params[var] = val

        return lay_params

    def create_dut_schematics(self, sch_params_list: Sequence[Dict[str, Any]],
                              cell_name_list: Sequence[str], sch_db: ModuleDB,
                              gen_wrappers: bool = True) -> [List[Any], Path, Path]:
        dut_lib = self.specs['dut_lib']
        dut_cell = self.specs['dut_cell']
        impl_lib = self.specs['impl_lib']
        wrapper_list = self.specs['dut_wrappers']

        info_list = []
        for sch_params, cur_name in zip(sch_params_list, cell_name_list):
            gen_cls = cast(Type[ModuleType], ModuleDB.get_schematic_class(dut_lib, dut_cell))
            sch_master = sch_db.new_master(gen_cls, sch_params)
            info_list.append((sch_master, cur_name))
            if gen_wrappers:
                for wrapper_config in wrapper_list:
                    wrapper_name = wrapper_config['name']
                    wrapper_lib = wrapper_config['lib']
                    wrapper_cell = wrapper_config['cell']
                    wrapper_params = wrapper_config['params'].copy()
                    wrapper_params['dut_lib'] = impl_lib
                    wrapper_params['dut_cell'] = cur_name
                    gen_cls_wrap = cast(Type[ModuleType],
                                        ModuleDB.get_schematic_class(wrapper_lib, wrapper_cell))
                    sch_master_wrap = sch_db.new_master(gen_cls_wrap, wrapper_params)
                    info_list.append((sch_master_wrap,
                                      self.get_wrapper_name(cur_name, wrapper_name)))

        sch_db.batch_schematic(info_list)
        print('DUT schematic creation done')

        netlist_type = self._prj.sim_access.netlist_type
        ext = get_extension(netlist_type)

        dut_cv_info_list = []
        dir_path = self._root_dir / 'designs' / impl_lib
        dir_path.mkdir(parents=True, exist_ok=True)
        dut_netlist_sim = dir_path / f'{dut_cell}.{ext}'
        print('Creating DUT netlist')
        sch_db.batch_schematic(info_list, output=netlist_type, top_subckt=True,
                               cv_info_out=dut_cv_info_list, fname=str(dut_netlist_sim))
        print('DUT netlist done')

        dut_netlist_cdl = dir_path / f'{dut_cell}.cdl'
        print('Creating DUT CDL netlist')
        sch_db.batch_schematic(info_list, output=DesignOutput.CDL, fname=str(dut_netlist_cdl))
        print('DUT CDL netlist done')

        return dut_cv_info_list, dut_netlist_cdl, dut_netlist_sim

    def create_dut_layouts(self, lay_params_list: Sequence[Dict[str, Any]],
                           cell_name_list: Sequence[str], temp_db: TemplateDB
                           ) -> Sequence[Dict[str, Any]]:
        """Create multiple layouts"""
        cls_package = self.specs['layout_package']
        cls_name = self.specs['layout_class']

        lay_module = importlib.import_module(cls_package)
        temp_cls = getattr(lay_module, cls_name)

        info_list, sch_params_list = [], []
        for lay_params, cell_name in zip(lay_params_list, cell_name_list):
            template = temp_db.new_template(params=lay_params, temp_cls=temp_cls, debug=False)
            info_list.append((template, cell_name))
            sch_params_list.append(template.sch_params)
        temp_db.batch_layout(info_list)
        return sch_params_list

    def get_design_name(self, combo_list: Sequence[Any, ...]) -> str:
        """Generate cell names based on sweep parameter values."""

        name_base = self.specs['dsn_basename']
        suffix = ''
        for var, val in zip(self.swp_var_list, combo_list):
            if isinstance(val, str):
                suffix += '_%s_%s' % (var, val)
            elif isinstance(val, int):
                suffix += '_%s_%d' % (var, val)
            elif isinstance(val, float):
                suffix += '_%s_%s' % (var, float_to_si_string(val))
            else:
                raise ValueError('Unsupported parameter type: %s' % (type(val)))

        return name_base + suffix
