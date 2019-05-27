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

"""This is the core bag module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional, Union, Type, Sequence, TypeVar

import os
import cProfile
import pstats

from pybag.enum import DesignOutput

from .interface import ZMQDealer
from .layout.routing.grid import RoutingGrid
from .layout.template import TemplateDB
from .layout.tech import TechInfo
from .concurrent.core import batch_async_task
from .env import (
    get_port_number, get_bag_config, get_bag_work_dir, create_routing_grid, get_bag_tmp_dir
)
from .util.importlib import import_class

if TYPE_CHECKING:
    from .interface.simulator import SimAccess
    from .interface.database import DbAccess
    from .layout.template import TemplateBase
    from .design.module import Module
    from .design.database import ModuleDB

    ModuleType = TypeVar('ModuleType', bound=Module)
    TemplateType = TypeVar('TemplateType', bound=TemplateBase)


class BagProject(object):
    """The main bag controller class.

    This class mainly stores all the user configurations, and issue
    high level bag commands.

    Attributes
    ----------
    bag_config : Dict[str, Any]
        the BAG configuration parameters dictionary.
    """

    def __init__(self) -> None:
        self.bag_config = get_bag_config()

        bag_tmp_dir = get_bag_tmp_dir()
        bag_work_dir = get_bag_work_dir()

        # get port files
        port, msg = get_port_number(bag_config=self.bag_config)
        if msg:
            print(f'*WARNING* {msg}.  Operating without Virtuoso.')

        # create ZMQDealer object
        dealer_kwargs = {}
        dealer_kwargs.update(self.bag_config['socket'])
        del dealer_kwargs['port_file']

        # create TechInfo instance
        self._grid = create_routing_grid()

        if port >= 0:
            # make DbAccess instance.
            dealer = ZMQDealer(port, **dealer_kwargs)
        else:
            dealer = None

        # create database interface object
        try:
            lib_defs_file = os.path.join(bag_work_dir, self.bag_config['lib_defs'])
        except ValueError:
            lib_defs_file = ''
        db_cls = import_class(self.bag_config['database']['class'])
        self.impl_db: DbAccess = db_cls(dealer, bag_tmp_dir, self.bag_config['database'],
                                        lib_defs_file)
        self._default_lib_path = self.impl_db.default_lib_path

        # make SimAccess instance.
        sim_cls = import_class(self.bag_config['simulation']['class'])
        self._sim: SimAccess = sim_cls(bag_tmp_dir, self.bag_config['simulation'])

    @property
    def tech_info(self) -> TechInfo:
        """TechInfo: the TechInfo object."""
        return self._grid.tech_info

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: the global routing grid object."""
        return self._grid

    @property
    def default_lib_path(self) -> str:
        return self._default_lib_path

    @property
    def sim_access(self) -> SimAccess:
        return self._sim

    def close_bag_server(self) -> None:
        """Close the BAG database server."""
        self.impl_db.close()
        self.impl_db = None

    def import_sch_cellview(self, lib_name: str, cell_name: str,
                            view_name: str = 'schematic') -> None:
        """Import the given schematic and symbol template into Python.

        This import process is done recursively.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        view_name : str
            view name.
        """
        self.impl_db.import_sch_cellview(lib_name, cell_name, view_name)

    def import_design_library(self, lib_name, view_name='schematic'):
        # type: (str, str) -> None
        """Import all design templates in the given library from CAD database.

        Parameters
        ----------
        lib_name : str
            name of the library.
        view_name : str
            the view name to import from the library.
        """
        self.impl_db.import_design_library(lib_name, view_name)

    def get_cells_in_library(self, lib_name):
        # type: (str) -> Sequence[str]
        """Get a list of cells in the given library.

        Returns an empty list if the given library does not exist.

        Parameters
        ----------
        lib_name : str
            the library name.

        Returns
        -------
        cell_list : Sequence[str]
            a list of cells in the library
        """
        return self.impl_db.get_cells_in_library(lib_name)

    def make_template_db(self, impl_lib, **kwargs):
        # type: (str, **Any) -> TemplateDB
        """Create and return a new TemplateDB instance.

        Parameters
        ----------
        impl_lib : str
            the library name to put generated layouts in.
        **kwargs : Any
            optional TemplateDB parameters.
        """
        tdb = TemplateDB(self.grid, impl_lib, prj=self, **kwargs)

        return tdb

    def make_module_db(self, impl_lib, **kwargs):
        # type: (str, **Any) -> ModuleDB
        """Create and return a new ModuleDB instance.

        Parameters
        ----------
        impl_lib : str
            the library name to put generated layouts in.
        **kwargs : Any
            optional ModuleDB parameters.
        """
        return ModuleDB(self.tech_info, impl_lib, prj=self, **kwargs)

    def generate_cell(self,  # type: BagProject
                      specs,  # type: Dict[str, Any]
                      temp_cls,  # type: Type[TemplateType]
                      sch_cls=None,  # type: Optional[Type[ModuleType]]
                      gen_lay=True,  # type: bool
                      gen_sch=False,  # type: bool
                      run_lvs=False,  # type: bool
                      run_rcx=False,  # type: bool
                      debug=False,  # type: bool
                      profile_fname='',  # type: str
                      **kwargs,
                      ):
        # type: (...) -> Optional[pstats.Stats]
        """Generate layout/schematic of a given cell from specification file.

        Parameters
        ----------
        specs : Dict[str, Any]
            the specification dictionary.
        temp_cls : Type[TemplateType]
            the layout generator class.
        sch_cls : Optional[Type[ModuleType]]
            the schematic generator class.
        gen_lay : bool
            True to generate layout.
        gen_sch : bool
            True to generate schematics.
        run_lvs : bool
            True to run LVS.
        run_rcx : bool
            True to run RCX.
        debug : bool
            True to print debug messages.
        profile_fname : str
            If not empty, profile layout generation, and save statistics to this file.
        **kwargs :
            Additional optional arguments.

        Returns
        -------
        stats : pstats.Stats
            If profiling is enabled, the statistics object.
        """
        prefix = kwargs.pop('prefix', '')
        suffix = kwargs.pop('suffix', '')
        output_lay = kwargs.pop('output_lay', DesignOutput.LAYOUT)
        output_sch = kwargs.pop('output_sch', DesignOutput.SCHEMATIC)
        options_lay = kwargs.pop('options_lay', {})
        options_sch = kwargs.pop('options_sch', {})

        impl_lib = specs['impl_lib']
        impl_cell = specs['impl_cell']
        params = specs['params']
        gen_sch = gen_sch and (sch_cls is not None)

        if gen_lay or gen_sch:
            temp_db = self.make_template_db(impl_lib, name_prefix=prefix, name_suffix=suffix)

            name_list = [impl_cell]
            print('computing layout...')
            if profile_fname:
                profiler = cProfile.Profile()
                profiler.runcall(temp_db.new_master, gen_cls=temp_cls, params=params,
                                 debug=False)
                profiler.dump_stats(profile_fname)
                result = pstats.Stats(profile_fname).strip_dirs()
            else:
                result = None

            temp = temp_db.new_master(temp_cls, params=params, debug=debug)
            print('computation done.')
            temp_list = [temp]

            if gen_lay:
                print('creating layout...')
                temp_db.batch_output(output_lay, temp_list, name_list=name_list, debug=debug,
                                     **options_lay)
                print('layout done.')

            if gen_sch:
                module_db = self.make_module_db(impl_lib, name_prefix=prefix, name_suffix=suffix)
                print('computing schematic...')
                dsn = module_db.new_master(sch_cls, params=params, debug=debug)
                print('computation done.')
                print('creating schematic...')
                module_db.batch_output(output_sch, [dsn], name_list=[impl_cell], debug=debug,
                                       **options_sch)
                print('schematic done.')
        else:
            result = None

        lvs_passed = False
        if run_lvs or run_rcx:
            print('running lvs...')
            lvs_passed, lvs_log = self.run_lvs(impl_lib, impl_cell)
            print('LVS log: %s' % lvs_log)
            if lvs_passed:
                print('LVS passed!')
            else:
                print('LVS failed...')
        if lvs_passed and run_rcx:
            print('running rcx...')
            rcx_passed, rcx_log = self.run_rcx(impl_lib, impl_cell)
            print('RCX log: %s' % rcx_log)
            if rcx_passed:
                print('RCX passed!')
            else:
                print('RCX failed...')

        return result

    def create_library(self, lib_name, lib_path=''):
        # type: (str, str) -> None
        """Create a new library if one does not exist yet.

        Parameters
        ----------
        lib_name : str
            the library name.
        lib_path : str
            directory to create the library in.  If Empty, use default location.
        """
        return self.impl_db.create_library(lib_name, lib_path=lib_path)

    def instantiate_schematic(self, lib_name, content_list, lib_path=''):
        # type: (str, Sequence[Any], str) -> None
        """Create the given schematic contents in CAD database.

        NOTE: this is BAG's internal method.  To create schematics, call batch_schematic() instead.

        Parameters
        ----------
        lib_name : str
            name of the new library to put the schematic instances.
        content_list : Sequence[Any]
            list of schematics to create.
        lib_path : str
            the path to create the library in.  If empty, use default location.
        """
        self.impl_db.instantiate_schematic(lib_name, content_list, lib_path=lib_path)

    def instantiate_layout_pcell(self, lib_name, cell_name, inst_lib, inst_cell, params,
                                 pin_mapping=None, view_name='layout'):
        # type: (str, str, str, str, Dict[str, Any], Optional[Dict[str, str]], str) -> None
        """Create a layout cell with a single pcell instance.

        Parameters
        ----------
        lib_name : str
            layout library name.
        cell_name : str
            layout cell name.
        inst_lib : str
            pcell library name.
        inst_cell : str
            pcell cell name.
        params : Dict[str, Any]
            the parameter dictionary.
        pin_mapping: Optional[Dict[str, str]]
            the pin renaming dictionary.
        view_name : str
            layout view name, default is "layout".
        """
        pin_mapping = pin_mapping or {}
        self.impl_db.instantiate_layout_pcell(lib_name, cell_name, view_name,
                                              inst_lib, inst_cell, params, pin_mapping)

    def instantiate_layout(self, lib_name, content_list, lib_path='', view='layout'):
        # type: (str, Sequence[Any], str, str) -> None
        """Create a batch of layouts.

        Parameters
        ----------
        lib_name : str
            layout library name.
        content_list : Sequence[Any]
            list of layouts to create
        lib_path : str
            the path to create the library in.  If empty, use default location.
        view : str
            layout view name.
        """
        self.impl_db.instantiate_layout(lib_name, content_list, lib_path=lib_path, view=view)

    def release_write_locks(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        """Release write locks from all the given cells.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        self.impl_db.release_write_locks(lib_name, cell_view_list)

    def refresh_cellviews(self, lib_name, cell_view_list):
        # type: (str, Sequence[Tuple[str, str]]) -> None
        """Refresh the given cellviews in the database.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_view_list : Sequence[Tuple[str, str]]
            list of cell/view name tuples.
        """
        self.impl_db.refresh_cellviews(lib_name, cell_view_list)

    def perform_checks_on_cell(self, lib_name, cell_name, view_name):
        # type: (str, str, str) -> None
        """Perform checks on the given cell.

        Parameters
        ----------
        lib_name : str
            the library name.
        cell_name : str
            the cell name.
        view_name : str
            the view name.
        """
        self.impl_db.perform_checks_on_cell(lib_name, cell_name, view_name)

    def run_lvs(self,  # type: BagProject
                lib_name,  # type: str
                cell_name,  # type: str
                **kwargs  # type: Any
                ):
        # type: (...) -> Tuple[bool, str]
        """Run LVS on the given cell.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        coro = self.impl_db.async_run_lvs(lib_name, cell_name, **kwargs)
        results = batch_async_task([coro])
        if results is None or isinstance(results[0], Exception):
            return False, ''
        return results[0]

    def run_rcx(self,  # type: BagProject
                lib_name,  # type: str
                cell_name,  # type: str
                **kwargs  # type: Any
                ):
        # type: (...) -> Tuple[Union[bool, Optional[str]], str]
        """Run RCX on the given cell.

        The behavior and the first return value of this method depends on the
        input arguments.  The second return argument will always be the RCX
        log file name.

        If create_schematic is True, this method will run RCX, then if it succeeds,
        create a schematic of the extracted netlist in the database.  It then returns
        a boolean value which will be True if RCX succeeds.

        If create_schematic is False, this method will run RCX, then return a string
        which is the extracted netlist filename. If RCX failed, None will be returned
        instead.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
            override RCX parameter values.
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : Union[bool, str]
            The return value, as described.
        log_fname : str
            name of the RCX log file.
        """
        create_schematic = kwargs.get('create_schematic', True)

        coro = self.impl_db.async_run_rcx(lib_name, cell_name, **kwargs)
        results = batch_async_task([coro])
        if results is None or isinstance(results[0], Exception):
            if create_schematic:
                return False, ''
            else:
                return None, ''
        return results[0]

    def export_layout(self, lib_name, cell_name, out_file, **kwargs):
        # type: (str, str, str, **Any) -> str
        """export layout.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell name.
        out_file : str
            output file name.
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.

        Returns
        -------
        log_fname : str
            log file name.  Empty if task cancelled.
        """
        coro = self.impl_db.async_export_layout(lib_name, cell_name, out_file, **kwargs)
        results = batch_async_task([coro])
        if results is None or isinstance(results[0], Exception):
            return ''
        return results[0]

    def batch_export_layout(self, info_list):
        # type: (Sequence[Tuple[Any, ...]]) -> Optional[Sequence[str]]
        """Export layout of all given cells

        Parameters
        ----------
        info_list:
            list of cell information.  Each element is a tuple of:

            lib_name : str
                library name.
            cell_name : str
                cell name.
            out_file : str
                layout output file name.
            view_name : str
                layout view name.  Optional.
            params : Optional[Dict[str, Any]]
                optional export parameter values.

        Returns
        -------
        results : Optional[Sequence[str]]
            If task is cancelled, return None.  Otherwise, this is a
            list of log file names.
        """
        coro_list = [self.impl_db.async_export_layout(*info) for info in info_list]
        temp_results = batch_async_task(coro_list)
        if temp_results is None:
            return None
        return ['' if isinstance(val, Exception) else val for val in temp_results]

    async def async_run_lvs(self, lib_name: str, cell_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """A coroutine for running LVS.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
        **kwargs : Any
            optional keyword arguments.  See Checker class for details.
            LVS parameters should be specified as lvs_params.

        Returns
        -------
        value : bool
            True if LVS succeeds
        log_fname : str
            name of the LVS log file.
        """
        return await self.impl_db.async_run_lvs(lib_name, cell_name, **kwargs)

    async def async_run_rcx(self,  # type: BagProject
                            lib_name: str,
                            cell_name: str,
                            **kwargs: Any
                            ) -> Tuple[Union[bool, Optional[str]], str]:
        """Run RCX on the given cell.

        The behavior and the first return value of this method depends on the
        input arguments.  The second return argument will always be the RCX
        log file name.

        If create_schematic is True, this method will run RCX, then if it succeeds,
        create a schematic of the extracted netlist in the database.  It then returns
        a boolean value which will be True if RCX succeeds.

        If create_schematic is False, this method will run RCX, then return a string
        which is the extracted netlist filename. If RCX failed, None will be returned
        instead.

        Parameters
        ----------
        lib_name : str
            library name.
        cell_name : str
            cell_name
            override RCX parameter values.
        **kwargs :
            optional keyword arguments.  See DbAccess class for details.

        Returns
        -------
        value : Union[bool, str]
            The return value, as described.
        log_fname : str
            name of the RCX log file.
        """
        return await self.impl_db.async_run_rcx(lib_name, cell_name, **kwargs)

    def create_schematic_from_netlist(self, netlist, lib_name, cell_name,
                                      sch_view=None, **kwargs):
        # type: (str, str, str, Optional[str], **Any) -> None
        """Create a schematic from a netlist.

        This is mainly used to create extracted schematic from an extracted netlist.

        Parameters
        ----------
        netlist : str
            the netlist file name.
        lib_name : str
            library name.
        cell_name : str
            cell_name
        sch_view : Optional[str]
            schematic view name.  The default value is implemendation dependent.
        **kwargs : Any
            additional implementation-dependent arguments.
        """
        return self.impl_db.create_schematic_from_netlist(netlist, lib_name, cell_name,
                                                          sch_view=sch_view, **kwargs)

    def create_verilog_view(self, verilog_file, lib_name, cell_name, **kwargs):
        # type: (str, str, str, **Any) -> None
        """Create a verilog view for mix-signal simulation.

        Parameters
        ----------
        verilog_file : str
            the verilog file name.
        lib_name : str
            library name.
        cell_name : str
            cell name.
        **kwargs : Any
            additional implementation-dependent arguments.
        """
        verilog_file = os.path.abspath(verilog_file)
        if not os.path.isfile(verilog_file):
            raise ValueError('%s is not a file.' % verilog_file)

        return self.impl_db.create_verilog_view(verilog_file, lib_name, cell_name, **kwargs)
