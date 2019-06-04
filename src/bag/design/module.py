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

"""This module defines base design module class and primitive design classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Any, Union, Iterable

import os
import abc
from itertools import zip_longest

from pybag.core import PySchCellView
from pybag.enum import TermType, DesignOutput, is_model_type, get_extension

from ..math import float_to_si_string
from ..util.cache import DesignMaster, Param
from .instance import SchInstance
from ..layout.tech import TechInfo


if TYPE_CHECKING:
    from .database import ModuleDB


class Module(DesignMaster):
    """The base class of all schematic generators.  This represents a schematic master.

    This class defines all the methods needed to implement a design in the CAD database.

    Parameters
    ----------
    yaml_fname : str
        the netlist information file name.
    database : ModuleDB
        the design database object.
    params : Param
        the parameters dictionary.
    copy_state : Optional[Dict[str, Any]]
        If not None, set content of this master from this dictionary.
    **kwargs : Any
        optional arguments
    """

    def __init__(self, yaml_fname: str, database: ModuleDB, params: Param, *,
                 copy_state: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._cv = None  # type: Optional[PySchCellView]
        if copy_state:
            self._netlist_dir = copy_state['netlist_dir']
            self._cv = copy_state['cv']
            self._pins = copy_state['pins']  # type: Dict[str, TermType]
            self._orig_cell_name = copy_state['orig_cell_name']
            self.instances = copy_state['instances']  # type: Dict[str, SchInstance]
        else:
            self._pins = {}  # type: Dict[str, TermType]
            if yaml_fname:
                # normal schematic
                yaml_fname = os.path.abspath(yaml_fname)
                self._netlist_dir = os.path.dirname(yaml_fname)
                self._cv = PySchCellView(yaml_fname, 'symbol')
                self._orig_cell_name = self._cv.cell_name
                self.instances = {name: SchInstance(database, ref)
                                  for (name, ref) in
                                  self._cv.inst_refs()}  # type: Dict[str, SchInstance]
                if not self.is_primitive():
                    self._cv.lib_name = database.lib_name
            else:
                # empty yaml file name, this is a BAG primitive
                self._netlist_dir = ''
                self._orig_cell_name = self.__class__.__name__.split('__')[1]
                self.instances = {}  # type: Dict[str, SchInstance]

        # initialize schematic master
        DesignMaster.__init__(self, database, params, copy_state=copy_state, **kwargs)

    @classmethod
    def get_hidden_params(cls) -> Dict[str, Any]:
        ans = DesignMaster.get_hidden_params()
        ans['model_params'] = None
        return ans

    @classmethod
    def is_primitive(cls) -> bool:
        """Returns True if this Module represents a BAG primitive.

        NOTE: This method is only used by BAG and schematic primitives.  This method prevents
        the module from being copied during design implementation.  Custom subclasses should
        not override this method.

        Returns
        -------
        is_primitive : bool
            True if this Module represents a BAG primitive.
        """
        return False

    def get_master_basename(self) -> str:
        return self.orig_cell_name

    def get_copy_state(self) -> Dict[str, Any]:
        base = DesignMaster.get_copy_state(self)
        new_cv = self._cv.get_copy()
        new_inst = {name: SchInstance(self.master_db, ref, master=self.instances[name].master)
                    for name, ref in new_cv.inst_refs()}

        base['netlist_dir'] = self._netlist_dir
        base['cv'] = new_cv
        base['pins'] = self._pins.copy()
        base['orig_cell_name'] = self._orig_cell_name
        base['instances'] = new_inst
        return base

    @property
    def tech_info(self) -> TechInfo:
        return self.master_db.tech_info

    @property
    def sch_scale(self) -> float:
        tech_info = self.master_db.tech_info
        return tech_info.resolution * tech_info.layout_unit

    @property
    def pins(self) -> Dict[str, TermType]:
        return self._pins

    @abc.abstractmethod
    def design(self, **kwargs: Any) -> None:
        """To be overridden by subclasses to design this module.

        To design instances of this module, you can
        call their :meth:`.design` method or any other ways you coded.

        To modify schematic structure, call:

        :meth:`.rename_pin`

        :meth:`.delete_instance`

        :meth:`.replace_instance_master`

        :meth:`.reconnect_instance_terminal`

        :meth:`.array_instance`
        """
        pass

    def design_model(self, model_params: Param, key: Any) -> None:
        self.params = self.params.copy(append=dict(model_params=model_params))
        self.update_signature(key)
        self._cv.cell_name = self.cell_name
        if 'view_name' not in model_params:
            # this is a hierarchical model
            if not self.instances:
                # found a leaf cell with no behavioral model
                raise ValueError('Schematic master has no instances and no behavioral model.')

            self.clear_children_key()
            for name, inst in self.instances.items():
                cur_params: Optional[Param] = model_params.get(name, None)
                if cur_params is None:
                    raise ValueError('Cannot find model parameters for instance {}'.format(name))
                inst.design_model(cur_params)
                if not inst.is_primitive:
                    self.add_child_key(inst.master_key)

    def set_param(self, key: str, val: Union[int, float, bool, str]) -> None:
        """Set schematic parameters for this master.

        This method is only used to set parameters for BAG primitives.

        Parameters
        ----------
        key : str
            parameter name.
        val : Union[int, float, bool, str]
            parameter value.
        """
        self._cv.set_param(key, val)

    def finalize(self) -> None:
        """Finalize this master instance.
        """
        # invoke design function, excluding model_params
        args = dict((k, v) for k, v in self.params.items() if k != 'model_params')
        self.design(**args)

        # get set of children master keys
        for inst in self.instances.values():
            if not inst.is_primitive:
                self.add_child_key(inst.master_key)

        if self._cv is not None:
            # get pins
            self._pins = dict(self._cv.terminals())
            # update cell name
            self._cv.cell_name = self.cell_name

        # call super finalize routine
        DesignMaster.finalize(self)

    def get_content(self, output_type: DesignOutput, rename_dict: Dict[str, str],
                    name_prefix: str, name_suffix: str) -> Tuple[str, Any]:
        cell_name = self.format_cell_name(self.cell_name, rename_dict, name_prefix, name_suffix)

        if self.is_primitive():
            return cell_name, (None, '')

        netlist = ''
        if is_model_type(output_type):
            model_params: Optional[Param] = self.params['model_params']
            if model_params is None:
                # model parameters is unset.  This happens if a behavioral model view is used
                # at a top level block, and this cell gets shadows out.
                # If this is the case, just return None so this cellview won't be netlisted.
                return cell_name, (None, '')
            view_name = model_params.get('view_name', None)
            if view_name is not None:
                ext = get_extension(output_type)
                if view_name:
                    fname = '{}.{}.{}'.format(self.orig_cell_name, view_name, ext)
                else:
                    # empty view_name, use default name
                    fname = '{}.{}'.format(self.orig_cell_name, ext)
                fname = os.path.join(os.path.dirname(self._netlist_dir), 'models', fname)
                netlist = self.master_db.generate_model_netlist(fname, cell_name, model_params)

        return cell_name, (self._cv, netlist)

    @property
    def cell_name(self) -> str:
        """The master cell name."""
        if self.is_primitive():
            return self.get_cell_name_from_parameters()
        return super(Module, self).cell_name

    @property
    def orig_cell_name(self) -> str:
        """The original schematic template cell name."""
        return self._orig_cell_name

    def should_delete_instance(self) -> bool:
        """Returns True if this instance should be deleted based on its parameters.

        This method is mainly used to delete 0 finger or 0 width transistors.  However,
        You can override this method if there exists parameter settings which corresponds
        to an empty schematic.

        Returns
        -------
        delete : bool
            True if parent should delete this instance.
        """
        return False

    def get_schematic_parameters(self) -> Dict[str, str]:
        """Returns the schematic parameter dictionary of this instance.

        NOTE: This method is only used by BAG primitives, as they are
        implemented with parameterized cells in the CAD database.  Custom
        subclasses should not override this method.

        Returns
        -------
        params : Dict[str, str]
            the schematic parameter dictionary.
        """
        return {}

    def get_cell_name_from_parameters(self) -> str:
        """Returns new cell name based on parameters.

        NOTE: This method is only used by BAG primitives.  This method
        enables a BAG primitive to change the cell master based on
        design parameters (e.g. change transistor instance based on the
        intent parameter).  Custom subclasses should not override this
        method.

        Returns
        -------
        cell : str
            the cell name based on parameters.
        """
        return self.orig_cell_name

    def rename_pin(self, old_pin: str, new_pin: str) -> None:
        """Renames an input/output pin of this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        old_pin : str
            the old pin name.
        new_pin : str
            the new pin name.
        """
        self._cv.rename_pin(old_pin, new_pin)

    def add_pin(self, new_pin: str, pin_type: Union[TermType, str]) -> None:
        """Adds a new pin to this schematic.

        NOTE: Make sure to call :meth:`.reconnect_instance_terminal` so that instances are
        connected to the new pin.

        Parameters
        ----------
        new_pin : str
            the new pin name.
        pin_type : Union[TermType, str]
            the new pin type.
        """
        if isinstance(pin_type, TermType):
            self._cv.add_pin(new_pin, pin_type)
        else:
            self._cv.add_pin(new_pin, TermType[pin_type].value)

    def remove_pin(self, remove_pin: str) -> bool:
        """Removes a pin from this schematic.

        Parameters
        ----------
        remove_pin : str
            the pin to remove.

        Returns
        -------
        success : bool
            True if the pin is successfully found and removed.
        """
        return self._cv.remove_pin(remove_pin)

    def set_pin_attribute(self, pin_name: str, key: str, val: str) -> None:
        """Set an attribute on the given pin.

        Parameters
        ----------
        pin_name : str
            the pin name.
        key : str
            the attribute name.
        val : str
            the attribute value.
        """
        self._cv.set_pin_attribute(pin_name, key, val)

    def rename_instance(self, old_name: str, new_name: str,
                        conn_list: Optional[Iterable[Tuple[str, str]]] = None) -> None:
        """Renames an instance in this schematic.

        Parameters
        ----------
        old_name : str
            the old instance name.
        new_name : str
            the new instance name.
        conn_list : Optional[Iterable[Tuple[str, str]]]
            an optional connection list.
        """
        self._cv.rename_instance(old_name, new_name)
        self.instances[new_name] = inst = self.instances.pop(old_name)
        if conn_list:
            for term, net in conn_list:
                inst.update_connection(new_name, term, net)

    def delete_instance(self, inst_name: str) -> bool:
        """Delete the instance with the given name.

        This method is identical to remove_instance().  It's here only for backwards
        compatibility.

        Parameters
        ----------
        inst_name : str
            the child instance to delete.

        Returns
        -------
        success : bool
            True if the instance is successfully found and removed.
        """
        success = self._cv.remove_instance(inst_name)
        if success:
            del self.instances[inst_name]
        return success

    def replace_instance_master(self, inst_name: str, lib_name: str, cell_name: str,
                                static: bool = False, keep_connections: bool = False) -> None:
        """Replace the master of the given instance.

        NOTE: all terminal connections will be reset.  Call reconnect_instance_terminal() to modify
        terminal connections.

        Parameters
        ----------
        inst_name : str
            the child instance to replace.
        lib_name : str
            the new library name.
        cell_name : str
            the new cell name.
        static : bool
            True if we're replacing instance with a static schematic instead of a design module.
        keep_connections : bool
            True to keep the old connections when the instance master changed.
        """
        if inst_name not in self.instances:
            raise ValueError('Cannot find instance with name: %s' % inst_name)

        self.instances[inst_name].change_generator(lib_name, cell_name, static=static,
                                                   keep_connections=keep_connections)

    def reconnect_instance_terminal(self, inst_name: str, term_name: str, net_name: str) -> None:
        """Reconnect the instance terminal to a new net.

        Parameters
        ----------
        inst_name : str
            the instance to modify.
        term_name : str
            the instance terminal name to reconnect.
        net_name : str
            the net to connect the instance terminal to.
        """
        inst = self.instances.get(inst_name, None)
        if inst is None:
            raise ValueError('Cannot find instance {}'.format(inst_name))

        inst.update_connection(inst_name, term_name, net_name)

    def reconnect_instance(self, inst_name: str, term_net_iter: Iterable[Tuple[str, str]]) -> None:
        """Reconnect all give instance terminals

        Parameters
        ----------
        inst_name : str
            the instance to modify.
        term_net_iter : Iterable[Tuple[str, str]]
            an iterable of (term, net) tuples.
        """
        inst = self.instances.get(inst_name, None)
        if inst is None:
            raise ValueError('Cannot find instance {}'.format(inst_name))

        for term, net in term_net_iter:
            inst.update_connection(inst_name, term, net)

    def array_instance(self, inst_name: str,
                       inst_name_list: Optional[List[str]] = None,
                       term_list: Optional[List[Dict[str, str]]] = None,
                       inst_term_list: Optional[List[Tuple[str, Iterable[Tuple[str, str]]]]] = None,
                       dx: int = 0, dy: int = 0) -> None:
        """Replace the given instance by an array of instances.

        This method will replace self.instances[inst_name] by a list of
        Modules.  The user can then design each of those modules.

        Parameters
        ----------
        inst_name : str
            the instance to array.
        inst_name_list : Optional[List[str]]
            a list of the names for each array item.
        term_list : Optional[List[Dict[str, str]]]
            a list of modified terminal connections for each array item.  The keys are
            instance terminal names, and the values are the net names to connect
            them to.  Only terminal connections different than the parent instance
            should be listed here.
            If None, assume terminal connections are not changed.
        inst_term_list : Optional[List[Tuple[str, List[Tuple[str, str]]]]]
            zipped version of inst_name_list and term_list.  If given, this is used instead.
        dx : int
            the X coordinate shift.  If dx = dy = 0, default to shift right.
        dy : int
            the Y coordinate shift.  If dx = dy = 0, default to shift right.
        """
        if inst_term_list is None:
            if inst_name_list is None:
                raise ValueError('inst_name_list cannot be None if inst_term_iter is None.')
            # get instance/terminal list iterator
            if term_list is None:
                inst_term_list = zip_longest(inst_name_list, [], fillvalue=[])
            elif len(inst_name_list) != len(term_list):
                raise ValueError('inst_name_list and term_list length mismatch.')
            else:
                inst_term_list = zip_longest(inst_name_list, (term.items() for term in term_list))
        else:
            inst_name_list = [arg[0] for arg in inst_term_list]
        # array instance
        self._cv.array_instance(inst_name, dx, dy, inst_term_list)

        # update instance dictionary
        orig_inst = self.instances.pop(inst_name)
        db = orig_inst.database
        for name in inst_name_list:
            inst_ptr = self._cv.get_inst_ref(name)
            self.instances[name] = SchInstance(db, inst_ptr, master=orig_inst.master)

    def design_dc_bias_sources(self,  vbias_dict: Optional[Dict[str, List[str]]],
                               ibias_dict: Optional[Dict[str, List[str]]],
                               vinst_name: str, iinst_name: str,
                               define_vdd: bool = True) -> None:
        """Convenience function for generating DC bias sources.

        Given DC voltage/current bias sources information, array the given voltage/current bias
        sources and configure the voltage/current.

        Each bias dictionary is a dictionary from bias source name to a 3-element list.  The first
        two elements are the PLUS/MINUS net names, respectively, and the third element is the DC
        voltage/current value as a string or float. A variable name can be given to define a
        testbench parameter.

        Parameters
        ----------
        vbias_dict : Optional[Dict[str, List[str]]]
            the voltage bias dictionary.  None or empty to disable.
        ibias_dict : Optional[Dict[str, List[str]]]
            the current bias dictionary.  None or empty to disable.
        vinst_name : str
            the DC voltage source instance name.
        iinst_name : str
            the DC current source instance name.
        define_vdd : bool
            True to include a supply voltage source connected to VDD/VSS, with voltage value 'vdd'.
        """
        if define_vdd and 'SUP' not in vbias_dict:
            vbias_dict = vbias_dict.copy()
            vbias_dict['SUP'] = ['VDD', 'VSS', 'vdd']

        for bias_dict, name_template, param_name, inst_name in \
                ((vbias_dict, 'V%s', 'vdc', vinst_name), (ibias_dict, 'I%s', 'idc', iinst_name)):
            if bias_dict:
                name_list, term_list, val_list, param_dict_list = [], [], [], []
                for name in sorted(bias_dict.keys()):
                    value_tuple = bias_dict[name]
                    pname, nname, bias_val = value_tuple[:3]
                    param_dict = value_tuple[3] if len(value_tuple) > 3 \
                        else None  # type: Optional[Dict]
                    term_list.append(dict(PLUS=pname, MINUS=nname))
                    name_list.append(name_template % name)
                    param_dict_list.append(param_dict)
                    if isinstance(bias_val, str):
                        val_list.append(bias_val)
                    elif isinstance(bias_val, int) or isinstance(bias_val, float):
                        val_list.append(float_to_si_string(bias_val))
                    else:
                        raise ValueError('value %s of type %s '
                                         'not supported' % (bias_val, type(bias_val)))

                self.array_instance(inst_name, name_list, term_list=term_list)
                for name, val, param_dict in zip(name_list, val_list, param_dict_list):
                    inst = self.instances[name]
                    inst.set_param(param_name, val)
                    if param_dict is not None:
                        for k, v in param_dict.items():
                            if isinstance(v, str):
                                pass
                            elif isinstance(v, int) or isinstance(v, float):
                                v = float_to_si_string(v)
                            else:
                                raise ValueError('value %s of type %s not supported' % (v, type(v)))

                            inst.set_param(k, v)
            else:
                self.delete_instance(inst_name)

    def design_dummy_transistors(self, dum_info: List[Tuple[Any]], inst_name: str, vdd_name: str,
                                 vss_name: str, net_map: Optional[Dict[str, str]] = None) -> None:
        """Convenience function for generating dummy transistor schematic.

        Given dummy information (computed by AnalogBase) and a BAG transistor instance,
        this method generates dummy schematics by arraying and modifying the BAG
        transistor instance.

        Parameters
        ----------
        dum_info : List[Tuple[Any]]
            the dummy information data structure.
        inst_name : str
            the BAG transistor instance name.
        vdd_name : str
            VDD net name.  Used for PMOS dummies.
        vss_name : str
            VSS net name.  Used for NMOS dummies.
        net_map : Optional[Dict[str, str]]
            optional net name transformation mapping.
        """
        if not dum_info:
            self.delete_instance(inst_name)
        else:
            num_arr = len(dum_info)
            arr_name_list = ['XDUMMY%d' % idx for idx in range(num_arr)]
            self.array_instance(inst_name, arr_name_list)

            for name, ((mos_type, w, lch, th, s_net, d_net), fg) in zip(arr_name_list, dum_info):
                if mos_type == 'pch':
                    cell_name = 'pmos4_standard'
                    sup_name = vdd_name
                else:
                    cell_name = 'nmos4_standard'
                    sup_name = vss_name
                if net_map is not None:
                    s_net = net_map.get(s_net, s_net)
                    d_net = net_map.get(d_net, d_net)
                s_name = s_net if s_net else sup_name
                d_name = d_net if d_net else sup_name
                inst = self.instances[name]
                inst.change_generator('BAG_prim', cell_name)
                inst.update_connection(name, 'G', sup_name)
                inst.update_connection(name, 'B', sup_name)
                inst.update_connection(name, 'D', d_name)
                inst.update_connection(name, 'S', s_name)
                inst.design(w=w, l=lch, nf=fg, intent=th)

    def design_transistor(self, inst_name: str, w: int, lch: int, seg: int,
                          intent: str, m: str = '', d: str = '', g: Union[str, List[str]] = '',
                          s: str = '', b: str = '', stack: int = 1, mos_type: str = '') -> None:
        """Design a BAG_prim transistor (with stacking support).

        This is a convenient method to design a stack transistor.  Additional transistors
        will be created on the right.  The intermediate nodes of each parallel segment are not
        shorted together.

        Parameters
        ----------
        inst_name : str
            name of the BAG_prim transistor instance.
        w : int
            the width of the transistor, in number of fins or resolution units.
        lch : int
            the channel length, in resolution units.
        seg : int
            number of parallel segments of stacked transistors.
        intent : str
            the threshold flavor.
        m : str
            base name of the intermediate nodes.  the intermediate nodes will be named
            'midX', where X is an non-negative integer.
        d : str
            the drain name.  Empty string to not rename.
        g : Union[str, List[str]]
            the gate name.  Empty string to not rename.
            If a list is given, then a NAND-gate structure will be built where the gate nets
            may be different.  Index 0 corresponds to the gate of the source transistor.
        s : str
            the source name.  Empty string to not rename.
        b : str
            the body name.  Empty string to not rename.
        stack : int
            number of series stack transistors.
        mos_type : str
            if non-empty, will change the transistor master to this type.
        """
        inst = self.instances[inst_name]
        if not issubclass(inst.master_class, MosModuleBase):
            raise ValueError('This method only works on BAG_prim transistors.')
        if stack <= 0 or seg <= 0:
            raise ValueError('stack and seg must be positive')

        if mos_type:
            cell_name = 'nmos4_standard' if mos_type == 'nch' else 'pmos4_standard'
            inst.change_generator('BAG_prim', cell_name, keep_connections=True)

        g_is_str = isinstance(g, str)
        if stack == 1:
            # design instance
            inst.design(w=w, l=lch, nf=seg, intent=intent)
            # connect terminals
            if not g_is_str:
                g = g[0]
            for term, net in (('D', d), ('G', g), ('S', s), ('B', b)):
                if net:
                    inst.update_connection(inst_name, term, net)
        else:
            if not m:
                raise ValueError('Intermediate node base name cannot be empty.')
            # design instance
            inst.design(w=w, l=lch, nf=1, intent=intent)
            # rename G/B
            if g_is_str and g:
                inst.update_connection(inst_name, 'G', g)
            if b:
                inst.update_connection(inst_name, 'B', b)
            if not d:
                d = inst.get_connection('D')
            if not s:
                s = inst.get_connection('S')

            if seg == 1:
                # only one segment, array instance via naming
                # rename instance
                new_name = inst_name + '<0:{}>'.format(stack - 1)
                self.rename_instance(inst_name, new_name)
                # rename D/S
                if stack > 2:
                    m += '<0:{}>'.format(stack - 2)
                new_s = s + ',' + m
                new_d = m + ',' + d
                inst.update_connection(new_name, 'D', new_d)
                inst.update_connection(new_name, 'S', new_s)
                if not g_is_str:
                    inst.update_connection(new_name, 'G', ','.join(g))
            else:
                # multiple segment and stacks, have to array instance
                # construct instance name/terminal map iterator
                inst_term_list = []
                last_cnt = (stack - 1) * seg
                g_cnt = 0
                for cnt in range(0, last_cnt + 1, seg):
                    d_suf = '<{}:{}>'.format(cnt + seg - 1, cnt)
                    s_suf = '<{}:{}>'.format(cnt - 1, cnt - seg)
                    iname = inst_name + d_suf
                    if cnt == 0:
                        s_name = s
                        d_name = m + d_suf
                    elif cnt == last_cnt:
                        s_name = m + s_suf
                        d_name = d
                    else:
                        s_name = m + s_suf
                        d_name = m + d_suf
                    term_list = [('S', s_name), ('D', d_name)]
                    if not g_is_str:
                        term_list.append(('G', g[g_cnt]))
                        g_cnt += 1
                    inst_term_list.append((iname, term_list))

                self.array_instance(inst_name, inst_term_list=inst_term_list)


class MosModuleBase(Module):
    """The base design class for the bag primitive transistor.
    """

    def __init__(self, yaml_fname: str, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, yaml_fname, database, params, **kwargs)
        self._pins = dict(G=TermType.inout, D=TermType.inout, S=TermType.inout, B=TermType.inout)

    @classmethod
    def is_primitive(cls) -> bool:
        return True

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w='transistor width, in resolution units or number of fins.',
            l='transistor length, in resolution units.',
            nf='transistor number of fingers.',
            intent='transistor threshold flavor.',
        )

    def design(self, w: int, l: int, nf: int, intent: str) -> None:
        pass

    def get_schematic_parameters(self) -> Dict[str, str]:
        w_res = self.tech_info.tech_params['mos']['width_resolution']
        l_res = self.tech_info.tech_params['mos']['length_resolution']
        scale = self.sch_scale

        w: int = self.params['w']
        l: int = self.params['l']
        nf: int = self.params['nf']

        wstr = float_to_si_string(int(round(w / w_res)) * w_res)
        lstr = float_to_si_string(int(round(l * scale / l_res)) * l_res)
        nstr = str(nf)

        return dict(w=wstr, l=lstr, nf=nstr)

    def get_cell_name_from_parameters(self) -> str:
        mos_type = self.orig_cell_name.split('_')[0]
        return '{}_{}'.format(mos_type, self.params['intent'])

    def should_delete_instance(self) -> bool:
        return self.params['nf'] == 0 or self.params['w'] == 0


class ResPhysicalModuleBase(Module):
    """The base design class for a real resistor parametrized by width and length.
    """

    def __init__(self, yaml_fname: str, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, yaml_fname, database, params, **kwargs)
        self._pins = dict(PLUS=TermType.inout, MINUS=TermType.inout, BULK=TermType.inout)

    @classmethod
    def is_primitive(cls) -> bool:
        return True

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w='resistor width, in resolution units.',
            l='resistor length, in resolution units.',
            intent='resistor flavor.',
        )

    def design(self, w: int, l: int, intent: str) -> None:
        pass

    def get_schematic_parameters(self) -> Dict[str, str]:
        w: int = self.params['w']
        l: int = self.params['l']
        scale = self.sch_scale
        wstr = float_to_si_string(w * scale)
        lstr = float_to_si_string(l * scale)

        return dict(w=wstr, l=lstr)

    def get_cell_name_from_parameters(self) -> str:
        return 'res_{}'.format(self.params['intent'])

    def should_delete_instance(self) -> bool:
        return self.params['w'] == 0 or self.params['l'] == 0


class ResMetalModule(Module):
    """The base design class for a metal resistor.
    """

    def __init__(self, yaml_fname: str, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, yaml_fname, database, params, **kwargs)
        self._pins = dict(PLUS=TermType.inout, MINUS=TermType.inout)

    @classmethod
    def is_primitive(cls) -> bool:
        return True

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w='resistor width, in resolution units.',
            l='resistor length, in resolution units.',
            layer='the metal layer ID.',
        )

    def design(self, w: int, l: int, layer: int) -> None:
        pass

    def get_schematic_parameters(self) -> Dict[str, str]:
        w: int = self.params['w']
        l: int = self.params['l']
        scale = self.sch_scale
        wstr = float_to_si_string(w * scale)
        lstr = float_to_si_string(l * scale)
        return dict(w=wstr, l=lstr)

    def get_cell_name_from_parameters(self) -> str:
        return 'res_metal_{}'.format(self.params['layer'])

    def should_delete_instance(self) -> bool:
        return self.params['w'] == 0 or self.params['l'] == 0
