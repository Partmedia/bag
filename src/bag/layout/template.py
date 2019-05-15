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

"""This module defines layout template classes.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Union, Dict, Any, List, TypeVar, Type, Optional, Tuple, Iterable, Mapping,
    Sequence
)
from bag.typing import PointType

import abc
from itertools import product

from pybag.enum import (
    PathStyle, BlockageType, BoundaryType, GeometryMode, DesignOutput, Orient2D,
    Orientation, Direction, MinLenMode, RoundMode
)
from pybag.core import (
    BBox, BBoxArray, PyLayCellView, Transform, PyLayInstRef, PyPath, PyBlockage, PyBoundary,
    PyRect, PyVia, PyPolygon, PyPolygon90, PyPolygon45, ViaParam, COORD_MIN, COORD_MAX
)

from ..util.immutable import ImmutableSortedDict
from ..util.cache import DesignMaster, MasterDB, Param
from ..util.interval import IntervalSet
from ..util.math import HalfInt
from ..io.file import write_yaml

from .core import PyLayInstance
from .tech import TechInfo
from .routing.base import Port, TrackID, WireArray
from .routing.grid import RoutingGrid
from .data import MOMCapInfo

GeoType = Union[PyRect, PyPolygon90, PyPolygon45, PyPolygon]
TemplateType = TypeVar('TemplateType', bound='TemplateBase')
DiffWarrType = Tuple[Optional[WireArray], Optional[WireArray]]

if TYPE_CHECKING:
    from bag.core import BagProject
    from bag.typing import TrackType, SizeType


class TemplateDB(MasterDB):
    """A database of all templates.

    This class is a subclass of MasterDB that defines some extra properties/function
    aliases to make creating layouts easier.

    Parameters
    ----------
    routing_grid : RoutingGrid
        the default RoutingGrid object.
    lib_name : str
        the cadence library to put all generated templates in.
    prj : Optional[BagProject]
        the BagProject instance.
    name_prefix : str
        generated layout name prefix.
    name_suffix : str
        generated layout name suffix.
    """

    def __init__(self, routing_grid: RoutingGrid, lib_name: str, prj: Optional[BagProject] = None,
                 name_prefix: str = '', name_suffix: str = '') -> None:
        MasterDB.__init__(self, lib_name, prj=prj, name_prefix=name_prefix, name_suffix=name_suffix)

        self._grid = routing_grid

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: The global RoutingGrid instance."""
        return self._grid

    @property
    def tech_info(self) -> TechInfo:
        return self._grid.tech_info

    def new_template(self, temp_cls: Type[TemplateType], params: Optional[Mapping[str, Any]] = None,
                     **kwargs: Any) -> TemplateType:
        """Alias for new_master() for backwards compatibility.
        """
        return self.new_master(temp_cls, params=params, **kwargs)

    def instantiate_layout(self, template: TemplateBase, top_cell_name: str = '',
                           output: DesignOutput = DesignOutput.LAYOUT, **kwargs: Any) -> None:
        """Alias for instantiate_master(), with default output type of LAYOUT.
        """
        self.instantiate_master(output, template, top_cell_name, **kwargs)

    def batch_layout(self, info_list: Sequence[Tuple[TemplateBase, str]],
                     output: DesignOutput = DesignOutput.LAYOUT, **kwargs: Any) -> None:
        """Alias for batch_output(), with default output type of LAYOUT.
        """
        self.batch_output(output, info_list, **kwargs)


def get_cap_via_extensions(info: MOMCapInfo, grid: RoutingGrid, bot_layer: int,
                           top_layer: int) -> Dict[int, int]:
    via_ext_dict: Dict[int, int] = {lay: 0 for lay in range(bot_layer, top_layer + 1)}
    # get via extensions on each layer
    for lay0 in range(bot_layer, top_layer):
        lay1 = lay0 + 1

        # port-to-port via extension
        bot_tr_w = info.get_port_tr_w(lay0)
        top_tr_w = info.get_port_tr_w(lay1)
        ext_pp = grid.get_via_extensions(Direction.LOWER, lay0, bot_tr_w, top_tr_w)

        w0, sp0, _, _ = info.get_cap_specs(lay0)
        w1, sp1, _, _ = info.get_cap_specs(lay1)
        # cap-to-cap via extension
        ext_cc = grid.get_via_extensions_dim(Direction.LOWER, lay0, w0, w1)
        # cap-to-port via extension
        ext_cp = grid.get_via_extensions_dim_tr(Direction.LOWER, lay0, w0, top_tr_w)
        # port-to-cap via extension
        ext_pc = grid.get_via_extensions_dim_tr(Direction.UPPER, lay1, w1, bot_tr_w)

        via_ext_dict[lay0] = max(via_ext_dict[lay0], ext_pp[0], ext_cc[0], ext_cp[0], ext_pc[0])
        via_ext_dict[lay1] = max(via_ext_dict[lay1], ext_pp[1], ext_cc[1], ext_cp[1], ext_pc[1])

    return via_ext_dict


class TemplateBase(DesignMaster):
    """The base template class.

    Parameters
    ----------
    temp_db : TemplateDB
        the template database.
    params : Param
        the parameter values.
    **kwargs : Any
        dictionary of the following optional parameters:

        grid : RoutingGrid
            the routing grid to use for this template.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        # initialize template attributes
        self._size: Optional[SizeType] = None
        self._ports = {}
        self._port_params = {}
        self._prim_ports = {}
        self._prim_port_params = {}
        self._array_box: Optional[BBox] = None
        self._fill_box: Optional[BBox] = None
        self.prim_top_layer = None
        self.prim_bound_box = None
        self._sch_params: Optional[Dict[str, Any]] = None

        # add hidden parameters
        DesignMaster.__init__(self, temp_db, params, **kwargs)

        tmp_grid = self.params['grid']
        if tmp_grid is None:
            self._grid: RoutingGrid = temp_db.grid
        else:
            self._grid: RoutingGrid = tmp_grid

        # create Cython wrapper object
        self._layout = PyLayCellView(self._grid, self.cell_name)

    @classmethod
    def get_hidden_params(cls) -> Dict[str, Any]:
        ans = DesignMaster.get_hidden_params()
        ans['grid'] = None
        ans['show_pins'] = True
        return ans

    @abc.abstractmethod
    def draw_layout(self) -> None:
        """Draw the layout of this template.

        Override this method to create the layout.

        WARNING: you should never call this method yourself.
        """
        pass

    def get_master_basename(self) -> str:
        """Returns the base name to use for this instance.

        Returns
        -------
        basename : str
            the base name for this instance.
        """
        return self.get_layout_basename()

    def get_layout_basename(self) -> str:
        """Returns the base name for this template.

        Returns
        -------
        base_name : str
            the base name of this template.
        """
        return self.__class__.__name__

    def get_content(self, output_type: DesignOutput, rename_dict: Dict[str, str],
                    name_prefix: str, name_suffix: str) -> Tuple[str, Any]:
        if not self.finalized:
            raise ValueError('This template is not finalized yet')

        cell_name = self.format_cell_name(self._layout.cell_name, rename_dict,
                                          name_prefix, name_suffix)
        return name_prefix + cell_name + name_suffix, self._layout

    def finalize(self) -> None:
        """Finalize this master instance.
        """
        # create layout
        self.draw_layout()

        # finalize this template
        grid = self.grid
        grid.tech_info.finalize_template(self)

        # construct port objects
        for net_name, port_params in self._port_params.items():
            pin_dict = port_params['pins']
            label = port_params['label']
            if port_params['show']:
                label = port_params['label']
                for wire_arr_list in pin_dict.values():
                    for warr in wire_arr_list:  # type: WireArray
                        self._layout.add_pin_arr(net_name, label, warr)
            self._ports[net_name] = Port(net_name, pin_dict, label)

        # construct primitive port objects
        for net_name, port_params in self._prim_port_params.items():
            pin_dict = port_params['pins']
            label = port_params['label']
            if port_params['show']:
                label = port_params['label']
                for layer_name, box_list in pin_dict.items():
                    for box in box_list:
                        self._layout.add_pin(layer_name, net_name, label, box)
            self._ports[net_name] = Port(net_name, pin_dict, label)

        # call super finalize routine
        DesignMaster.finalize(self)

    @property
    def show_pins(self) -> bool:
        """bool: True to show pins."""
        return self.params['show_pins']

    @property
    def sch_params(self) -> Optional[Dict[str, Any]]:
        """Optional[Dict[str, Any]]: The schematic parameters dictionary."""
        return self._sch_params

    @sch_params.setter
    def sch_params(self, new_params: Dict[str, Any]) -> None:
        self._sch_params = new_params

    @property
    def template_db(self) -> TemplateDB:
        """TemplateDB: The template database object"""
        # noinspection PyTypeChecker
        return self.master_db

    @property
    def is_empty(self) -> bool:
        """bool: True if this template is empty."""
        return self._layout.is_empty

    @property
    def grid(self) -> RoutingGrid:
        """RoutingGrid: The RoutingGrid object"""
        return self._grid

    @grid.setter
    def grid(self, new_grid: RoutingGrid) -> None:
        self._layout.set_grid(new_grid)
        self._grid = new_grid

    @property
    def array_box(self) -> Optional[BBox]:
        """Optional[BBox]: The array/abutment bounding box of this template."""
        return self._array_box

    @array_box.setter
    def array_box(self, new_array_box: BBox) -> None:
        if not self._finalized:
            self._array_box = new_array_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def fill_box(self) -> Optional[BBox]:
        """Optional[BBox]: The dummy fill bounding box of this template."""
        return self._fill_box

    @fill_box.setter
    def fill_box(self, new_box: BBox) -> None:
        if not self._finalized:
            self._fill_box = new_box
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def top_layer(self) -> int:
        """int: The top layer ID used in this template."""
        if self.size is None:
            if self.prim_top_layer is None:
                raise Exception('Both size and prim_top_layer are unset.')
            return self.prim_top_layer
        return self.size[0]

    @property
    def size(self) -> Optional[SizeType]:
        """Optional[SizeType]: The size of this template, in (layer, nx_blk, ny_blk) format."""
        return self._size

    @property
    def size_defined(self) -> bool:
        """bool: True if size or bounding box has been set."""
        return self.size is not None or self.prim_bound_box is not None

    @property
    def bound_box(self) -> Optional[BBox]:
        """Optional[BBox]: Returns the template BBox.  None if size not set yet."""
        mysize = self.size
        if mysize is None:
            if self.prim_bound_box is None:
                raise ValueError('Both size and prim_bound_box are unset.')
            return self.prim_bound_box

        wblk, hblk = self.grid.get_size_dimension(mysize)
        return BBox(0, 0, wblk, hblk)

    @size.setter
    def size(self, new_size: SizeType) -> None:
        if not self._finalized:
            self._size = new_size
        else:
            raise RuntimeError('Template already finalized.')

    @property
    def layout_cellview(self) -> PyLayCellView:
        """PyLayCellView: The internal layout object."""
        return self._layout

    def set_geometry_mode(self, mode: GeometryMode) -> None:
        """Sets the geometry mode of this layout.

        Parameters
        ----------
        mode : GeometryMode
            the geometry mode.
        """
        self._layout.set_geometry_mode(mode.value)

    def get_rect_bbox(self, lay_purp: Tuple[str, str]) -> BBox:
        """Returns the overall bounding box of all rectangles on the given layer.

        Note: currently this does not check primitive instances or vias.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.

        Returns
        -------
        box : BBox
            the overall bounding box of the given layer.
        """
        return self._layout.get_rect_bbox(lay_purp[0], lay_purp[1])

    def new_template_with(self, **kwargs: Any) -> TemplateBase:
        """Create a new template with the given parameters.

        This method will update the parameter values with the given dictionary,
        then create a new template with those parameters and return it.

        Parameters
        ----------
        **kwargs : Any
            a dictionary of new parameter values.

        Returns
        -------
        new_temp : TemplateBase
            A new layout master object.
        """
        # get new parameter dictionary.
        new_params = self.params.copy(append=kwargs)
        return self.template_db.new_template(self.__class__, params=new_params)

    def set_size_from_bound_box(self, top_layer_id: int, bbox: BBox, *, round_up: bool = False,
                                half_blk_x: bool = True, half_blk_y: bool = True):
        """Compute the size from overall bounding box.

        Parameters
        ----------
        top_layer_id : int
            the top level routing layer ID that array box is calculated with.
        bbox : BBox
            the overall bounding box
        round_up: bool
            True to round up bounding box if not quantized properly
        half_blk_x : bool
            True to allow half-block widths.
        half_blk_y : bool
            True to allow half-block heights.
        """
        grid = self.grid

        if bbox.xl != 0 or bbox.yl != 0:
            raise ValueError('lower-left corner of overall bounding box must be (0, 0).')

        if grid.size_defined(top_layer_id):
            self.size = grid.get_size_tuple(top_layer_id, bbox.w, bbox.h, round_up=round_up,
                                            half_blk_x=half_blk_x, half_blk_y=half_blk_y)
        else:
            self.prim_top_layer = top_layer_id
            self.prim_bound_box = bbox

    def set_size_from_array_box(self, top_layer_id: int) -> None:
        """Automatically compute the size from array_box.

        Assumes the array box is exactly in the center of the template.

        Parameters
        ----------
        top_layer_id : int
            the top level routing layer ID that array box is calculated with.
        """
        grid = self.grid

        array_box = self.array_box
        if array_box is None:
            raise ValueError("array_box is not set")

        dx = array_box.xl
        dy = array_box.yl
        if dx < 0 or dy < 0:
            raise ValueError('lower-left corner of array box must be in first quadrant.')

        # noinspection PyAttributeOutsideInit
        self.size = grid.get_size_tuple(top_layer_id, 2 * dx + self.array_box.width_unit,
                                        2 * dy + self.array_box.height_unit)

    def write_summary_file(self, fname: str, lib_name: str, cell_name: str) -> None:
        """Create a summary file for this template layout."""
        # get all pin information
        grid = self.grid
        tech_info = grid.tech_info
        pin_dict = {}
        res = grid.resolution
        for port_name in self.port_names_iter():
            pin_cnt = 0
            port = self.get_port(port_name)
            for pin_warr in port:
                for lay, _, bbox in pin_warr.wire_iter(grid):
                    if pin_cnt == 0:
                        pin_name = port_name
                    else:
                        pin_name = '%s_%d' % (port_name, pin_cnt)
                    pin_cnt += 1
                    pin_dict[pin_name] = dict(
                        layer=[lay, tech_info.pin_purpose],
                        netname=port_name,
                        xy0=[bbox.xl * res, bbox.yl * res],
                        xy1=[bbox.xh * res, bbox.yh * res],
                    )

        # get size information
        bnd_box = self.bound_box
        if bnd_box is None:
            raise ValueError("bound_box is not set")
        info = {
            lib_name: {
                cell_name: dict(
                    pins=pin_dict,
                    xy0=[0.0, 0.0],
                    xy1=[bnd_box.w * res, bnd_box.h * res],
                ),
            },
        }

        write_yaml(fname, info)

    def get_pin_name(self, name: str) -> str:
        """Get the actual name of the given pin from the renaming dictionary.

        Given a pin name, If this Template has a parameter called 'rename_dict',
        return the actual pin name from the renaming dictionary.

        Parameters
        ----------
        name : str
            the pin name.

        Returns
        -------
        actual_name : str
            the renamed pin name.
        """
        rename_dict = self.params.get('rename_dict', {})
        return rename_dict.get(name, name)

    def get_port(self, name: str = '') -> Port:
        """Returns the port object with the given name.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this template has only one port,
            then return it.

        Returns
        -------
        port : Port
            the port object.
        """
        if not name:
            if len(self._ports) != 1:
                raise ValueError('Template has %d ports != 1.' % len(self._ports))
            name = next(iter(self._ports))
        return self._ports[name]

    def has_port(self, port_name: str) -> bool:
        """Returns True if this template has the given port."""
        return port_name in self._ports

    def port_names_iter(self) -> Iterable[str]:
        """Iterates over port names in this template.

        Yields
        ------
        port_name : str
            name of a port in this template.
        """
        return self._ports.keys()

    def get_prim_port(self, name: str = '') -> Port:
        """Returns the primitive port object with the given name.

        Parameters
        ----------
        name : str
            the port terminal name.  If None or empty, check if this template has only one port,
            then return it.

        Returns
        -------
        port : Port
            the primitive port object.
        """
        if not name:
            if len(self._prim_ports) != 1:
                raise ValueError('Template has %d ports != 1.' % len(self._prim_ports))
            name = next(iter(self._ports))
        return self._prim_ports[name]

    def has_prim_port(self, port_name: str) -> bool:
        """Returns True if this template has the given primitive port."""
        return port_name in self._prim_ports

    def prim_port_names_iter(self) -> Iterable[str]:
        """Iterates over primitive port names in this template.

        Yields
        ------
        port_name : str
            name of a primitive port in this template.
        """
        return self._prim_ports.keys()

    def new_template(self, temp_cls: Type[TemplateType], *,
                     params: Optional[Mapping[str, Any]] = None) -> TemplateType:
        """Create a new template.

        Parameters
        ----------
        temp_cls : Type[TemplateType]
            the template class to instantiate.
        params : Optional[Mapping[str, Any]]
            the parameter dictionary.

        Returns
        -------
        template : TemplateType
            the new template instance.
        """
        if isinstance(params, ImmutableSortedDict):
            params = params.copy(append=dict(grid=self.grid))
        else:
            params['grid'] = self.grid
        return self.template_db.new_template(params=params, temp_cls=temp_cls)

    def add_instance(self,
                     master: TemplateBase,
                     *,
                     inst_name: str = '',
                     xform: Optional[Transform] = None,
                     nx: int = 1,
                     ny: int = 1,
                     spx: int = 0,
                     spy: int = 0,
                     commit: bool = True,
                     ) -> PyLayInstance:
        """Adds a new (arrayed) instance to layout.

        Parameters
        ----------
        master : TemplateBase
            the master template object.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        xform : Optional[Transform]
            the transformation object.
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : CoordType
            column pitch.  Used for arraying given instance.
        spy : CoordType
            row pitch.  Used for arraying given instance.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        inst : PyLayInstance
            the added instance.
        """
        if xform is None:
            xform = Transform()

        ref = self._layout.add_instance(master.layout_cellview, inst_name, xform, nx, ny,
                                        spx, spy, False)
        ans = PyLayInstance(self, master, ref)
        if commit:
            ans.commit()
        return ans

    def add_instance_primitive(self,
                               lib_name: str,
                               cell_name: str,
                               *,
                               xform: Optional[Transform] = None,
                               view_name: str = 'layout',
                               inst_name: str = '',
                               nx: int = 1,
                               ny: int = 1,
                               spx: int = 0,
                               spy: int = 0,
                               params: Optional[Dict[str, Any]] = None,
                               commit: bool = True,
                               **kwargs: Any,
                               ) -> PyLayInstRef:
        """Adds a new (arrayed) primitive instance to layout.

        Parameters
        ----------
        lib_name : str
            instance library name.
        cell_name : str
            instance cell name.
        xform : Optional[Transform]
            the transformation object.
        view_name : str
            instance view name.  Defaults to 'layout'.
        inst_name : Optional[str]
            instance name.  If None or an instance with this name already exists,
            a generated unique name is used.
        nx : int
            number of columns.  Must be positive integer.
        ny : int
            number of rows.  Must be positive integer.
        spx : CoordType
            column pitch.  Used for arraying given instance.
        spy : CoordType
            row pitch.  Used for arraying given instance.
        params : Optional[Dict[str, Any]]
            the parameter dictionary.  Used for adding pcell instance.
        commit : bool
            True to commit the object immediately.
        **kwargs : Any
            additional arguments.  Usually implementation specific.

        Returns
        -------
        ref : PyLayInstRef
            A reference to the primitive instance.
        """
        if not params:
            params = kwargs
        else:
            params.update(kwargs)
        if xform is None:
            xform = Transform()

        # TODO: support pcells
        if params:
            raise ValueError("layout pcells not supported yet; see developer")

        return self._layout.add_prim_instance(lib_name, cell_name, view_name, inst_name, xform,
                                              nx, ny, spx, spy, commit)

    def is_horizontal(self, layer: str) -> bool:
        """Returns True if the given layer has no direction or is horizontal."""
        lay_id = self._grid.tech_info.get_layer_id(layer)
        return (lay_id is None) or self._grid.is_horizontal(lay_id)

    def add_rect(self, lay_purp: Tuple[str, str], bbox: BBox, commit: bool = True) -> PyRect:
        """Add a new rectangle.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        bbox : BBox
            the rectangle bounding box.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        rect : PyRect
            the added rectangle.
        """
        return self._layout.add_rect(lay_purp[0], lay_purp[1], bbox, commit)

    def add_rect_arr(self, lay_purp: Tuple[str, str], barr: BBoxArray) -> None:
        """Add a new rectangle array.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        barr : BBoxArray
            the rectangle bounding box array.
        """
        self._layout.add_rect_arr(lay_purp[0], lay_purp[1], barr)

    def add_res_metal(self, layer_id: int, bbox: BBox) -> None:
        """Add a new metal resistor.

        Parameters
        ----------
        layer_id : int
            the metal layer ID.
        bbox : BBox
            the resistor bounding box.
        """
        for lay, purp in self._grid.tech_info.get_res_metal_layers(layer_id):
            self._layout.add_rect(lay, purp, bbox, True)

    def add_path(self, lay_purp: Tuple[str, str], width: int, points: List[PointType],
                 start_style: PathStyle, *, join_style: PathStyle = PathStyle.round,
                 stop_style: Optional[PathStyle] = None, commit: bool = True) -> PyPath:
        """Add a new path.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        width : int
            the path width.
        points : List[PointType]
            points defining this path.
        start_style : PathStyle
            the path beginning style.
        join_style : PathStyle
            path style for the joints.
        stop_style : Optional[PathStyle]
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        if stop_style is None:
            stop_style = start_style
        half_width = width // 2
        return self._layout.add_path(lay_purp[0], lay_purp[1], points, half_width, start_style,
                                     stop_style, join_style, commit)

    def add_path45_bus(self, lay_purp: Tuple[str, str], points: List[PointType], widths: List[int],
                       spaces: List[int], start_style: PathStyle, *,
                       join_style: PathStyle = PathStyle.round,
                       stop_style: Optional[PathStyle] = None, commit: bool = True) -> PyPath:
        """Add a path bus that only contains 45 degree turns.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        points : List[PointType]
            points defining this path.
        widths : List[int]
            width of each path in the bus.
        spaces : List[int]
            space between each path.
        start_style : PathStyle
            the path beginning style.
        join_style : PathStyle
            path style for the joints.
        stop_style : Optional[PathStyle]
            the path ending style.  Defaults to start style.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        path : PyPath
            the added path object.
        """
        if stop_style is None:
            stop_style = start_style
        return self._layout.add_path45_bus(lay_purp[0], lay_purp[1], points, widths, spaces,
                                           start_style, stop_style, join_style, commit)

    def add_polygon(self, lay_purp: Tuple[str, str], points: List[PointType],
                    commit: bool = True) -> PyPolygon:
        """Add a new polygon.

        Parameters
        ----------
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        points : List[PointType]
            vertices of the polygon.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        polygon : PyPolygon
            the added polygon object.
        """
        return self._layout.add_poly(lay_purp[0], lay_purp[1], points, commit)

    def add_blockage(self, layer: str, blk_type: BlockageType, points: List[PointType],
                     commit: bool = True) -> PyBlockage:
        """Add a new blockage object.

        Parameters
        ----------
        layer : str
            the layer name.
        blk_type : BlockageType
            the blockage type.
        points : List[PointType]
            vertices of the blockage object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        blockage : PyBlockage
            the added blockage object.
        """
        return self._layout.add_blockage(layer, blk_type, points, commit)

    def add_boundary(self, bnd_type: BoundaryType, points: List[PointType],
                     commit: bool = True) -> PyBoundary:
        """Add a new boundary.

        Parameters
        ----------
        bnd_type : str
            the boundary type.
        points : List[PointType]
            vertices of the boundary object.
        commit : bool
            True to commit the object immediately.

        Returns
        -------
        boundary : PyBoundary
            the added boundary object.
        """
        return self._layout.add_boundary(bnd_type, points, commit)

    def reexport(self, port: Port, *,
                 net_name: str = '', label: str = '', show: bool = True) -> None:
        """Re-export the given port object.

        Add all geometries in the given port as pins with optional new name
        and label.

        Parameters
        ----------
        port : Port
            the Port object to re-export.
        net_name : str
            the new net name.  If not given, use the port's current net name.
        label : str
            the label.  If not given, use net_name.
        show : bool
            True to draw the pin in layout.
        """
        net_name = net_name or port.net_name
        if not label:
            if net_name != port.net_name:
                label = net_name
            else:
                label = port.label

        if net_name not in self._port_params:
            self._port_params[net_name] = dict(label=label, pins={}, show=show)

        port_params = self._port_params[net_name]
        # check labels is consistent.
        if port_params['label'] != label:
            msg = 'Current port label = %s != specified label = %s'
            raise ValueError(msg % (port_params['label'], label))
        if port_params['show'] != show:
            raise ValueError('Conflicting show port specification.')

        # export all port geometries
        port_pins = port_params['pins']
        for wire_arr in port:
            layer_id = wire_arr.layer_id
            if layer_id not in port_pins:
                port_pins[layer_id] = [wire_arr]
            else:
                port_pins[layer_id].append(wire_arr)

    def add_pin_primitive(self, net_name: str, layer: str, bbox: BBox, *,
                          label: str = '', show: bool = True):
        """Add a primitive pin to the layout.

        Parameters
        ----------
        net_name : str
            the net name associated with the pin.
        layer : str
            the pin layer name.
        bbox : BBox
            the pin bounding box.
        label : str
            the label of this pin.  If None or empty, defaults to be the net_name.
            this argument is used if you need the label to be different than net name
            for LVS purposes.  For example, unconnected pins usually need a colon after
            the name to indicate that LVS should short those pins together.
        show : bool
            True to draw the pin in layout.
        """
        label = label or net_name
        if net_name in self._prim_port_params:
            port_params = self._prim_port_params[net_name]
        else:
            port_params = self._prim_port_params[net_name] = dict(label=label, pins={}, show=show)

        # check labels is consistent.
        if port_params['label'] != label:
            msg = 'Current port label = %s != specified label = %s'
            raise ValueError(msg % (port_params['label'], label))
        if port_params['show'] != show:
            raise ValueError('Conflicting show port specification.')

        port_pins = port_params['pins']

        if layer in port_pins:
            port_pins[layer].append(bbox)
        else:
            port_pins[layer] = [bbox]

    def add_label(self, label: str, lay_purp: Tuple[str, str], bbox: BBox) -> None:
        """Adds a label to the layout.

        This is mainly used to add voltage text labels.

        Parameters
        ----------
        label : str
            the label text.
        lay_purp: Tuple[str, str]
            the layer/purpose pair.
        bbox : BBox
            the label bounding box.
        """
        w = bbox.w
        text_h = bbox.h
        if text_h > w:
            orient = Orientation.R90
            text_h = w
        else:
            orient = Orientation.R0
        xform = Transform(bbox.xm, bbox.ym, orient)
        self._layout.add_label(lay_purp[0], lay_purp[1], xform, label, text_h)

    def add_pin(self, net_name: str, wire_arr_list: Union[WireArray, List[WireArray]],
                *, label: str = '', show: bool = True, edge_mode: int = 0) -> None:
        """Add new pin to the layout.

        If one or more pins with the same net name already exists,
        they'll be grouped under the same port.

        Parameters
        ----------
        net_name : str
            the net name associated with the pin.
        wire_arr_list : Union[WireArray, List[WireArray]]
            WireArrays representing the pin geometry.
        label : str
            the label of this pin.  If None or empty, defaults to be the net_name.
            this argument is used if you need the label to be different than net name
            for LVS purposes.  For example, unconnected pins usually need a colon after
            the name to indicate that LVS should short those pins together.
        show : bool
            if True, draw the pin in layout.
        edge_mode : int
            If <0, draw the pin on the lower end of the WireArray.  If >0, draw the pin
            on the upper end.  If 0, draw the pin on the entire WireArray.
        """
        label = label or net_name

        if net_name not in self._port_params:
            self._port_params[net_name] = dict(label=label, pins={}, show=show)

        port_params = self._port_params[net_name]

        # check labels is consistent.
        if port_params['label'] != label:
            msg = 'Current port label = %s != specified label = %s'
            raise ValueError(msg % (port_params['label'], label))
        if port_params['show'] != show:
            raise ValueError('Conflicting show port specification.')

        for warr in WireArray.wire_grp_iter(wire_arr_list):
            # add pin array to port_pins
            tid = warr.track_id
            layer_id = tid.layer_id
            if edge_mode != 0:
                # create new pin WireArray that's snapped to the edge
                cur_w = self.grid.get_wire_total_width(layer_id, tid.width)
                wl = warr.lower
                wu = warr.upper
                pin_len = min(cur_w * 2, wu - wl)
                if edge_mode < 0:
                    wu = wl + pin_len
                else:
                    wl = wu - pin_len
                warr = WireArray(tid, wl, wu)

            port_pins = port_params['pins']
            if layer_id not in port_pins:
                port_pins[layer_id] = [warr]
            else:
                port_pins[layer_id].append(warr)

    def add_via(self, bbox: BBox, bot_lay_purp: Tuple[str, str], top_lay_purp: Tuple[str, str],
                bot_dir: Orient2D, *, extend: bool = True, top_dir: Optional[Orient2D] = None,
                add_layers: bool = False, commit: bool = True) -> PyVia:
        """Adds an arrayed via object to the layout.

        Parameters
        ----------
        bbox : BBox
            the via bounding box, not including extensions.
        bot_lay_purp : Tuple[str. str]
            the bottom layer/purpose pair.
        top_lay_purp : Tuple[str, str]
            the top layer/purpose pair.
        bot_dir : Orient2D
            the bottom layer extension direction.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[Orient2D]
            top layer extension direction.  Defaults to be perpendicular to bottom layer direction.
        add_layers : bool
            True to add metal rectangles on top and bottom layers.
        commit : bool
            True to commit via immediately.

        Returns
        -------
        via : PyVia
            the new via object.
        """
        tech_info = self._grid.tech_info
        via_info = tech_info.get_via_info(bbox, Direction.LOWER, bot_lay_purp[0],
                                          top_lay_purp[0],
                                          bot_dir, purpose=bot_lay_purp[1],
                                          adj_purpose=top_lay_purp[1],
                                          extend=extend, adj_ex_dir=top_dir)

        if via_info is None:
            raise ValueError('Cannot create via between layers {} and {} '
                             'with BBox: {}'.format(bot_lay_purp, top_lay_purp, bbox))

        table = via_info['params']
        via_id = table['id']
        xform = table['xform']
        via_param = table['via_param']

        return self._layout.add_via(xform, via_id, via_param, add_layers, commit)

    def add_via_arr(self, barr: BBoxArray, bot_lay_purp: Tuple[str, str],
                    top_lay_purp: Tuple[str, str], bot_dir: Orient2D, *, extend: bool = True,
                    top_dir: Optional[Orient2D] = None, add_layers: bool = False) -> Dict[str, Any]:
        """Adds an arrayed via object to the layout.

        Parameters
        ----------
        barr : BBoxArray
            the BBoxArray representing the via bounding boxes, not including extensions.
        bot_lay_purp : Tuple[str. str]
            the bottom layer/purpose pair.
        top_lay_purp : Tuple[str, str]
            the top layer/purpose pair.
        bot_dir : Orient2D
            the bottom layer extension direction.
        extend : bool
            True if via extension can be drawn outside of the box.
        top_dir : Optional[Orient2D]
            top layer extension direction.  Defaults to be perpendicular to bottom layer direction.
        add_layers : bool
            True to add metal rectangles on top and bottom layers.

        Returns
        -------
        via_info : Dict[str, Any]
            the via information dictionary.
        """
        tech_info = self._grid.tech_info
        base_box = barr.base
        via_info = tech_info.get_via_info(base_box, Direction.LOWER, bot_lay_purp[0],
                                          top_lay_purp[0], bot_dir, purpose=bot_lay_purp[1],
                                          adj_purpose=top_lay_purp[1], extend=extend,
                                          adj_ex_dir=top_dir)

        if via_info is None:
            raise ValueError('Cannot create via between layers {} and {} '
                             'with BBox: {}'.format(bot_lay_purp, top_lay_purp, base_box))

        table = via_info['params']
        via_id = table['id']
        xform = table['xform']
        via_param = table['via_param']

        self._layout.add_via_arr(xform, via_id, via_param, add_layers, barr.nx, barr.ny,
                                 barr.spx, barr.spy)

        return via_info

    def add_via_primitive(self, via_type: str, xform: Transform, cut_width: int, cut_height: int,
                          *, num_rows: int = 1, num_cols: int = 1, sp_rows: int = 0,
                          sp_cols: int = 0, enc1: Tuple[int, int, int, int] = (0, 0, 0, 0),
                          enc2: Tuple[int, int, int, int] = (0, 0, 0, 0), nx: int = 1, ny: int = 1,
                          spx: int = 0, spy: int = 0) -> None:
        """Adds via(s) by specifying all parameters.

        Parameters
        ----------
        via_type : str
            the via type name.
        xform: Transform
            the transformation object.
        cut_width : CoordType
            via cut width.  This is used to create rectangle via.
        cut_height : CoordType
            via cut height.  This is used to create rectangle via.
        num_rows : int
            number of via cut rows.
        num_cols : int
            number of via cut columns.
        sp_rows : CoordType
            spacing between via cut rows.
        sp_cols : CoordType
            spacing between via cut columns.
        enc1 : Optional[List[CoordType]]
            a list of left, right, top, and bottom enclosure values on bottom layer.
            Defaults to all 0.
        enc2 : Optional[List[CoordType]]
            a list of left, right, top, and bottom enclosure values on top layer.
            Defaults to all 0.
        nx : int
            number of columns.
        ny : int
            number of rows.
        spx : int
            column pitch.
        spy : int
            row pitch.
        """
        l1, r1, t1, b1 = enc1
        l2, r2, t2, b2 = enc2
        param = ViaParam(num_cols, num_rows, cut_width, cut_height, sp_cols, sp_rows,
                         l1, r1, t1, b1, l2, r2, t2, b2)
        self._layout.add_via_arr(xform, via_type, param, True, nx, ny, spx, spy)

    def add_via_on_grid(self, tid1: TrackID, tid2: TrackID, *, extend: bool = True) -> None:
        """Add a via on the routing grid.

        Parameters
        ----------
        tid1 : TrackID
            the first TrackID
        tid2 : TrackID
            the second TrackID
        extend : bool
            True to extend outside the via bounding box.
        """
        self._layout.add_via_on_intersections(WireArray(tid1, COORD_MIN, COORD_MAX),
                                              WireArray(tid2, COORD_MIN, COORD_MAX),
                                              extend, False)

    def extend_wires(self, warr_list: Union[WireArray, List[Optional[WireArray]]], *,
                     lower: Optional[int] = None, upper: Optional[int] = None,
                     min_len_mode: Optional[int] = None) -> List[Optional[WireArray]]:
        """Extend the given wires to the given coordinates.

        Parameters
        ----------
        warr_list : Union[WireArray, List[Optional[WireArray]]]
            the wires to extend.
        lower : Optional[int]
            the wire lower coordinate.
        upper : Optional[int]
            the wire upper coordinate.
        min_len_mode : Optional[int]
            If not None, will extend track so it satisfy minimum length requirement.
            Use -1 to extend lower bound, 1 to extend upper bound, 0 to extend both equally.

        Returns
        -------
        warr_list : List[Optional[WireArray]]
            list of added wire arrays.
            If any elements in warr_list were None, they will be None in the return.
        """
        grid = self.grid

        new_warr_list = []
        for warr in WireArray.wire_grp_iter(warr_list):
            tid = warr.track_id
            if warr is None:
                new_warr_list.append(None)
            else:
                wlower = warr.lower
                wupper = warr.upper
                if lower is None:
                    cur_lower = wlower
                else:
                    cur_lower = min(lower, wlower)
                if upper is None:
                    cur_upper = wupper
                else:
                    cur_upper = max(upper, wupper)
                if min_len_mode is not None:
                    # extend track to meet minimum length
                    # make sure minimum length is even so that middle coordinate exists
                    min_len = grid.get_min_length(tid.layer_id, tid.width, even=True)
                    tr_len = cur_upper - cur_lower
                    if min_len > tr_len:
                        ext = min_len - tr_len
                        if min_len_mode < 0:
                            cur_lower -= ext
                        elif min_len_mode > 0:
                            cur_upper += ext
                        else:
                            cur_lower -= ext // 2
                            cur_upper = cur_lower + min_len

                new_warr = WireArray(tid, cur_lower, cur_upper)
                self._layout.add_warr(new_warr)
                new_warr_list.append(new_warr)

        return new_warr_list

    def add_wires(self, layer_id: int, track_idx: TrackType, lower: int, upper: int, *,
                  width: int = 1, num: int = 1, pitch: TrackType = 1) -> WireArray:
        """Add the given wire(s) to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        lower : CoordType
            the wire lower coordinate.
        upper : CoordType
            the wire upper coordinate.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : TrackType
            the wire pitch.

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        tid = TrackID(layer_id, track_idx, width=width, num=num, pitch=pitch)
        warr = WireArray(tid, lower, upper)
        self._layout.add_warr(warr)
        return warr

    def add_res_metal_warr(self, layer_id: int, track_idx: TrackType, lower: int, upper: int,
                           **kwargs: Any) -> WireArray:
        """Add metal resistor as WireArray to this layout.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        lower : CoordType
            the wire lower coordinate.
        upper : CoordType
            the wire upper coordinate.
        **kwargs :
            optional arguments to add_wires()

        Returns
        -------
        warr : WireArray
            the added WireArray object.
        """
        warr = self.add_wires(layer_id, track_idx, lower, upper, **kwargs)

        for _, _, box in warr.wire_iter(self.grid):
            self.add_res_metal(layer_id, box)

        return warr

    def add_mom_cap(self, cap_box: BBox, bot_layer: int, num_layer: int, *,
                    port_widths: Optional[Mapping[int, int]] = None,
                    port_plow: Optional[Mapping[int, bool]] = None,
                    array: bool = False,
                    cap_wires_list: Optional[List[Tuple[Tuple[str, str], Tuple[str, str],
                                                        BBoxArray, BBoxArray]]] = None,
                    cap_type: str = 'standard'
                    ) -> Dict[int, Tuple[List[WireArray], List[WireArray]]]:
        """Draw mom cap in the defined bounding box."""

        empty_dict = {}
        if num_layer <= 1:
            raise ValueError('Must have at least 2 layers for MOM cap.')
        if port_widths is None:
            port_widths = empty_dict
        if port_plow is None:
            port_plow = empty_dict

        grid = self.grid
        tech_info = grid.tech_info

        top_layer = bot_layer + num_layer - 1
        cap_info = MOMCapInfo(tech_info.tech_params['mom_cap'][cap_type], port_widths, port_plow)
        via_ext_dict = get_cap_via_extensions(cap_info, grid, bot_layer, top_layer)

        # find port locations and cap boundaries.
        port_tracks: Dict[int, Tuple[List[int], List[int]]] = {}
        cap_bounds: Dict[int, Tuple[int, int]] = {}
        cap_exts: Dict[int, Tuple[int, int]] = {}
        for cur_layer in range(bot_layer, top_layer + 1):
            cap_w, cap_sp, cap_margin, num_ports = cap_info.get_cap_specs(cur_layer)
            port_tr_w = cap_info.get_port_tr_w(cur_layer)
            port_tr_sep = grid.get_sep_tracks(cur_layer, port_tr_w, port_tr_w)

            dir_idx = grid.get_direction(cur_layer).value
            coord0, coord1 = cap_box.get_interval(1 - dir_idx)
            # get max via extension on adjacent layers
            adj_via_ext = 0
            if cur_layer != bot_layer:
                adj_via_ext = via_ext_dict[cur_layer - 1]
            if cur_layer != top_layer:
                adj_via_ext = max(adj_via_ext, via_ext_dict[cur_layer + 1])
            # find track indices
            if array:
                tidx0 = grid.coord_to_track(cur_layer, coord0)
                tidx1 = grid.coord_to_track(cur_layer, coord1)
            else:
                tidx0 = grid.find_next_track(cur_layer, coord0 + adj_via_ext, tr_width=port_tr_w,
                                             mode=RoundMode.GREATER_EQ)
                tidx1 = grid.find_next_track(cur_layer, coord1 - adj_via_ext, tr_width=port_tr_w,
                                             mode=RoundMode.LESS_EQ)

            if tidx0 + 2 * num_ports * port_tr_sep >= tidx1:
                raise ValueError('Cannot draw MOM cap; '
                                 f'not enough space between ports on layer {cur_layer}.')

            # compute space from MOM cap wires to port wires
            cap_margin = max(cap_margin, grid.get_min_space(cur_layer, port_tr_w))
            lower_tracks = [tidx0 + idx * port_tr_sep for idx in range(num_ports)]
            upper_tracks = [tidx1 - idx * port_tr_sep for idx in range(num_ports - 1, -1, -1)]

            tr_ll = grid.get_wire_bounds(cur_layer, lower_tracks[0], width=port_tr_w)[0]
            tr_lu = grid.get_wire_bounds(cur_layer, lower_tracks[num_ports - 1], width=port_tr_w)[1]
            tr_ul = grid.get_wire_bounds(cur_layer, upper_tracks[0], width=port_tr_w)[0]
            tr_uu = grid.get_wire_bounds(cur_layer, upper_tracks[num_ports - 1], width=port_tr_w)[1]
            port_tracks[cur_layer] = (lower_tracks, upper_tracks)
            cap_bounds[cur_layer] = (tr_lu + cap_margin, tr_ul - cap_margin)
            cap_exts[cur_layer] = (tr_ll, tr_uu)

        port_dict: Dict[int, Tuple[List[WireArray], List[WireArray]]] = {}
        cap_wire_dict: Dict[int, Tuple[Tuple[str, str], Tuple[str, str], BBoxArray, BBoxArray]] = {}
        # draw ports/wires
        for cur_layer in range(bot_layer, top_layer + 1):
            port_plow = cap_info.get_port_plow(cur_layer)
            port_tr_w = cap_info.get_port_tr_w(cur_layer)
            cap_w, cap_sp, cap_margin, num_ports = cap_info.get_cap_specs(cur_layer)

            # find port/cap wires lower/upper coordinates
            lower = COORD_MAX
            upper = COORD_MIN
            if cur_layer != top_layer:
                lower, upper = cap_exts[cur_layer + 1]
            if cur_layer != bot_layer:
                tmpl, tmpu = cap_exts[cur_layer - 1]
                lower = min(lower, tmpl)
                upper = max(upper, tmpu)

            via_ext = via_ext_dict[cur_layer]
            lower -= via_ext
            upper += via_ext

            # draw ports
            lower_tracks, upper_tracks = port_tracks[cur_layer]
            lower_warrs = [self.add_wires(cur_layer, tr_idx, lower, upper, width=port_tr_w)
                           for tr_idx in lower_tracks]
            upper_warrs = [self.add_wires(cur_layer, tr_idx, lower, upper, width=port_tr_w)
                           for tr_idx in upper_tracks]

            # assign port wires to positive/negative terminals
            num_ports = len(lower_warrs)
            if port_plow:
                if num_ports == 1:
                    plist = lower_warrs
                    nlist = upper_warrs
                else:
                    plist = [lower_warrs[0], upper_warrs[0]]
                    nlist = [lower_warrs[1], upper_warrs[1]]
            else:
                if num_ports == 1:
                    plist = upper_warrs
                    nlist = lower_warrs
                else:
                    plist = [lower_warrs[1], upper_warrs[1]]
                    nlist = [lower_warrs[0], upper_warrs[0]]

            # save ports
            port_dict[cur_layer] = plist, nlist

            # compute cap wires BBoxArray
            cap_bndl, cap_bndh = cap_bounds[cur_layer]
            cap_tot_space = cap_bndh - cap_bndl
            cap_pitch = cap_w + cap_sp
            num_cap_wires = cap_tot_space // cap_pitch
            cap_bndl += (cap_tot_space - (num_cap_wires * cap_pitch - cap_sp)) // 2

            cur_dir = grid.get_direction(cur_layer)
            cap_box0 = BBox(cur_dir, lower, upper, cap_bndl, cap_bndl + cap_w)
            lay_purp_list = tech_info.get_lay_purp_list(cur_layer)
            num_lay_purp = len(lay_purp_list)
            assert num_lay_purp <= 2, 'This method now only works for 1 or 2 colors.'
            num0 = (num_cap_wires + 1) // 2
            num1 = num_cap_wires - num0
            barr_pitch = cap_pitch * 2
            cap_box1 = cap_box0.get_move_by_orient(cur_dir, dt=0, dp=cap_pitch)
            barr0 = BBoxArray(cap_box0, cur_dir, np=num0, spp=barr_pitch)
            barr1 = BBoxArray(cap_box1, cur_dir, np=num1, spp=barr_pitch)
            if port_plow:
                capp_barr = barr1
                capn_barr = barr0
                capp_lp = lay_purp_list[-1]
                capn_lp = lay_purp_list[0]
            else:
                capp_barr = barr0
                capn_barr = barr1
                capp_lp = lay_purp_list[0]
                capn_lp = lay_purp_list[-1]

            # draw cap wires
            self.add_rect_arr(capp_lp, capp_barr)
            self.add_rect_arr(capn_lp, capn_barr)
            # save caps
            cap_barr_tuple = (capp_lp, capn_lp, capp_barr, capn_barr)
            cap_wire_dict[cur_layer] = cap_barr_tuple
            if cap_wires_list is not None:
                cap_wires_list.append(cap_barr_tuple)

            # connect port/cap wires to bottom port/cap
            if cur_layer != bot_layer:
                # connect ports to layer below
                bplist, bnlist = port_dict[cur_layer - 1]
                bcapp_lp, bcapn_lp, bcapp, bcapn = cap_wire_dict[cur_layer - 1]
                self._add_mom_cap_connect_ports(bplist, plist)
                self._add_mom_cap_connect_ports(bnlist, nlist)
                self._add_mom_cap_connect_cap_to_port(Direction.UPPER, capp_lp, capp_barr, bplist)
                self._add_mom_cap_connect_cap_to_port(Direction.UPPER, capn_lp, capn_barr, bnlist)
                self._add_mom_cap_connect_cap_to_port(Direction.LOWER, bcapp_lp, bcapp, plist)
                self._add_mom_cap_connect_cap_to_port(Direction.LOWER, bcapn_lp, bcapn, nlist)

        return port_dict

    def _add_mom_cap_connect_cap_to_port(self, cap_dir: Direction, cap_lp: Tuple[str, str],
                                         barr: BBoxArray, ports: List[WireArray]) -> None:
        num_ports = len(ports)
        if num_ports == 1:
            self.connect_bbox_to_tracks(cap_dir, cap_lp, barr, ports[0].track_id)
        else:
            port_dir = self.grid.get_direction(ports[0].layer_id)
            for idx, warr in enumerate(ports):
                new_barr = barr.get_sub_array(port_dir, num_ports, idx)
                self.connect_bbox_to_tracks(cap_dir, cap_lp, new_barr, warr.track_id)

    def _add_mom_cap_connect_ports(self, bot_ports: List[WireArray], top_ports: List[WireArray]
                                   ) -> None:
        for bot_warr, top_warr in product(bot_ports, top_ports):
            self.add_via_on_grid(bot_warr.track_id, top_warr.track_id, extend=True)

    def reserve_tracks(self, layer_id: int, track_idx: TrackType, *,
                       width: int = 1, num: int = 1, pitch: int = 0) -> None:
        """Reserve the given routing tracks so that power fill will not fill these tracks.

        Note: the size of this template should be set before calling this method.

        Parameters
        ----------
        layer_id : int
            the wire layer ID.
        track_idx : TrackType
            the smallest wire track index.
        width : int
            the wire width in number of tracks.
        num : int
            number of wires.
        pitch : TrackType
            the wire pitch.
        """
        # TODO: fix this method
        raise ValueError('Not implemented yet.')

    def connect_wires(self,  wire_arr_list: Union[WireArray, List[WireArray]], *,
                      lower: Optional[int] = None,
                      upper: Optional[int] = None,
                      debug: bool = False,
                      ) -> List[WireArray]:
        """Connect all given WireArrays together.

        all WireArrays must be on the same layer.

        Parameters
        ----------
        wire_arr_list : Union[WireArr, List[WireArr]]
            WireArrays to connect together.
        lower : Optional[CoordType]
            if given, extend connection wires to this lower coordinate.
        upper : Optional[CoordType]
            if given, extend connection wires to this upper coordinate.
        debug : bool
            True to print debug messages.

        Returns
        -------
        conn_list : List[WireArray]
            list of connection wires created.
        """
        grid = self._grid

        # record all wire ranges
        layer_id = None
        intv_set = IntervalSet()
        for wire_arr in WireArray.wire_grp_iter(wire_arr_list):

            tid = wire_arr.track_id
            lay_id = tid.layer_id
            tr_w = tid.width
            if layer_id is None:
                layer_id = lay_id
            elif lay_id != layer_id:
                raise ValueError('WireArray layer ID != {}'.format(layer_id))

            cur_range = wire_arr.lower, wire_arr.upper
            for tidx in tid:
                intv = grid.get_wire_bounds(lay_id, tidx, width=tr_w)
                intv_rang_item = intv_set.get_first_overlap_item(intv)
                if intv_rang_item is None:
                    range_set = IntervalSet()
                    range_set.add(cur_range)
                    intv_set.add(intv, val=(range_set, tidx, tr_w))
                elif intv_rang_item[0] == intv:
                    tmp_rang_set: IntervalSet = intv_rang_item[1][0]
                    tmp_rang_set.add(cur_range, merge=True, abut=True)
                else:
                    raise ValueError('wire interval {} overlap existing wires.'.format(intv))

        # draw wires, group into arrays
        new_warr_list = []
        base_start = None  # type: Optional[int]
        base_end = None  # type: Optional[int]
        base_tidx = None  # type: Optional[HalfInt]
        base_width = None  # type: Optional[int]
        count = 0
        pitch = 0
        last_tidx = 0
        for set_item in intv_set.items():
            intv = set_item[0]
            range_set: IntervalSet = set_item[1][0]
            cur_tidx: HalfInt = set_item[1][1]
            cur_tr_w: int = set_item[1][2]
            cur_start = range_set.start
            cur_end = range_set.stop
            if lower is not None and lower < cur_start:
                cur_start = lower
            if upper is not None and upper > cur_end:
                cur_end = upper

            if debug:
                print('wires intv: %s, range: (%d, %d)' % (intv, cur_start, cur_end))
            if count == 0:
                base_tidx = cur_tidx
                base_start = cur_start
                base_end = cur_end
                base_width = cur_tr_w
                count = 1
                pitch = 0
            else:
                assert base_tidx is not None, "count == 0 should have set base_intv"
                assert base_width is not None, "count == 0 should have set base_width"
                assert base_start is not None, "count == 0 should have set base_start"
                assert base_end is not None, "count == 0 should have set base_end"
                if cur_start == base_start and cur_end == base_end and base_width == cur_tr_w:
                    # length and width matches
                    cur_pitch = cur_tidx - last_tidx
                    if count == 1:
                        # second wire, set half pitch
                        pitch = cur_pitch
                        count += 1
                    elif pitch == cur_pitch:
                        # pitch matches
                        count += 1
                    else:
                        # pitch does not match, add current wires and start anew
                        track_id = TrackID(layer_id, base_tidx, width=base_width,
                                           num=count, pitch=pitch)
                        warr = WireArray(track_id, base_start, base_end)
                        new_warr_list.append(warr)
                        self._layout.add_warr(warr)
                        base_tidx = cur_tidx
                        count = 1
                        pitch = 0
                else:
                    # length/width does not match, add cumulated wires and start anew
                    track_id = TrackID(layer_id, base_tidx, width=base_width,
                                       num=count, pitch=pitch)
                    warr = WireArray(track_id, base_start, base_end)
                    new_warr_list.append(warr)
                    self._layout.add_warr(warr)
                    base_start = cur_start
                    base_end = cur_end
                    base_tidx = cur_tidx
                    base_width = cur_tr_w
                    count = 1
                    pitch = 0

            # update last lower coordinate
            last_tidx = cur_tidx

        if base_tidx is None:
            # no wires given at all
            return []

        assert base_tidx is not None, "count == 0 should have set base_intv"
        assert base_start is not None, "count == 0 should have set base_start"
        assert base_end is not None, "count == 0 should have set base_end"

        # add last wires
        track_id = TrackID(layer_id, base_tidx, base_width, num=count, pitch=pitch)
        warr = WireArray(track_id, base_start, base_end)
        self._layout.add_warr(warr)
        new_warr_list.append(warr)
        return new_warr_list

    def connect_bbox_to_tracks(self, layer_dir: Direction, lay_purp: Tuple[str, str],
                               box_arr: Union[BBox, BBoxArray], track_id: TrackID, *,
                               track_lower: Optional[int] = None,
                               track_upper: Optional[int] = None,
                               min_len_mode: MinLenMode = MinLenMode.NONE,
                               wire_lower: Optional[int] = None,
                               wire_upper: Optional[int] = None) -> WireArray:
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        layer_dir : Direction
            the primitive wire layer direction relative to the given tracks.  LOWER if
            the wires are below tracks, UPPER if the wires are above tracks.
        lay_purp : Tuple[str, str]
            the primitive wire layer/purpose name.
        box_arr : Union[BBox, BBoxArray]
            bounding box of the wire(s) to connect to tracks.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            The minimum length extension mode.
        wire_lower : Optional[int]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[int]
            if given, extend wire(s) to this upper coordinate.

        Returns
        -------
        wire_arr : WireArray
            WireArray representing the tracks created.
        """
        if isinstance(box_arr, BBox):
            box_arr = BBoxArray(box_arr)

        bnds = self._layout.connect_barr_to_tracks(layer_dir, lay_purp[0], lay_purp[1], box_arr,
                                                   track_id, track_lower, track_upper, min_len_mode,
                                                   wire_lower, wire_upper)
        tr_idx = 1 - layer_dir.value
        return WireArray(track_id, bnds[tr_idx][0], bnds[tr_idx][1])

    def connect_bbox_to_differential_tracks(self, p_lay_dir: Direction, n_lay_dir: Direction,
                                            p_lay_purp: Tuple[str, str],
                                            n_lay_purp: Tuple[str, str],
                                            pbox: Union[BBox, BBoxArray],
                                            nbox: Union[BBox, BBoxArray], tr_layer_id: int,
                                            ptr_idx: TrackType, ntr_idx: TrackType, *,
                                            width: int = 1, track_lower: Optional[int] = None,
                                            track_upper: Optional[int] = None,
                                            min_len_mode: MinLenMode = MinLenMode.NONE
                                            ) -> DiffWarrType:
        """Connect the given differential primitive wires to two tracks symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        p_lay_dir : Direction
            positive signal layer direction.
        n_lay_dir : Direction
            negative signal layer direction.
        p_lay_purp : Tuple[str, str]
            positive signal layer/purpose pair.
        n_lay_purp : Tuple[str, str]
            negative signal layer/purpose pair.
        pbox : Union[BBox, BBoxArray]
            positive signal wires to connect.
        nbox : Union[BBox, BBoxArray]
            negative signal wires to connect.
        tr_layer_id : int
            track layer ID.
        ptr_idx : TrackType
            positive track index.
        ntr_idx : TrackType
            negative track index.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_bbox_to_matching_tracks([p_lay_dir, n_lay_dir],
                                                          [p_lay_purp, n_lay_purp], [pbox, nbox],
                                                          tr_layer_id, [ptr_idx, ntr_idx],
                                                          width=width, track_lower=track_lower,
                                                          track_upper=track_upper,
                                                          min_len_mode=min_len_mode)
        return track_list[0], track_list[1]

    def fix_track_min_length(self, tr_layer_id: int, width: int, track_lower: int, track_upper: int,
                             min_len_mode: MinLenMode) -> Tuple[int, int]:
        even = min_len_mode is MinLenMode.MIDDLE
        tr_len = max(track_upper - track_lower, self.grid.get_min_length(tr_layer_id, width,
                                                                         even=even))
        if min_len_mode is MinLenMode.LOWER:
            track_lower = track_upper - tr_len
        elif min_len_mode is MinLenMode.UPPER:
            track_upper = track_lower + tr_len
        elif min_len_mode is MinLenMode.MIDDLE:
            track_lower = (track_upper + track_lower - tr_len) // 2
            track_upper = track_lower + tr_len

        return track_lower, track_upper

    def connect_bbox_to_matching_tracks(self, lay_dir_list: List[Direction],
                                        lay_purp_list: List[Tuple[str, str]],
                                        box_arr_list: List[Union[BBox, BBoxArray]],
                                        tr_layer_id: int, tr_idx_list: List[TrackType], *,
                                        width: int = 1, track_lower: Optional[int] = None,
                                        track_upper: Optional[int] = None,
                                        min_len_mode: MinLenMode = MinLenMode.NONE,
                                        ) -> List[Optional[WireArray]]:
        """Connect the given primitive wire to given tracks.

        Parameters
        ----------
        lay_dir_list : List[Direction]
            the primitive wire layer direction list.
        lay_purp_list : List[Tuple[str, str]]
            the primitive wire layer/purpose list.
        box_arr_list : List[Union[BBox, BBoxArray]]
            bounding box of the wire(s) to connect to tracks.
        tr_layer_id : int
            track layer ID.
        tr_idx_list : List[TrackType]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        Returns
        -------
        wire_arr : List[Optional[WireArray]]
            WireArrays representing the tracks created.
        """
        grid = self.grid
        tr_dir = grid.get_direction(tr_layer_id)
        w_dir = tr_dir.perpendicular()

        num = len(lay_dir_list)
        if len(lay_purp_list) != num or len(box_arr_list) != num or len(tr_idx_list) != num:
            raise ValueError('Connection list parameters have mismatch length.')
        if num == 0:
            raise ValueError('Connection lists are empty.')

        wl = None
        wu = None
        for lay_dir, (lay, purp), box_arr, tr_idx in zip(lay_dir_list, lay_purp_list,
                                                         box_arr_list, tr_idx_list):
            if isinstance(box_arr, BBox):
                box_arr = BBoxArray(box_arr)

            tid = TrackID(tr_layer_id, tr_idx, width=width)
            bnds = self._layout.connect_barr_to_tracks(lay_dir, lay, purp, box_arr, tid,
                                                       track_lower, track_upper, MinLenMode.NONE,
                                                       wl, wu)
            w_idx = lay_dir.value
            tr_idx = 1 - w_idx
            wl = bnds[w_idx][0]
            wu = bnds[w_idx][1]
            track_lower = bnds[tr_idx][0]
            track_upper = bnds[tr_idx][1]

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, width, track_lower,
                                                             track_upper, min_len_mode)
        # extend wires
        ans = []
        for (lay, purp), box_arr, tr_idx in zip(lay_purp_list, box_arr_list, tr_idx_list):
            if isinstance(box_arr, BBox):
                box_arr = BBoxArray(box_arr)
            else:
                box_arr = BBoxArray(box_arr.base, tr_dir, nt=box_arr.get_num(tr_dir),
                                    spt=box_arr.get_sp(tr_dir))

            box_arr.set_interval(w_dir, wl, wu)
            self._layout.add_rect_arr(lay, purp, box_arr)

            warr = WireArray(TrackID(tr_layer_id, tr_idx, width=width), track_lower, track_upper)
            self._layout.add_warr(warr)
            ans.append(warr)

        return ans

    def connect_to_tracks(self, wire_arr_list: Union[WireArray, List[WireArray]],
                          track_id: TrackID, *, wire_lower: Optional[int] = None,
                          wire_upper: Optional[int] = None, track_lower: Optional[int] = None,
                          track_upper: Optional[int] = None, min_len_mode: MinLenMode = None,
                          ret_wire_list: Optional[List[WireArray]] = None,
                          debug: bool = False) -> Optional[WireArray]:
        """Connect all given WireArrays to the given track(s).

        All given wires should be on adjacent layers of the track.

        Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_id : TrackID
            TrackID that specifies the track(s) to connect the given wires to.
        wire_lower : Optional[CoordType]
            if given, extend wire(s) to this lower coordinate.
        wire_upper : Optional[CoordType]
            if given, extend wire(s) to this upper coordinate.
        track_lower : Optional[CoordType]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[CoordType]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        ret_wire_list : Optional[List[WireArray]]
            If not none, extended wires that are created will be appended to this list.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Optional[WireArray]
            WireArray representing the tracks created.
        """
        # find min/max track Y coordinates
        tr_layer_id = track_id.layer_id
        tr_w = track_id.width

        # get top wire and bottom wire list
        warr_list_list = [[], []]
        for wire_arr in WireArray.wire_grp_iter(wire_arr_list):
            cur_layer_id = wire_arr.layer_id
            if cur_layer_id == tr_layer_id + 1:
                warr_list_list[1].append(wire_arr)
            elif cur_layer_id == tr_layer_id - 1:
                warr_list_list[0].append(wire_arr)
            else:
                raise ValueError(
                    'WireArray layer %d cannot connect to layer %d' % (cur_layer_id, tr_layer_id))

        if not warr_list_list[0] and not warr_list_list[1]:
            # no wires at all
            return None

        # connect wires together
        for warr in self.connect_wires(warr_list_list[0], lower=wire_lower, upper=wire_upper,
                                       debug=debug):
            bnds = self._layout.connect_warr_to_tracks(warr, track_id, None, None,
                                                       track_lower, track_upper)
            if ret_wire_list is not None:
                ret_wire_list.append(WireArray(warr.track_id, bnds[0][0], bnds[0][1]))
            track_lower = bnds[1][0]
            track_upper = bnds[1][1]
        for warr in self.connect_wires(warr_list_list[1], lower=wire_lower, upper=wire_upper,
                                       debug=debug):
            bnds = self._layout.connect_warr_to_tracks(warr, track_id, None, None,
                                                       track_lower, track_upper)
            if ret_wire_list is not None:
                ret_wire_list.append(WireArray(warr.track_id, bnds[1][0], bnds[1][1]))
            track_lower = bnds[0][0]
            track_upper = bnds[0][1]

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, tr_w, track_lower,
                                                             track_upper, min_len_mode)
        result = WireArray(track_id, track_lower, track_upper)
        self._layout.add_warr(result)
        return result

    def connect_to_track_wires(self, wire_arr_list: Union[WireArray, List[WireArray]],
                               track_wires: Union[WireArray, List[WireArray]], *,
                               min_len_mode: Optional[MinLenMode] = None,
                               debug: bool = False) -> Union[Optional[WireArray],
                                                             List[Optional[WireArray]]]:
        """Connect all given WireArrays to the given WireArrays on adjacent layer.

        Parameters
        ----------
        wire_arr_list : Union[WireArray, List[WireArray]]
            list of WireArrays to connect to track.
        track_wires : Union[WireArray, List[WireArray]]
            list of tracks as WireArrays.
        min_len_mode : MinLenMode
            the minimum length extension mode.
        debug : bool
            True to print debug messages.

        Returns
        -------
        wire_arr : Union[Optional[WireArray], List[Optional[WireArray]]]
            WireArrays representing the tracks created.  None if nothing to do.
        """
        ans = []  # type: List[Optional[WireArray]]
        for warr in WireArray.wire_grp_iter(track_wires):
            tr = self.connect_to_tracks(wire_arr_list, warr.track_id, track_lower=warr.lower,
                                        track_upper=warr.upper, min_len_mode=min_len_mode,
                                        debug=debug)
            ans.append(tr)

        if isinstance(track_wires, WireArray):
            return ans[0]
        return ans

    def connect_differential_tracks(self, pwarr_list: Union[WireArray, List[WireArray]],
                                    nwarr_list: Union[WireArray, List[WireArray]],
                                    tr_layer_id: int, ptr_idx: TrackType, ntr_idx: TrackType, *,
                                    width: int = 1, track_lower: Optional[int] = None,
                                    track_upper: Optional[int] = None
                                    ) -> Tuple[Optional[WireArray], Optional[WireArray]]:
        """Connect the given differential wires to two tracks symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        pwarr_list : Union[WireArray, List[WireArray]]
            positive signal wires to connect.
        nwarr_list : Union[WireArray, List[WireArray]]
            negative signal wires to connect.
        tr_layer_id : int
            track layer ID.
        ptr_idx : TrackType
            positive track index.
        ntr_idx : TrackType
            negative track index.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        track_list = self.connect_matching_tracks([pwarr_list, nwarr_list], tr_layer_id,
                                                  [ptr_idx, ntr_idx], width=width,
                                                  track_lower=track_lower, track_upper=track_upper)
        return track_list[0], track_list[1]

    def connect_differential_wires(self, pin_warrs: Union[WireArray, List[WireArray]],
                                   nin_warrs: Union[WireArray, List[WireArray]],
                                   pout_warr: WireArray, nout_warr: WireArray, *,
                                   track_lower: Optional[int] = None,
                                   track_upper: Optional[int] = None
                                   ) -> Tuple[Optional[WireArray], Optional[WireArray]]:
        """Connect the given differential wires to two WireArrays symmetrically.

        This method makes sure the connections are symmetric and have identical parasitics.

        Parameters
        ----------
        pin_warrs : Union[WireArray, List[WireArray]]
            positive signal wires to connect.
        nin_warrs : Union[WireArray, List[WireArray]]
            negative signal wires to connect.
        pout_warr : WireArray
            positive track wires.
        nout_warr : WireArray
            negative track wires.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.

        Returns
        -------
        p_track : Optional[WireArray]
            the positive track.
        n_track : Optional[WireArray]
            the negative track.
        """
        p_tid = pout_warr.track_id
        lay_id = p_tid.layer_id
        pidx = p_tid.base_index
        nidx = nout_warr.track_id.base_index
        width = p_tid.width

        if track_lower is None:
            tr_lower = pout_warr.lower
        else:
            tr_lower = min(track_lower, pout_warr.lower)
        if track_upper is None:
            tr_upper = pout_warr.upper
        else:
            tr_upper = max(track_upper, pout_warr.upper)

        return self.connect_differential_tracks(pin_warrs, nin_warrs, lay_id, pidx, nidx,
                                                width=width, track_lower=tr_lower,
                                                track_upper=tr_upper)

    def connect_matching_tracks(self, warr_list_list: List[Union[WireArray, List[WireArray]]],
                                tr_layer_id: int, tr_idx_list: List[TrackType], *,
                                width: int = 1,
                                track_lower: Optional[int] = None,
                                track_upper: Optional[int] = None,
                                min_len_mode: MinLenMode = MinLenMode.NONE
                                ) -> List[Optional[WireArray]]:
        """Connect wires to tracks with optimal matching.

        This method connects the wires to tracks in a way that minimizes the parasitic mismatches.

        Parameters
        ----------
        warr_list_list : List[Union[WireArray, List[WireArray]]]
            list of signal wires to connect.
        tr_layer_id : int
            track layer ID.
        tr_idx_list : List[TrackType]
            list of track indices.
        width : int
            track width in number of tracks.
        track_lower : Optional[int]
            if given, extend track(s) to this lower coordinate.
        track_upper : Optional[int]
            if given, extend track(s) to this upper coordinate.
        min_len_mode : MinLenMode
            the minimum length extension mode.

        Returns
        -------
        track_list : List[WireArray]
            list of created tracks.
        """
        # simple error checking
        num_tracks = len(tr_idx_list)  # type: int
        if num_tracks != len(warr_list_list):
            raise ValueError('Connection list parameters have mismatch length.')
        if num_tracks == 0:
            raise ValueError('Connection lists are empty.')

        wbounds = [[None, None], [None, None]]
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            tid = TrackID(tr_layer_id, tr_idx, width=width)
            for warr in WireArray.wire_grp_iter(warr_list):
                cur_lay_id = warr.layer_id
                if cur_lay_id == tr_layer_id + 1:
                    wb_idx = 1
                elif cur_lay_id == tr_layer_id - 1:
                    wb_idx = 0
                else:
                    raise ValueError(
                        'WireArray layer {} cannot connect to layer {}'.format(cur_lay_id,
                                                                               tr_layer_id))

                bnds = self._layout.connect_warr_to_tracks(warr, tid, wbounds[wb_idx][0],
                                                           wbounds[wb_idx][1], track_lower,
                                                           track_upper)
                wbounds[wb_idx] = bnds[wb_idx]
                track_lower = bnds[1 - wb_idx][0]
                track_upper = bnds[1 - wb_idx][1]

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, width, track_lower,
                                                             track_upper, min_len_mode)
        # extend wires
        ans = []
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            for warr in WireArray.wire_grp_iter(warr_list):
                wb_idx = (warr.layer_id - tr_layer_id + 1) // 2
                self._layout.add_warr(WireArray(warr.track_id, wbounds[wb_idx][0],
                                                wbounds[wb_idx][1]))

            warr = WireArray(TrackID(tr_layer_id, tr_idx, width=width), track_lower, track_upper)
            self._layout.add_warr(warr)
            ans.append(warr)

        return ans

    def draw_vias_on_intersections(self, bot_warr_list: Union[WireArray, List[WireArray]],
                                   top_warr_list: Union[WireArray, List[WireArray]]) -> None:
        """Draw vias on all intersections of the two given wire groups.

        Parameters
        ----------
        bot_warr_list : Union[WireArray, List[WireArray]]
            the bottom wires.
        top_warr_list : Union[WireArray, List[WireArray]]
            the top wires.
        """
        for bwarr in WireArray.wire_grp_iter(bot_warr_list):
            for twarr in WireArray.wire_grp_iter(top_warr_list):
                self._layout.add_via_on_intersections(bwarr, twarr, True, True)

    def mark_bbox_used(self, layer_id: int, bbox: BBox) -> None:
        """Marks the given bounding-box region as used in this Template."""
        # TODO: Fix this
        raise ValueError('Not implemented yet')

    def do_max_space_fill(self, layer_id: int, bound_box: Optional[BBox] = None,
                          fill_boundary: bool = True) -> None:
        """Draw density fill on the given layer."""
        if bound_box is None:
            bound_box = self.bound_box

        fill_info = self.grid.tech_info.get_max_space_fill_info(layer_id)
        self._layout.do_max_space_fill(layer_id, bound_box, fill_boundary, fill_info.info)


class BlackBoxTemplate(TemplateBase):
    """A black box template."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lib_name='The library name.',
            cell_name='The layout cell name.',
            top_layer='The top level layer.',
            size='The width/height of the cell, in resolution units.',
            ports='The port information dictionary.',
        )

    def get_layout_basename(self) -> str:
        return self.params['cell_name']

    def draw_layout(self) -> None:
        lib_name: str = self.params['lib_name']
        cell_name: str = self.params['cell_name']
        top_layer: int = self.params['top_layer']
        size: Tuple[int, int] = self.params['size']
        ports: Dict[str, Dict[str, Tuple[int, int, int, int]]] = self.params['ports']

        show_pins = self.show_pins
        for term_name, pin_dict in ports.items():
            for lay, bbox_list in pin_dict.items():
                for xl, yb, xr, yt in bbox_list:
                    box = BBox(xl, yb, xr, yt)
                    self._register_pin(lay, term_name, box, show_pins)

        self.add_instance_primitive(lib_name, cell_name)

        self.prim_top_layer = top_layer
        self.prim_bound_box = BBox(0, 0, size[0], size[1])

        for layer in range(1, top_layer + 1):
            self.mark_bbox_used(layer, self.prim_bound_box)

        self.sch_params = dict(
            lib_name=lib_name,
            cell_name=cell_name,
        )

    def _register_pin(self, lay: str, term_name: str, box: BBox, show_pins: bool) -> None:
        # TODO: find way to add WireArray if possible
        self.add_pin_primitive(term_name, lay, box, show=show_pins)
