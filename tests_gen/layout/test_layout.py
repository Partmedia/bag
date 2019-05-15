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

from typing import Dict, Any, Type, cast

import pathlib
import importlib
from shutil import copyfile

import pytest

from pybag.core import read_gds
from pybag.enum import DesignOutput, get_extension

from bag.layout.template import TemplateDB, TemplateBase, TemplateType
from bag.env import get_gds_layer_map, get_gds_object_map


def get_master(temp_db: TemplateDB, lay_design_params: Dict[str, Any]) -> TemplateBase:
    module_name = lay_design_params['module']
    cls_name = lay_design_params['class']
    params = lay_design_params['params']
    try:
        lay_module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError('Cannot find Python module {} for layout generator.  '
                          'Is it on your PYTHONPATH?'.format(module_name))

    if not hasattr(lay_module, cls_name):
        raise ImportError('Cannot find layout generator class {} '
                          'in module {}'.format(cls_name, module_name))

    gen_cls = cast(Type[TemplateType], getattr(lay_module, cls_name))
    ans = temp_db.new_master(gen_cls, params=params)
    return ans


@pytest.mark.parametrize("output_type", [
    DesignOutput.GDS,
])
def test_layout(tmpdir,
                temp_db: TemplateDB,
                lay_design_params: Dict[str, Any],
                output_type: DesignOutput,
                gen_output: bool,
                ) -> None:
    """Test design() method of each schematic generator."""
    if lay_design_params is None:
        # no layout tests
        return

    extension = get_extension(output_type)

    base = 'out'
    expect_fname = lay_design_params.get('{}_{}'.format(base, extension), '')
    if not expect_fname and not gen_output:
        pytest.skip('Cannot find expected output file.')

    dsn = get_master(temp_db, lay_design_params)
    assert dsn is not None

    out_base_name = '{}.{}'.format(base, extension)
    path = tmpdir.join(out_base_name)
    temp_db.instantiate_layout(dsn, 'PYTEST_TOP', output=output_type, fname=str(path))
    assert path.check(file=1)

    actual_fname = str(path)
    if gen_output:
        dir_name = pathlib.Path('pytest_output') / lay_design_params['test_output_dir']
        expect_fname = dir_name / out_base_name
        dir_name.mkdir(parents=True, exist_ok=True)
        copyfile(actual_fname, expect_fname)
    else:
        # currently only works for GDS
        lay_map = get_gds_layer_map()
        obj_map = get_gds_object_map()
        grid = temp_db.grid
        expect_cv_list = read_gds(expect_fname, lay_map, obj_map, grid)
        actual_cv_list = read_gds(actual_fname, lay_map, obj_map, grid)

        assert expect_cv_list == actual_cv_list
