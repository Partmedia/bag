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

from typing import Dict, Any, cast, Type

import re
import pathlib

import pytest

from pybag.enum import DesignOutput, get_extension, is_model_type

from bag.env import get_bag_work_dir
from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.io.string import read_yaml_str


def check_netlist(output_type: DesignOutput, actual: str, expect: str) -> None:
    if output_type == DesignOutput.CDL:
        inc_line = '^\\.INCLUDE (.*)$'
    elif (output_type == DesignOutput.VERILOG or
          output_type == DesignOutput.SYSVERILOG or
          output_type == DesignOutput.SPECTRE):
        inc_line = '^include "(.*)"$'
    else:
        inc_line = ''

    if not inc_line:
        assert actual == expect
    else:
        bag_work_dir = get_bag_work_dir()
        pattern = re.compile(inc_line)
        actual_lines = actual.splitlines()
        expect_lines = expect.splitlines()
        for al, el in zip(actual_lines, expect_lines):
            am = pattern.match(al)
            if am is None:
                assert al == el
            else:
                em = pattern.match(el)
                if em is None:
                    assert al == el
                else:
                    # both are include statements
                    apath = am.group(1)
                    epath = em.group(1)
                    arel = pathlib.Path(apath).relative_to(bag_work_dir)
                    assert epath.endswith(str(arel))


def get_sch_master(module_db: ModuleDB, sch_design_params: Dict[str, Any]) -> Module:
    lib_name = sch_design_params['lib_name']
    cell_name = sch_design_params['cell_name']
    params = sch_design_params['params']

    gen_cls = cast(Type[Module], module_db.get_schematic_class(lib_name, cell_name))
    ans = module_db.new_master(gen_cls, params=params)
    return ans


@pytest.mark.parametrize("output_type, options", [
    (DesignOutput.YAML, {}),
    (DesignOutput.CDL, {'flat': True, 'shell': False, 'rmin': 2000}),
    (DesignOutput.SPECTRE, {'flat': True, 'shell': False, 'rmin': 2000}),
    (DesignOutput.SPECTRE, {'flat': True, 'shell': False, 'top_subckt': False, 'rmin': 2000}),
    (DesignOutput.VERILOG, {'flat': True, 'shell': False, 'rmin': 2000}),
    (DesignOutput.VERILOG, {'flat': True, 'shell': True, 'rmin': 2000}),
    (DesignOutput.SYSVERILOG, {'flat': True, 'shell': False, 'rmin': 2000}),
])
def test_design(tmpdir,
                module_db: ModuleDB,
                sch_design_params: Dict[str, Any],
                output_type: DesignOutput,
                options: Dict[str, Any],
                gen_output: bool,
                ) -> None:
    """Test design() method of each schematic generator."""
    if sch_design_params is None:
        # No schematic tests
        return

    impl_cell = sch_design_params['top_cell_name']
    extension = get_extension(output_type)

    if is_model_type(output_type):
        model_params = sch_design_params.get('model_params', None)
        if model_params is None:
            pytest.skip('Cannot find model parameters.')
    else:
        model_params = None

    out_code = int(options.get('shell', False)) + 2 * (1 - int(options.get('top_subckt', True)))
    if output_type is DesignOutput.YAML:
        base = 'out'
    else:
        base = f'out_{out_code}'

    expect_fname = sch_design_params.get('{}_{}'.format(base, extension), '')
    if not expect_fname and not gen_output:
        pytest.skip('Cannot find expected output file.')

    dsn = get_sch_master(module_db, sch_design_params)

    out_base_name = '{}.{}'.format(base, extension)
    path = tmpdir.join(out_base_name)
    if is_model_type(output_type):
        module_db.batch_model([(dsn, impl_cell, model_params)], output=output_type,
                              fname=str(path), **options)
    else:
        module_db.instantiate_master(output_type, dsn, top_cell_name=impl_cell, fname=str(path),
                                     **options)

    assert path.check(file=1)

    with path.open('r') as f:
        actual = f.read()

    if gen_output:
        dir_name = pathlib.Path('pytest_output') / sch_design_params['test_output_dir']
        out_fname = dir_name / out_base_name
        dir_name.mkdir(parents=True, exist_ok=True)
        with out_fname.open('w') as f:
            f.write(actual)
        expect = actual
    else:
        with open(expect_fname, 'r') as f:
            expect = f.read()

    if output_type is DesignOutput.YAML:
        actual_dict = read_yaml_str(actual)
        expect_dict = read_yaml_str(expect)
        assert actual_dict == expect_dict
    else:
        check_netlist(output_type, actual, expect)
