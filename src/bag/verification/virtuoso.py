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

"""This module handles exporting schematic/layout from Virtuoso.
"""
from typing import TYPE_CHECKING, Optional, Dict, Any

import os
from abc import ABC

from ..io import write_file, open_temp
from .base import SubProcessChecker

if TYPE_CHECKING:
    from .base import ProcInfo


class VirtuosoChecker(SubProcessChecker, ABC):
    """the base Checker class for Virtuoso.

    This class implement layout/schematic export procedures.

    Parameters
    ----------
    tmp_dir : str
        temporary file directory.
    max_workers : int
        maximum number of parallel processes.
    cancel_timeout : float
        timeout for cancelling a subprocess.
    source_added_file : str
        file to include for schematic export.
    """

    def __init__(self, tmp_dir, max_workers, cancel_timeout, source_added_file):
        # type: (str, int, float, str) -> None
        SubProcessChecker.__init__(self, tmp_dir, max_workers, cancel_timeout)
        self._source_added_file = source_added_file

    def setup_export_layout(self, lib_name, cell_name, out_file, view_name='layout', params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        out_file = os.path.abspath(out_file)

        run_dir = os.path.dirname(out_file)
        out_name = os.path.basename(out_file)
        log_file = os.path.join(run_dir, 'layout_export.log')

        os.makedirs(run_dir, exist_ok=True)

        # fill in stream out configuration file.
        content = self.render_file_template('layout_export_config.txt',
                                            dict(
                                                lib_name=lib_name,
                                                cell_name=cell_name,
                                                view_name=view_name,
                                                output_name=out_name,
                                                run_dir=run_dir,
                                            ))

        with open_temp(prefix='stream_template', dir=run_dir, delete=False) as config_file:
            config_fname = config_file.name
            config_file.write(content)

        # run strmOut
        cmd = ['strmout', '-templateFile', config_fname]

        return cmd, log_file, None, os.environ['BAG_WORK_DIR']

    def setup_export_schematic(self, lib_name, cell_name, out_file, view_name='schematic',
                               params=None):
        # type: (str, str, str, str, Optional[Dict[str, Any]]) -> ProcInfo
        out_file = os.path.abspath(out_file)

        run_dir = os.path.dirname(out_file)
        out_name = os.path.basename(out_file)
        log_file = os.path.join(run_dir, 'schematic_export.log')

        # fill in stream out configuration file.
        content = self.render_file_template('si_env.txt',
                                            dict(
                                                lib_name=lib_name,
                                                cell_name=cell_name,
                                                view_name=view_name,
                                                output_name=out_name,
                                                source_added_file=self._source_added_file,
                                                run_dir=run_dir,
                                            ))

        # create configuration file.
        config_fname = os.path.join(run_dir, 'si.env')
        write_file(config_fname, content)

        # run command
        cmd = ['si', run_dir, '-batch', '-command', 'netlist']

        return cmd, log_file, None, os.environ['BAG_WORK_DIR']
