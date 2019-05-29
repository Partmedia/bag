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

from typing import List, Dict, Tuple

from pathlib import Path

import h5py
import numpy as np

from .data import MDArray


def save_md_array_hdf5(data: MDArray, hdf5_path: Path, compression='lzf') -> None:
    """Saves the given MDArray as a HDF5 file.

    The simulation environments are stored as fixed length byte strings,
    and the sweep parameters are stored as dimension label for each data.

    Parameters
    ----------
    data: MDArray
        the data.
    hdf5_path: Path
        the hdf5 file path.
    compression : str
        HDF5 compression method.  Defaults to 'lzf' for speed (use 'gzip' for space).
    """
    # create parent directory
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(hdf5_path), 'w', libver='latest') as f:
        f.create_dataset('corner', data=np.array(data.sim_envs, dtype='S'))
        for group in data.group_list:
            data.open_group(group)
            grp = f.create_group(group)
            for name, arr in data.items():
                dset = grp.create_dataset(name, data=arr, chunks=True, compression=compression)
                var_list = data.get_swp_params(name)
                if var_list:
                    for idx, var in enumerate(var_list):
                        dset.dims[idx].label = var


def load_md_array_hdf5(path: Path) -> MDArray:
    """Read simulation results from HDF5 file.

    Parameters
    ----------
    path : Path
        the file to read.

    Returns
    -------
    results : MDArray
        the data.
    """
    if not path.is_file():
        raise ValueError(f'{path} is not a file.')

    with h5py.File(str(path), 'r') as f:
        sim_envs = f['corner'][:].astype('U').tolist()
        data: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]] = {}
        for analysis, grp in f.items():
            if analysis != 'corner':
                arr_dict: Dict[str, np.ndarray] = {}
                swp_dict: Dict[str, List[str]] = {}
                for name, dset in grp.items():
                    arr_dict[name] = dset[:]
                    if dset.dims[0].label:
                        # has sweep parameters
                        swp_dict[name] = [d.label for d in dset.dims]

                data[analysis] = (arr_dict, swp_dict)
        return MDArray(sim_envs, data)
