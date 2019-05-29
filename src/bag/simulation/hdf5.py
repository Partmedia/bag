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


def save_md_array_hdf5(env_list: List[str],
                       data: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]],
                       hdf5_path: Path, compression='gzip') -> None:
    """Saves the given MDArray as a HDF5 file.

    Parameters
    ----------
    data: MDArray
        the data.
    path: Path
        the hdf5 file path.
    compression : str
        HDF5 compression method.  Defaults to 'gzip'.
    """
    # create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    sweep_info = results['sweep_params']
    with h5py.File(str(path), 'w') as f:
        for name, swp_vars in sweep_info.items():
            # store data
            data = np.asarray(results[name])
            if not data.shape:
                dset = f.create_dataset(name, data=data)
            else:
                dset = f.create_dataset(name, data=data, compression=compression)
            # h5py workaround: need to explicitly store unicode
            dset.attrs['sweep_params'] = [swp.encode(encoding=bag_encoding, errors=bag_codec_error)
                                          for swp in swp_vars]

            # store sweep parameter values
            for var in swp_vars:
                if var not in f:
                    swp_data = results[var]
                    if np.issubdtype(swp_data.dtype, np.unicode_):
                        # we need to explicitly encode unicode strings to bytes
                        swp_data = [v.encode(encoding=bag_encoding, errors=bag_codec_error) for v in swp_data]

                    f.create_dataset(var, data=swp_data, compression=compression)


def load_md_array_hdf5(path: Path) -> MDArray:
    """Read simulation results from HDF5 file.

    Parameters
    ----------
    fname : str
        the file to read.

    Returns
    -------
    results : dict[str, any]
        the result dictionary.
    """
    if not os.path.isfile(fname):
        raise ValueError('%s is not a file.' % fname)

    results = {}
    sweep_params = {}
    with h5py.File(fname, 'r') as f:
        for name in f:
            dset = f[name]
            dset_data = dset[()]
            if np.issubdtype(dset.dtype, np.bytes_):
                # decode byte values to unicode arrays
                dset_data = np.array([v.decode(encoding=bag_encoding, errors=bag_codec_error) for v in dset_data])

            if 'sweep_params' in dset.attrs:
                cur_swp = [swp.decode(encoding=bag_encoding, errors=bag_codec_error)
                           for swp in dset.attrs['sweep_params']]
                results[name] = SweepArray(dset_data, cur_swp)
                sweep_params[name] = cur_swp
            else:
                results[name] = dset_data

    results['sweep_params'] = sweep_params
    return results
