# -*- coding: utf-8 -*-

from typing import Dict, Any

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag_test__pin_array_0(Module):
    """Module for library bag_test cell pin_array_0.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'pin_array_0.yaml'))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            mult='number of pins in parallel',
        )

    def design(self, mult: int) -> None:
        if mult > 1:
            bus_name = f'<{mult - 1}:0>'
            in_name = f'vin{bus_name}'
            out_name = f'vout{bus_name}'
            self.rename_pin('vin', in_name)
            self.rename_pin('vout', out_name)
            self.rename_instance('XIN', f'XIN{bus_name}', [('noConn', in_name)])
            self.rename_instance('XOUT', f'XOUT{bus_name}', [('noConn', out_name)])
