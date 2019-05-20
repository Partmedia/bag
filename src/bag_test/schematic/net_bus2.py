# -*- coding: utf-8 -*-

from typing import Dict, Any

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag_test__net_bus2(Module):
    """Module for library bag_test cell net_bus2.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'net_bus2.yaml'))

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
            mult='number of bits'
        )

    def design(self, mult: int) -> None:
        self.instances['X0'].design(mult=mult)

        bus_name = f'<{mult - 1}:0>'
        in_name = f'vin{bus_name}'
        out_name = f'vout{bus_name}'

        self.rename_pin('vin', in_name)
        self.rename_pin('vout', f'vout{bus_name}')
        self.reconnect_instance_terminal('X0', f'vin{bus_name}', in_name)
        self.reconnect_instance_terminal('X0', f'vout{bus_name}', out_name)

