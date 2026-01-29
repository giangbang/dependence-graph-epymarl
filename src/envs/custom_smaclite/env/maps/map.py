from enum import Enum
import os
from smaclite.env.maps import MapInfo


def get_standard_map(map_name):
    return MapInfo.from_file(os.path.join(os.path.dirname(__file__),
                                          'smaclite_maps', f"{map_name}.json"))


class CustomMapPreset(Enum):

    @property
    def map_info(self) -> MapInfo:
        return self.value

    MAP_5M_VS_6M = get_standard_map('5m_vs_6m')
    MAP_25M_VS_30M = get_standard_map('25m_vs_30m')