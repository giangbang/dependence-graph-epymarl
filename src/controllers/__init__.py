REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC


from .dcg_controller import DeepCoordinationGraphMAC

REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC

from .dcg_noshare_controller import DCGnoshareMAC

REGISTRY["dcg_noshare_mac"] = DCGnoshareMAC
