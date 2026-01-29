import gymnasium as gym

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

# after this steps, subsequent `import custom_smaclite` will work
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

    # this call automatically register smaclite envs
    from smaclite.env.maps.map import MapPreset  # noqa
    import custom_smaclite

    for preset in MapPreset:
        map_info = preset.value
        gym.register(
            f"custom-smaclite/{map_info.name}-v0",
            entry_point="custom_smaclite.env:CustomSMACliteEnv",
            kwargs={"map_info": map_info},
        )
    gym.register(
        "custom-smaclite/custom-v0", entry_point="custom_smaclite.env:CustomSMACliteEnv"
    )

    from envs.custom_smaclite.env.maps import CustomMapPreset
    for preset in CustomMapPreset:
        map_info = preset.value
        gym.register(
            f"custom-smaclite/{map_info.name}-v0",
            entry_point="custom_smaclite.env:CustomSMACliteEnv",
            kwargs={"map_info": map_info},
        )