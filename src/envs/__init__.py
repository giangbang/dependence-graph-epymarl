import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .smaclite_wrapper import SMACliteWrapper
from .magent_wrapper import MAgentWrapper
from .mpe_wrapper import MPEWrapper
from .stag_hunt import StagHuntWrapper
from .predator_prey import PredatorPrey


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    if not kwargs["map_name"].startswith(
        "custom-smaclite"
    ):  # otherwise custom-smaclite
        assert kwargs[
            "common_reward"
        ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
        del kwargs["common_reward"]
        del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


def magent_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return MAgentWrapper(**kwargs)


def mpe_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return MPEWrapper(**kwargs)


def stag_hunt_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return StagHuntWrapper(**kwargs)


def predator_prey_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert (
        kwargs["common_reward"] == True
    ), "Not support separate reward in Predator Prey"
    return PredatorPrey(**kwargs)

def gridworld_fn(**kwargs) -> MultiAgentEnv: 
    from envs.gridworld_wrapper import GridworldWrapper
    return GridworldWrapper(**kwargs)


REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn
REGISTRY["magent"] = magent_fn
REGISTRY["mpe"] = mpe_fn
REGISTRY["stag_hunt"] = stag_hunt_fn
REGISTRY["predator_prey"] = predator_prey_fn
REGISTRY["gridworld"] = gridworld_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
