import os
import sys

from envs.multiagentenv import MultiAgentEnv

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
    from envs.smaclite_wrapper import SMACliteWrapper
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    from envs.gymma import GymmaWrapper
    import envs.custom_lbf  # noqa
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


def magent_fn(**kwargs) -> MultiAgentEnv:
    from envs.magent_wrapper import MAgentWrapper
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return MAgentWrapper(**kwargs)


def mpe_fn(**kwargs) -> MultiAgentEnv:
    from envs.mpe_wrapper import MPEWrapper
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return MPEWrapper(**kwargs)


def stag_hunt_fn(**kwargs) -> MultiAgentEnv:
    from envs.stag_hunt import StagHuntWrapper
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return StagHuntWrapper(**kwargs)


def predator_prey_fn(**kwargs) -> MultiAgentEnv:
    from envs.predator_prey import PredatorPrey
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert (
        kwargs["common_reward"] == True
    ), "Not support separate reward in Predator Prey"
    return PredatorPrey(**kwargs)

def gridworld_fn(**kwargs) -> MultiAgentEnv: 
    from envs.gridworld_wrapper import GridworldWrapper
    return GridworldWrapper(**kwargs)

def football_fn(**kwargs) -> MultiAgentEnv:
    from envs.football_wrapper import FootballWrapper
    return FootballWrapper(**kwargs)

def star_spread_mpe_fn(**kwargs) -> MultiAgentEnv:
    from envs.star_spread_mpe import StarSpreadMPEv2Env
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return StarSpreadMPEv2Env(**kwargs)

def line_spread_mpe_fn(**kwargs) -> MultiAgentEnv:
    from envs.line_spread_mpe import LineSpreadMPEEnv
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return LineSpreadMPEEnv(**kwargs)

REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn
REGISTRY["magent"] = magent_fn
REGISTRY["mpe"] = mpe_fn
REGISTRY["stag_hunt"] = stag_hunt_fn
REGISTRY["predator_prey"] = predator_prey_fn
REGISTRY["gridworld"] = gridworld_fn
REGISTRY["football"] = football_fn
REGISTRY["star_spread_mpe"] = star_spread_mpe_fn
REGISTRY["line_spread_mpe"] = line_spread_mpe_fn


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
