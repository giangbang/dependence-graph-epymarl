import custom_smaclite.env.units.unit_type as ut

from smaclite.env.units.unit import Unit as BaseUnit


class Unit(BaseUnit):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.targeter = ut.TARGETER_CACHE[self.type.stats.name]
