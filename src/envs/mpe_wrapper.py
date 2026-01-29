from .gymma import GymmaWrapper


class MPEWrapper(GymmaWrapper):

    def get_state(self):
        return self._env.unwrapped._env.state()

    def get_state_size(self):
        return self.get_state().shape[0]
    
    def render(self, **kwargs):
        return self._env.unwrapped._env.render()