
import numpy as np
from envs.gridworld.gridworld_env import *

class NRoom(GridworldEnv):
    def __init__(self, nroom, *args, **kwargs):
        self.nroom = nroom
        super().__init__(*args, **kwargs)


    def _read_grid_map(self, grid_map_path):
        nroom = self.nroom
        assert nroom > 1
        map_width, map_height = 10, nroom * 2-1
        self.grid_map = np.empty((map_height, map_width), dtype='O')
        self.agent_list = []
        self.n_goals = 0
        
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                self.grid_map[i, j] = WorldObj(i, j, EMPTY, canpassby=True)

        # wall 
        for i in range(nroom-1):
            for j in range(map_width):
                self.set(i*2 + 1, j, WorldObj(i*2 + 1, j, WALL))

        # agent
        self.n_goals = 1
        self.set(0, map_width-1, WorldObj(0, map_width-1, TARGET, canpassby=True))
        self.agent_list.append(WorldObj(0, 0, AGENT, target=(0, map_width-1)))
        for agent in range(1, nroom):
            self.agent_list.append(WorldObj(agent*2, 0, AGENT, target=None))

        # door
        self.set(0, map_width//2, WorldObj(0, map_width//2, DOOR))
        true_switch = np.random.randint(1, nroom)
        self.set(true_switch*2, map_width-1, WorldObj(true_switch*2, map_width-1, SWITCH, 
                                            target=(0, map_width//2), 
                            canpassby=True, open_on_hold=True))

        # fake switch
        for switch in range(1, nroom):
            if switch == true_switch: continue
            self.set(switch*2, map_width-1, WorldObj(switch*2, map_width-1, SWITCH, 
                                            target=None, 
                            canpassby=True, open_on_hold=True))

        return copy.deepcopy(self.grid_map)
    
    
if __name__ == "__main__":
    env = NRoom(nroom=10)
    env.reset()
    import cv2
    cv2.imwrite(f"render_nroom.png", 
                cv2.cvtColor(env.render(), 
                             cv2.COLOR_RGB2BGR))
    print(env)