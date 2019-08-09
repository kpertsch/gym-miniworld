import numpy as np
from gym import spaces
from ..miniworld import MiniWorldEnv
from ..entity import Box, MarkerFrame
from ..params import DEFAULT_PARAMS


SCALE = 27      # scale from mujoco layout to miniworld environment


class Multiroom3d(MiniWorldEnv):
    """
    Multiroom environment in which the agent has to reach a red box
    """

    def __init__(self, **kwargs):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 2.0)
        params.set('turn_step', 45)

        self.layout_params = kwargs.pop('layout_params')
        self.rooms_per_side = kwargs.pop('rooms_per_side')
        self.doors = kwargs.pop('doors')
        self.heading_smoothing = kwargs.pop('heading_smoothing')
        self.room_size = self.mj2mw(self.layout_params.room_size)
        self.door_size = self.mj2mw(self.layout_params.door_size)
        self.textures = self.layout_params.textures
        super().__init__(**kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self, reset_state=None):
        self.add_rooms()
        self.add_doors()

        # add start and goal
        if reset_state is None:
            self.goal = self.place_entity(Box(color='red'))
            self.place_agent()
        else:
            self.goal = self.place_entity(Box(color='red'),
                                          pos=np.array((reset_state.goal[0], 0.1, reset_state.goal[1])))
            self.place_agent(pos=np.array((reset_state.start_pos[0], 0.1, reset_state.start_pos[1])),
                             dir=reset_state.start_angle)

    def _step_continuous_action(self, action):
        """d_angle in deg."""
        assert action.size == 2  # can currently only support [dx, dz] action
        dx, dz = action
        current_heading = self.agent.dir * 180 / np.pi

        heading = np.arctan2(-dz, dx) * 180 / np.pi
        distance = np.sqrt(dx**2 + dz**2)

        def get_angle_dist(a1, a2):
            mod = lambda a, n: a - np.floor(a/n) * n
            return mod((a1 - a2 + 180), 360) - 180

        d_heading = get_angle_dist(heading, current_heading)
        self.turn_agent(-current_heading + heading)
        self.move_agent(distance, fwd_drift=0)
        self.turn_agent(current_heading - heading + d_heading * self.heading_smoothing)

    def step(self, action):
        obs, reward, done, agent_pos = super().step(action)

        if self.near(self.goal):
            reward += self._reward()
            done = True

        return obs, reward, done, np.delete(agent_pos, [1])

    @staticmethod
    def mj2mw(val):
        scaled = val * SCALE
        if isinstance(val, float):
            return scaled
        elif val.size == 2 or len(val.shape) == 2:
            return np.array([scaled[0], -scaled[1]])
        else:
            raise ValueError("Scale function does not support inputs of shape {}".format(val.shape))

    @staticmethod
    def mw2mj(val):
        scaled = val / SCALE
        if isinstance(val, float):
            return scaled
        elif val.size == 2 or len(val.shape) == 2:
            return np.array([scaled[0], -scaled[1]])
        else:
            raise ValueError("Scale function does not support inputs of shape {}".format(val.shape))

    def add_rooms(self):

        def add_room(xmin, xmax, zmin, zmax):
            # adds rooms to member self.rooms automatically
            self.add_rect_room(
                min_x=xmin,
                max_x=xmax,
                min_z=zmin,
                max_z=zmax,
                wall_tex=self.textures[len(self.rooms) % len(self.textures)],
                floor_tex='asphalt',
                no_ceiling=True,
            )

        # add rooms
        offset = self.rooms_per_side * self.room_size / 2
        for x in range(self.rooms_per_side):
            for y in range(self.rooms_per_side):
                add_room(x * self.room_size - offset, (x+1) * self.room_size - offset,
                         y * self.room_size - offset, (y+1) * self.room_size - offset)

    def add_doors(self):
        # prune some walls to connect rooms
        def connect_horizontal(room_1, room_2):
            assert room_1.min_z == room_2.min_z and room_1.max_z == room_2.max_z  # can only connect neighboring rooms
            door_center = (room_1.max_z - room_1.min_z) / 2 + room_1.min_z
            self.connect_rooms(room_1, room_2, min_z=door_center - self.door_size / 2,
                               max_z=door_center + self.door_size / 2)

        def connect_vertical(room_1, room_2):
            assert room_1.min_x == room_2.min_x and room_1.max_x == room_2.max_x  # can only connect neighboring rooms
            door_center = (room_1.max_x - room_1.min_x) / 2 + room_1.min_x
            self.connect_rooms(room_1, room_2, min_x=door_center - self.door_size / 2,
                               max_x=door_center + self.door_size / 2)

        # create doors between rooms
        for door in self.doors:
            if door[1] - door[0] == 1:  # vertical door
                connect_vertical(self.rooms[door[0]], self.rooms[door[1]])
            else:
                connect_horizontal(self.rooms[door[0]], self.rooms[door[1]])

    def add_marker(self, room, marker_num, side):
        EPS = self.room_size * 0.02
        if side == 'left':
            pos1 = (room.min_x + EPS, 1.5, (room.max_z - room.min_z) / 2 + room.min_z - self.room_size / 4)
            pos2 = (room.min_x + EPS, 1.5, (room.max_z - room.min_z) / 2 + room.min_z + self.room_size / 4)
            dir = 0
        elif side == 'right':
            pos1 = (room.max_x - EPS, 1.5, (room.max_z - room.min_z) / 2 + room.min_z - self.room_size / 4)
            pos2 = (room.max_x - EPS, 1.5, (room.max_z - room.min_z) / 2 + room.min_z + self.room_size / 4)
            dir = np.pi
        elif side == 'top':
            pos1 = ((room.max_x - room.min_x) / 2 + room.min_x - self.room_size / 4, 1.5, room.min_z + EPS)
            pos2 = ((room.max_x - room.min_x) / 2 + room.min_x + self.room_size / 4, 1.5, room.min_z + EPS)
            dir = 3 * np.pi / 2
        elif side == 'bottom':
            pos1 = ((room.max_x - room.min_x) / 2 + room.min_x - self.room_size / 4, 1.5, room.max_z - EPS)
            pos2 = ((room.max_x - room.min_x) / 2 + room.min_x + self.room_size / 4, 1.5, room.max_z - EPS)
            dir = np.pi / 2
        else:
            raise ValueError("Marker placement '{}' not supported!".format(side))
        self.markers.append(self.place_entity(MarkerFrame(pos=(0, 0, 0), dir=0,
                                                          marker_num=marker_num, width=1.5),
                                              pos=pos1, dir=dir))
        self.markers.append(self.place_entity(MarkerFrame(pos=(0, 0, 0), dir=0,
                                                          marker_num=marker_num, width=1.5),
                                              pos=pos2, dir=dir))

