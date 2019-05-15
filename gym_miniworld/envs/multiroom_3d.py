import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, MarkerFrame
from ..params import DEFAULT_PARAMS
from ..random import RandGen


SCALE = 27      # scale from mujoco layout to miniworld environment
HEADING_SMOOTHING_CONSTANT = 0.3


class Multiroom3d(MiniWorldEnv):
    """
    Multiroom environment in which the agent has to reach a red box
    """

    def __init__(self, **kwargs):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 2.0)
        params.set('turn_step', 45)

        self.room_size = self.mj2mw(1/3)
        self.door_size = self.mj2mw(1.5 * 0.0667)
        self.wall_size = self.mj2mw(0.005)
        super().__init__(**kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self, reset_state=None):
        textures = ["lg_style_01_wall_{}".format(s) for s in ['green_bright', 'blue', 'cerise', 'green', 'purple',
                                                              'red', 'yellow']] + \
                   ["lg_style_03_wall_{}".format(s) for s in ['cyan', 'orange']]

        def add_room(xmin, xmax, zmin, zmax):
            # adds rooms to member self.rooms automatically
            self.add_rect_room(
                min_x=xmin,
                max_x=xmax,
                min_z=zmin,
                max_z=zmax,
                wall_tex=textures[len(self.rooms) % len(textures)],
                floor_tex='asphalt',
                no_ceiling=True,
            )

        # add rooms
        add_room(-(1.5 * self.room_size + self.wall_size), -(0.5 * self.room_size + self.wall_size),
                 -(1.5 * self.room_size + self.wall_size), -(0.5 * self.room_size + self.wall_size))
        add_room(-(1.5 * self.room_size + self.wall_size), -(0.5 * self.room_size + self.wall_size),
                 -(0.5 * self.room_size), +(0.5 * self.room_size))
        add_room(-(1.5 * self.room_size + self.wall_size), -(0.5 * self.room_size + self.wall_size),
                 +(0.5 * self.room_size + self.wall_size), +(1.5 * self.room_size + self.wall_size))

        add_room(-(0.5 * self.room_size), +(0.5 * self.room_size),
                 -(1.5 * self.room_size + self.wall_size), -(0.5 * self.room_size + self.wall_size))
        add_room(-(0.5 * self.room_size), +(0.5 * self.room_size),
                 -(0.5 * self.room_size), +(0.5 * self.room_size))
        add_room(-(0.5 * self.room_size), +(0.5 * self.room_size),
                 +(0.5 * self.room_size + self.wall_size), +(1.5 * self.room_size + self.wall_size))

        add_room(+(0.5 * self.room_size + self.wall_size), +(1.5 * self.room_size + self.wall_size),
                 -(1.5 * self.room_size + self.wall_size), -(0.5 * self.room_size + self.wall_size))
        add_room(+(0.5 * self.room_size + self.wall_size), +(1.5 * self.room_size + self.wall_size),
                 -(0.5 * self.room_size), +(0.5 * self.room_size))
        add_room(+(0.5 * self.room_size + self.wall_size), +(1.5 * self.room_size + self.wall_size),
                 +(0.5 * self.room_size + self.wall_size), +(1.5 * self.room_size + self.wall_size))

        # prune some walls to connect rooms
        def connect_horizontal(room_1, room_2):
            assert room_1.min_z == room_2.min_z and room_1.max_z == room_2.max_z  # can only connect neighboring rooms
            door_center = (room_1.max_z - room_1.min_z) / 2 + room_1.min_z
            self.connect_rooms(room_1, room_2, min_z=door_center - self.door_size / 2, max_z=door_center + self.door_size / 2)

        def connect_vertical(room_1, room_2):
            assert room_1.min_x == room_2.min_x and room_1.max_x == room_2.max_x    # can only connect neighboring rooms
            door_center = (room_1.max_x - room_1.min_x) / 2 + room_1.min_x
            self.connect_rooms(room_1, room_2, min_x=door_center - self.door_size/2, max_x=door_center + self.door_size/2)

        # create doors between rooms
        connect_horizontal(self.rooms[0], self.rooms[3])
        connect_horizontal(self.rooms[3], self.rooms[6])
        connect_horizontal(self.rooms[1], self.rooms[4])
        connect_horizontal(self.rooms[4], self.rooms[7])
        connect_horizontal(self.rooms[5], self.rooms[8])

        connect_vertical(self.rooms[1], self.rooms[2])
        connect_vertical(self.rooms[3], self.rooms[4])
        connect_vertical(self.rooms[7], self.rooms[8])

        # add markers to the rooms
        self.markers = []
        def add_marker(room, marker_num, side):
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

        # add_marker(self.rooms[0], 0, 'left')
        # add_marker(self.rooms[0], 0, 'top')
        # add_marker(self.rooms[0], 0, 'bottom')
        #
        # add_marker(self.rooms[1], 1, 'top')
        # add_marker(self.rooms[1], 1, 'left')
        #
        # add_marker(self.rooms[2], 2, 'left')
        # add_marker(self.rooms[2], 2, 'right')
        # add_marker(self.rooms[2], 2, 'bottom')
        #
        # add_marker(self.rooms[3], 3, 'top')
        #
        # add_marker(self.rooms[4], 4, 'bottom')
        #
        # add_marker(self.rooms[5], 5, 'left')
        # add_marker(self.rooms[5], 5, 'top')
        # add_marker(self.rooms[5], 5, 'bottom')
        #
        # add_marker(self.rooms[6], 6, 'right')
        # add_marker(self.rooms[6], 6, 'top')
        # add_marker(self.rooms[6], 6, 'bottom')
        #
        # add_marker(self.rooms[7], 7, 'right')
        # add_marker(self.rooms[7], 7, 'top')
        #
        # add_marker(self.rooms[8], 8, 'right')
        # add_marker(self.rooms[8], 8, 'bottom')

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

        target_heading = HEADING_SMOOTHING_CONSTANT * heading + (1 - HEADING_SMOOTHING_CONSTANT) * current_heading
        d_heading = get_angle_dist(heading, current_heading)

        print("Current: {}, Heading: {}, Diff: {}".format(current_heading, heading, d_heading))

        self.turn_agent(-current_heading + heading)
        self.move_agent(distance, fwd_drift=0)
        self.turn_agent(current_heading - heading + d_heading * HEADING_SMOOTHING_CONSTANT)

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
