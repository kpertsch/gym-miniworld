import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, MarkerFrame
from ..params import DEFAULT_PARAMS
from ..random import RandGen


SCALE = 27      # scale from mujoco layout to miniworld environment


def _mj2mw(val):
    return val * SCALE


def _mw2mj(val):
    return val / SCALE


class Multiroom3d(MiniWorldEnv):
    """
    Multiroom environment in which the agent has to reach a red box
    """

    def __init__(self, **kwargs):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 2.0)
        params.set('turn_step', 45)

        self.room_size = _mj2mw(1/3)
        self.door_size = _mj2mw(1.5 * 0.0667)
        self.wall_size = _mj2mw(0.005)
        super().__init__(**kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        textures = ["lg_style_01_wall_{}".format(s) for s in ['green_bright', 'blue', 'cerise', 'green', 'purple',
                                                              'red', 'yellow']] + \
                   ["lg_style_03_wall_{}".format(s) for s in ['cyan', 'orange']]
        maze_rand_gen = RandGen(42)  # fix maze layout

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
        self.box = self.place_entity(Box(color='red'))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
