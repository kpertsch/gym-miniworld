import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame
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
        params.set('forward_step', 0.7)
        params.set('turn_step', 45)

        self.room_size = _mj2mw(1/3)
        self.door_size = _mj2mw(1.5 * 0.0667)
        self.wall_size = _mj2mw(0.005)
        super().__init__(**kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        textures = ["brick_wall"] #, "cinder_blocks", "drywall", "marble", "metal_grill",
                    #"wood", "wood_planks"]
        maze_rand_gen = RandGen(42)  # fix maze layout

        def add_room(xmin, xmax, zmin, zmax):
            # adds rooms to member self.rooms automatically
            self.add_rect_room(
                min_x=xmin,
                max_x=xmax,
                min_z=zmin,
                max_z=zmax,
                wall_tex=maze_rand_gen.subset(textures, 1)[0],
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

        connect_horizontal(self.rooms[0], self.rooms[3])
        connect_horizontal(self.rooms[3], self.rooms[6])
        connect_horizontal(self.rooms[1], self.rooms[4])
        connect_horizontal(self.rooms[4], self.rooms[7])
        connect_horizontal(self.rooms[5], self.rooms[8])

        connect_vertical(self.rooms[1], self.rooms[2])
        connect_vertical(self.rooms[3], self.rooms[4])
        connect_vertical(self.rooms[7], self.rooms[8])

        self.box = self.place_entity(Box(color='red'))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
