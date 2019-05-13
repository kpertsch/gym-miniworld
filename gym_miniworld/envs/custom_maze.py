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


class CustomMaze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=8,
        num_cols=8,
        room_size=3,
        max_episode_steps=None,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        super().__init__(
            max_episode_steps = max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        textures = ["brick_wall"] #, "cinder_blocks", "drywall", "marble", "metal_grill",
                    #"wood", "wood_planks"]
        rows = []

        maze_rand_gen = RandGen(42)  # fix maze layout

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex=maze_rand_gen.subset(textures, 1)[0],  # 'brick_wall',
                    floor_tex='asphalt',
                    no_ceiling=True,
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = maze_rand_gen.subset([(0,1), (0,-1), (-1,0), (1,0)], 4)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        # prune some walls to make maze multi-modal
        self.connect_rooms(rows[3][0], rows[4][0], min_x=rows[3][0].min_x, max_x=rows[3][0].max_x)
        self.connect_rooms(rows[2][1], rows[2][2], min_z=rows[2][1].min_z, max_z=rows[2][1].max_z)

        self.box = self.place_entity(Box(color='red'))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

class CustomMazeS2(CustomMaze):
    def __init__(self):
        super().__init__(num_rows=2, num_cols=2)

class CustomMazeS3(CustomMaze):
    def __init__(self):
        super().__init__(num_rows=3, num_cols=3)

class CustomMazeFast(CustomMaze):
    def __init__(self, forward_step=0.7, turn_step=45, nrow=3, ncol=3):

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        max_steps = int(300 * (nrow*ncol/9))

        super().__init__(
            num_rows=nrow,
            num_cols=ncol,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=False
        )


class CustomMazeS3Fast(CustomMazeFast):
    def __init__(self):
        super().__init__(nrow=3, ncol=3)


class CustomMazeS4Fast(CustomMazeFast):
    def __init__(self):
        super().__init__(nrow=4, ncol=4)


class CustomMazeS5Fast(CustomMazeFast):
    def __init__(self):
        super().__init__(nrow=5, ncol=5)
