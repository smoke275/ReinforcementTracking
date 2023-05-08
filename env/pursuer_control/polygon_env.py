import csv

import gym
import numpy as np
import pygame
from bidict import bidict
from gym import spaces
import visilibity as vis

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

file_name = 'resources/sites_poly2.csv'
EPSILON = 0.0000001


class PolygonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 700  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 700.0, shape=(2,), dtype=float),
            }
        )

        # We have 8 actions, corresponding to "right", "up", "left", "down", "u-right", "u-left", "d-right", "d-left"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([-1.0, 0.0]),
            3: np.array([0.0, -1.0]),
            4: np.array([0.70710, 0.70710]),
            5: np.array([-0.70710, 0.70710]),
            6: np.array([0.70710, -0.70710]),
            7: np.array([-0.70710, -0.70710]),  # 1/root(2)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the env is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.poly = []
        with open(file_name) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                x_val = float(row[0]) * 0.7 + 350
                y_val = float(row[1]) * 0.7 + 350
                self.poly.append((x_val, y_val))

    def _get_obs(self):
        return {"agent": self._agent_location, }

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._agent_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        polygon = Polygon(self.poly)

        # Choose the agent's location uniformly at random
        self._agent_location = None

        while self._agent_location is None:
            test_pt = self.np_random.uniform(low=0.0, high=700.0, size=2)
            pursuer = Point(test_pt[0], test_pt[1])
            if polygon.contains(pursuer):
                self._agent_location = test_pt

        self._evader_location = None

        observer = vis.Point(self._agent_location[0], self._agent_location[1])
        poly_cop = [vis.Point(c[0], c[1]) for c in self.poly]
        walls = vis.Polygon([ele for ele in reversed(poly_cop)])
        self.env = vis.Environment([walls])

        self.env.PRINTING_DEBUG_DATA = False

        observer.snap_to_boundary_of(self.env, EPSILON)
        observer.snap_to_vertices_of(self.env, EPSILON)

        vispol = vis.Visibility_Polygon(observer, self.env, EPSILON)  # of each vertex in cluster

        while self._evader_location is None:
            test_pt = self.np_random.uniform(low=0.0, high=700.0, size=2)
            evader = vis.Point(test_pt[0], test_pt[1])
            if evader._in(vispol, EPSILON):
                self._evader_location = test_pt

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = self._agent_location + direction

        # An episode is done iff the agent has reached the target
        terminated = False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            # display_surface = pygame.display.get_surface()
            # display_surface.blit(pygame.transform.flip(display_surface, False, True), dest=(0, 0))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(0)

        # basic polygon
        pygame.draw.polygon(
            canvas,
            (255, 255, 255),
            self.poly,
            width=6,
        )

        observer = vis.Point(self._agent_location[0],
                             self._agent_location[1])
        vispol = vis.Visibility_Polygon(observer, self.env, EPSILON)
        point_x, point_y = poly_to_points(vispol)
        vispol_pt = [(point_x[it], point_y[it]) for it in range(len(point_y))]

        # visibility polygon of the pursuer(blue)
        # pygame.draw.polygon(
        #     canvas,
        #     (30, 30, 230),
        #     vispol_pt,
        #     width=3,
        # )

        for pt in self.poly:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                pt,
                5,
            )

        corners = []

        for i in range(len(self.poly)):
            a = self.poly[(i - 1) % len(self.poly)]
            b = self.poly[i]
            c = self.poly[(i + 1) % len(self.poly)]

            a = vis.Point(a[0], a[1])
            b = vis.Point(b[0], b[1])
            c = vis.Point(c[0], c[1])

            s = [b.x() - a.x(), b.y() - a.y()]
            t = [c.x() - b.x(), c.y() - b.y()]
            if s[0] * t[1] - t[0] * s[1] > 0:
                corners.append(i)

        for it in corners:
            col = (0, 255, 0)
            pygame.draw.circle(
                canvas,
                col,
                self.poly[it],
                5,
            )

        # evader
        pygame.draw.circle(
            canvas,
            pygame.Color('tomato2'),
            self._evader_location,
            5,
        )

        # pursuer
        pygame.draw.circle(
            canvas,
            pygame.Color('dodgerblue3'),
            self._agent_location,
            5,
        )

        vispol_inv = [(vispol[it].x(), vispol[it].y()) for it in reversed(range(vispol.n()))]
        bi = bidict()  # mapping from old to new

        for i in range(len(self.poly)):
            for j in range(vispol.n()):
                if np.linalg.norm(np.array(self.poly[i]) - np.array(vispol_inv[j])) < 0.001:
                    bi[i] = j

        vispol_corrected = vispol_inv.copy()
        new_vertices = {}  # store new vertices in vispol and the direction of free edge
        for j in range(vispol.n()):
            if bi.inverse.get(j) is None:
                new_vertices[j] = -1
            else:
                vispol_corrected[j] = self.poly[bi.inverse.get(j)]

        for i in new_vertices.keys():
            n_old = len(self.poly)
            n_new = vispol.n()
            pr = (i - 1) % n_new
            nx = (i + 1) % n_new
            pr_old = bi.inverse.get(pr)
            nx_old = bi.inverse.get(nx)
            if pr_old is None or nx_old is None:
                if nx_old is None:
                    new_vertices[i] = pr
                else:
                    new_vertices[i] = nx
            else:
                a = Point(self.poly[(pr_old - 1) % n_old][0], self.poly[(pr_old - 1) % n_old][1])
                b = Point(self.poly[pr_old][0], self.poly[pr_old][1])
                c = Point(self.poly[(pr_old + 1) % n_old][0], self.poly[(pr_old + 1) % n_old][1])
                det = (b.x - a.x) * (c.y - b.y) - (c.x - b.x) * (b.y - a.y)
                sign = 1 if det > 0 else -1
                vec_ba = Point(b.x - a.x, b.y - a.y)
                vec_cb = Point(c.x - b.x, c.y - b.y)
                dot_prod = (vec_ba.x * vec_cb.x + vec_ba.y * vec_cb.y) / (a.distance(b) * b.distance(c))
                angle = np.arccos(dot_prod)
                angle *= sign

                a1 = a
                b1 = Point(vispol_corrected[pr][0], vispol_corrected[pr][1])
                c1 = Point(vispol_corrected[(pr + 1) % n_new][0], vispol_corrected[(pr + 1) % n_new][1])

                det1 = (b1.x - a1.x) * (c1.y - b1.y) - (c1.x - b1.x) * (b1.y - a1.y)
                sign1 = 1 if det1 > 0 else -1
                vec_ba1 = Point(b1.x - a1.x, b1.y - a1.y)
                vec_cb1 = Point(c1.x - b1.x, c1.y - b1.y)
                dot_prod1 = (vec_ba1.x * vec_cb1.x + vec_ba1.y * vec_cb1.y) / (a1.distance(b1) * b1.distance(c1))
                angle_new = np.arccos(dot_prod1)
                angle_new *= sign1

                if np.abs(angle_new - angle) < 0.001:
                    new_vertices[i] = nx
                else:
                    new_vertices[i] = pr

        print(bi)
        print(new_vertices)

        observer1 = vis.Point(self._evader_location[0],
                             self._evader_location[1])
        env = vis.Environment([vispol])
        vispol1 = vis.Visibility_Polygon(observer1, env, EPSILON)
        point_x, point_y = poly_to_points(vispol1)
        vispol1_pt = [(point_x[it], point_y[it]) for it in range(len(point_y))]

        pygame.draw.polygon(
            canvas,
            pygame.Color('green'),
            vispol1_pt,
            width=3,
        )

        for k, v in new_vertices.items():
            pygame.draw.line(
                canvas,
                pygame.Color('pink'),
                vispol_corrected[k],
                vispol_corrected[v],
                width=3,
            )

        for it in range(vispol.n()):
            pygame.draw.circle(
                canvas,
                pygame.Color('yellow'),
                (vispol[it].x(), vispol[it].y()),
                5,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(pygame.transform.flip(canvas, False, True), canvas.get_rect())
            pygame.display.flip()
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def poly_to_points(polygon):
    end_pos_x = []
    end_pos_y = []
    for i in range(polygon.n()):
        x = polygon[i].x()
        y = polygon[i].y()

        end_pos_x.append(x)
        end_pos_y.append(y)

    return end_pos_x, end_pos_y
