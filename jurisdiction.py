import itertools
import threading
import traceback
from enum import Enum, auto

import pyclipper
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygon, QColor

from PyQt5.QtCore import Qt, QRect, QPoint, QTimer

import sys
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn import svm

# draw
import os
from itertools import accumulate, combinations

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import csv
import time
import visilibity as vis
import math
import numpy as np
from math import cos, sin
import shapely
from shapely.geometry import LineString
import pyvisgraph as vg
import random
import shapely.geometry as sp

# Used to plot the example
import matplotlib.pylab as p

ITERATIONS = 200
DEPLOYMENT_ITERATIONS = 100
SHOW_CIRCLE = False
SHOW_VORONOI = False
JUST_POLYGON = False
DELAY = 100
EPSILON = 0.0000001
file_name = 'resources/sites_poly2.csv'
env = None
p_walls = None
asso = {}
corners = []
SPEED = 7
sem = threading.Semaphore()

random.seed(43)

if ITERATIONS == 1:
    DELAY = float('inf')


class OPERATION(Enum):
    point = auto()
    line = auto()
    dotted_line = auto()
    circle = auto()
    filled_circle = auto()
    border_circle = auto()
    polygon = auto()
    filled_polygon = auto()
    dotted_polygon = auto()
    border_polygon = auto()


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = "Simulation"

        self.action_stack = []

        self.main_stack = []

        self.InitWindow()

        timer = QTimer(self)

        # adding action to the timer
        # update the whole code
        timer.timeout.connect(self.update)

        # setting start time of timer i.e 1 second
        timer.start(1000)

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(400, 600, 700, 700)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setWindow(QRect(-500, -500, 1000, 1000))
        painter.setViewport(QRect(0, 0, 700, 700))
        painter.scale(1, -1)

        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        points = [QPoint(-500, 500), QPoint(500, 500), QPoint(500, -500), QPoint(-500, -500), QPoint(-500, 500)]
        poly = QPolygon(points)
        painter.drawPolygon(poly)
        #
        # painter.drawLine(0, 0, 500, -500)
        sem.acquire()
        my_stack = self.main_stack
        sem.release()
        for i in my_stack:
            if i[0] == OPERATION.point:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.drawPoint(i[1], i[2])
            elif i[0] == OPERATION.line:
                painter.setPen(QPen(i[6], i[5], Qt.SolidLine))
                painter.drawLine(i[1], i[2], i[3], i[4])
            elif i[0] == OPERATION.dotted_line:
                painter.setPen(QPen(i[6], i[5], Qt.DotLine))
                painter.drawLine(i[1], i[2], i[3], i[4])
            elif i[0] == OPERATION.circle:
                painter.setPen(QPen(i[5], i[4], Qt.SolidLine))
                # painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                painter.drawEllipse(QPoint(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.filled_circle:
                painter.setPen(QPen(i[5], i[4], Qt.SolidLine))
                painter.setBrush(QBrush(i[5], Qt.SolidPattern))
                painter.drawEllipse(QPoint(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.border_circle:
                painter.setPen(QPen(Qt.black, i[4], Qt.SolidLine))
                painter.setBrush(QBrush(i[5], Qt.SolidPattern))
                painter.drawEllipse(QPoint(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.polygon:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.setBrush(QBrush(Qt.NoBrush))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPoint(x_val[j], y_val[j]))
                poly = QPolygon(points)
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.dotted_polygon:
                painter.setPen(QPen(i[4], i[3], Qt.DotLine))
                painter.setBrush(QBrush(Qt.NoBrush))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPoint(x_val[j], y_val[j]))
                poly = QPolygon(points)
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.filled_polygon:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.setBrush(QBrush(i[4], Qt.SolidPattern))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPoint(x_val[j], y_val[j]))
                poly = QPolygon(points)
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.border_polygon:
                painter.setPen(QPen(Qt.black, i[3], Qt.SolidLine))
                painter.setBrush(QBrush(i[4], Qt.SolidPattern))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPoint(x_val[j], y_val[j]))
                poly = QPolygon(points)
                painter.drawPolygon(poly)

    def draw(self, value):
        self.action_stack.append(value)

    def execute(self):
        sem.acquire()
        self.main_stack.clear()
        self.main_stack = self.action_stack
        self.action_stack = []
        sem.release()
        self.update()

    def draw_eq_triangle(self, color1, point, radius, angle):
        dx = 0
        dy = radius
        deg = 2 * math.pi / 3
        theta1 = deg + angle
        theta2 = -deg + angle
        cx = dx * math.cos(angle) - dy * math.sin(angle) + point.x()
        cy = dx * math.sin(angle) + dy * math.cos(angle) + point.y()
        bx = dx * math.cos(theta1) - dy * math.sin(theta1) + point.x()
        by = dx * math.sin(theta1) + dy * math.cos(theta1) + point.y()
        ax = dx * math.cos(theta2) - dy * math.sin(theta2) + point.x()
        ay = dx * math.sin(theta2) + dy * math.cos(theta2) + point.y()
        self.draw([OPERATION.border_polygon, [ax, bx, cx], [ay, by, cy], 1, Qt.darkGreen])

    def draw_guard(self, color, point, radius, angle):
        dx = 0
        dy = radius
        deg = 2 * math.pi / 3
        theta1 = deg + angle
        theta2 = -deg + angle
        cx = dx * math.cos(angle) - dy * math.sin(angle) + point.x()
        cy = dx * math.sin(angle) + dy * math.cos(angle) + point.y()
        dy *= 0.5
        bx = dx * math.cos(theta1) - dy * math.sin(theta1) + point.x()
        by = dx * math.sin(theta1) + dy * math.cos(theta1) + point.y()
        ax = dx * math.cos(theta2) - dy * math.sin(theta2) + point.x()
        ay = dx * math.sin(theta2) + dy * math.cos(theta2) + point.y()
        self.draw([OPERATION.border_polygon, [ax, bx, cx], [ay, by, cy], 1, Qt.blue])

    def run(self):
        # polys = [[vg.Point(0.0, 1.0), vg.Point(3.0, 1.0), vg.Point(1.5, 4.0)],
        #              [vg.Point(4.0, 4.0), vg.Point(7.0, 4.0), vg.Point(5.5, 8.0)]]
        # g = vg.VisGraph()
        # g.build(polys,workers= 3, status=False)
        # shortest = g.shortest_path(vg.Point(1.5, 0.0), vg.Point(4.0, 6.0))
        # print(shortest)

        tile1 = [vis.Point(-327.0, -14.0), vis.Point(-348.0, -140.0),
                 vis.Point(-305.0, -118.0), vis.Point(-305.0, -5.0)]

        tile2 = [vis.Point(-237.0, 121.0), vis.Point(-282.0, 105.0),
                 vis.Point(-286.0, 89.0), vis.Point(-223.0, 74.0)]

        tile3 = [vis.Point(-149.0, 131.0), vis.Point(-130.0, 135.0),
                 vis.Point(-96.0, 124.0), vis.Point(-86.0, 100.0)]

        tile4 = [vis.Point(-25.0, 75.0), vis.Point(-5.0, 70.0),
                 vis.Point(-48.0, 42.0), vis.Point(-58.0, 45.0)]

        tile5 = [vis.Point(76.0, 34.0), vis.Point(78.0, -20.0),
                 vis.Point(50.0, -22.0), vis.Point(60.0, 30.0)]

        tile6 = [vis.Point(83.0, -136.0), vis.Point(83.0, -180.0),
                 vis.Point(24.0, -150.0), vis.Point(29.0, -125.0)]

        tile7 = [vis.Point(135.0, -147.0), vis.Point(183.0, -159.0),
                 vis.Point(123.0, -190.0)]

        tile8 = [vis.Point(221.0, 21.0), vis.Point(250.0, 25.0),
                 vis.Point(266.0, -9.0), vis.Point(233.0, -27.0)]

        tile9 = [vis.Point(277.0, -103.0), vis.Point(277.0, -59.0),
                 vis.Point(290.0, -65.0), vis.Point(300.0, -88.0)]

        tile = [tile1, tile2, tile3, tile4,
                tile5, tile6, tile7, tile8,
                tile9]

        guard_path1 = [vis.Point(-320.0, -140.0), vis.Point(-326.0, -130.0)]
        guard_path2 = [vis.Point(-320.0, -140.0), vis.Point(0.0, 65.0)]
        guard_path3 = [vis.Point(-320.0, -140.0), vis.Point(-237.0, 119.0)]
        guard_path4 = [vis.Point(-320.0, -140.0), vis.Point(-90.0, 90.0)]
        guard_path5 = [vis.Point(-320.0, -140.0), vis.Point(60.0, 25.0)]
        guard_path6 = [vis.Point(-320.0, -140.0), vis.Point(45.0, -134.0)]
        guard_path7 = [vis.Point(-320.0, -140.0), vis.Point(183.0, -160.0)]
        guard_path8 = [vis.Point(-320.0, -140.0), vis.Point(280.0, -100.0)]
        guard_path9 = [vis.Point(-320.0, -140.0), vis.Point(247.0, -10.0)]

        class Guard:
            def __init__(self, init_path, step_size, trigger, to_be_deployed=True):
                self.iterator = 0
                self.trigger = trigger
                self.active = False
                self.to_be_deployed = to_be_deployed
                self.path = init_path
                self.step_size = step_size
                self.point = vis.Point(0.0, 0.0)
                self.current_path = None
                self.distance = None
                self.distance_cumulative = None
                self.pair = None
                self.view_point = self.path[-1]
                self.deployed = False

            def setup(self, point):
                self.current_path = 0
                self.distance = []
                self.path[0] = point
                cv1 = 0
                while cv1 < len(self.path) - 1:
                    a1 = evader_path[cv1]
                    b1 = evader_path[cv1 + 1]
                    self.distance.append(distance(a1, b1))
                    cv1 += 1
                self.distance_cumulative = list(accumulate(self.distance))

            def setup_pair(self, pair):
                self.pair = pair

        x_val = []
        y_val = []

        evader_path = [vis.Point(-320.0, -140.0), vis.Point(-290.0, -20.0),
                       vis.Point(-167.0, 63.0), vis.Point(-200.0, 110.0),
                       vis.Point(-80.0, 110.0), vis.Point(10.0, -70.0),
                       vis.Point(83.0, -136.0), vis.Point(160.0, -180.0),
                       vis.Point(260.0, -70.0), vis.Point(250.0, 0.0)]

        deployment_checkpoints = len(evader_path)

        evader_path_post_deployment = \
            [vis.Point(280.0, -70.0), vis.Point(140.0, -180.0),
             vis.Point(-10.0, -70.0), vis.Point(-140.0, 53.0),
             vis.Point(-10.0, -90.0), vis.Point(140.0, -180.0),
             vis.Point(280.0, -70.0)]

        evader_path.extend(evader_path_post_deployment)

        current_path = 0
        runner = vis.Point(evader_path[current_path].x(), evader_path[current_path].y())

        current_path_guard = 0
        guard = vis.Point(evader_path[current_path].x(), evader_path[current_path].y())
        last_pos = vis.Point(evader_path[current_path].x(), evader_path[current_path].y())
        heading = vis.Point(evader_path[current_path].x(), evader_path[current_path].y())

        current_path_guard = 0
        camera = vis.Point(-320.0, -160.0)

        evader_distance = []

        cv = 0
        while cv < len(evader_path) - 1:
            a = evader_path[cv]
            b = evader_path[cv + 1]
            evader_distance.append(distance(a, b))
            cv += 1

        evader_distance_cumulative = list(accumulate(evader_distance))
        guard_count = 0
        show_vis = False

        step_size = evader_distance_cumulative[deployment_checkpoints - 1] / DEPLOYMENT_ITERATIONS

        x = []
        y = []

        alpha = 0.6  # ve/vp

        guard_step_size = step_size * 1 / alpha
        guards = [Guard(guard_path1, guard_step_size, 0),
                  Guard(guard_path2, guard_step_size, 10),
                  Guard(guard_path3, guard_step_size, 27),
                  Guard(guard_path4, guard_step_size, 6, False),
                  Guard(guard_path5, guard_step_size, 13, False),
                  Guard(guard_path6, guard_step_size, 36),
                  Guard(guard_path7, guard_step_size, 75),
                  Guard(guard_path8, guard_step_size, 76, False),
                  Guard(guard_path9, guard_step_size, 92)]

        edge_guards = [(guards[1], guards[4]),
                       (guards[6], guards[7]),
                       (guards[0], guards[3])]

        inactive_guards = 0
        for g in guards:
            if not g.to_be_deployed:
                inactive_guards += 1

        for i in edge_guards:
            i[0].setup_pair(i[1])
            i[1].setup_pair(i[0])

        active_guards = len(guards) - inactive_guards

        d = 39.204592

        d_max = d * alpha

        poly = []

        with open(file_name) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                x_val = float(row[0])
                y_val = float(row[1])
                x.append(x_val)
                y.append(y_val)
                poly.append(vis.Point(x_val, y_val))

        observer = vis.Point(-64, -5)
        pursuer2 = vis.Point(25, -14)
        # pursuers = [vis.Point(25, -14), vis.Point(30, -25), vis.Point(50, 50)]
        evader = vis.Point(-110, 10)

        new_evader = None

        walls = vis.Polygon([ele for ele in reversed(poly)])
        global p_walls
        p_walls = walls
        global env
        env = vis.Environment([walls])

        env.PRINTING_DEBUG_DATA = False

        observer.snap_to_boundary_of(env, EPSILON)
        observer.snap_to_vertices_of(env, EPSILON)

        global corners

        # check corners
        for i in range(len(poly)):
            a = poly[(i - 1) % len(poly)]
            b = poly[i]
            c = poly[(i + 1) % len(poly)]

            s = [b.x() - a.x(), b.y() - a.y()]
            t = [c.x() - b.x(), c.y() - b.y()]
            if s[0] * t[1] - t[0] * s[1] > 0:
                corners.append(i)

        self.draw([OPERATION.polygon, x, y, 3, Qt.white])

        for b in corners:
            self.draw([OPERATION.filled_circle, poly[b].x(), poly[b].y(), 6, 1, Qt.blue])

        self.execute()
        time.sleep(2)

        global asso
        asso = {}
        for j in corners:
            vispol = vis.Visibility_Polygon(poly[j], env, EPSILON)

            a = poly[(j - 1) % len(poly)]
            b = poly[j]
            c = poly[(j + 1) % len(poly)]

            evader_x, evader_y = poly_to_points(vispol)
            evader_x.reverse()
            evader_y.reverse()

            vector_1 = [b.x() - a.x(), b.y() - a.y()]
            vector_2 = [b.x() - c.x(), b.y() - c.y()]

            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

            unit_vector_1 *= 10000
            unit_vector_2 *= 10000

            x1 = [b.x(), unit_vector_1[0] + b.x(), unit_vector_2[0] + b.x()]
            y1 = [b.y(), unit_vector_1[1] + b.y(), unit_vector_2[1] + b.y()]

            subj = []
            for k in range(len(evader_x)):
                subj.append((evader_x[k], evader_y[k]))
            clip = []
            for k in range(len(x1)):
                clip.append((x1[k], y1[k]))

            pc = pyclipper.Pyclipper()
            pc.AddPath(clip, pyclipper.PT_CLIP, True)
            pc.AddPath(subj, pyclipper.PT_SUBJECT, True)

            solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

            asso[j] = solution[0]

        print(corners)

        # b = vis.Point(0.0, 0.0)
        # rad = 10
        # samples = 40
        #
        # pol = jurisdiction(poly, (b, rad, samples, self))
        # if pol is not None:
        #     x1 = []
        #     y1 = []
        #     for k in range(len(pol)):
        #         x1.append(pol[k][0])
        #         y1.append(pol[k][1])
        #     self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.gray])
        #
        # self.draw([OPERATION.polygon, x, y, 3, Qt.red])
        #
        # self.draw([OPERATION.border_circle, b.x(), b.y(), 1, 2 * rad, Qt.darkRed])
        #
        # self.draw([OPERATION.filled_circle, b.x(), b.y(), 6, 1, Qt.blue])
        #
        # self.execute()
        # time.sleep(10)

        history = []
        boundary = [*range(len(poly))]
        current_corners = corners.copy()
        corner_history = []
        corner_clusters = {k: [k] for k in corners}

        vertex_clusters = {k: {k} for k in corners}

        while len(current_corners) > 0:
            history.append(boundary)
            corner_history.append(current_corners)
            random.shuffle(current_corners)

            if len(current_corners) == 1:
                last_corner = current_corners[0]
                for ii in boundary:
                    vertex_clusters[last_corner].add(ii)
            mod_id = None
            mod_k = None
            for (j, k) in list(itertools.product(current_corners, [-1, 1])):
                d = boundary.index(j)
                m = (d + 2 * k) % len(boundary)
                m1 = (d + k) % len(boundary)
                shortest_path = env.shortest_path(poly[j], poly[boundary[m]], EPSILON)
                shortest_path = shortest_path.path()
                if len(shortest_path) == 2:
                    corner_clusters[j].append(boundary[m])
                    corner_clusters[j].append(boundary[m1])
                    vertex_clusters[j].add(boundary[m1])
                    mod_id = j
                    mod_k = (d + k) % len(boundary)
                    break

            boundary = boundary[:]
            boundary.remove(boundary[mod_k])

            removal = []
            for corner in current_corners:
                d = boundary.index(corner)

                a = poly[boundary[(d - 1) % len(boundary)]]
                b = poly[boundary[d]]
                c = poly[boundary[(d + 1) % len(boundary)]]

                s = [b.x() - a.x(), b.y() - a.y()]
                t = [c.x() - b.x(), c.y() - b.y()]
                if s[0] * t[1] - t[0] * s[1] > 0:
                    removal.append(corner)

            current_corners = removal

            x1 = []
            y1 = []
            for k in boundary:
                x1.append(poly[k].x())
                y1.append(poly[k].y())

            self.draw([OPERATION.polygon, x1, y1, 3, Qt.darkGreen])

            self.draw([OPERATION.polygon, x, y, 3, Qt.white])

            for b in current_corners:
                self.draw([OPERATION.filled_circle, poly[b].x(), poly[b].y(), 6, 1, Qt.blue])

            self.execute()
            time.sleep(0.2)

        for key, value in vertex_clusters.items():
            for i in value:
                vispol = vis.Visibility_Polygon(poly[i], env, EPSILON)  # of each vertex in cluster
                point_x, point_y = poly_to_points(vispol)
                point_x.reverse()
                point_y.reverse()

                clip = []
                for k in range(len(point_x)):
                    clip.append((point_x[k], point_y[k]))
                pc = pyclipper.Pyclipper()
                pc.AddPath(clip, pyclipper.PT_CLIP, True)
                pc.AddPath(asso[key], pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                pc.Clear()
                if len(solution) > 0:
                    asso[key] = solution[0]
                else:
                    asso[key] = None
            # if corner_history[-1][0] == key: #intersect current boundary
            #     pc = pyclipper.Pyclipper()
            #     clip = []
            #     for k in history[-1]:
            #         clip.append((poly[k].x(), poly[k].y()))
            #     pc.AddPath(clip, pyclipper.PT_CLIP, True)
            #     pc.AddPath(asso[key], pyclipper.PT_SUBJECT, True)
            #     solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            #     pc.Clear()
            #     if len(solution) > 0:
            #         asso[key] = solution[0]
            #     else:
            #         asso[key] = None

        colors = sns.color_palette("husl", len(corners))

        for j, n in zip(corners, [*range(len(corners))]):
            x1 = []
            y1 = []
            visi_pol = asso[j]
            for k in range(len(visi_pol)):
                x1.append(visi_pol[k][0])
                y1.append(visi_pol[k][1])

            col = QColor()
            col.setRgbF(colors[n][0], colors[n][1], colors[n][2], 0.3)

            self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

            self.draw([OPERATION.polygon, x, y, 3, Qt.white])

            self.draw([OPERATION.filled_circle, poly[j].x(), poly[j].y(), 6, 1, Qt.blue])

            # pol = jurisdiction(poly, asso[j])
            #
            # if pol is not None:
            #     x1 = []
            #     y1 = []
            #     for k in range(len(pol)):
            #         x1.append(pol[k][0])
            #         y1.append(pol[k][1])
            #     self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.gray])

            self.execute()
            time.sleep(0.2)

        for j, n in zip(corners, [*range(len(corners))]):
            x1 = []
            y1 = []
            visi_pol = asso[j]
            for k in range(len(visi_pol)):
                x1.append(visi_pol[k][0])
                y1.append(visi_pol[k][1])

            col = QColor()
            col.setRgbF(colors[n][0], colors[n][1], colors[n][2], 0.3)

            self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

        self.draw([OPERATION.polygon, x, y, 3, Qt.white])

        self.execute()
        time.sleep(0.2)
        print(corner_clusters)

        corner_set = corners
        best_set = set()
        L = self.apriori(corner_set)

        while len(corner_set) > 0:
            L = self.apriori(corner_set)
            xx = random.choice(tuple(L[len(L)]))
            corner_set = [i for i in corner_set if i not in list(xx)]
            best_set.add(xx)
        f_set = [list(xx) for xx in best_set]
        print(f_set)

        colors = sns.color_palette("husl", len(f_set))

        tile_store = []

        for ele, n in zip(f_set, [*range(len(f_set))]):
            pol = asso[ele[0]]
            for i in ele[1:]:
                pc = pyclipper.Pyclipper()
                pc.AddPath(asso[i], pyclipper.PT_CLIP, True)
                pc.AddPath(pol, pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                pc.Clear()
                if len(solution) > 0:
                    pol = solution[0]
                else:
                    pol = None

            tile_store.append(pol)

            x1 = []
            y1 = []
            for k in range(len(pol)):
                x1.append(pol[k][0])
                y1.append(pol[k][1])

            col = QColor()
            col.setRgbF(colors[n][0], colors[n][1], colors[n][2], 0.3)

            self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.darkCyan])

        self.draw([OPERATION.polygon, x, y, 3, Qt.white])

        self.execute()
        time.sleep(1)

        for (tile_i, d) in zip(f_set, [*range(len(f_set))]):
            col = QColor()
            col.setRgbF(colors[d][0], colors[d][1], colors[d][2], 0.3)

            j = tile_store[d]
            x1 = []
            y1 = []
            for k in range(len(j)):
                x1.append(j[k][0])
                y1.append(j[k][1])

            pol = jurisdiction(poly, j)

            if pol is not None:
                x2 = []
                y2 = []
                for k in range(len(pol)):
                    x2.append(pol[k][0])
                    y2.append(pol[k][1])
                self.draw([OPERATION.filled_polygon, x2, y2, 3, Qt.gray])

            self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.darkCyan])
            self.draw([OPERATION.polygon, x, y, 3, Qt.white])

            for ct in tile_i:
                self.draw([OPERATION.filled_circle, poly[ct].x(), poly[ct].y(), 6, 1, Qt.blue])

            self.execute()
            time.sleep(0.6)

        colors = sns.color_palette("husl", len(corners))

        for j, n in zip(corners, [*range(len(corners))]):
            cluster = corner_clusters[j]
            sets = cluster[1:]

            corner_point = cluster[0]
            col = QColor()
            col.setRgbF(colors[n][0], colors[n][1], colors[n][2], 0.3)
            for w in range(0, len(sets), 2):
                a = poly[corner_point]
                b = poly[sets[w]]
                c = poly[sets[w + 1]]
                x1 = [a.x(), b.x(), c.x()]
                y1 = [a.y(), b.y(), c.y()]
                self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

            if corner_history[-1][0] == j:
                x1 = []
                y1 = []
                for i in history[-1]:
                    x1.append(poly[i].x())
                    y1.append(poly[i].y())
                self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

        self.draw([OPERATION.polygon, x, y, 3, Qt.white])

        self.execute()
        time.sleep(2)

        # for j, n in zip(corners, [*range(len(corners))]):
        #     cluster = corner_clusters[j]
        #     sets = cluster[1:]
        #
        #     corner_point = cluster[0]
        #     col = QColor()
        #     col.setRgbF(colors[n][0], colors[n][1], colors[n][2], 0.3)
        #     for w in range(0, len(sets), 2):
        #         a = poly[corner_point]
        #         b = poly[sets[w]]
        #         c = poly[sets[w + 1]]
        #         x1 = [a.x(), b.x(), c.x()]
        #         y1 = [a.y(), b.y(), c.y()]
        #         self.draw([OPERATION.filled_polygon, x1, y1, 3, col])
        #
        #     if corner_history[-1][0] == j:
        #         x1 = []
        #         y1 = []
        #         for i in history[-1]:
        #             x1.append(poly[i].x())
        #             y1.append(poly[i].y())
        #         self.draw([OPERATION.filled_polygon, x1, y1, 3, col])
        #
        #     self.draw([OPERATION.polygon, x, y, 3, Qt.red])
        #
        #     self.execute()
        #     time.sleep(0.6)

        for (k, d) in zip(f_set, [*range(len(f_set))]):
            for j, n in zip(k, [*range(len(k))]):
                cluster = corner_clusters[j]
                sets = cluster[1:]

                corner_point = cluster[0]
                col = QColor()
                col.setRgbF(colors[n][0], colors[n][1], colors[n][2], 0.3)
                for w in range(0, len(sets), 2):
                    a = poly[corner_point]
                    b = poly[sets[w]]
                    c = poly[sets[w + 1]]
                    x1 = [a.x(), b.x(), c.x()]
                    y1 = [a.y(), b.y(), c.y()]
                    self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

                if corner_history[-1][0] == j:
                    x1 = []
                    y1 = []
                    for i in history[-1]:
                        x1.append(poly[i].x())
                        y1.append(poly[i].y())
                    self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

            x1 = []
            y1 = []
            j = tile_store[d]
            for k in range(len(j)):
                x1.append(j[k][0])
                y1.append(j[k][1])
            self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.darkCyan])
            self.draw([OPERATION.polygon, x, y, 3, Qt.white])

            self.execute()
            time.sleep(0.6)

        colors = sns.color_palette("husl", len(f_set))

        for (k, d) in zip(f_set, [*range(len(f_set))]):
            col = QColor()
            col.setRgbF(colors[d][0], colors[d][1], colors[d][2], 0.3)

            m = tile_store[d]
            pts = m[:]

            for j, n in zip(k, [*range(len(k))]):
                cluster = corner_clusters[j]
                sets = cluster[1:]

                corner_point = cluster[0]
                for w in range(0, len(sets), 2):
                    pts_local = pts[:]
                    a = poly[corner_point]
                    b = poly[sets[w]]
                    c = poly[sets[w + 1]]

                    pts_local.append([a.x(), a.y()])
                    pts_local.append([b.x(), b.y()])
                    pts_local.append([c.x(), c.y()])
                    np_pts = np.array(pts_local)
                    hull = ConvexHull(np_pts)

                    x1 = []
                    y1 = []
                    for bt_pts in hull.vertices:
                        x1.append(np_pts[bt_pts, 0])
                        y1.append(np_pts[bt_pts, 1])
                    self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

                if corner_history[-1][0] == j:
                    pts_local = pts[:]
                    for i in history[-1]:
                        pts_local.append([poly[i].x(), poly[i].y()])

                    np_pts = np.array(pts_local)
                    hull = ConvexHull(np_pts)

                    x1 = []
                    y1 = []
                    for bt_pts in hull.vertices:
                        x1.append(np_pts[bt_pts, 0])
                        y1.append(np_pts[bt_pts, 1])
                    self.draw([OPERATION.filled_polygon, x1, y1, 3, col])

            x1 = []
            y1 = []
            j = tile_store[d]
            for k in range(len(j)):
                x1.append(j[k][0])
                y1.append(j[k][1])
            self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.darkMagenta])
            self.draw([OPERATION.polygon, x, y, 3, Qt.white])

            self.execute()
            time.sleep(0.6)

        # make cluster
        corners_pts = []
        corners_labels = []
        corner_map = {}
        for (k, d) in zip(f_set, [*range(len(f_set))]):
            for ind in k:
                corners_pts.append([poly[ind].x(), poly[ind].y()])
                corners_labels.append(d)
                corner_map[ind] = d

        clf = svm.SVC(kernel='linear', random_state=32)
        clf.fit(corners_pts, corners_labels)

        for id in corners:
            if clf.predict([[poly[id].x(), poly[id].y()]]) != corner_map[id]:
                print(id)

        walls = vis.Polygon([ele for ele in reversed(poly)])

        poly_walls = [walls]

        polygons = []

        for tl in tile_store:
            polygon = []
            for point in tl:
                pt = vis.Point(point[0], point[1])
                polygon.append(pt)
            polygons.append(polygon)

        for j in polygons:
            poly_walls.append(vis.Polygon([ele for ele in reversed(j)]))

        env1 = vis.Environment(poly_walls)

        env1.PRINTING_DEBUG_DATA = False

        vis_graph = vis.Visibility_Graph(env1, EPSILON)

        ll = walls.n()
        tile_limits = []

        for tl in tile_store:
            lims = [ll, len(tl)]
            tile_limits.append(lims)
            ll += len(tl)

        def get_tile(x):
            for ct in range(len(tile_limits)):
                if x >= tile_limits[ct][0] and x < tile_limits[ct][0] + tile_limits[ct][1]:
                    return ct
            return -1

        record = np.zeros((len(tile_limits), len(tile_limits)))

        edges = set()
        for j in range(walls.n(), vis_graph.n()):
            for k in range(walls.n(), vis_graph.n()):
                a = env1(j)
                b = env1(k)
                if bool(vis_graph(j, k)):
                    c = get_tile(j)
                    d = get_tile(k)

                    if c != d:
                        if j != k:
                            if record[c][d] == 0 and record[d][c] == 0:
                                self.draw([OPERATION.line, a.x(), a.y(), b.x(), b.y(), 2, Qt.darkBlue])
                                edges.add(frozenset([j, k]))
                                record[c][d] = 1
                                record[d][c] = 1

        for tl in tile_store:
            x1 = []
            y1 = []
            for k in range(len(tl)):
                x1.append(tl[k][0])
                y1.append(tl[k][1])

            self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.darkCyan])

        print(edges)
        self.draw([OPERATION.polygon, x, y, 3, Qt.white])

        self.execute()

        # first choice made
        edge = random.choice(tuple(edges))
        tup_edge = tuple(edge)
        a = env1(tup_edge[0])
        b = env1(tup_edge[1])

        cols = [Qt.blue, Qt.darkGreen]

        # for ratio in np.linspace(0, 1, 20):
        indi = 0
        ratio = 0.5

        self.draw([OPERATION.polygon, x, y, 3, Qt.white])
        self.draw([OPERATION.line, a.x(), a.y(), b.x(), b.y(), 2, Qt.darkBlue])

        for t in tup_edge:
            x1 = []
            y1 = []
            tl = tile_store[get_tile(t)]
            sets_tile = f_set[get_tile(t)]
            for k in range(len(tl)):
                x1.append(tl[k][0])
                y1.append(tl[k][1])

            self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.darkCyan])
            for ii in sets_tile:
                self.draw([OPERATION.filled_circle, poly[ii].x(), poly[ii].y(), 6, 1, cols[indi]])
            indi += 1

        vec = np.array([a.x() - b.x(), a.y() - b.y()])
        dist = np.linalg.norm(vec)
        val = dist * ratio
        vec = vec / np.linalg.norm(vec)
        unit_vector_p = vec
        vec = val * vec
        end = vis.Point(vec[0] + b.x(), vec[1] + b.y())

        self.draw([OPERATION.filled_circle, end.x(), end.y(), 6, 1, Qt.darkCyan])

        self.execute()
        time.sleep(0.5)

        vispol = vis.Visibility_Polygon(end, env, EPSILON)
        point_x, point_y = poly_to_points(vispol)
        point_x.reverse()
        point_y.reverse()

        temp_pol = []

        for i in range(vispol.n()):
            temp_pol.append((vispol[i].x(), vispol[i].y()))
        temp_pol.append((vispol[0].x(), vispol[0].y()))
        sp_poly = sp.Polygon(temp_pol)
        point_in_poly = get_random_point_in_polygon(sp_poly)

        for gg in range(15):
            point_in_poly = get_random_point_in_polygon(sp_poly)

        ev_pt = vis.Point(point_in_poly.x, point_in_poly.y)

        min_alpha = float('inf')

        ev_vispol = vis.Visibility_Polygon(ev_pt, env, EPSILON)

        dis_set = [[], []]

        indi = 0

        for t in tup_edge:
            tl = tile_store[get_tile(t)]
            sets_tile = f_set[get_tile(t)]
            dp = dist

            for ii in sets_tile:
                if poly[ii]._in(ev_vispol, EPSILON):  # if visible
                    de = distance(poly[ii], ev_pt)
                    dis_set[indi].append(de)  # de
                    min_alpha = min(min_alpha, de / dp)
            indi += 1

        print(min_alpha)

        # generate random motion
        delt = 0.1
        vel_e = 30
        pr_ratio = 0.5

        for i in range(100):
            flag = False
            test_pt = None
            while not flag:
                vec = randomvector(2)
                test_pt = vis.Point(ev_pt.x() + vec[0] * vel_e * delt, ev_pt.y() + vec[1] * vel_e * delt)
                if test_pt._in(walls, EPSILON):
                    flag = True
            # point found
            ev_pt.set_x(test_pt.x())
            ev_pt.set_y(test_pt.y())

            print(f'Iteration {i + 1} :: ({ev_pt.x()},{ev_pt.y()})')

            ev_vispol = vis.Visibility_Polygon(ev_pt, env, EPSILON)
            ab = dist
            min_corner = -1
            min_distance = 999999999
            de_min = 0

            for t in range(len(tup_edge)):
                sets_tile = f_set[get_tile(tup_edge[t])]
                da = ab * ratio
                db = ab * (1 - ratio)

                for ii in sets_tile:
                    if poly[ii]._in(ev_vispol, EPSILON):  # if visible
                        de = distance(poly[ii], ev_pt)
                        dis = 0
                        if t == 0:
                            dis = de - min_alpha * da
                        else:
                            dis = de - min_alpha * db

                        if dis < min_distance:
                            min_corner = t
                            min_distance = dis
                            de_min = de

            delta_ratio = (1 / min_alpha) * vel_e * delt / ab
            if min_corner == 1:
                delta_ratio *= -1
            ratio += delta_ratio
            if ratio <= 0.0:
                ratio = 0.0
            elif ratio >= 1.0:
                ratio = 1.0
            self.draw([OPERATION.polygon, x, y, 3, Qt.white])
            self.draw([OPERATION.line, a.x(), a.y(), b.x(), b.y(), 2, Qt.darkBlue])

            for t in range(len(tup_edge)):
                x1 = []
                y1 = []
                tl = tile_store[get_tile(tup_edge[t])]
                sets_tile = f_set[get_tile(tup_edge[t])]
                for k in range(len(tl)):
                    x1.append(tl[k][0])
                    y1.append(tl[k][1])

                self.draw([OPERATION.filled_polygon, x1, y1, 3, Qt.darkCyan])
                for ii in sets_tile:
                    self.draw([OPERATION.filled_circle, poly[ii].x(), poly[ii].y(), 6, 1, cols[t]])

            val = ab * ratio
            vec = val * unit_vector_p
            end = vis.Point(vec[0] + b.x(), vec[1] + b.y())

            self.draw([OPERATION.filled_circle, end.x(), end.y(), 7, 1, Qt.yellow])

            self.draw([OPERATION.filled_circle, ev_pt.x(), ev_pt.y(), 6, 1, Qt.darkRed])

            self.execute()
            time.sleep(0.5)

    def apriori(self, corner_set):
        C1ItemSet = set([frozenset([i]) for i in corner_set])
        # Final result, global frequent itemset
        globalFreqItemSet = dict()
        # Storing global itemset with support count

        L1ItemSet = C1ItemSet
        currentLSet = L1ItemSet
        k = 2

        # Calculating frequent item set
        while (currentLSet):
            # Storing frequent itemset
            globalFreqItemSet[k - 1] = currentLSet
            # Self-joining Lk
            candidateSet = self.getUnion(currentLSet, k)
            # Perform subset testing and remove pruned supersets
            candidateSet = self.pruning(candidateSet, currentLSet, k - 1)
            # Scanning itemSet for counting support
            currentLSet = self.checkEnvironment(candidateSet)
            k += 1

        return globalFreqItemSet

    def pruning(self, candidateSet, prevFreqSet, length):
        tempCandidateSet = candidateSet.copy()
        for item in candidateSet:
            subsets = combinations(item, length)
            for subset in subsets:
                # if the subset is not in previous K-frequent get, then remove the set
                if frozenset(subset) not in prevFreqSet:
                    tempCandidateSet.remove(item)
                    break
        return tempCandidateSet

    def getUnion(self, itemSet, length):
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

    def checkEnvironment(self, itemSet):
        freqItemSet = set()
        for item in itemSet:
            if self.checkIntersection(list(item)) is not None:
                freqItemSet.add(item)

        return freqItemSet

    def checkIntersection(self, values):
        if len(values) == 0:
            return None
        elif len(values) == 1:
            return asso[values[0]]
        else:
            pc = pyclipper.Pyclipper()
            polygon = asso[values[0]]
            for i in range(1, len(values)):
                pc.AddPath(polygon, pyclipper.PT_CLIP, True)
                pc.AddPath(asso[values[i]], pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                pc.Clear()
                if len(solution) > 0:
                    polygon = solution[0]
                else:
                    return None
                if len(polygon) == 0:
                    return None
            return polygon


def point_to_xy_vectors(points):
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    return x, y


def points_collinear(p1: vis.Point, p2: vis.Point, p3: vis.Point):
    a = -1 * (p1.x() * (p2.y() - p3.y()) + p2.x() * (p3.y() - p1.y()) + p3.x() * (p1.y() - p2.y()))
    if a < 0.0001:
        return True
    else:
        return False


def get_shapely_point(point: vis.Point):
    return shapely.geometry.Point(point.x(), point.y())


def distance(p1, p2):
    return math.sqrt((p1.x() - p2.x()) ** 2 +
                     (p1.y() - p2.y()) ** 2)


def minimum_distance_with_closest_point(pt_a: vis.Point, pt_b: vis.Point, pt_e: vis.Point):
    A = pt_e.x() - pt_a.x()
    B = pt_e.y() - pt_a.y()
    C = pt_b.x() - pt_a.x()
    D = pt_b.y() - pt_a.y()

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1

    if len_sq != 0:
        # in case of 0 length line
        param = dot / len_sq

    pt = None

    if param < 0:
        pt = vis.Point(pt_a.x(), pt_a.y())
    elif param > 1:
        pt = vis.Point(pt_b.x(), pt_b.y())
    else:
        pt = vis.Point(pt_a.x() + param * C,
                       pt_a.y() + param * D)

    dx = pt_e.x() - pt.x()
    dy = pt_e.y() - pt.y()

    return math.sqrt(dx * dx + dy * dy), pt


def jurisdiction(poly, obj):
    pc = pyclipper.Pyclipper()
    initpoints = []
    for k in range(len(poly)):
        initpoints.append((poly[k].x(), poly[k].y()))
    polygon = initpoints

    if type(obj) is tuple:  # circle
        center = obj[0]
        radius = obj[1]
        samples = obj[2]
        ob = obj[3]
        # u = np.random.uniform(0, 1, size=samples)
        u = np.linspace(0., 1., num=samples)

        x = radius * np.cos(2 * np.pi * u)
        y = radius * np.sin(2 * np.pi * u)

        for i in range(1, samples):
            # walls = vis.Polygon([ele for ele in reversed(poly)])
            # global env
            # env = vis.Environment([walls])
            # env.PRINTING_DEBUG_DATA = False
            pt_x, pt_y = center.x() + x[i], center.y() + y[i]
            pt = vis.Point(pt_x, pt_y)
            if pt._in(p_walls, EPSILON):
                ob.draw([OPERATION.filled_circle, pt.x(), pt.y(), 6, 1, Qt.blue])
                vispol = vis.Visibility_Polygon(pt, env, EPSILON)
                point_x, point_y = poly_to_points(vispol)
                point_x.reverse()
                point_y.reverse()

                clip = []
                for k in range(len(point_x)):
                    clip.append((point_x[k], point_y[k]))
                pc.AddPath(clip, pyclipper.PT_CLIP, True)
                pc.AddPath(polygon, pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
                pc.Clear()
                if len(solution) > 0:
                    polygon = solution[0]
                else:
                    return None
                if len(polygon) == 0:
                    return None

    else:
        if len(obj) == 0:
            return None
        elif len(obj) == 1:
            return obj
        for i in range(1, len(obj)):  # poly
            # walls = vis.Polygon([ele for ele in reversed(poly)])
            # global env
            # env = vis.Environment([walls])
            # env.PRINTING_DEBUG_DATA = False
            vispol = vis.Visibility_Polygon(vis.Point(obj[i][0], obj[i][1]), env, EPSILON)
            point_x, point_y = poly_to_points(vispol)
            point_x.reverse()
            point_y.reverse()

            clip = []
            for k in range(len(point_x)):
                clip.append((point_x[k], point_y[k]))
            pc.AddPath(clip, pyclipper.PT_CLIP, True)
            pc.AddPath(polygon, pyclipper.PT_SUBJECT, True)
            solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            pc.Clear()
            if len(solution) > 0:
                polygon = solution[0]
            else:
                return None
            if len(polygon) == 0:
                return None
    return polygon


def minimum_distance(pt_a: vis.Point, pt_b: vis.Point, pt_e: vis.Point):
    # vector AB
    AB = np.array([pt_b.x() - pt_a.x(), pt_b.y() - pt_a.y()])

    # vector BP
    BE = np.array([pt_e.x() - pt_b.x(), pt_e.y() - pt_b.y()])

    # vector AP
    AE = np.array([pt_e.x() - pt_a.x(), pt_e.y() - pt_a.y()])

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # Minimum distance from
    # point E to the line segment
    reqAns = 0

    # Case 1
    if AB_BE > 0:

        # Finding the magnitude
        y = pt_e.y() - pt_b.y()
        x = pt_e.x() - pt_b.x()
        req_ans = math.sqrt(x * x + y * y)

    # Case 2
    elif AB_AE < 0:
        y = pt_e.y() - pt_a.y()
        x = pt_e.x() - pt_a.x()
        req_ans = math.sqrt(x * x + y * y)

    # Case 3
    else:

        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = math.sqrt(x1 * x1 + y1 * y1)
        req_ans = abs(x1 * y2 - y1 * x2) / mod

    return req_ans


def poly_to_points_vis(polygon):
    end_pos_x = []
    end_pos_y = []
    for i in range(len(polygon)):
        x = polygon[i].x()
        y = polygon[i].y()

        end_pos_x.append(x)
        end_pos_y.append(y)

    return end_pos_x, end_pos_y


def poly_to_points(polygon):
    end_pos_x = []
    end_pos_y = []
    for i in range(polygon.n()):
        x = polygon[i].x()
        y = polygon[i].y()

        end_pos_x.append(x)
        end_pos_y.append(y)

    return end_pos_x, end_pos_y


def get_random_point_in_polygon(pol):
    minx, miny, maxx, maxy = pol.bounds
    while True:
        p = sp.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if pol.contains(p):
            return p


def randomvector(n):
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x * x for x in components))
    v = [x / r for x in components]
    return v


def get_color(v):
    x = v % 8
    if x == 0:
        return Qt.darkGreen
    elif x == 1:
        return QColor('#52489c')
    elif x == 2:
        return QColor('#43aa8b')
    elif x == 3:
        return Qt.darkCyan
    elif x == 4:
        return QColor('#d1ffc6')
    elif x == 5:
        return Qt.darkRed
    elif x == 6:
        return QColor('#59c3c3')
    elif x == 7:
        return QColor('#c97d60')
    elif x == 8:
        return QColor('#edc7cf')
    elif x == 9:
        return QColor('#52489c')
    else:
        return Qt.darkGray


def startup():
    App = QApplication(sys.argv)
    window = Window()
    # window.run()
    x = threading.Thread(target=window.run, args=())
    x.start()
    sys.exit(App.exec())
