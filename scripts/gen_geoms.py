import cv2
import os
import numpy as np

res = 512
out_dir = "../gym_miniworld/textures/geoms"

out_file_tmpl = "geom_{}.png"


def draw_circle(img, color):
    center = (int(res/2), int(res/2))
    radius = int(res/4)
    cv2.circle(img, center, radius, color, -1)


def draw_rectangle(img, color):
    center = int(res/2)
    extent = int(res/4)
    cv2.rectangle(img, (center - extent, center - extent), (center + extent, center + extent), color, -1)


def draw_triangle_right(img, color):
    center = int(res / 2)
    extent = int(res / 4)
    pt1 = (center + extent, center); pt2 = (center - extent, center - extent); pt3 = (center - extent, center + extent)
    cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, color, -1)


def draw_triangle_left(img, color):
    center = int(res / 2)
    extent = int(res / 4)
    pt1 = (center - extent, center); pt2 = (center + extent, center - extent); pt3 = (center + extent, center + extent)
    cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, color, -1)


def draw_triangle_up(img, color):
    center = int(res / 2)
    extent = int(res / 4)
    pt1 = (center, center - extent); pt2 = (center - extent, center + extent); pt3 = (center + extent, center + extent)
    cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, color, -1)


def draw_triangle_down(img, color):
    center = int(res / 2)
    extent = int(res / 4)
    pt1 = (center, center + extent); pt2 = (center - extent, center - extent); pt3 = (center + extent, center - extent)
    cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, color, -1)


draw_fcns = [draw_circle, draw_rectangle, draw_triangle_right, draw_triangle_left, draw_triangle_up, draw_triangle_down]

cs = np.asarray(np.linspace(0, 255, 5), dtype=np.uint8)
cs = cv2.applyColorMap(cs, cv2.COLORMAP_RAINBOW)[:, 0]

i = 0
for fcn in draw_fcns:
    for c in cs:
        im = np.asarray(0.4 * 255 * np.ones((res, res, 3)), dtype=np.uint8)
        fcn(im, [int(x) for x in c])
        cv2.imwrite(os.path.join(out_dir, out_file_tmpl.format(i)), im)
        i += 1



