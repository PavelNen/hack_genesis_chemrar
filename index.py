#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 00:12:43 2019

@author: alexey
"""

import cv2
import numpy as np
import matplotlib
import random
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import time
import poly_point_isect as bot

start_time = time.time()
def circle(a, b, c, d):
    return np.sqrt((a - c)**2 + (b - d)**2) < radius

n = 13
img = cv2.imread(f'imgs/img{n}.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 2  # minimum number of pixels making up a line
max_line_gap = 16  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on
radius = 16
# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

points = []
for line in lines:
    for x1,y1,x2,y2 in line:
        points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
        cv2.line(line_image,(x1,y1),(x2,y2), (255, 0, 0), thickness=5)

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
# cv2.imshow('lines_edges', lines_edges)
# cv2.imwrite('hough_lines.jpg', lines_edges)

# print(points)
intersections = bot.isect_segments(points)
# print(intersections)

for idx, inter in enumerate(intersections):
    a, b = inter
    match = 0
    for id_other, other_inter in enumerate(intersections[idx+1:]):
        c, d = other_inter
        if circle(a, b, c, d):
            match = 1
            intersections[idx] = ((c+a)/2, (d+b)/2)
            intersections.remove(other_inter)

    if match == 0:
        intersections.remove(inter)


def line_length(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def coordinates(intersections):
    Coordinates = {}
    for point in intersections:
        x, y = point
        yes = False
        for key in Coordinates.keys():
            curr_x, curr_y = np.mean(Coordinates[key], axis=0)
            if circle(curr_x, curr_y, x, y):
                Coordinates[key].append(point)
                yes = True
        if not yes:
            Coordinates[tuple(point)] = [point]

    for key, value in Coordinates.items():
        Coordinates[key] = np.mean(value, axis=0)

    return list(Coordinates.values())

intersections = coordinates(intersections)

for _ in range(6):
    min_inter_length = 200
    a, b = random.choice(intersections)
    for id_other, other_inter in enumerate(intersections):
        if a == c and b == d: pass

        c, d = other_inter
        current_length = line_length(a, b, c, d)
        if 1 < current_length < min_inter_length:
            min_inter_length = current_length
    print(min_inter_length)


for inter in intersections:
    a, b = inter
    for i in range(3):
        for j in range(3):
            lines_edges[int(b) + i, int(a) + j] = [0, 0, 255]

cv2.imwrite(f'result/hough_lines{n}.jpg', lines_edges)

# Время работы алгоритма
print(time.time() - start_time)
