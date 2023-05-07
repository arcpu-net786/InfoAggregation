import numpy as np
import matplotlib.path as mplPath
from math import sqrt, ceil, floor


def poly_points(vertices): #given a list of (x, y) vertex tuples, return a list of all points in the shape
    poly_path = mplPath.Path(np.array(vertices))
    pts_in_poly = []

    rightmost_x  = -float('inf')
    leftmost_x   = float('inf')
    topmost_y    = float('inf')
    bottommost_y = -float('inf')

    #find the topmost and bottommost y-coordinate, rightmost and leftmost x-coordinate
    for x, y in vertices:
        if x > rightmost_x:
            rightmost_x = x
        if x < leftmost_x:
            leftmost_x = x
        if y > bottommost_y:
            bottommost_y = y
        if y < topmost_y:
            topmost_y = y
    
    #check all points within square
    for i in range(int(floor(leftmost_x)), int(ceil(rightmost_x+1))):
        for j in range(int(floor(topmost_y)), int(ceil(bottommost_y+1))):
            point = (i, j)
            if poly_path.contains_point(point):
                pts_in_poly.append((i, j))

    return pts_in_poly


def distance(pt1, pt2): #find the Euclidian distance between two points
    x1, y1 = pt1
    x2, y2 = pt2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)