import json
from shapely.geometry import Point, Polygon

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def create_point(x, y):
    return Point(x, y)

def create_polygon(x_arr, y_arr):
    assert len(x_arr) == len(y_arr), "x_arr and y_arr must have the same length"
    coords = []

    for id, x in enumerate(x_arr):
        z = y_arr[id]
        coords.append((x,z))

    return Polygon(coords)

def check_point_within_polygon(point_x, point_y, poly_x, poly_y):
    point = create_point(point_x, point_y)
    poly = create_polygon(poly_x, poly_y)
    return poly.contains(point)