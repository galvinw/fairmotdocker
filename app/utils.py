import json
from shapely.geometry import Point, Polygon

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Create Point objects
p1 = Point(0.2312, 4.623)
p2 = Point(1.31, 3.52)
p3 = Point(0.25, 0.25)
p4 = Point(3.71, 1.39)
p5 = Point(4.25, 4.25)

p6 = Point(2.66, 4.51)
p7 = Point(2.2, 3.4)
p8 = Point(3, 3)
p9 = Point(4.2, 3.45)
p10 = Point(2.4, 1.4)
p11 = Point(2.66, 4.71)

# Create a Polygon

position_x = [2.66, 5, 3.63, 4, 1.9, 0.72]
position_z = [4.71, 3.5, 2.52, 1.6, 1, 2.28]

coords = []
for id, x in enumerate(position_x):
    z = position_z[id]
    coords.append((x,z))

poly = Polygon(coords)

assert not p1.within(poly)
assert not p2.within(poly)
assert not p3.within(poly)
assert not p4.within(poly)
assert not p5.within(poly)
assert p6.within(poly)
assert p7.within(poly)
assert p8.within(poly)
assert p9.within(poly)
assert p10.within(poly)
assert not p11.within(poly)
print("Test Completed!")