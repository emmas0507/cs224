import numpy as np

def get_circumcenter(tri_cordinates):
    (x1, y1) = tri_cordinates[0]
    (x2, y2) = tri_cordinates[1]
    (x3, y3) = tri_cordinates[2]

    denomenator = (x3 - x1)*(y1 - y2) - (x2 - x1)*(y1 - y3)
    numerator = (x1 + x3) * (x3 - x1) * (y1 - y2) - (x1 + x2) * (x2 - x1) * (y1 - y3) + (y2 - y3) * (y1 - y2) * (y1 - y3)
    x = numerator / denomenator / 1.0 / 2.0
    y = ((2 * x - x1 - x3) * (x3 - x1) / (y1 - y3) + (y1 + y3)) / 2
    return (x, y)

tri_cordinates = [(3,2), (1,4), (5, 4)]

center = get_circumcenter(tri_cordinates)

def get_radius(center, tri_cordinates):
    return [np.sqrt((x-center[0])**2 + (y-center[1])**2) for (x, y) in tri_cordinates]

print center
radius = get_radius(center, tri_cordinates)
print radius

# def distance_to_line(point, line_x, line_y):
