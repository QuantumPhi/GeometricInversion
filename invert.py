import abc
import matplotlib
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np

def distance(point, other):
    dx = point[0] - other[0]
    dy = point[1] - other[1]
    return np.hypot(dx, dy)

def invert(point, shape):
    circle = shape.get_circle(point)
    t = np.arctan2(point[1] - circle.C[1], point[0] - circle.C[0])
    dr = distance(point, circle.C)
    r = circle.R**2 / dr
    return (circle.C[0] + r * np.cos(t), circle.C[1] + r * np.sin(t))

class Shape(metaclass=abc.ABCMeta):
    def __init__(self, centroid, interp, res, color):
        self.C = centroid
        self.color = color
        if(interp):
            self.points = self.__interpolate__(res)
        else:
            self.points = self.vertices

    def __get_centroid__(self, points):
        area = sum([points[i][0] * points[(i+1) % len(points)][1] - points[(i+1) % len(points)][0] * points[i][1] for i in range(len(points))]) / 2

        c_x = sum([(points[i][0] + points[(i+1) % len(points)][0]) * (points[i][0] * points[(i+1) % len(points)][1] - points[(i+1) % len(points)][0] * points[i][1]) for i in range(len(points))]) / (6 * area)
        c_y = sum([(points[i][1] + points[(i+1) % len(points)][1]) * (points[i][0] * points[(i+1) % len(points)][1] - points[(i+1) % len(points)][0] * points[i][1]) for i in range(len(points))]) / (6 * area)

        return (c_x, c_y)

    @abc.abstractmethod
    def __interpolate__(self, res):
        return

    def __repr__(self):
        return "Shape - Center: %s" % (self.C)

    @abc.abstractmethod
    def get_circle(self, point):
        return

    def invert(self, shape, color=None):
        if(color is None):
            color = self.color
        closed = shape is not Line
        points = [invert(point, shape) for point in self.points]
        return Polygon(points, closed=closed, interp=False, color=color)

    def draw(self):
        plt.plot(self.C[0], self.C[1], marker='o', color=self.color, linestyle=' ')
        plt.gcf().gca().add_patch(self.__plot__)

class Line(Shape):
    def __init__(self, point, slope, res=128, color='k'):
        self.slope = slope
        super(Line, self).__init__(point, True, res, color)
        self.__plot__ = patch.Polygon(self.points, closed=False, color=color)

    def __interpolate__(self, res):
        xlim = plt.xlim()

        xvals = np.linspace(xlim[0], xlim[1], res)
        yvals = [self.slope * (x - self.C[0]) + self.C[1] for x in xvals]

        return list(zip(xvals, yvals))

    def get_circle(self, point):
        return None

class Circle(Shape):
    def __init__(self, point, radius, res=128, color='k'):
        self.R = radius
        super(Circle, self).__init__(point, True, res, color)
        self.__plot__ = patch.Circle(point, radius, fill=False, color=color)

    def __interpolate__(self, res):
        return [(self.C[0] + self.R * np.cos(t * 2 * np.pi / res), self.C[1] + self.R * np.sin(t * 2 * np.pi / res)) for t in range(res+1)]

    def __repr__(self):
        return "Circle - Center: %s, Radius: %s" % (self.C, self.R)

    def get_circle(self, point):
        return self

class Polygon(Shape):
    def __init__(self, vertices, closed=True, interp=True, res=128, color='k'):
        self.vertices = vertices
        super(Polygon, self).__init__(self.__get_centroid__(vertices), interp, res, color)
        self.__plot__ = patch.Polygon(vertices, closed=closed, fill=False, color=color)

    def __interpolate__(self, res):
        points = []
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i+1) % len(self.vertices)]

            x_equals = v1[0] - v2[0] == 0

            if(x_equals):
                v1 = tuple(reversed(v1))
                v2 = tuple(reversed(v2))

            xvals = np.linspace(v1[0], v2[0], res)

            if(v1[0] <= v2[0]):
                yvals = np.interp(xvals, [v1[0], v2[0]], [v1[1], v2[1]])
            else:
                yvals = np.interp(list(reversed(xvals)), [v2[0], v1[0]], [v1[1], v2[1]])

            if(x_equals):
                xvals, yvals = yvals, xvals

            points += list(zip(xvals, yvals))[1:]
        return points

    def __repr__(self):
        return "Polygon - Centroid: %s, Vertices: %d" % (self.C, len(self.vertices)-1)

    def get_circle(self, point):
        rad_point = None
        dx = point[0] - self.C[0]
        dy = point[1] - self.C[1]
        m = dy / dx

        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i+1) % len(self.vertices)]

            dx_v = v1[0] - v2[0]
            dy_v = v1[1] - v2[1]

            def check_point(x, y):
                dx_p = x - self.C[0]
                dy_p = y - self.C[1]
    
                in_range = (v1[0] <= x <= v2[0] or v1[0] >= x >= v2[0]) and (v1[1] <= y <= v2[1] or v1[1] >= y >= v2[1])
                colinear = np.abs(np.arctan2(dy_p, dx_p) - np.arctan2(dy, dx)) <= 1e-12
    
                return in_range and colinear

            if(dx == 0):
                m_v = dy_v / dx_v

                x = self.C[0]
                y = m_v * (self.C[0] - v1[0]) + v1[1]

                if(check_point(x, y)):
                    rad_point = (x, y)
                    break
            # NOTE(taona): Should be able to get rid of this.
            # BUG(taona): If section below is removed, need to check for case where m and m_v are 0
            elif(dy_v == 0):
                x = (v1[1] - self.C[1]) / m + self.C[0]
                y = v1[1]

                if(check_point(x, y)):
                    rad_point = (x, y)
                    break
            elif(dx_v == 0):
                x = v1[0]
                y = m * (v1[0] - self.C[0]) + self.C[1]

                dx_p = x - self.C[0]
                dy_p = y - self.C[1]

                in_range = v1[1] <= y <= v2[1] or v1[1] >= y >= v2[1]
                colinear = -1e-12 <= np.arctan2(dy_p, dx_p) - np.arctan2(dy, dx) <= 1e-12

                if(check_point(x, y)):
                    rad_point = (x, y)
                    break
            else:
                m_v = dy_v / dx_v

                A = np.matrix([[1, -m], [1, -m_v]])
                B = np.matrix([[m * -self.C[0] + self.C[1]], [m_v * -v1[0] + v1[1]]])
                X = np.linalg.solve(A, B)

                x = X[1, 0]
                y = X[0, 0]

                if(check_point(x, y)):
                    rad_point = (x, y)
                    break

        if(rad_point != None):
            return Circle(self.C, distance(self.C, rad_point), color=self.color)
        else:
            print(point)
            plt.plot(point[0], point[1], marker='x', color='r', linestyle=' ')
            return Circle(self.C, 0)

plt.gcf().gca().set_aspect('equal')

plt.xlim(-15, 15)
plt.ylim(-15, 15)

c1 = Circle((0, 0), 4, color='m', res=1024)
# c2 = Circle((1, 0), 1.4125, color='g', res=2048)
# c3 = Circle((3, 7), 3, color='r', res=2048)
# p1 = Polygon([(-5, 5), (4, 4), (5, -5), (-4, -4)], color='g', res=1024)
# p2 = Polygon([(-4, 3), (7, 0), (0, -3)], color='g', res=1024)
p3 = Polygon([(-4, 4), (4, 4), (4, -4), (-4, -4)], color='b', res=1024)
# l1 = Line((0, 1), 3, color='g', res=8192)
# i_c1c2 = c2.invert(c1, color='r')
# i_c1c3 = c3.invert(c1, color='r')
# i_c1p1 = p1.invert(c1, color='r')
# i_p1c1 = c1.invert(p1, color='r')
# i_p2c1 = c1.invert(p2, color='r')
# i_c1l1 = l1.invert(c1, color='r')
# i_p1l1 = l1.invert(p1, color='r')
# i_p2l1 = l1.invert(p2, color='r')
# i_p1p2 = p2.invert(p1, color='r')
# i_p3p1 = p1.invert(p3, color='r')
i_p3c1 = c1.invert(p3, color='r')

c1.draw()
# c2.draw()
# c3.draw()
# p1.draw()
# p2.draw()
p3.draw()
# l1.draw()
# i_c1c2.draw()
# i_c1c3.draw()
# i_c1p1.draw()
# i_p1c1.draw()
# i_p2c1.draw()
# i_c1l1.draw()
# i_p1l1.draw()
# i_p2l1.draw()
# i_p1p2.draw()
# i_p3p1.draw()
i_p3c1.draw()

plt.show()

### Magic code for debugging specific unruly points
# check = point == (SOME, POINT)
# if(check):
    #     print(v1)
    #     print(v2)
    #     print()
    #     print(point)
    #     print(self.centroid)
    #     print((x, y))
    #     print()
    #     print((dx_p, dy_p))
    #     print(np.arctan2(dy_p, dx_p))
    #     print()
    #     print((dx, dy))
    #     print(np.arctan2(dy, dx))
    #     print()
    #     print(np.arctan2(dy_p, dx_p) - np.arctan2(dy, dx))
    #     print()
    #     print(in_range)
    #     print(colinear)
    #     print("----------")
    #     plt.plot(x, y, marker='x', color='r')