import numpy as np
import matplotlib.pyplot as plt

def plotit(x1, y1, x2, y2, x3=[], y3=[], pad=False):
    points = list(zip(x1, y1)) + list(zip(x2, y2))
    if x3 != []:
        points += list(zip(x3, y3))

    from scipy.spatial import Voronoi, voronoi_plot_2d

    vor = Voronoi(points)
    voronoi_plot_2d(vor, show_points=False, show_vertices=False)

    plt.plot(x1, y1, marker=".", linestyle='None')
    plt.plot(x2, y2, marker="v", linestyle='None')

    plt.ylabel("Error rate")
    plt.xlabel("Value of h")

    plt.show()


# 4.a

x1 = [10, 0, 5]
y1 = [0, -10, -2]

x2 = [5, 0, 5]
y2 = [10, 5, 5]

plotit(x1, y1, x2, y2)

# 4.b
# trouver moyenne des points 1, et celles des points 2, puis faire comme en a

moy1 = np.array([np.sum(x1), np.sum(y1)]) / len(x1)
moy2 = np.array([np.sum(x2), np.sum(y2)]) / len(x2)

plotit([moy1[0]], [moy1[1]], [moy2[0]], [moy2[1]], [0, 1], [0, 1])