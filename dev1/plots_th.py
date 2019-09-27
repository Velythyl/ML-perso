import numpy as np
import matplotlib.pyplot as plt

def plotit(x1, y1, x2, y2, x3=[], y3=[], vor=True):
    points = list(zip(x1, y1)) + list(zip(x2, y2))
    if x3 != []:
        points += list(zip(x3, y3))

    if vor:
        from scipy.spatial import Voronoi, voronoi_plot_2d

        vor = Voronoi(points)
        voronoi_plot_2d(vor, show_points=False, show_vertices=False)

    plt.plot(x1, y1, marker=".", linestyle='None')
    plt.plot(x2, y2, marker="v", linestyle='None')
    if x3 != []:
        plt.plot(x3, y3, marker="s", linestyle='None')

    plt.ylabel("x2")
    plt.xlabel("x1")

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

print(moy1, moy2)   #[ 5. -4.] [3.33333333 6.66666667]

# on ne peut pas utiliser plotit: Qhull n'aime pas lorsqu'on a moins que 4 points a evaluer!
# Donc, on prend les points donnes par print et on les met sur http://alexbeutel.com/webgl/voronoi.html
# {"sites":[1166.665,1100,777.777,33.333],"queries":[]} avec 1400 14000

# 4.c.a

x3 = [2, -5, 10]
y3 = [8, 2, -4]

plotit(x1, y1, x2, y2, x3, y3)

# 4.c.b

moy3 = np.array([np.sum(x3), np.sum(y3)]) / len(x3)

print(moy1, moy2, moy3) #[ 5. -4.] [3.33333333 6.66666667] [2.33333333 2.        ]
