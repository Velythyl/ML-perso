import matplotlib.pyplot as plt

# 4.a

x11 = [10, 0, 5]
x21 = [0, -10, -2]

x12 = [5, 0, 5]
x22 = [10, 5, 5]

plt.plot(x11, x21, marker=".", linestyle = 'None')
plt.plot(x12, x22, marker="v", linestyle = 'None')

"""
points = list(zip(x11, x21)) + list(zip(x12, x22))

from scipy.spatial import Voronoi, voronoi_plot_2d

vor = Voronoi(points)
fig = voronoi_plot_2d(vor)

for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]

    polygon = [vor.vertices[i] for i in region]
    plt.fill(*zip(*polygon), color=[1, 1, 0, 1])
"""

plt.ylabel("Error rate")
plt.xlabel("Value of h")
plt.show()


#