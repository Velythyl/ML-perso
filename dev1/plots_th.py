import matplotlib.pyplot as plt

# 4.a

x11 = [10, 0, 5]
x21 = [0, -10, -2]

x12 = [5, 0, 5]
x22 = [10, 5, 5]

points = list(zip(x11, x21)) + list(zip(x12, x22))

from scipy.spatial import Voronoi, voronoi_plot_2d

vor = Voronoi(points)
fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False)

plt.plot(x11, x21, marker=".", linestyle = 'None')
plt.plot(x12, x22, marker="v", linestyle = 'None')

plt.ylabel("Error rate")
plt.xlabel("Value of h")

plt.show()


#