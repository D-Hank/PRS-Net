import numpy as np
import point_cloud_utils as pcu
import matplotlib.pyplot as plt

from mayavi import mlab

v, f = pcu.load_mesh_vf("model_normalized.obj")
#ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], color = "yellow", alpha = 0.2)

x = v[ : , 0]
y = v[ : , 1]
z = v[ : , 2]
mlab.triangular_mesh(x, y, z, f)
mlab.orientation_axes()

sample = np.load("sample.npy")
x = sample[ : , 0]
y = sample[ : , 1]
z = sample[ : , 2]
mlab.points3d(x, y, z, line_width = 0.005, scale_factor = 0.01)

mlab.show()
