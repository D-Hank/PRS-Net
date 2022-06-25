import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu

from mayavi import mlab
from settings import *

N_VIS = 10

# ----------------------BASE VISUALIZER---------------------
# --------------------Implemented by subclass---------------
class Visualization():
    def __init__(self, sample_points: np.ndarray, model_path: str, title: str):
        x = np.linspace(-0.4, 0.4, N_VIS)
        y = np.linspace(-0.4, 0.4, N_VIS)
        self.x, self.y = np.meshgrid(x, y)

        self.t = np.linspace(-0.5, 0.5, N_VIS)
        self.z_norm = np.array([0.0, 0.0, 1.0])
        self.x_norm = np.array([1.0, 0.0, 0.0])
        self.cos_thres = np.sqrt(0.5)

    # @abstract method
    def add_reflect(self, trans_points: np.ndarray, params: np.ndarray):
        pass

    # @abstract method
    def add_rotate(self, trans_points: np.ndarray, params: np.ndarray):
        pass

    # @abstract method
    def match_point(self, trans_points: np.ndarray, close_points: np.ndarray):
        pass

    # @abstract method
    def save_fig(self, fig_name):
        pass

# Implemented by matplotlib
class MatPlotVisualization(Visualization):
    def __init__(self, sample_points: np.ndarray, model_path: str, title: str):
        # points: (N_sample, 3)
        # voxel: (VSIZE, VSIZE, VSIZE)

        super().__init__(sample_points, model_path, title)
        fig = plt.figure(figsize = (10, 10))
        plt.xlabel(title, {"family": "Times New Roman"})
        x = sample_points[ : , 0]
        y = sample_points[ : , 1]
        z = sample_points[ : , 2]

        self.ax = fig.add_subplot(111, projection = "3d")
        self.ax.set_zlim(-0.5, 0.5)
        #self.ax.scatter(x, y, z, c = "g", marker = "o", s = 0.5)
        self.ax.axis()

        v, f = pcu.load_mesh_vf(model_path)
        self.ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], alpha = 0.3, color = "yellow")

    def add_reflect(self, trans_points: np.ndarray, params: np.ndarray):
        # points: (N_sample, 3)
        # params: (4, )
        x = trans_points[ : , 0]
        y = trans_points[ : , 1]
        z = trans_points[ : , 2]
        self.ax.scatter(x, y, z, c = "r", marker = "+", s = 0.5)

        # draw plane
        # guarantee normal[2]!=0
        # Check angle between normal vector and z-axis
        cos_z = np.abs(np.dot(params[0 : 3], self.z_norm))
        cos_x = np.abs(np.dot(params[0 : 3], self.x_norm))
        # Use xy to express z
        if cos_z >= self.cos_thres:
            z = (-params[0] * self.x - params[1] * self.y - params[3]) / params[2]
            self.ax.plot_surface(self.x, self.y, z, color = 'green', alpha = 0.4)
        # else if: use yz to express x
        elif cos_x >= self.cos_thres:
            x = (-params[1] * self.y - params[2] * self.x - params[3]) / params[0]
            self.ax.plot_surface(x, self.y, self.x, color = 'green', alpha = 0.4)
        # use xz to express y
        else:
            y = (-params[0] * self.x - params[2] * self.y - params[3]) / params[1]
            self.ax.plot_surface(self.x, y, self.y, color = 'green', alpha = 0.4)

    def add_rotate(self, trans_points: np.ndarray, params: np.ndarray):
        # points: (N_sample, 3)
        # params: (4, )
        x = trans_points[ : , 0]
        y = trans_points[ : , 1]
        z = trans_points[ : , 2]
        self.ax.scatter(x, y, z, c = "r", marker = "+", s = 0.5)

        # Get rotation axis and angle
        cos = params[0]
        sin = np.sqrt(1 - cos ** 2)
        u = params[1 : ] / (sin + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)

        self.ax.plot(
            u[0] * self.t,
            u[1] * self.t,
            u[2] * self.t,
            linewidth = 1.0,
            color = "orange",
            alpha = 0.9
        )

    def match_point(self, trans_points: np.ndarray, close_points: np.ndarray):
        # original: (N_sample, 3)
        # trans: (N_sample, 3)
        for i in range(0, trans_points.shape[0]):
            self.ax.plot(
                [trans_points[i][0], close_points[i][0]],
                [trans_points[i][1], close_points[i][1]],
                [trans_points[i][2], close_points[i][2]],
                linestyle = ":",
                color = "lightblue",
                linewidth = 0.5,
                alpha = 1.0
            )

    def save_fig(self, fig_name):
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        plt.show()
        #plt.savefig(fig_name, format = "svg")
        #plt.cla()

# Implemented by mayavi
class MayaVisualization(Visualization):
    def __init__(self, sample_points: np.ndarray, model_path: str, title: str):
        # points: (N_sample, 3)
        # voxel: (VSIZE, VSIZE, VSIZE)
        super().__init__(sample_points, model_path, title)
        fig = mlab.figure(figure = title, size = (1000, 1000), bgcolor = (1, 1, 1))

        v, f = pcu.load_mesh_vf(model_path)
        x = v[ : , 0]
        y = v[ : , 1]
        z = v[ : , 2]
        mlab.triangular_mesh(x, y, z, f, color = (0.6, 0.6, 0.6))
        mlab.orientation_axes()

        # Depth peeling
        fig.scene.renderer.use_depth_peeling = 1

    def add_reflect(self, trans_points: np.ndarray, params: np.ndarray):
        # points: (N_sample, 3)
        # params: (4, )

        # draw plane
        # guarantee normal[2]!=0
        # Check angle between normal vector and z-axis
        cos_z = np.abs(np.dot(params[0 : 3], self.z_norm))
        cos_x = np.abs(np.dot(params[0 : 3], self.x_norm))
        # Use xy to express z
        if cos_z >= self.cos_thres:
            z = (-params[0] * self.x - params[1] * self.y - params[3]) / params[2]
            draw_x = self.x
            draw_y = self.y
            draw_z = z
        # else if: use yz to express x
        elif cos_x >= self.cos_thres:
            x = (-params[1] * self.y - params[2] * self.x - params[3]) / params[0]
            draw_x = x
            draw_y = self.y
            draw_z = self.x
        # use xz to express y
        else:
            y = (-params[0] * self.x - params[2] * self.y - params[3]) / params[1]
            draw_x = self.x
            draw_y = y
            draw_z = self.y

        mlab.mesh(draw_x, draw_y, draw_z, color = (0.4, 1.0, 0.4), opacity = 0.3)

    def add_rotate(self, trans_points: np.ndarray, params: np.ndarray):
        # points: (N_sample, 3)
        # params: (4, )

        # Get rotation axis and angle
        cos = params[0]
        sin = np.sqrt(1 - cos ** 2)
        u = params[1 : ] / (sin + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)

        mlab.plot3d(
            u[0] * self.t,
            u[1] * self.t,
            u[2] * self.t,
            tube_radius = 0.005,
            opacity = 0.4,
            color = (0.0, 0.6, 1.0)
        )

    def save_fig(self, fig_name):
        mlab.show()
        #pass
