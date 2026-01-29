"""
Visualization module for active network simulation
Handles all plotting and animation functions
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """Visualize filament networks"""

    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.viz_cfg = config.viz

        # Setup plot parameters
        self._setup_plot_params()

    def _setup_plot_params(self):
        """Initialize plotting parameters"""
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]}
        )

        self.fsize = self.viz_cfg.fsize
        self.margin = self.viz_cfg.margin

        # Colors
        self.color_rods = self.viz_cfg.color_rods
        self.color_Pend = self.viz_cfg.color_Pend
        self.color_Mend = self.viz_cfg.color_Mend

        Ntot = self.network.Ntot
        self.color_rand = 0.3 + np.random.uniform(0, 0.7, (Ntot, 3))

        # Colormap for orientations
        self.cmap = plt.get_cmap("twilight")

        # Compute plot limits
        R_cnts = self.network.R_cnts
        xCtr = R_cnts[:, 0].reshape((Ntot, 1))
        yCtr = R_cnts[:, 1].reshape((Ntot, 1))

        x_corners, y_corners = self._box_corners(xCtr, yCtr, gap=1)
        self.x_corners = x_corners
        self.y_corners = y_corners

        xmin, xmax = min(x_corners), max(x_corners)
        ymin, ymax = min(y_corners), max(y_corners)

        self.XLIM = np.array([self.margin * xmin, self.margin * xmax])
        self.YLIM = np.array([self.margin * ymin, self.margin * ymax])
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

        # Text settings
        self.s0 = self.fsize[0] / 2
        self.txtsize = 7 * self.fsize[0] / 5

        logger.info("Visualization parameters initialized")

    @staticmethod
    def _box_corners(xcnt0, ycnt0, gap):
        """Compute bounding box corners"""
        x_corners = [
            np.min(xcnt0 - gap),
            np.max(xcnt0 + gap),
            np.max(xcnt0 + gap),
            np.min(xcnt0 - gap),
            np.min(xcnt0 - gap),
        ]
        y_corners = [
            np.min(ycnt0 - gap),
            np.min(ycnt0 - gap),
            np.max(ycnt0 + gap),
            np.max(ycnt0 + gap),
            np.min(ycnt0 - gap),
        ]
        return x_corners, y_corners

    def plot_rods(self, sim_state, tt=0, plot_radius=True, plt_show=False):
        """
        Plot filaments as rods

        Args:
            sim_state: simulation state object
            tt: time index
            plot_radius: show cutoff radius
        """
        lens = self.network.lens
        phi = sim_state.phi
        xCtr, yCtr = sim_state.xCtr, sim_state.yCtr

        xMend = xCtr - (lens / 2) * np.cos(phi)
        yMend = yCtr - (lens / 2) * np.sin(phi)
        xPend = xCtr + (lens / 2) * np.cos(phi)
        yPend = yCtr + (lens / 2) * np.sin(phi)

        R_Mend = np.concatenate((xMend, yMend), axis=1)
        R_Pend = np.concatenate((xPend, yPend), axis=1)
        R_segments = [[R_Mend[i], R_Pend[i]] for i in range(len(R_Mend))]

        px, py = sim_state.px, sim_state.py
        color_ang = self.cmap((np.pi + np.arctan2(py, px)) / (2 * np.pi))

        fig = plt.figure(facecolor="w", figsize=self.fsize, dpi=100)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().set_facecolor("k")

        lc = mc.LineCollection(R_segments, color=color_ang, linewidths=2)
        plt.gca().add_collection(lc)

        # plt.scatter(xMend, yMend, color=self.color_Mend, s=self.s0, zorder=2)
        # plt.scatter(xPend, yPend, color=self.color_Pend, s=self.s0, zorder=2)

        dt = self.config.sim.dt
        plt.text(
            0.9 * self.xmin,
            1.07 * self.ymax,
            "time = " + str(np.around(tt * dt, str(dt)[::-1].find("."))),
            c="w",
            alpha=1,
            fontsize=self.txtsize,
        )

        plt.plot(self.x_corners, self.y_corners, "gray", linewidth=1)
        plt.xlim(self.XLIM)
        plt.ylim(self.YLIM)

        if self.viz_cfg.save_img:
            fname = f"Simulation_Movie/actnet_t{tt:07d}.tif"
            plt.savefig(fname)
            logger.debug(f"Saved frame {tt}")

        plt.pause(0.001)
        if plt_show:
            plt.show()

    def plot_network(
        self,
        R_cnts,
        edges,
        tt=0,
        plot_edges=True,
        plot_radius=True,
        color_scatter=None,
        plt_show=False,
    ):
        """
        Plot network as points and edges

        Args:
            R_cnts: particle positions
            edges: list of edge coordinates
            tt: time index
            plot_edges: show edges
            plot_radius: show cutoff radius
            color_scatter: point colors
        """
        if color_scatter is None:
            color_scatter = self.color_rand

        xs, ys = R_cnts[:, 0], R_cnts[:, 1]

        fig = plt.figure(facecolor="w", figsize=self.fsize, dpi=100)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().set_facecolor("k")

        if plot_edges:
            lc = mc.LineCollection(
                edges, color=[0.3, 0.4, 0.5], alpha=0.7, linewidths=0.9
            )
            plt.gca().add_collection(lc)

        plt.scatter(xs, ys, color=color_scatter, s=self.s0, zorder=2)

        dt = self.config.sim.dt
        seednum = self.config.sim.seed
        mpf = self.config.net.mpf
        len_avg = self.config.net.xi * self.config.net.dens0**-0.5

        plt.text(
            0.9 * self.xmin,
            1.07 * self.ymax,
            "time = " + str(np.around(tt * dt, str(dt)[::-1].find("."))),
            c="w",
            alpha=1,
            fontsize=self.txtsize,
        )
        plt.text(
            0.9 * self.xmin,
            1.11 * self.ymin,
            "seed = " + str(seednum),
            c="w",
            alpha=1,
            fontsize=self.txtsize,
        )
        plt.text(
            0.65 * self.xmax,
            1.11 * self.ymin,
            "mpf = " + str(mpf),
            c="w",
            alpha=1,
            fontsize=self.txtsize,
        )
        plt.text(
            0.92 * self.xmax,
            self.ymax + 1.05,
            r"$\langle \ell \rangle$",
            c="w",
            alpha=1,
            fontsize=self.txtsize,
        )

        plt.plot(self.x_corners, self.y_corners, "gray", linewidth=1)
        plt.plot(
            [self.x_corners[2] - len_avg, self.x_corners[2]],
            [self.y_corners[2], self.y_corners[2]],
            "-",
            color=[0.6, 0, 0.2],
            lw=5,
            alpha=1,
            zorder=3,
        )

        if plot_radius:
            SumLens = 2 * len_avg
            Rcutoff = self.config.net.Rc_0
            plt.gca().add_patch(
                plt.Circle(
                    (self.x_corners[2], self.y_corners[2]),
                    SumLens * Rcutoff,
                    color=[0, 0.4, 0.6],
                    alpha=0.7,
                    clip_on=False,
                )
            )

        plt.xlim(self.XLIM)
        plt.ylim(self.YLIM)

        if self.viz_cfg.save_img:
            fname = f"Simulation_Movie/actnet_t{tt:07d}.tif"
            plt.savefig(fname)
            logger.debug(f"Saved frame {tt}")

        plt.pause(0.001)
        if plt_show:
            plt.show()

    def plot_timeseries(self, Xt, timeseries, VarName=""):
        """Plot time series data"""
        ts = self.config.sim.dt * np.array(timeseries)

        fig = plt.figure(facecolor="w", dpi=100, figsize=self.fsize)
        plt.plot(ts, Xt)
        plt.xlabel(r"\it{time}", fontsize=20)
        plt.ylabel(VarName, fontsize=20)

        if self.viz_cfg.save_img:
            fname = f"R2_C_{self.config.net.dens0}.png"
            plt.savefig(fname)

        plt.show()


def create_plot_callback(visualizer, tplot):
    """
    Create callback function for simulation visualization

    Args:
        visualizer: NetworkVisualizer instance
        tplot: list of timesteps to plot

    Returns:
        callback function
    """

    def callback(tt, sim_state, net_data):
        if tt in tplot:
            logger.info(f"Plotting frame at t={tt}")

            # Update orientation colors
            px, py = sim_state.px, sim_state.py
            color_ang = visualizer.cmap((np.pi + np.arctan2(py, px)) / (2 * np.pi))

            # # Plot network and rods
            # visualizer.plot_network(
            #     visualizer.network.R_cnts,
            #     net_data["edges"],
            #     tt,
            #     plot_edges=True,
            #     plot_radius=False,
            #     color_scatter=color_ang,
            # )
            visualizer.plot_rods(sim_state, tt, plot_radius=False)

    return callback
