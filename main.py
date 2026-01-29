"""
Author: Shahriar Shadkhoo

Main entry point for active network simulation

Example usage:
    python main.py

    # Or with custom config:
    from main import run_simulation
    from config import Config

    cfg = Config()
    cfg.sim.T_tot = 200
    cfg.net.dens0 = 2.0
    run_simulation(cfg)
"""

import logging
import numpy as np

from AN_utils.config import Config
from AN_utils.network import FilamentNetwork
from AN_utils.simulation import ActiveNetworkSimulation
from AN_utils.visualization import NetworkVisualizer, create_plot_callback
from AN_utils.helpers import setup_logging, set_random_seed, create_output_directory
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_simulation(config=None):
    """
    Run active network simulation

    Args:
        config: Config object (creates default if None)

    Returns:
        sim: ActiveNetworkSimulation object with final state
        network: FilamentNetwork object
    """
    # Setup
    if config is None:
        config = Config()

    setup_logging(level=logging.INFO)

    if config.sim.use_seeding:
        set_random_seed(config.sim.seed)

    if config.viz.save_img:
        create_output_directory()

    logger.info("=" * 60)
    logger.info("Starting Active Network Simulation")
    logger.info("=" * 60)

    # Initialize network
    logger.info("Initializing network...")
    network = FilamentNetwork(config)
    network.initialize()

    # Initialize simulation
    logger.info("Initializing simulation...")
    sim = ActiveNetworkSimulation(config, network)
    sim.initialize_state()

    # Setup visualization
    visualizer = None
    callback = None

    if config.viz.plot_img:
        logger.info("Setting up visualization...")
        visualizer = NetworkVisualizer(config, network)

        # Determine plot timesteps
        Nt = config.sim.Nt
        N_frame = config.sim.N_frame
        tplot = list(np.unique(np.around(np.linspace(0, Nt, N_frame + 1)).astype(int)))
        logger.info(f"Will plot {len(tplot)} frames")

        callback = create_plot_callback(visualizer, tplot)

        # Plot initial state
        logger.info("Plotting initial state...")
        px, py = sim.px, sim.py
        color_ang = visualizer.cmap((np.pi + np.arctan2(py, px)) / (2 * np.pi))
        # visualizer.plot_network(
        #     network.R_cnts,
        #     network.network["edges"],
        #     0,
        #     plot_edges=True,
        #     plot_radius=False,
        #     color_scatter=color_ang,
        #     plt_show=False,
        # )
        visualizer.plot_rods(sim, 0, plot_radius=False, plt_show=False)

    # Run simulation
    logger.info("Running simulation...")
    sim.run(callback=callback)

    # Plot final state
    if visualizer:
        logger.info("Plotting final state...")
        tt = config.sim.Nt - 1
        px, py = sim.px, sim.py
        color_ang = visualizer.cmap((np.pi + np.arctan2(py, px)) / (2 * np.pi))
        # visualizer.plot_network(
        #     network.R_cnts,
        #     network.network['edges'],
        #     tt + 1,
        #     plot_edges=True,
        #     plot_radius=False,
        #     color_scatter=color_ang
        #     plt_show=False
        # )
        visualizer.plot_rods(sim, tt + 1, plot_radius=False, plt_show=False)

    plt.show()

    logger.info("=" * 60)
    logger.info("Simulation completed successfully!")
    logger.info("=" * 60)

    return sim, network


def main():
    """Main entry point with default configuration"""
    config = Config()

    # You can customize config here
    # config.sim.T_tot = 200
    # config.net.dens0 = 2.0
    # config.motor.c_mm = 0.2

    sim, network = run_simulation(config)

    return sim, network


if __name__ == "__main__":
    sim, network = main()
