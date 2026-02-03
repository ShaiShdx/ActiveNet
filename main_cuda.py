"""
Author: Shahriar Shadkhoo

Main entry point for active network simulation with CUDA support
Automatically uses GPU if available, falls back to CPU

Example usage:
    # Auto-detect CUDA
    python main_cuda.py

    # Force CPU
    python main_cuda.py --cpu

    # From code:
    from main_cuda import run_simulation
    sim, network = run_simulation(use_cuda=True)
"""

import logging
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

from AN_utils.config import Config
from AN_utils.network import FilamentNetwork
from AN_utils.simulation_cuda import ActiveNetworkSimulationCUDA
from AN_utils.visualization import NetworkVisualizer, create_plot_callback
from AN_utils.helpers import setup_logging, set_random_seed, create_output_directory
from AN_utils.helpers_cuda import get_device_manager

logger = logging.getLogger(__name__)


def run_simulation(config=None, use_cuda=True):
    """
    Run active network simulation with CUDA support

    Args:
        config: Config object (creates default if None)
        use_cuda: Try to use CUDA if available (default: True)

    Returns:
        sim: ActiveNetworkSimulationCUDA object with final state
        network: FilamentNetwork object
    """
    # Setup
    if config is None:
        config = Config()

    setup_logging(level=logging.INFO)

    # Initialize device manager
    device = get_device_manager(use_cuda=use_cuda)

    if config.sim.use_seeding:
        set_random_seed(config.sim.seed)

    if config.viz.save_img:
        create_output_directory()

    logger.info("=" * 60)
    logger.info("Active Network Simulation with CUDA Support")
    logger.info("=" * 60)

    # Initialize network (CPU only - sparse matrices)
    logger.info("Initializing network...")
    network = FilamentNetwork(config)
    network.initialize()

    # Initialize simulation (with CUDA support)
    logger.info("Initializing simulation...")
    sim = ActiveNetworkSimulationCUDA(config, network, use_cuda=use_cuda)
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

        # Create callback that handles GPU->CPU transfer
        def cuda_callback(tt, sim_state, net_data):
            if tt in tplot:
                logger.info(f"Plotting frame at t={tt}")

                # Transfer state to CPU for plotting
                px_cpu = device.to_numpy(sim_state.px)
                py_cpu = device.to_numpy(sim_state.py)

                # Update R_cnts for plotting
                network.R_cnts[:, 0] = device.to_numpy(sim_state.xCtr).ravel()
                network.R_cnts[:, 1] = device.to_numpy(sim_state.yCtr).ravel()

                # Update orientation colors
                color_ang = visualizer.cmap(
                    (np.pi + np.arctan2(py_cpu, px_cpu)) / (2 * np.pi)
                )

                # Create temporary simulation state for visualization
                class TempState:
                    def __init__(self, sim_state, device):
                        self.xCtr = device.to_numpy(sim_state.xCtr)
                        self.yCtr = device.to_numpy(sim_state.yCtr)
                        self.phi = device.to_numpy(sim_state.phi)
                        self.px = device.to_numpy(sim_state.px)
                        self.py = device.to_numpy(sim_state.py)

                temp_state = TempState(sim_state, device)

                # Plot
                visualizer.plot_network(
                    network.R_cnts,
                    net_data["edges"],
                    tt,
                    plot_edges=True,
                    plot_radius=False,
                    color_scatter=color_ang,
                )
                visualizer.plot_rods(temp_state, tt, plot_radius=False)

            plt.show()

        callback = cuda_callback

        # Plot initial state
        logger.info("Plotting initial state...")
        callback(0, sim, network.network)

    # Run simulation
    logger.info("Running simulation...")
    sim.run(callback=callback)

    # Plot final state
    if visualizer:
        logger.info("Plotting final state...")
        tt = config.sim.Nt - 1
        callback(tt + 1, sim, network.network)

    logger.info("=" * 60)
    logger.info("Simulation completed successfully!")
    logger.info("=" * 60)

    return sim, network


def main():
    """Main entry point with command-line arguments"""
    parser = argparse.ArgumentParser(description="Active network simulation with CUDA")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    config = Config()

    # You can customize config here
    # config.sim.T_tot = 200
    # config.net.dens0 = 2.0

    use_cuda = not args.cpu
    sim, network = run_simulation(config, use_cuda=use_cuda)

    return sim, network


if __name__ == "__main__":
    ti = time.time()

    sim, network = main()

    print("\n TOTAL TIME = ", time.time() - ti)
