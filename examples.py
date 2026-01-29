"""
Example usage patterns for active network simulation
"""

import logging
from AN_utils.config import Config
from main import run_simulation
from AN_utils.helpers import setup_logging
import numpy as np


def example_basic():
    """Basic simulation with default parameters"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Simulation")
    print("=" * 60)

    setup_logging(level=logging.INFO)
    sim, network = run_simulation()

    print(f"\nFinal state: {network.Ntot} filaments")
    print(f"Network bonds: {network.network['N_bonds']}")


def example_custom_config():
    """Simulation with customized parameters"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    setup_logging(level=logging.INFO)

    cfg = Config()

    # Modify simulation parameters
    cfg.sim.T_tot = 50
    cfg.sim.dt = 0.01
    cfg.sim.N_frame = 10

    # Denser network
    cfg.net.dens0 = 2.0
    cfg.net.XY_lens = [20, 20]

    # More active motors
    cfg.motor.c_mm = 0.3
    cfg.motor.V_mm = -2.0

    # Disable saving
    cfg.viz.save_img = False

    sim, network = run_simulation(cfg)

    print(f"\nSimulated {cfg.sim.Nt} timesteps")


def example_parameter_sweep():
    """Run multiple simulations with different densities"""
    print("\n" + "=" * 60)
    print("Example 3: Parameter Sweep")
    print("=" * 60)

    setup_logging(level=logging.WARNING)  # Less verbose

    densities = [0.5, 1.0, 2.0]
    results = []

    for dens in densities:
        print(f"\nRunning simulation with density = {dens}")

        cfg = Config()
        cfg.sim.T_tot = 20
        cfg.sim.N_frame = 5
        cfg.net.dens0 = dens
        cfg.viz.plot_img = False  # Skip plotting for speed

        sim, network = run_simulation(cfg)

        results.append(
            {
                "density": dens,
                "N_filaments": network.Ntot,
                "N_bonds": network.network["N_bonds"],
            }
        )

    print("\n" + "-" * 60)
    print("Results:")
    for r in results:
        print(
            f"  Density {r['density']}: {r['N_filaments']} filaments, {r['N_bonds']} bonds"
        )


def example_triangular_lattice():
    """Use triangular instead of square lattice"""
    print("\n" + "=" * 60)
    print("Example 4: Triangular Lattice")
    print("=" * 60)

    setup_logging(level=logging.INFO)

    cfg = Config()
    cfg.net.unit_cell = "triangular"
    cfg.net.disorder = 0.5  # Less disorder
    cfg.sim.T_tot = 30
    cfg.sim.N_frame = 8

    sim, network = run_simulation(cfg)


def example_high_activity():
    """Simulation with strong active forces"""
    print("\n" + "=" * 60)
    print("Example 5: High Activity")
    print("=" * 60)

    setup_logging(level=logging.INFO)

    cfg = Config()

    # All minus-minus motors
    cfg.motor.c_el = 0.0
    cfg.motor.c_mm = 1.0
    cfg.motor.c_pp = 0.0

    # Strong activity
    cfg.motor.a_mm = 2.0
    cfg.motor.V_mm = -3.0

    # Enable noise
    cfg.phys.nois_on = True
    cfg.phys.Temp_t = 5.0

    cfg.sim.T_tot = 40

    sim, network = run_simulation(cfg)


def example_with_callback():
    """Custom callback for monitoring simulation"""
    print("\n" + "=" * 60)
    print("Example 6: Custom Callback")
    print("=" * 60)

    setup_logging(level=logging.INFO)

    from network import FilamentNetwork
    from simulation import ActiveNetworkSimulation

    cfg = Config()
    cfg.sim.T_tot = 30
    cfg.viz.plot_img = False

    network = FilamentNetwork(cfg)
    network.initialize()

    sim = ActiveNetworkSimulation(cfg, network)
    sim.initialize_state()

    # Custom callback to track kinetic energy
    energies = []

    def energy_callback(tt, sim_state, net_data):
        KE = 0.5 * cfg.phys.M0 * np.sum(sim_state.Vx**2 + sim_state.Vy**2)
        energies.append(KE)
        if tt % 100 == 0:
            print(f"  t={tt}: KE = {KE:.3f}")

    print("\nMonitoring kinetic energy:")
    sim.run(callback=energy_callback)

    print(f"\nAverage KE: {np.mean(energies):.3f}")
    print(f"Final KE: {energies[-1]:.3f}")


if __name__ == "__main__":
    # Run all examples
    example_basic()
    example_custom_config()
    example_parameter_sweep()
    example_triangular_lattice()
    example_high_activity()
    example_with_callback()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
