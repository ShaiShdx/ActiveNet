"""
Configuration module for active network simulation
"""

import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Core simulation parameters"""

    T_tot: float = 100.0
    dt: float = 0.01
    N_frame: int = 20
    seed: int = 20
    use_seeding: bool = True

    def __post_init__(self):
        self.N_frame = min(self.N_frame, int(self.T_tot / self.dt))
        self.Nt = round(self.T_tot / self.dt)
        logger.info(f"Simulation: T_tot={self.T_tot}, dt={self.dt}, Nt={self.Nt}")


@dataclass
class NetworkConfig:
    """Network geometry and topology parameters"""

    XY_lens: list = field(default_factory=lambda: [30, 30])
    dens0: float = 1.0
    xi: float = 1.5  # len_avg * sqrt(dens)
    unit_cell: str = "square"
    disorder: float = 1.0
    phi_range: list = field(default_factory=lambda: [-np.pi, np.pi])
    phi_avg: float = 0.0

    # Network constraints
    mpf: int = 20  # max motors per filament
    Rc_0: float = 0.5  # cutoff distance in units of Avg_Lens

    def __post_init__(self):
        if self.unit_cell != "square":
            self.XY_lens[1] *= (3 / 2) ** 0.5
            self.XY_lens[1] -= (1 + round(self.XY_lens[1] * self.dens0**0.5)) % 2
        logger.info(f"Network: {self.unit_cell} lattice, density={self.dens0}")


@dataclass
class MotorConfig:
    """Motor and protein concentrations"""

    # Concentrations (normalized automatically)
    c_el: float = 0.0  # elastic linkers
    c_mm: float = 0.1  # minus-minus motors
    c_pp: float = 0.0  # plus-plus motors

    # Motor properties
    e_mm: float = 1.0  # binding energies
    e_pp: float = 1.0
    e_mp: float = 1.0

    a_el: float = 0.0  # activity units
    a_mm: float = 1.0
    a_pp: float = 1.0

    V_el: float = 0.0  # walking velocities
    V_mm: float = -1.0
    V_pp: float = 1.0

    k_mm: float = 10.0  # spring constants
    k_pp: float = 10.0
    l_rest: float = 0.1

    def __post_init__(self):
        # Normalize concentrations
        total = self.c_el + self.c_mm + self.c_pp
        if total > 0:
            self.c_el /= total
            self.c_mm /= total
            self.c_pp /= total

        # Create numpy arrays (torch-compatible structure)
        self.ccs = np.array([self.c_el, self.c_mm, self.c_pp], dtype=np.float64)
        self.A_type = np.array([self.a_el, self.a_mm, self.a_pp], dtype=np.float64)
        self.V_type = np.array([self.V_el, self.V_mm, self.V_pp], dtype=np.float64)

        logger.info(
            f"Motors: c_el={self.c_el:.2f}, c_mm={self.c_mm:.2f}, c_pp={self.c_pp:.2f}"
        )


@dataclass
class PhysicsConfig:
    """Physical parameters and flags"""

    # Interaction strengths
    self_int: float = -0.5
    fil_int: float = 1.0

    # Material properties
    M0: float = 1.0
    I0: float = 10.0
    tau0: float = 0.0
    mu: float = 10.0
    diam: float = 1.0

    # Temperatures
    Temp_t: float = 10.0  # translational
    Temp_a: float = 100.0  # angular

    # Feature flags
    actv_on: bool = True
    elas_on: bool = True
    drag_on: bool = True
    nois_on: bool = False

    # Rearrangement
    rearr: bool = True
    T_rearr: float = 1.0

    def __post_init__(self):
        logger.info(
            f"Physics: actv={self.actv_on}, elas={self.elas_on}, drag={self.drag_on}"
        )


@dataclass
class VisualizationConfig:
    """Plotting and output settings"""

    plot_img: bool = True
    save_img: bool = False
    read_img: bool = False

    fsize: tuple = (10, 10)
    margin: float = 1.2

    color_rods: list = field(default_factory=lambda: [0.8, 0.8, 0.8])
    color_Pend: list = field(default_factory=lambda: [0.6, 0.0, 0.4])
    color_Mend: list = field(default_factory=lambda: [0.0, 0.6, 0.4])

    def __post_init__(self):
        logger.info(f"Visualization: plot={self.plot_img}, save={self.save_img}")


class Config:
    """Master configuration container"""

    def __init__(self):
        self.sim = SimulationConfig()
        self.net = NetworkConfig()
        self.motor = MotorConfig()
        self.phys = PhysicsConfig()
        self.viz = VisualizationConfig()

        logger.info("Configuration initialized")

    def update(self, **kwargs):
        """Update configuration from kwargs"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                for k, v in value.items():
                    if hasattr(config_obj, k):
                        setattr(config_obj, k, v)
                        logger.debug(f"Updated {key}.{k} = {v}")
