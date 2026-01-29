"""
Simulation module for active network dynamics
Handles time evolution, force/torque calculations
"""

import logging
import torch
import numpy as np
from time import time

logger = logging.getLogger(__name__)


class ForceCalculator:
    """Compute forces on filaments"""

    def __init__(self, config):
        self.config = config
        self.phys = config.phys
        self.motor = config.motor

    def compute_active_forces(self, network, Vr, Pr):
        """
        Compute active motor forces

        Args:
            network: network connectivity dict
            Vr: velocities at rod ends (2*Nbonds, 2)
            Pr: orientations (Ntot, 2)

        Returns:
            Fax, Fay: active forces
        """
        B12, B21, Bas = network["B12"], network["B21"], network["Bas"]
        V_mot, A_mot = network["V_mot"], network["A_mot"]

        fil_int = self.phys.fil_int
        self_int = self.phys.self_int

        # Filament interaction term
        Fact = fil_int * B21.T.dot(
            (
                A_mot
                * (
                    V_mot
                    - np.sum((B12.dot(Pr)) * (Bas.dot(Vr)), axis=1).reshape(
                        B12.shape[0], 1
                    )
                )
            )
            * B12.dot(Pr)
        )

        # Self interaction term
        Fact += self_int * B21.T.dot(
            (
                A_mot
                * (
                    V_mot
                    + np.sum((B21.dot(Pr)) * (B21.dot(Vr)), axis=1).reshape(
                        B21.shape[0], 1
                    )
                )
            )
            * B21.dot(Pr)
        )

        Ntot = Pr.shape[0]
        Fax = Fact[:, 0].reshape(Ntot, 1)
        Fay = Fact[:, 1].reshape(Ntot, 1)

        return Fax, Fay

    def compute_elastic_forces(self, network, xCtr, yCtr, xMend, yMend, xPend, yPend):
        """
        Compute elastic spring forces between filament ends

        Returns:
            Fpx, Fpy: elastic forces
        """
        incid_el = network["incid_el"]

        DxM, DyM = incid_el.dot(xMend), incid_el.dot(yMend)
        DxP, DyP = incid_el.dot(xPend), incid_el.dot(yPend)
        DrM = np.sqrt(DxM**2 + DyM**2)
        DrP = np.sqrt(DxP**2 + DyP**2)

        kmm, kpp = self.motor.k_mm, self.motor.k_pp
        Lrest = self.motor.l_rest * self.config.net.xi * self.config.net.dens0**-0.5

        # Attractive forces
        FpxA = -incid_el.T.dot(
            kmm * np.exp(-((DrM - 2 * Lrest) ** 2) / (2 * Lrest**2)) * (DxM / DrM)
            + kpp * np.exp(-((DrP - 2 * Lrest) ** 2) / (2 * Lrest**2)) * (DxP / DrP)
        )
        FpyA = -incid_el.T.dot(
            kmm * np.exp(-((DrM - 2 * Lrest) ** 2) / (2 * Lrest**2)) * (DyM / DrM)
            + kpp * np.exp(-((DrP - 2 * Lrest) ** 2) / (2 * Lrest**2)) * (DyP / DrP)
        )

        # Repulsive forces
        FpxR = +incid_el.T.dot(
            10 * kmm * np.exp(-((DrM - 1 * Lrest) ** 2) / (1 * Lrest**2)) * (DxM / DrM)
            + 10
            * kpp
            * np.exp(-((DrP - 1 * Lrest) ** 2) / (1 * Lrest**2))
            * (DxP / DrP)
        )
        FpyR = +incid_el.T.dot(
            10 * kmm * np.exp(-((DrM - 1 * Lrest) ** 2) / (1 * Lrest**2)) * (DyM / DrM)
            + 10
            * kpp
            * np.exp(-((DrP - 1 * Lrest) ** 2) / (1 * Lrest**2))
            * (DyP / DrP)
        )

        Fpx = FpxA + FpxR
        Fpy = FpyA + FpyR

        return Fpx, Fpy

    def compute_drag_forces(self, Vx, Vy, px, py, len_avg):
        """
        Compute viscous drag forces

        Returns:
            Fdx, Fdy: drag forces
        """
        mu = self.phys.mu
        diam = self.phys.diam

        # Parallel drag
        FparX = -mu * np.pi * (diam**2) * (+px * Vx + py * Vy) * px
        FparY = -mu * np.pi * (diam**2) * (+px * Vx + py * Vy) * py

        # Perpendicular drag
        FprpX = -mu * diam * len_avg * (+px * Vy - py * Vx) * (-py)
        FprpY = -mu * diam * len_avg * (+px * Vy - py * Vx) * (+px)

        Fdx = FparX + FprpX
        Fdy = FparY + FprpY

        return Fdx, Fdy


class ActiveNetworkSimulation:
    """Main simulation driver"""

    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.force_calc = ForceCalculator(config)

        # Time parameters
        self.dt = config.sim.dt
        self.Nt = config.sim.Nt
        self.tseries = range(self.Nt)

        # Physics parameters
        self.MM = config.phys.M0
        self.II = config.phys.I0

        # State arrays
        self.xCtr = None
        self.yCtr = None
        self.Vx = None
        self.Vy = None
        self.av = None
        self.phi = None
        self.px = None
        self.py = None

        # Rearrangement schedule
        self.t_rearr = []
        if config.phys.rearr:
            mft = int(max(1, np.ceil(config.phys.T_rearr / self.dt)))
            self.t_rearr = np.around(np.arange(mft, self.Nt - 1, mft)).astype(int)
            logger.info(f"Rearrangement every {mft} steps ({len(self.t_rearr)} total)")

    def initialize_state(self):
        """Initialize simulation state from network"""
        logger.info("Initializing simulation state")

        Ntot = self.network.Ntot
        R_cnts = self.network.R_cnts
        self.phi = self.network.phi  # Already numpy array

        # Center of mass positions
        self.xCtr = R_cnts[:, 0].reshape((Ntot, 1))
        self.yCtr = R_cnts[:, 1].reshape((Ntot, 1))

        # Velocities
        self.Vx = np.zeros((Ntot, 1))
        self.Vy = np.zeros((Ntot, 1))
        self.av = np.zeros((Ntot, 1))

        # Orientations
        self.px = np.cos(self.phi)
        self.py = np.sin(self.phi)

        logger.info(f"State initialized for {Ntot} filaments")

    def compute_filament_ends(self, lens):
        """Compute filament end positions"""
        xMend = self.xCtr - (lens / 2) * self.px
        yMend = self.yCtr - (lens / 2) * self.py
        xPend = self.xCtr + (lens / 2) * self.px
        yPend = self.yCtr + (lens / 2) * self.py
        return xMend, yMend, xPend, yPend

    def run(self, callback=None):
        """
        Run simulation

        Args:
            callback: function called at each timestep with (tt, self)
        """
        logger.info("Starting simulation run")
        ti = time()

        lens = self.network.lens
        len_avg = self.config.net.xi * self.config.net.dens0**-0.5
        net_data = self.network.network
        R_cnts = self.network.R_cnts

        # Feature flags
        actv_on = self.config.phys.actv_on
        elas_on = self.config.phys.elas_on
        drag_on = self.config.phys.drag_on
        nois_on = self.config.phys.nois_on

        Temp_t = self.config.phys.Temp_t
        Ntot = self.network.Ntot

        for tt in self.tseries:

            # Rearrange network connectivity if needed
            if tt in self.t_rearr:
                logger.debug(f"Rearranging network at t={tt}")
                net_data = self.network.update_connectivity(
                    R_cnts, self.config.net.Rc_0
                )

            # Update filament end positions
            xMend, yMend, xPend, yPend = self.compute_filament_ends(lens)

            # Compute velocities at rod ends
            lev_r = len_avg / 2
            Vr = np.hstack(
                (
                    self.Vx - lev_r * self.av * self.py,
                    self.Vy + lev_r * self.av * self.px,
                )
            )
            Pr = np.hstack((self.px, self.py))

            # ========== Force Calculation ==========
            Fax = Fay = 0
            if actv_on:
                Fax, Fay = self.force_calc.compute_active_forces(net_data, Vr, Pr)

            Fpx = Fpy = 0
            if elas_on:
                Fpx, Fpy = self.force_calc.compute_elastic_forces(
                    net_data, self.xCtr, self.yCtr, xMend, yMend, xPend, yPend
                )

            Fdx = Fdy = 0
            if drag_on:
                Fdx, Fdy = self.force_calc.compute_drag_forces(
                    self.Vx, self.Vy, self.px, self.py, len_avg
                )

            # Total acceleration
            ax = (Fpx + Fax + Fdx) / self.MM
            ay = (Fpy + Fay + Fdy) / self.MM

            if nois_on:
                ax += Temp_t**0.5 * np.random.normal(0, 1, (Ntot, 1))
                ay += Temp_t**0.5 * np.random.normal(0, 1, (Ntot, 1))

            # Update positions and velocities
            self.xCtr += self.Vx * self.dt + (ax * self.dt**2) / 2
            self.yCtr += self.Vy * self.dt + (ay * self.dt**2) / 2
            self.Vx += ax * self.dt
            self.Vy += ay * self.dt

            # Update positions in R_cnts
            R_cnts[:, 0] = self.xCtr.ravel()
            R_cnts[:, 1] = self.yCtr.ravel()

            # ========== Torque Calculation ==========
            tau0 = self.config.phys.tau0
            if tau0:
                incid_el = net_data["incid_el"]
                Tau = (
                    -tau0 * np.sin(incid_el.T.dot(incid_el).dot(self.phi))
                    - self.av * self.config.phys.mu * len_avg**2 / 2
                )
                ang_ax = tau0 * Tau / self.II

                if nois_on:
                    Temp_a = self.config.phys.Temp_a
                    ang_ax += (Temp_a**0.5 / self.II) * np.random.normal(
                        0, 1, (Ntot, 1)
                    )

                self.phi += self.av * self.dt + (ang_ax * self.dt**2) / 2
                self.av += ang_ax * self.dt
                self.px, self.py = np.cos(self.phi), np.sin(self.phi)

            # Callback for visualization/logging
            if callback:
                callback(tt, self, net_data)

        tf = time()
        logger.info(f"Simulation completed in {tf - ti:.2f} seconds")
