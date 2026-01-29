"""
CUDA-enabled simulation module for active network dynamics
Automatically uses GPU if available, falls back to CPU
"""

import logging
import numpy as np
from time import time
from AN_utils.helpers_cuda import get_device_manager

logger = logging.getLogger(__name__)


class ForceCalculatorCUDA:
    """Compute forces on filaments with CUDA support"""

    def __init__(self, config, device_manager=None):
        self.config = config
        self.phys = config.phys
        self.motor = config.motor
        self.device = device_manager or get_device_manager()

    def compute_active_forces(self, network, Vr, Pr):
        """
        Compute active motor forces (GPU-accelerated)

        Args:
            network: network connectivity dict (sparse matrices on CPU)
            Vr: velocities at rod ends (on device)
            Pr: orientations (on device)

        Returns:
            Fax, Fay: active forces (on device)
        """
        # Sparse matrix operations stay on CPU
        B12, B21, Bas = network["B12"], network["B21"], network["Bas"]
        V_mot = self.device.to_tensor(network["V_mot"])
        A_mot = self.device.to_tensor(network["A_mot"])

        fil_int = self.phys.fil_int
        self_int = self.phys.self_int

        # Convert to numpy for sparse operations
        Vr_np = self.device.to_numpy(Vr)
        Pr_np = self.device.to_numpy(Pr)

        # Sparse matrix multiplications (CPU)
        B12_Pr = B12.dot(Pr_np)
        B21_Pr = B21.dot(Pr_np)
        Bas_Vr = Bas.dot(Vr_np)
        B12_Vr = B12.dot(Vr_np)
        B21_Vr = B21.dot(Vr_np)

        # Convert back to device for element-wise operations
        B12_Pr = self.device.to_tensor(B12_Pr)
        B21_Pr = self.device.to_tensor(B21_Pr)
        Bas_Vr = self.device.to_tensor(Bas_Vr)
        B21_Vr = self.device.to_tensor(B21_Vr)

        # Element-wise operations (GPU if available)
        Ntot = Pr.shape[0]

        # Filament interaction term
        interaction_term = self.device.sum(B12_Pr * Bas_Vr, axis=1).reshape(
            B12_Pr.shape[0], 1
        )
        factor1 = A_mot * (V_mot - interaction_term) * B12_Pr

        # Convert to numpy for sparse transpose multiplication
        factor1_np = self.device.to_numpy(factor1)
        Fact = self.device.to_tensor(B21.T.dot(factor1_np)) * fil_int

        # Self interaction term
        self_interaction_term = self.device.sum(B21_Pr * B21_Vr, axis=1).reshape(
            B21_Pr.shape[0], 1
        )
        factor2 = A_mot * (V_mot + self_interaction_term) * B21_Pr

        factor2_np = self.device.to_numpy(factor2)
        Fact += self.device.to_tensor(B21.T.dot(factor2_np)) * self_int

        # Extract x and y components
        if self.device.torch_available:
            Fax = Fact[:, 0].reshape(Ntot, 1)
            Fay = Fact[:, 1].reshape(Ntot, 1)
        else:
            Fax = Fact[:, 0].reshape(Ntot, 1)
            Fay = Fact[:, 1].reshape(Ntot, 1)

        return Fax, Fay

    def compute_elastic_forces(self, network, xCtr, yCtr, xMend, yMend, xPend, yPend):
        """
        Compute elastic spring forces (GPU-accelerated)

        Returns:
            Fpx, Fpy: elastic forces (on device)
        """
        incid_el = network["incid_el"]

        # Sparse operations on CPU
        xCtr_np = self.device.to_numpy(xCtr)
        yCtr_np = self.device.to_numpy(yCtr)
        xMend_np = self.device.to_numpy(xMend)
        yMend_np = self.device.to_numpy(yMend)
        xPend_np = self.device.to_numpy(xPend)
        yPend_np = self.device.to_numpy(yPend)

        DxM = self.device.to_tensor(incid_el.dot(xMend_np))
        DyM = self.device.to_tensor(incid_el.dot(yMend_np))
        DxP = self.device.to_tensor(incid_el.dot(xPend_np))
        DyP = self.device.to_tensor(incid_el.dot(yPend_np))

        DrM = self.device.sqrt(DxM**2 + DyM**2)
        DrP = self.device.sqrt(DxP**2 + DyP**2)

        kmm, kpp = self.motor.k_mm, self.motor.k_pp
        Lrest = self.motor.l_rest * self.config.net.xi * self.config.net.dens0**-0.5

        # Dense operations on GPU
        exp_M_attr = self.device.exp(-((DrM - 2 * Lrest) ** 2) / (2 * Lrest**2))
        exp_P_attr = self.device.exp(-((DrP - 2 * Lrest) ** 2) / (2 * Lrest**2))
        exp_M_repl = self.device.exp(-((DrM - 1 * Lrest) ** 2) / (1 * Lrest**2))
        exp_P_repl = self.device.exp(-((DrP - 1 * Lrest) ** 2) / (1 * Lrest**2))

        # Attractive forces
        Fx_attr = kmm * exp_M_attr * (DxM / DrM) + kpp * exp_P_attr * (DxP / DrP)
        Fy_attr = kmm * exp_M_attr * (DyM / DrM) + kpp * exp_P_attr * (DyP / DrP)

        # Repulsive forces
        Fx_repl = 10 * kmm * exp_M_repl * (DxM / DrM) + 10 * kpp * exp_P_repl * (
            DxP / DrP
        )
        Fy_repl = 10 * kmm * exp_M_repl * (DyM / DrM) + 10 * kpp * exp_P_repl * (
            DyP / DrP
        )

        # Sparse transpose operations on CPU
        Fx_attr_np = self.device.to_numpy(Fx_attr)
        Fy_attr_np = self.device.to_numpy(Fy_attr)
        Fx_repl_np = self.device.to_numpy(Fx_repl)
        Fy_repl_np = self.device.to_numpy(Fy_repl)

        FpxA = self.device.to_tensor(-incid_el.T.dot(Fx_attr_np))
        FpyA = self.device.to_tensor(-incid_el.T.dot(Fy_attr_np))
        FpxR = self.device.to_tensor(incid_el.T.dot(Fx_repl_np))
        FpyR = self.device.to_tensor(incid_el.T.dot(Fy_repl_np))

        Fpx = FpxA + FpxR
        Fpy = FpyA + FpyR

        return Fpx, Fpy

    def compute_drag_forces(self, Vx, Vy, px, py, len_avg):
        """
        Compute viscous drag forces (GPU-accelerated)

        Returns:
            Fdx, Fdy: drag forces (on device)
        """
        mu = self.phys.mu
        diam = self.phys.diam

        # All operations on device
        FparX = -mu * np.pi * (diam**2) * (px * Vx + py * Vy) * px
        FparY = -mu * np.pi * (diam**2) * (px * Vx + py * Vy) * py

        FprpX = -mu * diam * len_avg * (px * Vy - py * Vx) * (-py)
        FprpY = -mu * diam * len_avg * (px * Vy - py * Vx) * px

        Fdx = FparX + FprpX
        Fdy = FparY + FprpY

        return Fdx, Fdy


class ActiveNetworkSimulationCUDA:
    """Main simulation driver with CUDA support"""

    def __init__(self, config, network, use_cuda=True):
        self.config = config
        self.network = network
        self.device = get_device_manager(use_cuda=use_cuda)
        self.force_calc = ForceCalculatorCUDA(config, self.device)

        # Time parameters
        self.dt = config.sim.dt
        self.Nt = config.sim.Nt
        self.tseries = range(self.Nt)

        # Physics parameters
        self.MM = config.phys.M0
        self.II = config.phys.I0

        # State arrays (will be on device)
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
        logger.info("Initializing simulation state on device")

        Ntot = self.network.Ntot
        R_cnts = self.network.R_cnts
        phi_np = self.network.phi

        # Move state to device
        self.phi = self.device.to_tensor(phi_np)
        self.xCtr = self.device.to_tensor(R_cnts[:, 0].reshape((Ntot, 1)))
        self.yCtr = self.device.to_tensor(R_cnts[:, 1].reshape((Ntot, 1)))

        # Initialize velocities on device
        self.Vx = self.device.zeros((Ntot, 1))
        self.Vy = self.device.zeros((Ntot, 1))
        self.av = self.device.zeros((Ntot, 1))

        # Compute orientations
        if self.device.torch_available:
            import torch

            self.px = torch.cos(self.phi)
            self.py = torch.sin(self.phi)
        else:
            self.px = np.cos(self.phi)
            self.py = np.sin(self.phi)

        logger.info(
            f"State initialized on {self.device.device if self.device.torch_available else 'CPU'}"
        )

    def compute_filament_ends(self, lens):
        """Compute filament end positions"""
        if self.device.torch_available:
            import torch

            xMend = self.xCtr - (lens / 2) * torch.cos(self.phi)
            yMend = self.yCtr - (lens / 2) * torch.sin(self.phi)
            xPend = self.xCtr + (lens / 2) * torch.cos(self.phi)
            yPend = self.yCtr + (lens / 2) * torch.sin(self.phi)
        else:
            xMend = self.xCtr - (lens / 2) * np.cos(self.phi)
            yMend = self.yCtr - (lens / 2) * np.sin(self.phi)
            xPend = self.xCtr + (lens / 2) * np.cos(self.phi)
            yPend = self.yCtr + (lens / 2) * np.sin(self.phi)
        return xMend, yMend, xPend, yPend

    def run(self, callback=None):
        """
        Run simulation with CUDA acceleration

        Args:
            callback: function called at each timestep with (tt, self)
        """
        logger.info("Starting CUDA-accelerated simulation")
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

            # Rearrange network connectivity if needed (CPU operation)
            if tt in self.t_rearr:
                logger.debug(f"Rearranging network at t={tt}")
                # Update R_cnts with current positions
                R_cnts[:, 0] = self.device.to_numpy(self.xCtr).ravel()
                R_cnts[:, 1] = self.device.to_numpy(self.yCtr).ravel()
                net_data = self.network.update_connectivity(
                    R_cnts, self.config.net.Rc_0
                )

            # Update filament end positions
            xMend, yMend, xPend, yPend = self.compute_filament_ends(lens)

            # Compute velocities at rod ends
            lev_r = len_avg / 2
            if self.device.torch_available:
                import torch

                Vr = torch.hstack(
                    (
                        self.Vx - lev_r * self.av * self.py,
                        self.Vy + lev_r * self.av * self.px,
                    )
                )
                Pr = torch.hstack((self.px, self.py))
            else:
                Vr = np.hstack(
                    (
                        self.Vx - lev_r * self.av * self.py,
                        self.Vy + lev_r * self.av * self.px,
                    )
                )
                Pr = np.hstack((self.px, self.py))

            # ========== Force Calculation (GPU) ==========
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
            if self.device.torch_available and not isinstance(Fax, int):
                ax = (Fpx + Fax + Fdx) / self.MM
                ay = (Fpy + Fay + Fdy) / self.MM
            else:
                ax = (Fpx + Fax + Fdx) / self.MM if not isinstance(Fpx, int) else 0
                ay = (Fpy + Fay + Fdy) / self.MM if not isinstance(Fpy, int) else 0

            if nois_on:
                noise_x = self.device.randn((Ntot, 1)) * (Temp_t**0.5)
                noise_y = self.device.randn((Ntot, 1)) * (Temp_t**0.5)
                ax += noise_x
                ay += noise_y

            # Update positions and velocities (GPU)
            self.xCtr += self.Vx * self.dt + (ax * self.dt**2) / 2
            self.yCtr += self.Vy * self.dt + (ay * self.dt**2) / 2
            self.Vx += ax * self.dt
            self.Vy += ay * self.dt

            # Callback for visualization (transfers to CPU)
            if callback:
                callback(tt, self, net_data)

        tf = time()
        logger.info(f"Simulation completed in {tf - ti:.2f} seconds")
