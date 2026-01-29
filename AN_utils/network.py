"""
Network generation and topology module
Handles lattice creation and network connectivity
"""

import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
import networkx as nx

logger = logging.getLogger(__name__)


class LatticeGenerator:
    """Generate lattice points with different unit cells"""

    @staticmethod
    def generate(XY_lens, dens, disorder=1.0, unit_cell="square", spdim=2):
        """
        Generate lattice points

        Args:
            XY_lens: [Lx, Ly] system dimensions
            dens: point density
            disorder: positional disorder magnitude
            unit_cell: 'square' or 'triangular'
            spdim: spatial dimension

        Returns:
            R_cnts: positions as numpy array
            Ntot: number of points
        """
        logger.info(f"Generating {unit_cell} lattice: L={XY_lens}, dens={dens}")

        nx_pts = int(round(XY_lens[0] * dens**0.5))
        ny_pts = int(round(XY_lens[1] * dens**0.5))

        if unit_cell == "square":
            R_lst = [[i, j] for i in range(nx_pts) for j in range(ny_pts)]
        else:
            # Triangular lattice
            R_lst_0 = [
                [(1 + (-1) ** (j + 1)) / 4, j * 3**0.5 / 2] for j in range(1, ny_pts, 2)
            ]
            R_lst_1 = [
                [i + (1 + (-1) ** (j + 1)) / 4, j * 3**0.5 / 2]
                for i in range(1, nx_pts)
                for j in range(ny_pts)
            ]
            R_lst = R_lst_0 + R_lst_1

        R_cnts = np.asarray(R_lst).astype(float)
        R_cnts -= np.mean(R_cnts, axis=0)
        R_cnts *= np.asarray([dens**-0.5, dens**-0.5])

        Ntot = len(R_cnts)
        R_cnts += np.random.uniform(low=-disorder, high=+disorder, size=(Ntot, spdim))

        logger.info(f"Generated {Ntot} lattice points")
        return R_cnts, Ntot


class NetworkTopology:
    """Manage network connectivity and motor assignments"""

    def __init__(self, config):
        self.config = config
        self.motor_cfg = config.motor
        self.net_cfg = config.net

    def build_connectivity(self, R_cnts, SumLens, Rcutoff):
        """
        Build network connectivity based on distance cutoff

        Args:
            R_cnts: particle positions (N, 2)
            SumLens: sum of filament lengths
            Rcutoff: cutoff radius

        Returns:
            Dictionary with network operators and properties
        """
        NN = len(R_cnts)
        logger.info(f"Building connectivity for {NN} particles, Rc={Rcutoff:.3f}")

        # Compute pairwise distances
        ds = pdist(R_cnts, "euclidean")
        ds_rad = np.heaviside(-ds + SumLens * Rcutoff, 0).astype(int)

        # Limit motors per filament if specified
        mpf = self.net_cfg.mpf
        if mpf:
            ds_vals = ds * ds_rad
            ds_vals_mat = squareform(ds_vals) * np.triu(np.ones((NN, NN)), 1)
            ds_vals_sym = ds_vals_mat + ds_vals_mat.T
            inds = np.argsort(ds_vals_sym, axis=1)[:, ::-1][:, :mpf]

            ds_sq = np.zeros((NN, NN))
            for vv in range(NN):
                nz_inds = inds[vv][ds_vals_sym[vv][inds[vv]] != 0]
                ds_sq[vv][nz_inds] = 1
            ds_sq += ds_sq.T
            ds_sq[ds_sq != 0] = 1
            ds_sq = np.triu(ds_sq, 1)
        else:
            ds_sq = squareform(ds_rad) * np.triu(np.ones((NN, NN)), 1)

        # Build incidence matrices
        v1 = np.where(ds_sq != 0)[0]
        v2 = np.where(ds_sq != 0)[1]
        N_bonds = len(v1)

        logger.info(f"Network has {N_bonds} bonds")

        B12 = np.zeros((2 * N_bonds, NN))
        B12[np.arange(N_bonds), v2] = 1
        B12[np.arange(N_bonds, 2 * N_bonds), v1] = 1

        B21 = np.vstack((B12[N_bonds : 2 * N_bonds], B12[0:N_bonds]))
        B12 = csr_matrix(B12)
        B21 = csr_matrix(B21)
        Bas = B21 - B12

        # Assign motor types
        edges = list(zip(R_cnts[v1], R_cnts[v2]))

        # Get motor type arrays (already numpy)
        ccs_np = self.motor_cfg.ccs
        V_type_np = self.motor_cfg.V_type
        A_type_np = self.motor_cfg.A_type

        # Assign motor types using numpy multinomial
        MotorType = np.where(np.random.multinomial(1, ccs_np, 2 * N_bonds))[1]

        V_mot = V_type_np[MotorType].reshape((2 * N_bonds, 1))
        A_mot = A_type_np[MotorType].reshape((2 * N_bonds, 1))

        # Create NetworkX graph
        G = nx.Graph()
        ed_list = list(zip(v1, v2))
        G.add_edges_from(ed_list)

        logger.debug(f"Motor distribution: {np.bincount(MotorType)}")

        return {
            "B12": B12,
            "B21": B21,
            "Bas": Bas,
            "V_mot": V_mot,
            "A_mot": A_mot,
            "edges": edges,
            "graph": G,
            "N_bonds": N_bonds,
            "incid_el": Bas[0:N_bonds],
        }


class FilamentNetwork:
    """Represents the complete filament network state"""

    def __init__(self, config):
        self.config = config
        self.lattice_gen = LatticeGenerator()
        self.topology = NetworkTopology(config)

        # State variables (will be initialized)
        self.R_cnts = None
        self.Ntot = None
        self.phi = None
        self.lens = None
        self.network = None

    def initialize(self):
        """Initialize network geometry and topology"""
        logger.info("Initializing filament network")

        # Generate lattice
        net_cfg = self.config.net
        self.R_cnts, self.Ntot = self.lattice_gen.generate(
            net_cfg.XY_lens,
            net_cfg.dens0,
            disorder=net_cfg.disorder,
            unit_cell=net_cfg.unit_cell,
        )

        # Initialize orientations
        phi_range = net_cfg.phi_range
        self.phi = (
            np.random.uniform(phi_range[0], phi_range[1], (self.Ntot, 1))
            + net_cfg.phi_avg
        )

        # Initialize lengths
        len_avg = net_cfg.xi * net_cfg.dens0**-0.5
        self.lens = len_avg

        # Build network connectivity
        SumLens = 2 * self.lens
        self.network = self.topology.build_connectivity(
            self.R_cnts, SumLens, net_cfg.Rc_0
        )

        logger.info(
            f"Network initialized: {self.Ntot} filaments, {self.network['N_bonds']} bonds"
        )

        return self

    def update_connectivity(self, R_cnts, Rcutoff):
        """Rebuild network connectivity (for rearrangement)"""
        logger.debug("Updating network connectivity")
        SumLens = 2 * self.lens
        self.network = self.topology.build_connectivity(R_cnts, SumLens, Rcutoff)
        return self.network
