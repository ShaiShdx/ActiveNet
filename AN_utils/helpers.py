"""
Helper functions for active network simulation
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Configure logging for the simulation

    Args:
        level: logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Logging configured")


def convert_pattern_to_points(img, density, XY_lens, save_converted=False):
    """
    Convert grayscale image pattern to point positions

    Args:
        img: grayscale image array
        density: desired point density
        XY_lens: [Lx, Ly] domain size
        save_converted: save output image

    Returns:
        R_cnts: point positions
        Ntot: number of points
        img_converted: binary image with points
        dens: actual density
    """
    logger.info("Converting image pattern to points")

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    img_gray = img
    if img.ndim >= 3:
        img_gray = np.around(rgb2gray(img))

    Ly, Lx = img_gray.shape

    RegInds = list(np.where(img_gray.ravel() != 0)[0])
    Area = len(RegInds)
    dens = density * np.prod(XY_lens) / Area
    pks = (dens / np.max(img_gray)) * img_gray.ravel()[RegInds]

    img_converted = np.zeros((Lx * Ly, 1))
    img_converted[RegInds] = np.random.binomial(1, pks, Area).reshape((Area, 1))
    img_converted = np.reshape(img_converted, [Ly, Lx])

    y_cnts, x_cnts = np.where(img_converted != 0)
    Ntot = len(x_cnts)

    R_cnts = np.concatenate((x_cnts.reshape(Ntot, 1), y_cnts.reshape(Ntot, 1)), axis=1)
    R_cnts = R_cnts * np.array([1, -1])
    R_cnts -= np.mean(R_cnts, axis=0).astype(int)
    R_cnts = R_cnts.astype(float)
    x_cnts = R_cnts[:, 0]
    y_cnts = R_cnts[:, 1]

    logger.info(f"Converted pattern: {Ntot} points, density={dens:.3f}")

    if save_converted:
        fig = plt.figure(facecolor="w", figsize=(5, 5), dpi=100)
        plt.scatter(x_cnts, y_cnts, color=[0.8, 0.8, 0.8], s=1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().set_facecolor("black")
        plt.xlim([1.2 * np.min(x_cnts), 1.2 * np.max(x_cnts)])
        plt.ylim([1.2 * np.min(y_cnts), 1.2 * np.max(y_cnts)])
        plt.savefig("output_pattern.tiff")
        plt.show()
        logger.info("Saved converted pattern image")

    return R_cnts, Ntot, img_converted, dens


def set_random_seed(seed):
    """
    Set random seeds for reproducibility

    Args:
        seed: random seed value
    """
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def create_output_directory(path="Simulation_Movie"):
    """
    Create output directory for saving frames

    Args:
        path: directory path
    """
    import os

    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created output directory: {path}")
