#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28.01.2026
@author: Shahriar Shadkhoo MacBook Pro
"""

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from copy import deepcopy
import os
from tqdm import tqdm
from time import time

# Auto-detect CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def main(shape_img=None):

    plot_centers = False
    plot_quivers = True

    # ["Circle", "Ellipse","Hexagon", "Rectangle", "Triangle", "Hexagram"]
    shape_geometry = "Hexagon"

    Lx, Ly = 40, 40
    AR_x, AR_y, AR_fr = 1, 0.6, 0.8

    Temp = 0.1
    gamma = 200

    xi = 5  # <L>/density
    linkact = 1  # must be between 0 and 1
    link_fr = 1  # link fractions
    densact = 1

    T_tot = 200
    dt = 0.1
    tseries = range(round(T_tot / dt))
    TauLink = T_tot / 10  # activation/binding timescale of links
    N_t = len(tseries)

    diam = 1
    K_Link = 10
    l_rest = 0.05

    if shape_img:
        [R_cnts, Ntot] = Convert_Pattern(shape_img, density_scale=0.1)
        # Convert to torch
        R_cnts = torch.from_numpy(R_cnts).float().to(device)
    else:
        Global_Geometry = shape_geometry
        dens1 = 50
        dens2 = dens1 * densact
        dens = np.sqrt(dens1 * dens2)
        len_avg = xi * (dens) ** (-1 / 2)
        len_std = len_avg / 10

        N1 = round(dens1 * Lx / 2 * Ly)
        N2 = 0
        Ntot = N1 + N2

        # Generate on GPU
        R1_cnts = torch.rand(N1, 2, device=device) * torch.tensor(
            [Lx, Ly], device=device
        ) - torch.tensor([Lx / 2, Ly / 2], device=device)
        R2_cnts = torch.rand(N2, 2, device=device) * torch.tensor(
            [Lx / 2, Ly / 2], device=device
        ) - torch.tensor([0, Ly / 4], device=device)
        R_cnts = torch.cat((R1_cnts, R2_cnts), dim=0)

        if Global_Geometry == "Circle" or Global_Geometry == "Ellipse":
            size = torch.max(R_cnts, dim=0)[0] - torch.min(R_cnts, dim=0)[0]
            r0 = (torch.min(size) / 2) * AR_fr
            corners = []
            cent = Point([0, 0])
            shape = cent.buffer(r0.cpu().item())

        elif Global_Geometry == "Hexagon":
            size = torch.max(R_cnts, dim=0)[0] - torch.min(R_cnts, dim=0)[0]
            side_len = (torch.min(size) * np.sqrt(3) / 4) * AR_fr
            side_len_val = side_len.cpu().item()
            corners = [
                (-side_len_val, 0),
                (-side_len_val / 2, +side_len_val * np.sqrt(3) / 2),
                (+side_len_val / 2, +side_len_val * np.sqrt(3) / 2),
                (+side_len_val, 0),
                (+side_len_val / 2, -side_len_val * np.sqrt(3) / 2),
                (-side_len_val / 2, -side_len_val * np.sqrt(3) / 2),
            ]
            shape = Polygon(corners)

        elif Global_Geometry == "Rectangle":
            size = torch.max(R_cnts, dim=0)[0] - torch.min(R_cnts, dim=0)[0]
            side_len_x = (torch.min(size) * AR_fr * AR_x).cpu().item()
            side_len_y = (torch.min(size) * AR_fr * AR_y).cpu().item()
            corners = [
                (-side_len_x / 2, -side_len_y / 2),
                (-side_len_x / 2, +side_len_y / 2),
                (+side_len_x / 2, +side_len_y / 2),
                (+side_len_x / 2, -side_len_y / 2),
            ]
            shape = Polygon(corners)

        elif Global_Geometry == "Triangle":
            size = (torch.max(R_cnts, dim=0)[0] - torch.min(R_cnts, dim=0)[0]) * AR_fr
            H = torch.min(size).cpu().item()
            side_len = H * 2 / np.sqrt(3)
            cent_coord = (0, side_len * (1 - np.sqrt(3)) / 4)
            corners = [
                (-side_len / 2, -H / 2 - cent_coord[1]),
                (0, H / 2 - cent_coord[1]),
                (+side_len / 2, -H / 2 - cent_coord[1]),
            ]
            shape = Polygon(corners)

        elif Global_Geometry == "Hexagram":
            size = (
                (1 / 2)
                * (torch.max(R_cnts, dim=0)[0] - torch.min(R_cnts, dim=0)[0])
                * AR_fr
            )
            H = torch.min(size).cpu().item()
            h = H * np.sqrt(3) / 3
            angs = np.asarray(list(range(0, 360, 30))) * np.pi / 180
            dists = np.asarray([h, H, h, H, h, H, h, H, h, H, h, H])
            Xhex = dists * np.cos(angs)
            Yhex = dists * np.sin(angs)
            hex_verts = list(zip(Xhex, Yhex))
            corners = hex_verts
            shape = Polygon(corners)

        # Filter points (need CPU for shapely)
        R_cnts_cpu = R_cnts.cpu().numpy()
        pts_outside = [
            _
            for _ in range(Ntot)
            if (not shape.contains(Point(R_cnts_cpu[_].tolist())))
        ]
        pts_inside = list(set([_ for _ in range(Ntot)]) - set(pts_outside))

        # Keep only inside points, convert back to GPU
        R_cnts = (
            torch.from_numpy(np.delete(R_cnts_cpu, pts_outside, axis=0))
            .float()
            .to(device)
        )
        Ntot = len(R_cnts)

    # Generate lengths and orientations on GPU
    lens = torch.abs(torch.randn(Ntot, 1, device=device) * len_std + len_avg)
    phi = torch.rand(Ntot, 1, device=device) * 2 * np.pi
    px = lens * torch.cos(phi)
    py = lens * torch.sin(phi)
    II = gamma * lens / 1200

    xcnt = R_cnts[:, 0].reshape((Ntot, 1))
    ycnt = R_cnts[:, 1].reshape((Ntot, 1))
    xend0 = xcnt - (lens / 2) * torch.cos(phi)
    yend0 = ycnt - (lens / 2) * torch.sin(phi)

    # Compute plot limits
    margin = 1.5
    xmin = int(round(margin * torch.min(xend0).cpu().item()))
    xmax = int(round(margin * torch.max(xend0).cpu().item()))
    ymin = int(round(margin * torch.min(yend0).cpu().item()))
    ymax = int(round(margin * torch.max(yend0).cpu().item()))
    XLIM, YLIM = [xmin, xmax], [ymin, ymax]

    # Compute connectivity (sparse matrices stay on CPU)
    exp_lens = torch.exp(lens)
    exp_sumLens = torch.kron(exp_lens, exp_lens.T)
    sum_lens = torch.log(exp_sumLens)
    sum_lens_triu = sum_lens[torch.triu_indices(Ntot, Ntot, offset=1).unbind()]

    # Need CPU for scipy operations
    R_cnts_cpu = R_cnts.cpu().numpy()
    ds = pdist(R_cnts_cpu, "euclidean")
    ds_len = ds - sum_lens_triu.cpu().numpy() / 2
    ds_len[ds_len > 0] = 0
    ds_len[ds_len != 0] = 1
    ds_len_sym = csr_matrix(squareform(ds_len) * np.triu(np.ones((Ntot, Ntot)), 1))
    ds_len_sym += ds_len_sym.T

    Act = linkact * torch.ones((Ntot, 1), device=device)

    # Sparse operations on CPU, convert results to GPU
    Degrees = np.sum(ds_len_sym, axis=0)
    Delta = deepcopy(-ds_len_sym)
    Delta_Diag = np.zeros((Ntot, Ntot))
    np.fill_diagonal(Delta_Diag, Degrees.tolist()[0])
    Delta += Delta_Diag
    Diag = torch.from_numpy(np.asarray(Degrees).T).float().to(device)

    # Keep Delta as sparse on CPU for efficiency
    Delta_sparse = csr_matrix(np.asarray(Delta) * Act.cpu().numpy())

    # Helper function for sparse-dense multiplication
    def sparse_dot(sparse_mat, dense_tensor):
        """Multiply sparse CPU matrix with dense GPU tensor"""
        dense_cpu = dense_tensor.cpu().numpy()
        result_cpu = sparse_mat.dot(dense_cpu)
        return torch.from_numpy(result_cpu).float().to(device)

    xpos = torch.where(xcnt > 0)[0]
    xneg = torch.where(xcnt < 0)[0]
    Nm1 = len(xpos)
    Nm2 = len(xneg)

    km = torch.zeros((Ntot, 1), device=device)

    N_frame = np.min((10, int(T_tot / dt)))
    if (plot_centers or plot_quivers) and N_frame != 0:
        tplot = np.around(np.linspace(0, N_t, N_frame)).astype(int)

    for tt in tqdm(tseries):

        link_fr_t = link_fr * (1 - np.exp(-tt * dt / TauLink))

        xend = xcnt - (lens / 2) * torch.cos(phi)
        yend = ycnt - (lens / 2) * torch.sin(phi)

        # Sparse operations
        Dx = sparse_dot(Delta_sparse, xend)
        Dy = sparse_dot(Delta_sparse, yend)

        # Random binomial on GPU
        km1_samples = torch.rand(Nm1, 1, device=device) < (linkact * link_fr_t)
        km2_samples = torch.rand(Nm2, 1, device=device) < link_fr_t
        km1 = K_Link * km1_samples.float()
        km2 = K_Link * km2_samples.float()
        km[xpos] = km1
        km[xneg] = km2

        # Force calculations on GPU
        Dr = torch.sqrt(Dx**2 + Dy**2)
        Fx = -km * (torch.abs(Dx) - l_rest * Diag) * (Dx / Dr)
        Fy = -km * (torch.abs(Dy) - l_rest * Diag) * (Dy / Dr)
        Tau = -torch.sin(sparse_dot(Delta_sparse, phi))

        # Noise on GPU
        noise_vx = torch.randn(Ntot, 1, device=device) * Temp
        noise_vy = torch.randn(Ntot, 1, device=device) * Temp
        noise_phi = torch.randn(Ntot, 1, device=device) * Temp * (2 * np.pi / 180)

        # Gamma calculations on GPU
        gamma_x = gamma * (
            torch.cos(phi) ** 2 + lens * (torch.sin(phi) ** 2) / (np.pi * diam)
        )
        gamma_y = gamma * (
            torch.sin(phi) ** 2 + lens * (torch.cos(phi) ** 2) / (np.pi * diam)
        )

        # Update velocities on GPU
        Vx = Fx / gamma_x + noise_vx
        Vy = Fy / gamma_y + noise_vy
        Ls = Tau / II + noise_phi

        # Update positions on GPU
        xcnt += Vx * dt
        ycnt += Vy * dt
        phi += Ls * dt

        # Plotting (transfer to CPU)
        if plot_quivers:
            if tt in tplot or tt == N_t - 1:
                xcnt_cpu = xcnt.cpu().numpy()
                ycnt_cpu = ycnt.cpu().numpy()
                px_cpu = px.cpu().numpy()
                py_cpu = py.cpu().numpy()

                fig = plt.figure(facecolor="black", figsize=(5, 5), dpi=100)
                plt.quiver(
                    xcnt_cpu,
                    ycnt_cpu,
                    px_cpu,
                    py_cpu,
                    color=[0.8, 0.8, 0.8],
                    scale_units="xy",
                    scale=1,
                    headwidth=5,
                    pivot="mid",
                )
                plt.gca().set_aspect("equal", adjustable="box")
                plt.gca().set_facecolor("black")
                plt.xlim(XLIM)
                plt.ylim(YLIM)
                plt.axis("off")
                plt.pause(0.001)

        if plot_centers:
            if tt in tplot or tt == N_t - 1:
                xcnt_cpu = xcnt.cpu().numpy()
                ycnt_cpu = ycnt.cpu().numpy()

                fig = plt.figure(facecolor="black", figsize=(5, 5), dpi=100)
                plt.scatter(xcnt_cpu, ycnt_cpu, color=[0.8, 0.8, 0.8], s=20.0)
                plt.gca().set_aspect("equal", adjustable="box")
                plt.gca().set_facecolor("black")
                plt.xlim(XLIM)
                plt.ylim(YLIM)
                plt.axis("off")
                plt.pause(0.001)

    plt.show()
    print(f"Total Number of Filaments = {Ntot} \n")


if __name__ == "__main__":

    time_init = time()

    shape_img = None
    path_to_image = "input_pattern.png"

    if os.path.exists(path_to_image):
        print("File 'input_pattern' not found")
        try:
            from shape_conversion import Convert_Pattern

            shape_img = plt.imread(path_to_image)

        except ImportError:
            print("shape_conversion not found")
    else:
        print("Image file not found")

    main(shape_img)

    print(f"\n Total Time: {time() - time_init}")
