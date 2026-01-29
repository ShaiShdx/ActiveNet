#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28.01.2026
@author: Shahriar Shadkhoo MacBook Pro
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

# Auto-detect CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def Convert_Pattern(img, density_scale):

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    img_gray = img
    if img.ndim >= 3:
        img_gray = np.around(rgb2gray(img))

    Ly, Lx = img_gray.shape
    N0 = int(Lx * Ly * density_scale)

    # Convert to torch
    img_gray_t = torch.from_numpy(img_gray).to(device)

    # Generate random points on GPU
    points = torch.rand(N0, 2, device=device) * torch.tensor(
        [Ly - 1, Lx - 1], device=device
    )
    pts_rnd = torch.round(points).long()

    # Get intensities
    pts_intensity = img_gray_t[pts_rnd[:, 0], pts_rnd[:, 1]]
    pts_intensity = pts_intensity.reshape(N0, 1) / torch.max(pts_intensity - 1)

    pts_rnd = pts_rnd.float()
    pts_rnd *= pts_intensity

    # Filter points
    ind_final = torch.where(pts_intensity.squeeze() == 1)[0]
    R_cnts = pts_rnd[ind_final]
    R_cnts = torch.flip(R_cnts, [1]) * torch.tensor([1, -1], device=device)
    Ntot = len(R_cnts)

    # Center the points
    x_cnts = R_cnts[:, 0] - torch.mean(R_cnts[:, 0])
    y_cnts = R_cnts[:, 1] - torch.mean(R_cnts[:, 1])

    # Convert back to numpy for plotting
    x_cnts_np = x_cnts.cpu().numpy()
    y_cnts_np = y_cnts.cpu().numpy()
    R_cnts_np = R_cnts.cpu().numpy()

    fig = plt.figure()
    plt.scatter(x_cnts_np, y_cnts_np, s=0.001)
    plt.axis("equal")
    plt.xlim([-Lx / 2, Lx / 2])
    plt.ylim([-Ly / 2, Ly / 2])
    plt.savefig("output_pattern")
    plt.show()

    return R_cnts_np, Ntot


if __name__ == "__main__":
    img = plt.imread("input_pattern")
    [R_cnts, Ntot] = Convert_Pattern(img, density_scale=0.1)
