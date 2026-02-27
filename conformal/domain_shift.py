import torch
import numpy as np
from torch.optim import Adam
from torch.nn import Parameter
from torchcp.classification.scores import THR
from .conformal_methods import *


def empirical_cdf_torch(values, points, weights=None):
    if weights is None:
        weights = torch.ones_like(values) / len(values)


    diff = points.unsqueeze(1) - values.unsqueeze(0) 

    indicator = torch.sigmoid(diff * 100) 

    cdf = torch.matmul(indicator, weights)  

    return cdf

def kde_torch(x, data, weights, bandwidth):
    """Kernel Density Estimation (KDE) in PyTorch."""
    diff = x.unsqueeze(1) - data.unsqueeze(0)  
    normalization = (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi))).to(x.device)  
    kernels = torch.exp(-0.5 * (diff / bandwidth) ** 2) / normalization
    weighted_kernels = kernels * weights
    return torch.sum(weighted_kernels, dim=1)

def TV_loss(p_P, F_P, F_Q_down, F_Q_up, s_eval):
    ds_c = torch.diff(s_eval, prepend=s_eval[:1]) 
    loss = torch.sum(
        p_P * (
            torch.abs(F_P - F_Q_up) +
            torch.abs(F_P - F_Q_down) +
            torch.abs(F_Q_down - F_Q_up)
        ) * ds_c
    )
    return loss


def learn_calibration_weights(
    calib_preds,
    calib_labels,
    test_preds,
    nscore,
    lambda_raps=0.001,
    k_raps=1,
    num_iterations=3000,
    lr=1e-7,
    device=None
):
    if nscore == "weighted_aps":
        score_fn = APS()
    elif nscore == "weighted_lac":
        score_fn = THR(score_type="Identity")
    elif nscore == "weighted_raps":
        score_fn = RAPS(lambda_raps, k_raps)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calib_preds = calib_preds.to(device)
    calib_labels = calib_labels.to(device)
    test_preds = test_preds.to(device)
    with torch.no_grad():
        S_P = score_fn(calib_preds, calib_labels).detach()

    S_P = S_P.reshape(-1).to(device)
    S_P_sorted, _ = torch.sort(S_P)
    with torch.no_grad():
        S_Q_all = score_fn(test_preds)

    s_Q_down = S_Q_all.min(dim=1).values
    s_Q_up   = S_Q_all.max(dim=1).values
    w = Parameter(torch.ones(len(S_P), device=device) / len(S_P))
    optimizer = Adam([w], lr=lr)

    def scott_bandwidth(data):
        n = len(data)
        std = torch.std(data)
        return std * (n ** (-1 / 5))

    bandwidth = scott_bandwidth(S_P_sorted)

    for _ in range(num_iterations):

        w_normalized = torch.nn.functional.softmax(w, dim=0)

        s_eval = S_P_sorted

        p_P = kde_torch(s_eval, S_P_sorted, w_normalized, bandwidth)

        F_P = empirical_cdf_torch(S_P_sorted, s_eval, weights=w_normalized)
        F_Q_down = empirical_cdf_torch(s_Q_down, s_eval)
        F_Q_up   = empirical_cdf_torch(s_Q_up, s_eval)

        loss = TV_loss(p_P, F_P, F_Q_down, F_Q_up, s_eval)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.nn.functional.softmax(w, dim=0).detach()