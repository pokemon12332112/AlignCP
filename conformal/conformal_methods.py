import torch
import time

from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import THR
from torchcp.classification.scores.base import BaseScore

import numpy as np

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def weighted_quantile(values, weights, q):
    v = _to_numpy(values).reshape(-1)
    w = _to_numpy(weights).reshape(-1)

    if v.shape[0] != w.shape[0]:
        raise ValueError("values and weights must have same length")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    sw = w.sum()
    if sw <= 0:
        raise ValueError("sum(weights) must be > 0")

    q = float(q)
    q = min(max(q, 0.0), 1.0)

    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted) / sw

    idx = np.searchsorted(cdf, q, side="left")
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])

def weighted_lac(calib_preds, calib_labs, weight_calib, val_preds, alpha):
    score_fn = THR(score_type="Identity")

    t0 = time.time()

    calib_scores = score_fn(calib_preds, calib_labs)

    q_hat = weighted_quantile(
        _to_numpy(calib_scores),
        _to_numpy(weight_calib),
        q=1 - alpha
    )

    conformal_predictor = SplitPredictor(score_fn)
    conformal_predictor.q_hat = q_hat

    t1 = time.time()
    time_fit = t1 - t0

    t2 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    t3 = time.time()
    time_infer = t3 - t2

    return val_pred_sets, time_fit, time_infer


def weighted_aps(calib_preds, calib_labs, weight_calib, val_preds, alpha):
    score_fn = APS()

    t0 = time.time()

    calib_scores = score_fn(calib_preds, calib_labs)
   

    q_hat = weighted_quantile(
        _to_numpy(calib_scores),
        _to_numpy(weight_calib),
        q=1 - alpha
    )

    conformal_predictor = SplitPredictor(score_fn)
    conformal_predictor.q_hat = q_hat

    t1 = time.time()
    time_fit = t1 - t0

    t2 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    t3 = time.time()
    time_infer = t3 - t2

    return val_pred_sets, time_fit, time_infer

def weighted_raps(calib_preds, calib_labs, weight_calib, val_preds, alpha, lambda_raps=None, k_raps=None):
    score_fn = RAPS(lambda_raps, k_raps)

    t0 = time.time()

    calib_scores = score_fn(calib_preds, calib_labs)

    q_hat = weighted_quantile(
        _to_numpy(calib_scores),
        _to_numpy(weight_calib),
        q=1 - alpha
    )

    conformal_predictor = SplitPredictor(score_fn)
    conformal_predictor.q_hat = q_hat

    t1 = time.time()
    time_fit = t1 - t0

    t2 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    t3 = time.time()
    time_infer = t3 - t2

    return val_pred_sets, time_fit, time_infer


def lac(calib_preds, calib_labs, val_preds, alpha):
    conformal_predictor = SplitPredictor(THR(score_type="Identity"))

    time_conf_fit_i_1 = time.time()
    conformal_predictor.calculate_threshold(calib_preds, calib_labs, alpha)
    time_conf_fit_i_2 = time.time()
    time_fit = time_conf_fit_i_2 - time_conf_fit_i_1

    time_conf_infer_i_1 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    time_conf_infer_i_2 = time.time()
    time_infer = time_conf_infer_i_2 - time_conf_infer_i_1

    return val_pred_sets, time_fit, time_infer


def aps(calib_preds, calib_labs, val_preds, alpha):
    conformal_predictor = SplitPredictor(APS())

    time_conf_fit_i_1 = time.time()
    conformal_predictor.calculate_threshold(calib_preds, calib_labs, alpha)
    time_conf_fit_i_2 = time.time()
    time_fit = time_conf_fit_i_2 - time_conf_fit_i_1

    time_conf_infer_i_1 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    time_conf_infer_i_2 = time.time()
    time_infer = time_conf_infer_i_2 - time_conf_infer_i_1

    return val_pred_sets, time_fit, time_infer


def raps(calib_preds, calib_labs, val_preds, alpha, lambda_raps, k_raps):
    assert lambda_raps is not None, 'lambda_raps can not be None.'
    assert k_raps is not None, 'k_raps can not be None.'

    conformal_predictor = SplitPredictor(RAPS(lambda_raps, k_raps))

    time_conf_fit_i_1 = time.time()
    conformal_predictor.calculate_threshold(calib_preds, calib_labs, alpha)
    time_conf_fit_i_2 = time.time()
    time_fit = time_conf_fit_i_2 - time_conf_fit_i_1

    time_conf_infer_i_1 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    time_conf_infer_i_2 = time.time()
    time_infer = time_conf_infer_i_2 - time_conf_infer_i_1

    return val_pred_sets, time_fit, time_infer


def conformal_method(method, calib_preds, calib_labs, val_preds, alpha, weighted_calib = None, lambda_raps=0.001, k_raps=1):
    if method == 'aps':
        return aps(calib_preds, calib_labs, val_preds, alpha)
    elif method == 'raps':
        return raps(calib_preds, calib_labs, val_preds, alpha, lambda_raps, k_raps)
    elif method == 'lac':
        return lac(calib_preds, calib_labs, val_preds, alpha)
    elif method == 'weighted_lac':
        return weighted_lac(calib_preds, calib_labs, weighted_calib, val_preds, alpha)
    elif method == 'weighted_aps':
        return weighted_aps(calib_preds, calib_labs, weighted_calib, val_preds, alpha)
    elif method == 'weighted_raps':
        return weighted_raps(calib_preds, calib_labs, weighted_calib, val_preds, alpha, lambda_raps, k_raps)
    else:
        raise NotImplementedError


class APS(BaseScore):
    def __call__(self, logits, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        if label is None:
            return self._calculate_all_label(logits)
        else:
            return self._calculate_single_label(logits, label)

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


class RAPS(APS):
    def __init__(self, penalty, kreg=0):

        if penalty <= 0:
            raise ValueError("The parameter 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("The parameter 'kreg' must be a nonnegative value.")
        if type(kreg) != int:
            raise TypeError("The parameter 'kreg' must be a integer.")
        super(RAPS, self).__init__()
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores_first_rank = U * ordered[idx] + reg
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
