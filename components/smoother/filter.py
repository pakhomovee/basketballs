"""Kalman filter with missing-observation support."""

import numpy as np

from kalmanlib import KalmanFilter


def filter_signal_with_missing(
    kf: KalmanFilter,
    xs: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward filter pass with missing observations.
    xs: (2, T) observations; use B@apriori when observed[t]=False (prediction-only step).
    observed: (T,) bool, True where we have real observation.
    Returns: filteredSignal, errs, aprSignals, aprErrs (all 4xT or 4x4xT).
    """
    T = xs.shape[1]
    filtered = np.zeros((4, T))
    errs = np.zeros((4, 4, T))
    apr_signals = np.zeros((4, T))
    apr_errs = np.zeros((4, 4, T))

    apost = kf.startMean.copy()
    apost_err = kf.startCov.copy()

    for t in range(T):
        if observed[t]:
            obs = xs[:, t]
        else:
            apriori = kf.A @ apost
            obs = kf.B @ apriori
        apr_sig, apr_err, apost_sig, apost_err, _ = kf.filterStep(apost, apost_err, obs)
        apost = apost_sig
        apost_err = apost_err
        filtered[:, t] = apost
        errs[:, :, t] = apost_err
        apr_signals[:, t] = apr_sig
        apr_errs[:, :, t] = apr_err

    return filtered, errs, apr_signals, apr_errs
