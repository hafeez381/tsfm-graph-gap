"""The two floors every model must beat (Lec 3 §2.1): last observed value, historical average per slot of the week."""
import numpy as np

H, L = 12, 12
SLOTS = 2016                                                          # 7 days x 288 five-minute slots


def week_slot(ts: np.ndarray) -> np.ndarray:
    """Slot of the week, 0 = Monday 00:00, from unix seconds."""
    dow = (ts // 86400 + 3) % 7                                       # 1970-01-01 was a Thursday
    return (dow * 288 + (ts % 86400) // 300).astype(np.int64)


def assert_train_only_fit(fit_steps: np.ndarray, train_end: int) -> None:
    if fit_steps.size and fit_steps.max() >= train_end:
        raise AssertionError("historical average fitted on steps at or after train_end")


def historical_average(X: np.ndarray, M: np.ndarray, ts: np.ndarray, train_end: int) -> np.ndarray:
    """(2016, N) mean observed speed per slot of the week and sensor, from training steps only.
    A slot with no observed training reading falls back to the sensor's mean observed training speed."""
    steps = np.arange(train_end)
    assert_train_only_fit(steps, train_end)
    slot, x, m = week_slot(ts[steps]), X[steps].astype(np.float64), M[steps].astype(np.float64)
    N = X.shape[1]
    total, count = np.zeros((SLOTS, N)), np.zeros((SLOTS, N))
    np.add.at(total, slot, x * m)
    np.add.at(count, slot, m)
    sensor_mean = (x * m).sum(0) / np.maximum(m.sum(0), 1)
    return np.where(count > 0, total / np.maximum(count, 1), sensor_mean[None, :]).astype(np.float32)


def predict_ha(table: np.ndarray, ts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n, N, H) historical-average forecast for the targets t+1 .. t+H."""
    slots = week_slot(ts[t[:, None] + np.arange(1, H + 1)[None, :]])  # (n, H)
    return table[slots].transpose(0, 2, 1)


def last_value(X: np.ndarray, M: np.ndarray, ts: np.ndarray, t: np.ndarray, ha_table: np.ndarray) -> np.ndarray:
    """(n, N, H): the last observed reading in the 12-step lookback, repeated H times.
    A sensor window with no observed lookback falls back to the historical average."""
    lb = t[:, None] - np.arange(L)[None, :]                           # (n, L), lb[:, 0] = t (most recent)
    m = M[lb].astype(bool)                                            # (n, L, N)
    first = m.argmax(axis=1)                                          # index of the most recent observed step
    last = np.take_along_axis(X[lb], first[:, None, :], axis=1)[:, 0, :]   # (n, N)
    yhat = np.repeat(last[:, :, None], H, axis=2)
    return np.where(m.any(axis=1)[:, :, None], yhat, predict_ha(ha_table, ts, t)).astype(np.float32)
