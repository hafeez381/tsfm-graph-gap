"""Road graph A and the union neighbourhood mask (Decision 2).

Direction convention: A[j, i] is the edge j -> i. tsl builds dist[j, i] from distances_la.csv rows
(from=j, to=i, cost), the road distance driving from sensor j to sensor i, so A[j, i] > 0 means j is
just upstream of i. Checked on sensors 0 and 37: dist[0, 37] = 2,124 m but dist[37, 0] = 8,554 m, and
only A[0, 37] survives the threshold.
"""
import numpy as np


def build(dataset, threshold: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """A (N, N) float32 Gaussian-kernel weights thresholded at `threshold`, no self loops;
    nbr (N, N) bool, nbr[i, j] = j in N(i) = {j : A[j, i] > 0 or A[i, j] > 0} plus i itself (eq. 3)."""
    A = dataset.get_connectivity(method="distance", threshold=threshold, include_self=False,
                                 layout="dense").astype(np.float32)
    nbr = (A > 0) | (A.T > 0) | np.eye(A.shape[0], dtype=bool)
    assert (np.diag(A) == 0).all() and (nbr == nbr.T).all()
    return A, nbr
