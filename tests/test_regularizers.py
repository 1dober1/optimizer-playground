import numpy as np
from src.regularizers import L1, Elastic_Net


def test_l1_prox_soft_thresholding():
    reg = L1(alpha=0.5)
    lr = 0.1
    t = reg.alpha * lr

    w = np.array([0.01, -0.02, 0.05, 0.2, -0.3])
    wp = reg.prox(w, lr)

    assert wp[0] == 0.0
    assert wp[1] == 0.0
    assert wp[2] == 0.0
    assert np.isclose(wp[3], 0.2 - t)
    assert np.isclose(wp[4], -(0.3 - t))


def test_elasticnet_prox_no_l1_ratio_no_change():
    reg = Elastic_Net(alpha=0.5, l1_ratio=0.0)
    w = np.array([0.1, -0.2, 0.0, 3.0])
    wp = reg.prox(w.copy(), lr=0.123)
    assert np.allclose(wp, w)
