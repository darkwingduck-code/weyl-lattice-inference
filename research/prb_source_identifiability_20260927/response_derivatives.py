"""Analytic derivatives of the existing three-channel RPA contraction."""
import numpy as np


def contracted_derivatives(response, Q, lr, ls):
    """Return derivatives with respect to g, Ua, lambda_r, lambda_s.

    response is already dressed through the charge/Gamma5 interaction block.
    The two interaction derivatives use d chi / d U_i = chi[:,i] chi[i,:].
    """
    rows = np.array([[1., 0., 0.], [0., 1., lr]])
    cols = np.array([[1., 0.], [0., 1.], [0., ls]])
    derivatives = []
    for channel, scale in ((0, Q**-2), (1, 1.)):
        outer = response[:, :, channel, None] * response[:, None, channel, :]
        derivatives.append(scale*np.einsum("ai,kij,jb->kab", rows, outer, cols))
    dr, ds = np.zeros_like(derivatives[0]), np.zeros_like(derivatives[0])
    dr[:, 1, 0] = response[:, 2, 0]
    dr[:, 1, 1] = response[:, 2, 1] + ls*response[:, 2, 2]
    ds[:, 0, 1] = response[:, 0, 2]
    ds[:, 1, 1] = response[:, 1, 2] + lr*response[:, 2, 2]
    return [*derivatives, dr, ds]
