import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from das import das


@partial(jit, static_argnums=(3, 4))
def lag_one_coherence(iq, t_tx, t_rx, fs, fd):
    """
    Lag-one coherence of the receive aperture (DOI: 10.1109/TUFFC.2018.2855653).
    The LOC measures the quality of a signal relative to its noise, and can be
    used to select acoustic output.
    """
    iq = jnp.transpose(iq, (1, 0, 2))  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, jnp.eye(iq.shape[0]))  # Get rx channel data
    # Compute the correlation coefficient
    xy = jnp.real(jnp.nansum(rxdata[:-1] * jnp.conj(rxdata[1:]), axis=0))
    xx = jnp.nansum(jnp.abs(rxdata[:-1]) ** 2, axis=0)
    yy = jnp.nansum(jnp.abs(rxdata[1:]) ** 2, axis=0)
    ncc = xy / jnp.sqrt(xx * yy)
    return ncc


@partial(jit, static_argnums=(3, 4))
def coherence_factor(iq, t_tx, t_rx, fs, fd):
    """
    The coherence factor of the receive aperture (DOI: 10.1121/1.410562).
    The CF is a focusing criterion used to measure the amount of aberration in
    an image.
    """
    iq = jnp.transpose(iq, (1, 0, 2))  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, jnp.eye(iq.shape[0]))  # Get rx channel data
    num = jnp.abs(jnp.nansum(rxdata, axis=0))
    den = jnp.nansum(jnp.abs(rxdata), axis=0)
    return num / den


@partial(jit, static_argnums=(3, 4))
def speckle_brightness(iq, t_tx, t_rx, fs, fd):
    """
    The speckle brightness criterion (DOI: 10.1121/1.397889)
    Speckle brightness can be used to measure the focusing quality.
    """
    return jnp.nanmean(jnp.abs(das(iq, t_tx, t_rx, fs, fd)))


@jit
def total_variation(c):
    """
    Total variation of sound speed map in x and z.
    The sound speed map c should be specified as a 2D matrix of size [nx, nz]
    """
    tvx = jnp.nanmean(jnp.square(jnp.diff(c, axis=0)))
    tvz = jnp.nanmean(jnp.square(jnp.diff(c, axis=1)))
    return tvx + tvz


@partial(jit, static_argnums=(3, 4, 5))
def phase_error(iq, t_tx, t_rx, fs, fd, thresh=0.9):
    """
    The phase error between translating transmit and receive apertures.
    This error is closesly related to the "Translated Transmit Apertures" algorithm
    (DOI: 10.1109/58.585209), where translated transmit and receive apertures
    with common midpoint should have perfect speckle correlation by the van
    Cittert Zernike theorem (DOI: 10.1121/1.418235). High correlation will
    result in high-quality phase shift estimates (DOI: 10.1121/10.0000809).
    CUTE also takes a similar approach (DOI: 10.1016/j.ultras.2020.106168),
    but in the angular basis instead of the element basis.
    """
    # Compute the IQ data for given transmit and receive subapertures.
    # The IQ data matrix will look as follows:
    #               (Tx index, Rx index)
    #   A B C    A: (2, 0)   B: (2, 1)   C: (2, 2)
    #   D E F    D: (1, 0)   E: (1, 1)   F: (1, 2)
    #   G H I    G: (0, 0)   H: (0, 1)   I: (0, 2)
    # The diagonals correspond tx/rx pairs with common midpoints, e.g.:
    #   A, E, and I have a midpoint at 1.
    #   D and H have a midpoint at 0.5.
    #   G has a midpoint at 0.
    #   B and F have a midpoint at 1.5.
    #   C has a midpoint at 2.
    #
    # We create tx and rx subapertures of size 2*halfsa+1 elements, with
    # spacing determined by dx. These are made using das_subap.
    nrx, ntx, nsamps = iq.shape
    mask = np.zeros((nrx, ntx))
    halfsa = 8  # Half of a subaperture
    dx = 1  # Subaperture increment
    for diag in range(-halfsa, halfsa + 1):
        mask = mask + jnp.diag(jnp.ones((ntx - abs(diag),)), diag)
    mask = mask[halfsa : mask.shape[0] - halfsa : dx]
    At = mask[::-1]
    Ar = mask
    iqfoc = das(iq, t_tx, t_rx, fs, fd, At, Ar)

    # Now compute the correlation between neighboring pulse-echo signals with
    # common midpoints. If <A,B> is the correlation between A and B, we want
    #   <A, E>, <E, I>, <B, F>, <D, H>. The corners are naturally cut off.
    xy = iqfoc[:-1, :-1] * jnp.conj(iqfoc[+1:, +1:])
    xx = iqfoc[:-1, :-1] * jnp.conj(iqfoc[:-1, :-1])
    yy = iqfoc[+1:, +1:] * jnp.conj(iqfoc[+1:, +1:])
    # Use jax "double where" trick to remove correlations with only one signal
    valid1 = (iqfoc[:-1, :-1] != 0) & (iqfoc[1:, 1:] != 0)
    xy = jnp.where(valid1, jnp.where(valid1, xy, 0), 0)
    xx = jnp.where(valid1, jnp.where(valid1, xx, 0), 0)
    yy = jnp.where(valid1, jnp.where(valid1, yy, 0), 0)
    # Determine where the correlation coefficient is high enough to use
    xy = jnp.sum(xy, axis=-1)  # Sum over kernel
    xx = jnp.sum(xx, axis=-1)  # Sum over kernel
    yy = jnp.sum(yy, axis=-1)  # Sum over kernel
    ccsq = jnp.square(jnp.abs(xy)) / (jnp.abs(xx) * jnp.abs(yy))
    valid2 = ccsq > thresh * thresh
    xy = jnp.where(valid2, jnp.where(valid2, xy, 0), 0)
    # Convert
    xy = xy[::-1]  # Anti-diagonal --> diagonal
    xy = jnp.reshape(xy, (*xy.shape[:2], -1))
    xy = jnp.transpose(xy, (2, 0, 1))  # Place subap dimensions inside
    xy = jnp.triu(xy) + jnp.transpose(jnp.conj(jnp.tril(xy)), (0, 2, 1))
    dphi = jnp.angle(xy)  # Compute the phase shift.
    return dphi
