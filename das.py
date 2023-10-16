import jax.numpy as jnp
from jax import jit, vmap, checkpoint
from jax.lax import map
from functools import partial


@partial(jit, static_argnums=(3, 4))
def das(iqraw, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):
    """
    Delay-and-sum IQ data according to a given time delay profile.
    @param iqraw   [na, nb, nsamps]  Raw IQ data (baseband)
    @param tA      [na, *pixdims]    Time delays to apply to dimension 0 of iq
    @param tB      [nb, *pixdims]    Time delays to apply to dimension 1 of iq
    @param fs      scalar            Sampling frequency to convert from time to samples
    @param fd      scalar            Demodulation frequency (0 for RF modulated data)
    @param A       [*na_out, na]     Linear combination of dimension 0 of iqraw
    @param B       [*nb_out, nb]     Linear combination of dimension 1 of iqraw
    @param apoA    [na, *pixdims]    Broadcastable apodization on dimension 0 of iq
    @param apoB    [nb, *pixdims]    Broadcastable apodization on dimension 1 of iq
    @param interp  string            Interpolation method to use
    @return iqfoc  [*na_out, *nb_out, *pixel_dims]   Beamformed IQ data

    The tensors A and B specify how to combine the "elements" in dimensions 0 and 1 of
    iqraw via a tensor contraction. If A or B are None, they default to a vector of ones,
    i.e. a simple sum over all elements. If A or B are identity matrices, the result will
    be the delayed-but-not-summed output. A and B can be arbitrary tensors of arbitrary
    size, as long as the inner most dimension matches na or nb, respectively. Another
    alternative use case is for subaperture beamforming.

    Note that via acoustic reciprocity, it does not matter whether a or b correspond to
    the transmit or receive "elements".
    """
    # The default linear combination is to sum all elements.
    if A is None:
        A = jnp.ones((iqraw.shape[0],))
    if B is None:
        B = jnp.ones((iqraw.shape[1],))

    # Choose the interpolating function
    fints = {
        "nearest": interp_nearest,
        "linear": interp_linear,
        "cubic": interp_cubic,
        "lanczos3": lambda x, t: interp_lanczos(x, t, nlobe=3),
        "lanczos5": lambda x, t: interp_lanczos(x, t, nlobe=5),
    }
    fint = fints[interp]

    # Baseband interpolator
    def bbint(iq, t):
        iqfoc = fint(iq, fs * t)
        return iqfoc * jnp.exp(2j * jnp.pi * fd * t)

    # # Delay-and-sum beamforming (vmap inner, vmap outer)
    # # This method uses vmap to push both the inner and outer loops into XLA, which uses
    # # uses more memory, but can take advantage of XLA's parallelization.  However, it is
    # # slower when memory bandwidth is a bottleneck.
    # def das_b(iq_i, tA_i):
    #     return jnp.tensordot(B, vmap(bbint)(iq_i, tA_i + tB) * apoB, (-1, 0))
    # return jnp.tensordot(A, vmap(das_b)(iqraw, tA) * apoA, (-1, 0))

    # Delay-and-sum beamforming (vmap inner, map outer)
    # This method does not vmap the outer loop and thus cannot take advantage of XLA's
    # parallelization. However, it uses less memory and is faster when memory bandwidth
    # is a bottleneck.
    @checkpoint
    def das_b(x):
        iq_i, tA_i = x
        return jnp.tensordot(B, vmap(bbint)(iq_i, tA_i + tB) * apoB, (-1, 0))

    return jnp.tensordot(A, map(das_b, (iqraw, tA)) * apoA, (-1, 0))


def safe_access(x: jnp.ndarray, s):
    """Safe access to array x at indices s.
    @param x: Array to access
    @param s: Indices to access at
    @return: Array of values at indices s
    """
    s = s.astype("int32")
    valid = (s >= 0) & (s < x.size)
    return jnp.where(valid, jnp.where(valid, x[s], 0), 0)


def interp_nearest(x: jnp.ndarray, si: jnp.ndarray):
    """1D nearest neighbor interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    return x[jnp.clip(jnp.round(si), 0, x.shape[0] - 1).astype("int32")]


def interp_linear(x: jnp.ndarray, si: jnp.ndarray):
    """1D linear interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    f, s = jnp.modf(si)  # Extract fractional, integer parts
    x0 = safe_access(x, s + 0)
    x1 = safe_access(x, s + 1)
    return (1 - f) * x0 + f * x1


def interp_cubic(x: jnp.ndarray, si: jnp.ndarray):
    """1D cubic Hermite interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    f, s = jnp.modf(si)  # Extract fractional, integer parts
    # Values
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    # Coefficients
    a0 = 0 + f * (-1 + f * (+2 * f - 1))
    a1 = 2 + f * (+0 + f * (-5 * f + 3))
    a2 = 0 + f * (+1 + f * (+4 * f - 3))
    a3 = 0 + f * (+0 + f * (-1 * f + 1))
    return (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3) / 2


def _lanczos_helper(x, nlobe=3):
    """Lanczos kernel"""
    a = (nlobe + 1) / 2
    return jnp.where(jnp.abs(x) < a, jnp.sinc(x) * jnp.sinc(x / a), 0)


def interp_lanczos(x: jnp.ndarray, si: jnp.ndarray, nlobe=3):
    """Lanczos interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @param nlobe: Number of lobes of the sinc function (e.g., 3 or 5)
    @return: Interpolated signal
    """
    f, s = jnp.modf(si)  # Extract fractional, integer parts
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    a0 = _lanczos_helper(f + 1, nlobe)
    a1 = _lanczos_helper(f + 0, nlobe)
    a2 = _lanczos_helper(f - 1, nlobe)
    a3 = _lanczos_helper(f - 2, nlobe)
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
