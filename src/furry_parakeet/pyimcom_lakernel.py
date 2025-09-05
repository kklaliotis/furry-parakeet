"""
Linear algebra kernel for PyIMCOM.

Functions
---------
BruteForceKernel
    Brute force version of the kernel. Slow and useful only for comparisons.
CKernel
    PyIMCOM kernel wrapped around C routines.
CKernelMulti
    PyIMCOM kernel, generating multiple images.
get_coadd_matrix_discrete
    PyIMCOM kernel, but built out of a basis of discrete choices in the Lagrange multiplier.
test_get_coadd_matrix_discrete
    Test function for get_coadd_matrix_discrete.
testkernel
    Test case for the kernel.
testinterp
    Test interpolation functions.

"""

import time
import warnings

import numpy as np
import scipy
from astropy.io import fits
from scipy.linalg import LinAlgError

from . import pyimcom_croutines


def BruteForceKernel(A, mBhalf, C, targetleak, kCmin=1e-16, kCmax=1e16, nbis=53):
    """
    Brute force version of the kernel. Slow and useful only for comparisons.

    Implements PyIMCOM operations with a for loop (hence slow).

    Parameters
    ----------
    A : np.array
        System matrix, shape = (n, n)
    mBhalf : np.array
        The matrix -0.5*B: target overlap matrix, shape = (m, n)
    C : float
        Target normalization.
    targetleak : float
        Allowable leakage of target PSF.
    kCmin : float, optional
        Minimum value of kappa/C to test.
    kCmax : float, optional
        Maximum value of kappa/C to test.
    nbis : int, optional
        Number of bisections to use in search for the best kappa.
        (These are done in log kappa space.)

    Returns
    -------
    kappa : np.array
        Lagrange multiplier per output pixel, shape=(m,)
    Sigma : np.array
        Output noise amplification, shape=(m,)
    UC : np.array
        Fractional squared error in PSF, shape=(m,)
    T : np.array
        Coaddition matrix, shape=(m, n)

    """

    # get dimensions
    (m, n) = np.shape(mBhalf)

    # eigensystem
    lam, Q = np.linalg.eigh(A)
    # -P/2 matrix
    mPhalf = mBhalf @ Q

    # allocate targets
    kappa = np.zeros((m,))
    T = np.zeros((m, n))
    UC = np.zeros((m,))
    Sigma = np.zeros((m,))

    # now loop over pixels
    for a in range(m):
        factor = np.sqrt(kCmax / kCmin)
        kappa[a] = np.sqrt(kCmax * kCmin)
        for ibis in range(nbis + 1):
            factor = np.sqrt(factor)
            UC[a] = 1 - np.sum((lam + 2 * kappa[a]) / (lam + kappa[a]) ** 2 * mPhalf[a, :] ** 2) / C
            if ibis != nbis:
                if UC[a] > targetleak:
                    kappa[a] /= factor
                else:
                    kappa[a] *= factor
        T[a, :] = Q @ (mPhalf[a, :] / (lam + kappa[a]))
        Sigma[a] = np.sum((mPhalf[a, :] / (lam + kappa[a])) ** 2)

    return (kappa, Sigma, UC, T)


def CKernel(A, mBhalf, C, targetleak, kCmin=1e-16, kCmax=1e16, nbis=53):
    """
    PyIMCOM kernel wrapped around C routines.


    Parameters
    ----------
    A : np.array
        System matrix, shape = (n, n)
    mBhalf : np.array
        The matrix -0.5*B: target overlap matrix, shape = (m, n)
    C : float
        Target normalization.
    targetleak : float
        Allowable leakage of target PSF.
    kCmin : float, optional
        Minimum value of kappa/C to test.
    kCmax : float, optional
        Maximum value of kappa/C to test.
    nbis : int, optional
        Number of bisections to use in search for the best kappa.
        (These are done in log kappa space.)

    Returns
    -------
    kappa : np.array
        Lagrange multiplier per output pixel, shape=(m,)
    Sigma : np.array
        Output noise amplification, shape=(m,)
    UC : np.array
        Fractional squared error in PSF, shape=(m,)
    T : np.array
        Coaddition matrix, shape=(m, n)

    """

    # get dimensions
    (m, n) = np.shape(mBhalf)

    # eigensystem
    lam, Q = np.linalg.eigh(A)
    # -P/2 matrix
    mPhalf = mBhalf @ Q

    # output arrays
    kappa = np.zeros((m,))
    Sigma = np.zeros((m,))
    UC = np.zeros((m,))
    tt = np.zeros((m, n))

    pyimcom_croutines.lakernel1(lam, Q, mPhalf, C, targetleak, kCmin, kCmax, nbis, kappa, Sigma, UC, tt, 1e49)
    T = tt @ Q.T
    return (kappa, Sigma, UC, T)


def CKernelMulti(A, mBhalf, C, targetleak, kCmin=1e-16, kCmax=1e16, nbis=53, smax=1e8):
    """
    PyIMCOM kernel, generating multiple images.

    There are ``nt`` target PSFs. If `mBhalf` is a 2D array, assumes nt=1.

    Parameters
    ----------
    A : np.array
        System matrix, shape = (n, n)
    mBhalf : np.array
        The matrix -0.5*B: target overlap matrix, shape = (nt, m, n)
    C : np.array
        Target normalization, shape = (nt,)
    targetleak : np.array
        Allowable leakage of target PSF, shape = (nt,)
    kCmin : float, optional
        Minimum value of kappa/C to test.
    kCmax : float, optional
        Maximum value of kappa/C to test.
    nbis : int, optional
        Number of bisections to use in search for the best kappa.
        (These are done in log kappa space.)
    smax : float, optional
        Maximum allowed noise metric Sigma.

    Returns
    -------
    kappa : np.array
        Lagrange multiplier per output pixel, shape = (nt, m)
    Sigma : np.array
        Output noise amplification, shape = (nt, m)
    UC : np.array
        Fractional squared error in PSF, shape = (nt, m)
    T : np.array
        Coaddition matrix, shape = (nt, m, n)

    See Also
    --------
    CKernel : Similar but with only one target PSF.

    """

    # eigensystem
    lam, Q = np.linalg.eigh(A)

    # get dimensions and mPhalf matrix
    if mBhalf.ndim == 2:
        nt = 1
        (m, n) = np.shape(mBhalf)
        mBhalf_image = mBhalf.reshape((1, m, n))
        C_s = np.array([C])
        targetleak_s = np.array([targetleak])
    else:
        (nt, m, n) = np.shape(mBhalf)
        mBhalf_image = mBhalf
        C_s = C
        targetleak_s = targetleak

    # output arrays
    kappa = np.zeros((nt, m))
    Sigma = np.zeros((nt, m))
    UC = np.zeros((nt, m))
    T = np.zeros((nt, m, n))
    tt = np.zeros((m, n))

    for k in range(nt):
        pyimcom_croutines.lakernel1(
            lam,
            Q,
            mBhalf_image[k, :, :] @ Q,
            C_s[k],
            targetleak_s[k],
            kCmin,
            kCmax,
            nbis,
            kappa[k, :],
            Sigma[k, :],
            UC[k, :],
            tt,
            smax,
        )
        T[k, :, :] = tt @ Q.T
    return (kappa, Sigma, UC, T)


####
# alternative to CKernelMulti, almost same functionality but has a range of kappa
####


def get_coadd_matrix_discrete(A, mBhalf, C, kappa_array, ucmin, smax=0.5):
    """
    PyIMCOM kernel, but built out of a basis of discrete choices in the Lagrange multiplier.

    Parameters
    ----------
    A : np.array
        System matrix, shape = (n, n)
    mBhalf : np.array
        The matrix -0.5*B: target overlap matrix, shape = (n_out, m, n)
    C : np.array
        Target normalization, shape = (n_out,)
    kappa_array : np.array
        Lagrange multiplier nodes in ascending order, shape = (nv,)
    ucmin : float
        Target leakage U/C
    smax : float, optional
        Maximum allowed noise metric Sigma.

    Returns
    -------
    kappa : np.array
        Lagrange multiplier per output pixel, shape = (n_out, m)
    Sigma : np.array
        Output noise amplification, shape = (n_out, m)
    UC : np.array
        Fractional squared error in PSF, shape = (n_out, m)
    T : np.array
        Coaddition matrix, shape = (n_out, m, n)

    """

    # get parameters
    (n_out, m, n) = np.shape(mBhalf)
    nv = np.size(kappa_array)

    # Cholesky decompositions for each eigenvalue node
    L = np.zeros((nv, n, n))
    di = np.diag_indices(n)
    AA = np.copy(A)
    for j in range(nv):
        if j > 0:
            AA[di] += kappa_array[j] - kappa_array[j - 1]
        else:
            AA[di] += kappa_array[0]
        try:
            L[j, :, :] = scipy.linalg.cholesky(AA, lower=True, check_finite=False)
        except LinAlgError:
            # if matrix is not quite positive definite, we can rectify it
            w, v = np.linalg.eigh(A)
            AA[di] += kappa_array[j] + np.abs(w[0])
            del v
            warnings.warn(
                f"Warning: pyimcom_lakernel Cholesky decomposition failed; "
                f"fixed negative eigenvalue {w[0]:19.12e}"
            )
            L[j, :, :] = scipy.linalg.cholesky(AA, lower=True, check_finite=False)
    del AA
    del di

    # outputs
    T_ = np.zeros((n_out, m, n), dtype=np.float32)
    UC_ = np.zeros((n_out, m), dtype=np.float32)
    S_ = np.zeros((n_out, m), dtype=np.float32)
    k_ = np.zeros((n_out, m), dtype=np.float32)

    Tpi = np.zeros((nv, m, n))
    #
    # loop over output PSFs
    for j_out in range(n_out):
        # build values at nodes
        for p in range(nv):
            Tpi[p, :, :] = scipy.linalg.cho_solve(
                (L[p, :, :], True), mBhalf[j_out, :, :].T, check_finite=False
            ).T
        Dp = np.einsum("ai,pai->ap", mBhalf[j_out, :, :], Tpi)
        Npq = np.einsum("pai,qai->apq", Tpi, Tpi)
        Ep = np.einsum("qai,ai->aq", Tpi, mBhalf[j_out, :, :])
        Epq = np.zeros((m, nv, nv))
        for p in range(nv):
            for q in range(p):
                Epq[:, q, p] = Epq[:, p, q] = Ep[:, q] - kappa_array[p] * Npq[:, p, q]
            Epq[:, p, p] = Ep[:, p] - kappa_array[p] * Npq[:, p, p]

        # now make outputs and call C function
        out_kappa = np.zeros((m,))
        out_Sigma = np.zeros((m,))
        out_UC = np.zeros((m,))
        out_w = np.zeros((m * nv,))
        pyimcom_croutines.build_reduced_T_wrap(
            Npq.flatten(),
            Dp.flatten() / C[j_out],
            Epq.flatten() / C[j_out],
            kappa_array / C[j_out],
            ucmin,
            smax,
            out_kappa,
            out_Sigma,
            out_UC,
            out_w,
        )

        # make outputs
        k_[j_out, :] = out_kappa * C[j_out]
        S_[j_out, :] = out_Sigma
        UC_[j_out, :] = out_UC
        T_[j_out, :, :] = np.einsum("pai,ap->ai", Tpi, out_w.reshape((m, nv)))

    return (k_, S_, UC_, T_)


# test circuit


def test_get_coadd_matrix_discrete():
    """Test function for get_coadd_matrix_discrete."""

    # number of outputs to print
    # npr = 4

    # number of layers to test multi-ouptut
    nt = 3

    sigma = 1.75
    # u = numpy.array([0.2, 0.1])

    # test grid: interpolate an m1xm1 image from n1xn1
    m1 = 50
    n1 = 80
    n = n1 * n1
    m = m1 * m1

    x = np.zeros((n,))
    y = np.zeros((n,))
    for i in range(n1):
        y[n1 * i : n1 * i + n1] = i
        x[i::n1] = i
    xout = np.zeros((m,))
    yout = np.zeros((m,))
    for i in range(m1):
        yout[m1 * i : m1 * i + m1] = 5 + 0.25 * i
        xout[i::m1] = 5 + 0.25 * i

    # make sample image
    # thisImage = numpy.exp(2 * numpy.pi * 1j * (u[0] * x + u[1] * y))
    # desiredOutput = numpy.exp(2 * numpy.pi * 1j * (u[0] * xout + u[1] * yout))

    A = np.zeros((n, n))
    mBhalf = np.zeros((m, n))
    mBhalfPoly = np.zeros((nt, m, n))
    C = np.ones((nt,))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-1.0 / sigma**2 * ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))
        for a in range(m):
            mBhalf[a, i] = np.exp(-1.0 / sigma**2 * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2))
            for k in range(nt):
                mBhalfPoly[k, a, i] = np.exp(
                    -1.0 / sigma**2 / (0.5 + 0.5 * 1.05**k) * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2)
                )
    for k in range(nt):
        C[k] = (1 + 1.05**k) ** 2 / 4.0 / 1.05**k

    # rescale everything
    A *= 0.7
    mBhalf *= 0.7
    mBhalfPoly *= 0.7
    C *= 0.7

    fits.PrimaryHDU(A).writeto("A.fits", overwrite=True)
    fits.PrimaryHDU(mBhalfPoly).writeto("mBhalf.fits", overwrite=True)
    print("C=", C)

    kappa_array = np.logspace(-6, -2, 3)
    print("kappa_array=", kappa_array)

    print("n", n, "m", m, "nt", nt, "nv", np.size(kappa_array))

    t1a = time.perf_counter()
    print("begin", t1a)

    (kappa_, Sigma_, UC_, T_) = get_coadd_matrix_discrete(A, mBhalfPoly, C, kappa_array, smax=0.5)

    t1b = time.perf_counter()
    print("get_coadd time", t1b - t1a)

    # print information
    fits.PrimaryHDU(T_).writeto("T.fits", overwrite=True)
    fits.PrimaryHDU(UC_).writeto("UC.fits", overwrite=True)
    fits.PrimaryHDU(Sigma_).writeto("Sigma.fits", overwrite=True)
    fits.PrimaryHDU(kappa_).writeto("kappa_ind.fits", overwrite=True)


def testkernel(sigma, u):
    """
    Test case for the kernel.

    This is nothing fancy.  The test interpolates an image containing a single sine wave, with Gaussian PSF.

    Parameters
    ----------
    sigma : float
        The 1 sigma width of PSF (Gaussian)
    u : np.array or list
        Shape (2,). Fourier wave vector of sine wave, (x,y) ordering.

    Returns
    -------
    None

    """

    # number of outputs to print
    npr = 4

    # number of layers to test multi-ouptut
    nt = 3

    # test grid: interpolate an m1xm1 image from n1xn1
    m1 = 25
    n1 = 33
    n = n1 * n1
    m = m1 * m1

    x = np.zeros((n,))
    y = np.zeros((n,))
    for i in range(n1):
        y[n1 * i : n1 * i + n1] = i
        x[i::n1] = i
    xout = np.zeros((m,))
    yout = np.zeros((m,))
    for i in range(m1):
        yout[m1 * i : m1 * i + m1] = 5 + 0.25 * i
        xout[i::m1] = 5 + 0.25 * i

    # make sample image
    thisImage = np.exp(2 * np.pi * 1j * (u[0] * x + u[1] * y))
    desiredOutput = np.exp(2 * np.pi * 1j * (u[0] * xout + u[1] * yout))

    print("build matrices", time.perf_counter())

    A = np.zeros((n, n))
    mBhalf = np.zeros((m, n))
    mBhalfPoly = np.zeros((nt, m, n))
    C = 1.0
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-1.0 / sigma**2 * ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))
        for a in range(m):
            mBhalf[a, i] = np.exp(-1.0 / sigma**2 * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2))
            for k in range(nt):
                mBhalfPoly[k, a, i] = np.exp(
                    -1.0 / (1.05**k * sigma) ** 2 * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2)
                )

    # rescale everything
    A *= 0.7
    mBhalf *= 0.7
    mBhalfPoly *= 0.7
    C *= 0.7

    t1a = time.perf_counter()
    print("kernel, brute force", t1a)

    # brute force version of kernel
    (kappa, Sigma, UC, T) = BruteForceKernel(A, mBhalf, C, 1e-8)

    print("** brute force kernel **")
    print("kappa =", kappa[:npr])
    print("Sigma =", Sigma[:npr])
    print("UC =", UC[:npr])
    print("Image residual =")
    print(np.abs(T @ thisImage - desiredOutput).reshape((m1, m1))[:npr])

    t1b = time.perf_counter()
    print("kernel, C", t1b)

    # C version of kernel
    (kappa2, Sigma2, UC2, T2) = CKernel(A, mBhalf, C, 1e-8)

    print("** C kernel **")
    print("kappa =", kappa2[:npr])
    print("Sigma =", Sigma2[:npr])
    print("UC =", UC2[:npr])
    print("Image residual =")
    print(np.abs(T2 @ thisImage - desiredOutput).reshape((m1, m1))[:npr])

    t1c = time.perf_counter()

    (kappa3, Sigma3, UC3, T3) = CKernelMulti(
        A, mBhalfPoly, C * 1.05 ** (2 * np.array(range(nt))), 1e-8 * np.ones((nt,))
    )
    print("Sigma3 =", Sigma3[:, :npr])
    print("output =", (T2 @ thisImage)[:npr], (T3 @ thisImage)[:, :npr])

    t1d = time.perf_counter()
    print("end -->", t1d)

    print("timing: ", t1b - t1a, t1c - t1b, t1d - t1c)


def testinterp(u):
    """
    Test interpolation functions.

    Parameters
    ----------
    u : np.array or list
        Shape (2,). Fourier wave vector of sine wave, (x,y) ordering.

    Returns
    -------
    None

    """

    ny = 1024
    nx = 1024
    indata = np.zeros((3, ny, nx))
    indata[0, :, :] = 1.0
    for ix in range(nx):
        indata[1, :, ix] = u[0] * ix + u[1] * np.linspace(0, ny - 1, ny)
    indata[2, :, :] = np.cos(2 * np.pi * indata[1, :, :])

    no = 32768
    xout = np.linspace(8, 9, no)
    yout = np.linspace(10, 10.5, no)

    fout = np.zeros((3, no))

    t1a = time.perf_counter()
    pyimcom_croutines.iD5512C(indata, xout, yout, fout)
    # pyimcom_croutines.iD5512C(indata[2,:,:].reshape((1,ny,nx)), xout, yout, fout[2,:].reshape((1,no)))
    t1b = time.perf_counter()

    pred = u[0] * xout + u[1] * yout

    # print(fout)
    # print(pred)
    # print(numpy.cos(2*numpy.pi*pred))
    print("errors:")
    print(fout[0, :] - 1)
    print(fout[1, :] - pred)
    print(fout[2, :] - np.cos(2 * np.pi * pred))

    print(f"timing interp = {t1b-t1a:9.6f} s")


# tests to run if this is the main function
if __name__ == "__main__":
    testkernel(4.0, [0.2, 0.1])
