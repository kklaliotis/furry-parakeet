Quick start guide
=================

You can access these functions by calling::

    from furry_parakeet import pyimcom_interface

There are a few types of functions.

PSF overlap class
-----------------

The ``PSF_Overlap`` class, called with ::

    pyimcom_interface.PSF_Overlap(psf_in_list, psf_out_list, dsample, nsample,
        s_in, distort_matrices=None)

computes all of the inner products of various PSFs. This is an expensive operation if the PSFs are big, but if the PSF is slowly varying, you may need to make a ``PSF_Overlap`` class only every few output stamps. Therefore its functionality is placed separately from the image combination routines.

Then you want to call the ``get_coadd_matrix`` function::

    get_coadd_matrix(psfobj, psf_oversamp_factor, targetleak, ctrpos,
        distort_matrices, in_stamp_dscale, in_stamp_shape, out_stamp_dscale,
        out_stamp_shape, in_mask, tbdy_radius, smax=1., flat_penalty=0.,
        use_kappa_arr=None, choose_outputs="CKMSTU")

for each stamp that you co-add. This function returns a Python dictionary containing the image combination matrix (T-matrix). (Note you can request more than one output PSF at a time if you want, but they all have to have the same input mask.) There is also a lot of metadata that goes with it. The ``choose_outputs`` allows you to pull out any of the IMCOM system solution data, but beware your memory needs can get big really fast!

The user can then use the T-matrix for any of the usual operations, like operating on a set of real data, simulated data, noise images, etc. ... because the T-matrix only has to be built once, but storing all the T-matrices for the full survey is likely prohibitive, we recommend processing on as many input images at the same time as possible.

Operations such as tiling many output postage stamps, identifying which input images overlap a given region, the coordinate transformations to put them into the same system as the output images, etc. have to go in the calling script; they are not part of ``pyimcom_interface``.

Utilities
---------

These make (2, 2) numpy array matrices::

    rotmatrix(theta)    # rotation matrix
    shearmatrix(g1,g2)  # shear matrix

We've used these mostly for testing, since the "real" distortion matrices in Roman applications are obtained by taking the derivatives of the World Coordinate System (WCS).

Simple PSF models
-----------------

You will likely need these only for simple test runs, since they don't correspond to the PSF of a real system. **For realistic cases, you will input your own PSFs.** In each case, the output is a numpy array of shape (n, n).

+------------------+------------------------------------------------------+
| Case             | Command                                              |
+------------------+------------------------------------------------------+
| 2D Gaussian PSF  | ``psf_gaussian(n,sigmax,sigmay)``                    |
+------------------+------------------------------------------------------+
| "Simple" Airy PSF| ``psf_simple_airy(n,ldp,tophat_conv=0.,sigma=0.)``   |
| convolved with   |                                                      |
| tophat and       |                                                      |
| Gaussian         |                                                      |
+------------------+------------------------------------------------------+
| "Complex" Airy   |  ``psf_cplx_airy(n,ldp,tophat_conv=0.,sigma=0.,      |
| PSF with some    |  features=0)``                                       |
| artificial       |                                                      |
| diffraction      |                                                      |
| spikes and (if   |                                                      |
| ``features`` is  |                                                      |
| used) some messy |                                                      |
| features tacked  |                                                      |
| on               |                                                      |
+------------------+------------------------------------------------------+
