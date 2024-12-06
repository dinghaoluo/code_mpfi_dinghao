# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:19:38 2024
Modified on Fri 15 Nov 16:16:14 2024

@author: Jingyu Cao
@modified by Dinghao Luo

calculate std map for movies
"""


#%% imports
import numpy as np
from numpy.linalg import norm
from scipy.ndimage import maximum_filter, uniform_filter, gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.stats import mode as most_common_value
import matplotlib.pyplot as plt
from warnings import warn
import time
from tqdm import tqdm

# GPU support
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
except ModuleNotFoundError:
    print('cupy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'An error occurred: {e}')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    print('GPU-acceleartion unavailable')


#%% functions
def neuropil_subtraction(mov: np.ndarray, filter_size: int) -> None:
    '''Apply spatial low-pass filter to help ignore neuropil
    
    The uniform filter of size "filter_size" is applied to each frame
    and divided by the a 2D plane of ones with feathered edges.
    This is then subtracted from the original frame. 
    

    Parameters
    ----------
    mov : np.ndarray
        Input movie of shape (n_bins, y, x)
    filter_size : int
        Size of filter size for uniform_filter in pixel

    Returns
    -------
    mov_out : np.ndarray
        Low-pass filtered movie of shape (n_bins, y, x)
    '''
    # plane with feathered edges
    _, Ly, Lx = mov.shape
    c1 = uniform_filter(np.ones((Ly, Lx)), size=filter_size, mode="constant")

    mov_out = np.zeros_like(mov)
    for frame_old, frame_new in zip(mov, mov_out):
        frame_filt = uniform_filter(frame_old, size=filter_size, mode="constant") / c1
        frame_new[:] = frame_old - frame_filt
    return mov_out



def spatially_downsample(mov: np.ndarray, n_scales: int, filter_size=3) -> np.ndarray:
    '''Downsample movie at multiple spatial scales

    Spatially downsample the movie `n_scales` times by a factor of 2.
    Applies smoothing with 2D uniform filter of size `filter_size` before
    each downsampling step.
    Also returns meshgrid of downsampled grid points.

    Parameters
    ----------
    mov : np.ndarray
        Input movie of shape (n_bins, y, x)
    n_scales : int
        Number of times to downsample

    Returns
    -------
    mov_down : list
        List of downsampled movies
    grid_down : list
        List of downsampled grid points
    '''
    _, Lyc, Lxc = mov.shape
    grid_list = np.meshgrid(range(Lxc), range(Lyc))
    grid = np.array(grid_list).astype("float32")

    # variables to be downsampled
    mov_d, grid_d = mov, grid

    # collect downsampled movies and grids
    mov_down, grid_down = [], []

    # downsample multiple times
    for _ in range(n_scales):
        # smooth (downsampled) movie
        smoothed = square_convolution_2d(mov_d, filter_size=filter_size)
        mov_down.append(smoothed)

        # downsample movie TODO why x2?
        mov_d = 2 * downsample(mov_d, taper_edge=True)

        # downsample grid
        grid_down.append(grid_d)
        grid_d = downsample(grid_d, taper_edge=False)

    return mov_down, grid_down



def square_convolution_2d(mov: np.ndarray, filter_size: int) -> np.ndarray:
    '''Returns movie convolved by uniform kernel with width "filter_size".

    Parameters
    ----------
    mov : np.ndarray
        Input movie of shape (n_bins, y, x)
    filter_size : int
        Size of filter size for uniform_filter in pixel

    Returns
    -------
    mov_out : np.ndarray
        Convolved movie of shape (n_bins, y, x)
    '''
    mov_out = np.zeros_like(mov, dtype=np.float32)
    for frame_old, frame_new in zip(mov, mov_out):
        frame_filt = filter_size * \
            uniform_filter(frame_old, size=filter_size, mode="constant")
        frame_new[:] = frame_filt
    return mov_out



def downsample(mov: np.ndarray, taper_edge: bool = True) -> np.ndarray:
    """
    Returns a pixel-downsampled movie from "mov", tapering the edges of "taper_edge" is True.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to downsample
    taper_edge: bool
        Whether to taper the edges

    Returns
    -------
    filtered_mov:
        The downsampled frames
    """
    n_frames, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((n_frames, int(np.ceil(Ly / 2)), Lx), "float32")
    movd[:, :Ly // 2, :] = np.mean([mov[:, 0:-1:2, :], mov[:, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:, -1, :] = mov[:, -1, :] / 2 if taper_edge else mov[:, -1, :]

    # bin along X
    mov2 = np.zeros((n_frames, int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), "float32")
    mov2[:, :, :Lx // 2] = np.mean([movd[:, :, 0:-1:2], movd[:, :, 1::2]], axis=0)
    if Lx % 2 == 1:
        mov2[:, :, -1] = movd[:, :, -1] / 2 if taper_edge else movd[:, :, -1]

    return mov2



def set_scale_and_thresholds(mov_norm_down, grid_down, spatial_scale, threshold_scaling):
    '''Find best spatial scale and set thresholds for ROI detection

    Parameters
    ----------
    mov_norm_down : list
        List of downsampled movies
    grid_down : list
        List of downsampled grid points
    spatial_scale : int
        If > 0, use this as the spatial scale, otherwise estimate it
    threshold_scaling : float
        

    Returns
    -------
    scale_pix : int
        Spatial scale in pixels
    thresh_peak : float
        Threshold: `threshold_scaling` * 5 * `scale`
    thresh_multiplier : float
        Threshold multiplier: max(1, n_bins / 1200)
    vcorr : np.ndarray
        Correlation map
    '''
    
    # spline approximation of max projection of downsampled movies
    maxproj_splined = spline_over_scales(mov_norm_down, grid_down)

    scale, estimate_mode = find_best_scale(maxproj_splined, spatial_scale=spatial_scale)


    # define thresholds based on spatial scale
    scale_pix = scale_in_pixel(scale)
    # threshold for accepted peaks (scale it by spatial scale) TODO why hardcode 5
    thresh_peak = threshold_scaling * 5 * scale
    thresh_multiplier = max(1, mov_norm_down[0].shape[0] / 1200)
    print(
        "NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f "
        % (
            estimate_mode,
            scale_pix,
            thresh_multiplier,
            thresh_multiplier * thresh_peak,
        )
    )

    return scale_pix, thresh_peak, thresh_multiplier


def spline_over_scales(mov_down, grid_down):
    '''Spline approximation of max projection of downsampled movies

    Uses RectBivariateSpline to upsample the downsampled max projection
    across time at each spatial scale.

    Parameters
    ----------
    mov_down : list
        List of downsampled movies
    grid_down : list
        List of downsampled grid points

    Returns
    -------
    img_up : np.ndarray
        Array with upsampled max proj images across time of shape (n_scales, y, x)
    '''

    grid = grid_down[0]

    img_up = []
    for mov_d, grid_d in zip(mov_down, grid_down):
        img_d = mov_d.var(axis=0)
        upsample_model = RectBivariateSpline(
            x=grid_d[1, :, 0],
            y=grid_d[0, 0, :],
            z=img_d,
            kx=min(3, grid_d.shape[1] - 1),
            ky=min(3, grid_d.shape[2] - 1),
        )
        up = upsample_model(grid[1, :, 0], grid[0, 0, :])
        img_up.append(up)

    img_up = np.array(img_up)

    return img_up


def find_best_scale(maxproj_splined: np.ndarray, spatial_scale: int, max_scale=4):
    '''Find best spatial

    Returns best scale (between 1 and `max_scale`) 
    and estimation method ("FORCED" or "estimated").

    Parameters
    ----------
    maxproj_splined : np.ndarray
        Array with upsampled max proj images across time of shape (n_scales, y, x)
    spatial_scale : int
        If > 0, use this as the spatial scale, otherwise estimate it

    Returns
    -------
    scale : int
        Best spatial scale
    mode : str
        Estimation mode
    '''
    modes = { # to mirror former Enum class
        'frc': "FORCED",
        'est': "estimated",
    }
    if spatial_scale > 0:
        scale = max(1, min(max_scale, spatial_scale))
        mode = modes['frc']
    else:
        scale = estimate_spatial_scale(maxproj_splined)
        mode = modes['est']

    if not scale > 0:
        warn(
            "Spatial scale estimation failed.  Setting spatial scale to 1 in order to continue."
        )
        scale = 1
        mode = modes['frc']

    return scale, mode


def estimate_spatial_scale(I: np.ndarray) -> int:
    '''Estimate spatial scale based on max projection

    Parameters
    ----------
    I : np.ndarray
        Array with upsampled max proj images across time of shape (n_scales, y, x)

    Returns
    -------
    im : int
        Best spatial scale
    '''
    I0 = I.max(axis=0)
    imap = np.argmax(I, axis=0).flatten()
    ipk = np.abs(I0 - maximum_filter(I0, size=(11, 11))).flatten() < 1e-4
    isort = np.argsort(I0.flatten()[ipk])[::-1]
    im, _ = most_common_value(imap[ipk][isort[:50]], keepdims=False)
    return im



def scale_in_pixel(scale):
    "Convert scale integer to number of pixels"
    return int(3 * 2**scale)



def threshold_reduce(mov: np.ndarray, intensity_threshold: float) -> np.ndarray:
    """
    Returns standard deviation of pixels, thresholded by "intensity_threshold".
    Run in a loop to reduce memory footprint.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    intensity_threshold: float
        The threshold to use

    Returns
    -------
    Vt: Ly x Lx
        The standard deviation of the non-thresholded pixels
    """
    nbinned, Lyp, Lxp = mov.shape
    Vt = np.zeros((Lyp, Lxp), "float32")
    for t in range(nbinned):
        Vt += mov[t]**2 * (mov[t] > intensity_threshold)
    Vt = Vt**.5
    return Vt

def temporal_high_pass_filter(mov: np.ndarray, width: int, use_overlapping: bool = False) -> np.ndarray:
    """
    Returns hp-filtered mov over time, selecting an algorithm for computational performance based on the kernel width.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width
    use_overlapping: bool, default False
        If True, use overlapping instead of non-overlapping rolling mean
        Note that this only takes effect if width > 10

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames
    """

    return hp_gaussian_filter(mov, width) if width < 10 else hp_rolling_mean_filter(
        mov, width, use_overlapping)  # gaussian is slower

def hp_gaussian_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a high-pass-filtered copy of the 3D array "mov" using a gaussian kernel.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The kernel width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered video
    """
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter(mov[:, j, :], [width, 0])
    return mov


def hp_rolling_mean_filter(mov: np.ndarray, width: int, use_overlapping: bool = False) -> np.ndarray:
    """
    Returns a high-pass-filtered copy of the 3D array "mov" using a rolling mean kernel over time.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width
    use_overlapping: bool, default False
        If True, use overlapping windows

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames

    """
    mov = mov.copy()
    if use_overlapping:
        for i in range(mov.shape[0]):
            mov[i, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    else:
        for i in range(0, mov.shape[0], width):
            mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov


def max_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a filtered copy of the 3D array "mov" using a rolling max kernel over time.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames

    """
    mov = mov.copy()
    for i in range(mov.shape[0]):
        mov[i, :, :] = mov[i:i + width, :, :].max(axis=0)
    return mov

def mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a filtered copy of the 3D array "mov" using a rolling mean kernel over time.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames

    """
    mov = mov.copy()
    for i in tqdm(range(mov.shape[0]), desc='Mean-filtering:'):
        mov[i, :, :] = np.mean(mov[i:i + width, :, :], axis=0)
    return mov

def median_filter(mov, width, GPU_AVAILABLE=False):
    """
    Returns a filtered copy of the 3D array "mov" using a rolling mean kernel over time.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames

    """
    if GPU_AVAILABLE:
        print('using GPU')
        new_mov = cp.zeros(mov.shape)
        for i in tqdm(range(mov.shape[0]), desc='median filtering:'):
            new_mov[i, :, :] = cp.median(cp.asarray(mov[i:i+width, :, :]), axis=0)
        return new_mov.get()
    else:
        print('not using GPU')
        new_mov = np.zeros(mov.shape)
        for i in tqdm(range(mov.shape[0]), desc='median filtering:'):
            new_mov[i, :, :] = np.median(mov[i:i + width, :, :], axis=0)
        return new_mov


def standard_deviation_over_time(mov: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Returns standard deviation of difference between pixels across time, computed in batches of batch_size.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    batch_size: int
        The batch size

    Returns
    -------
    filtered_mov: Ly x Lx
        The statistics for each pixel
    """
    nbins, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Ly, Lx), "float32")
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix + batch_size, :, :], axis=0)**2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov
    

def bin_movie(f_reg, bin_size, yrange=None, xrange=None, badframes=None):
    """ bin registered movie """
    n_frames = f_reg.shape[0]
    good_frames = ~badframes if badframes is not None else np.ones(n_frames, dtype=bool)
    batch_size = min(good_frames.sum(), 500)
    Lyc = yrange[1] - yrange[0]
    Lxc = xrange[1] - xrange[0]

    # Number of binned frames is rounded down when binning frames
    num_binned_frames = n_frames // bin_size
    mov = np.zeros((num_binned_frames, Lyc, Lxc), np.float32)
    curr_bin_number = 0
    t0 = time.time()

    # Iterate over n_frames to maintain binning over TIME
    for k in np.arange(0, n_frames, batch_size):
        data = f_reg[k:min(k + batch_size, n_frames)]

        # exclude badframes
        good_indices = good_frames[k:min(k + batch_size, n_frames)]
        if good_indices.mean() > 0.5:
            data = data[good_indices]

        # crop to valid region
        if yrange is not None and xrange is not None:
            data = data[:, slice(*yrange), slice(*xrange)]

        # bin in time
        if data.shape[0] > bin_size:
            # Downsample by binning via reshaping and taking mean of each bin
            # only if current batch size exceeds or matches bin_size
            n_d = data.shape[0]
            data = data[:(n_d // bin_size) * bin_size]
            data = data.reshape(-1, bin_size, Lyc, Lxc).astype(np.float32).mean(axis=1)
        else:
            # Current batch size is below bin_size (could have many bad frames in this batch)
            # Downsample taking the mean of batch to get a single bin
            data = data.mean(axis=0)[np.newaxis, :, :]
        # Only fill in binned data if not exceeding the number of bins mov has
        if mov.shape[0] > curr_bin_number:
            # Fill in binned data
            n_bins = data.shape[0]
            mov[curr_bin_number:curr_bin_number + n_bins] = data
            curr_bin_number += n_bins

    print("Binned movie of size [%d,%d,%d] created in %0.2f sec." %
          (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
    return mov

def add_square(yi, xi, lx, Ly, Lx):
    """return square of pixels around peak with norm 1

    Parameters
    ----------------

    yi : int
        y-center

    xi : int
        x-center

    lx : int
        x-width

    Ly : int
        full y frame

    Lx : int
        full x frame

    Returns
    ----------------

    y0 : array
        pixels in y

    x0 : array
        pixels in x

    mask : array
        pixel weightings

    """
    lhf = int((lx - 1) / 2)
    ipix = np.tile(np.arange(-lhf, -lhf + lx, dtype=np.int32), reps=(lx, 1))
    x0 = xi + ipix
    y0 = yi + ipix.T
    mask = np.ones_like(ipix, dtype=np.float32)
    ix = np.all((y0 >= 0, y0 < Ly, x0 >= 0, x0 < Lx), axis=0)
    x0 = x0[ix]
    y0 = y0[ix]
    mask = mask[ix]
    mask = mask / norm(mask)
    return y0, x0, mask


def iter_extend(ypix, xpix, mov, Lyc, Lxc, active_frames, thresh_act_pix):
    """extend mask based on activity of pixels on active frames
    ACTIVE frames determined by threshold

    Parameters
    ----------------

    ypix : array
        pixels in y

    xpix : array
        pixels in x

    mov : 2D array
        binned residual movie [nbinned x Lyc*Lxc]

    active_frames : 1D array
        list of active frames

    Returns
    ----------------
    ypix : array
        extended pixels in y

    xpix : array
        extended pixels in x
    lam : array
        pixel weighting
    """

    # only active frames
    mov_act = mov[active_frames]

    while True:
        npix_old = ypix.size  # roi size before extension

        # extend by 1 pixel on each side
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)

        # mean activity in roi
        roi_act = mov_act[:, ypix * Lxc + xpix]
        lam = roi_act.mean(axis=0)

        # select active pixels
        thresh_lam = max(0, lam.max() * thresh_act_pix) # @@@@@@@@@@@ max(0, lam.max() / 5)
        pix_act = lam > thresh_lam

        if not np.any(pix_act):  # stop if no pixels are active
            break

        ypix, xpix, lam = ypix[pix_act], xpix[pix_act], lam[pix_act]
        npix_new = ypix.size  # after extension

        if npix_new <= npix_old:  # stop if no pixels were added
            break

        if npix_new >= 400:  # stop if too many pixels
            break

    # normalize by standard deviation
    lam = lam / np.sum(lam**2) ** 0.5

    return ypix, xpix, lam, thresh_lam


def iter_extend_fixedth(ypix, xpix, mov, Lyc, Lxc, active_frames, thresh_intensity):
    """extend mask based on activity of pixels on active frames
    ACTIVE frames determined by threshold

    Parameters
    ----------------

    ypix : array
        pixels in y

    xpix : array
        pixels in x

    mov : 2D array
        binned residual movie [nbinned x Lyc*Lxc]

    active_frames : 1D array
        list of active frames

    Returns
    ----------------
    ypix : array
        extended pixels in y

    xpix : array
        extended pixels in x
    lam : array
        pixel weighting
    """

    # only active frames
    mov_act = mov[active_frames]

    while True:
        npix_old = ypix.size  # roi size before extension

        # extend by 1 pixel on each side
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)

        # mean activity in roi
        roi_act = mov_act[:, ypix * Lxc + xpix]
        lam = roi_act.mean(axis=0)

        # select active pixels
        thresh_lam = max(0, thresh_intensity) # @@@@@@@@@@@@ lam.max() / 15) # @@@@@@@@@@@ max(0, lam.max() / 5)
        pix_act = lam > thresh_lam

        if not np.any(pix_act):  # stop if no pixels are active
            break

        ypix, xpix, lam = ypix[pix_act], xpix[pix_act], lam[pix_act]
        npix_new = ypix.size  # after extension

        if npix_new <= npix_old:  # stop if no pixels were added
            break

        if npix_new >= 400:  # stop if too many pixels
            break

    # normalize by standard deviation
    lam = lam / np.sum(lam**2) ** 0.5

    return ypix, xpix, lam


def extendROI(ypix, xpix, Ly, Lx, niter=1):
    '''Extend ypix and xpix by `niter` pixel(s) on each side

    Parameters
    ----------
    ypix : np.ndarray
        1D array of y pixel indices
    xpix : np.ndarray
        1D array of x pixel indices
    Ly : int
        y dimension of movie
    Lx : int
        x dimension of movie
    niter : int, optional
        Number of iterations to extend, by default 1

    Returns
    -------
    ypix : np.ndarray
        1D array of extended y pixel indices
    xpix : np.ndarray
        1D array of extended x pixel indices
    '''
    for _ in range(niter):
        yx = (
            (ypix, ypix, ypix, ypix - 1, ypix + 1),
            (xpix, xpix + 1, xpix - 1, xpix, xpix),
        )
        yx = np.array(yx)
        yx = yx.reshape((2, -1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix


def two_comps(mpix0, lam, Th2):
    """check if splitting ROI increases variance explained

    Parameters
    ----------------

    mpix0 : 2D array
        binned movie for pixels in ROI [nbinned x npix]

    lam : array
        pixel weighting

    Th2 : float
        intensity threshold


    Returns
    ----------------

    vrat : array
        extended pixels in y

    ipick : tuple
        new ROI

    """
    # TODO add comments
    mpix = mpix0.copy()
    xproj = mpix @ lam
    gf0 = xproj > Th2

    mpix[gf0, :] -= np.outer(xproj[gf0], lam)
    vexp0 = np.sum(mpix0**2) - np.sum(mpix**2)

    k = np.argmax(np.sum(mpix * np.float32(mpix > 0), axis=1))
    mu = [lam * np.float32(mpix[k] < 0), lam * np.float32(mpix[k] > 0)]

    mpix = mpix0.copy()
    goodframe = []
    xproj = []
    for mu0 in mu:
        mu0[:] /= norm(mu0) + 1e-6
        xp = mpix @ mu0
        mpix[gf0, :] -= np.outer(xp[gf0], mu0)
        goodframe.append(gf0)
        xproj.append(xp[gf0])

    flag = [False, False]
    V = np.zeros(2)
    for _ in range(3):
        for k in range(2):
            if flag[k]:
                continue
            mpix[goodframe[k], :] += np.outer(xproj[k], mu[k])
            xp = mpix @ mu[k]
            goodframe[k] = xp > Th2
            V[k] = np.sum(xp**2)
            if np.sum(goodframe[k]) == 0:
                flag[k] = True
                V[k] = -1
                continue
            xproj[k] = xp[goodframe[k]]
            mu[k] = np.mean(mpix[goodframe[k], :] *
                            xproj[k][:, np.newaxis], axis=0)
            mu[k][mu[k] < 0] = 0
            mu[k] /= 1e-6 + np.sum(mu[k] ** 2) ** 0.5
            mpix[goodframe[k], :] -= np.outer(xproj[k], mu[k])
    k = np.argmax(V)
    vexp = np.sum(mpix0**2) - np.sum(mpix**2)
    vrat = vexp / vexp0
    return vrat, (mu[k], xproj[k], goodframe[k])


def multiscale_mask(ypix, xpix, lam, Ly_down, Lx_down):
    '''Downsample masks across spatial scales

    Parameters
    ----------
    ypix : np.ndarray
        1D array of y pixel indices
    xpix : np.ndarray
        1D array of x pixel indices
    lam : np.ndarray
        1D array of pixel weights
    Ly_down : list
        List of y dimensions at each spatial scale
    Lx_down : list
        List of x dimensions at each spatial scale

    Returns
    -------
    ypix_down : list
        List of downsampled y pixel indices at each spatial scale
    xpix_down : list
        List of downsampled x pixel indices at each spatial scale
    lam_down : list
        List of downsampled pixel weights at each spatial scale
    '''
    # initialize at original scale
    xpix_down = [xpix]
    ypix_down = [ypix]
    lam_down = [lam]
    for j in range(1, len(Ly_down)):
        ipix, ind = np.unique(
            np.int32(xpix_down[j - 1] / 2) + np.int32(ypix_down[j - 1] / 2) * Lx_down[j],
            return_inverse=True,
        )
        lam_d = np.zeros(len(ipix))
        for i in range(len(xpix_down[j - 1])):
            lam_d[ind[i]] += lam_down[j - 1][i] / 2
        lam_down.append(lam_d)
        ypix_down.append(np.int32(ipix / Lx_down[j]))
        xpix_down.append(np.int32(ipix % Lx_down[j]))
    for j in range(len(Ly_down)):
        ypix_down[j], xpix_down[j], lam_down[j] = extend_mask(
            ypix_down[j], xpix_down[j], lam_down[j], Ly_down[j], Lx_down[j])
    return ypix_down, xpix_down, lam_down


def extend_mask(ypix, xpix, lam, Ly, Lx):
    """extend mask into 8 surrrounding pixels"""
    # TODO add docstring and comments
    nel = len(xpix)
    yx = (
        (ypix, ypix, ypix, ypix - 1, ypix - 1,
         ypix - 1, ypix + 1, ypix + 1, ypix + 1),
        (xpix, xpix + 1, xpix - 1, xpix, xpix +
         1, xpix - 1, xpix, xpix + 1, xpix - 1),
    )
    yx = np.array(yx)
    yx = yx.reshape((2, -1))
    yu, ind = np.unique(yx, axis=1, return_inverse=True)
    LAM = np.zeros(yu.shape[1])
    for j in range(len(ind)):
        LAM[ind[j]] += lam[j % nel] / 3
    ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
    ypix1, xpix1 = yu[:, ix]
    lam1 = LAM[ix]
    return ypix1, xpix1, lam1

#%% params
n_scales = 5
spatial_scale = 1
threshold_scaling = 0.5
max_iterations = 200
n_iter_refine = 3
thresh_split=1.25
new_ops = {}
batch_size = 200

#%% loading test data
# if using data.bin file

p_ops = p_data = r"Z:\Jingyu\2P_Recording\AC955\AC955-20240911\02\RegOnly\suite2p\plane0"

# p_data = r"Z:\Dinghao\2p_recording\A094i\A094i-20240716\A094i-20240716-01\processed\suite2p\plane0"
# p_ops = r"Z:\Dinghao\2p_recording\A094i\A094i-20240716\A094i-20240716-01\processed\suite2p\plane0"
ops = np.load(p_ops+r'\ops.npy', allow_pickle=True).item()
reg_dlight = np.memmap(p_data+r'\data.bin', mode='r', dtype='int16',shape=(5000, 512, 512))
mov = reg_dlight.astype('float32')

high_pass=ops["high_pass"]
use_overlapping=ops["wang:high_pass_overlapping"]
rolling = 'median'
width=ops["wang:rolling_width"]
percentile=ops.get("active_percentile", 0.0)
thresh_act_pix=ops["wang:thresh_act_pix"]

#%% spatial and temporal filtering
movc = mov.copy()

# rolling max filter:
if rolling == 'max':
    movc = max_filter(movc, width=int(width))
if rolling == 'mean':
    movc = mean_filter(movc, width=int(width))
if rolling == 'median':
    movc = median_filter(movc, width=int(width), GPU_AVAILABLE=GPU_AVAILABLE)

# neuropil subtraction
movc = neuropil_subtraction(movc, filter_size=15)

# high-pass filter movie
movc = temporal_high_pass_filter(movc, width=int(high_pass), use_overlapping=0)

# test plot: Mean of Each Pixel Over Time
plt.imshow(np.mean(movc, axis=0), cmap='gray')
plt.title('Mean of Each Pixel Over Time\noverall mean={}'.format(np.mean(movc)))
plt.axis('off') 


#%% Normalization

# Find the maximum pixel intensity across the entire image stack
mov_max_intensity = np.max(movc)
mov_min_intensity = np.min(movc)
 
# Normalize each image by the maximum intensity
max_min_intensity = np.array([mov_max_intensity, np.abs(mov_min_intensity)])
mov_norm = movc / np.max(max_min_intensity)

# Normalize each image by the standard deviation
# mov_sd = standard_deviation_over_time(movc, batch_size=batch_size)
# mov_norm = movc / mov_sd

# test plot: Mean of Each Pixel Over Time after normalization
plt.imshow(np.mean(mov_norm, axis=0), cmap='gray')
plt.title('Mean of Each Pixel Over Time [normalized]\noverall mean={}'.format(np.mean(mov_norm)))
plt.axis('off') 
#%% standard deviation of mov_norm
# calculating standard deviation
mov_sd = standard_deviation_over_time(mov_norm, batch_size=batch_size)
# test plot: Standard Deviation of Each Pixel Over Time
plt.imshow(mov_sd, cmap='viridis', vmin=np.percentile(mov_sd, 1), vmax=np.percentile(mov_sd,100))
plt.title('Standard Deviation of Each Pixel Over Time')
plt.colorbar()
plt.axis('off') 

# mean of the standard deviation of all the pixels in mov_norm  
mean_mov_sd = np.mean(mov_sd)
# mean of all pixels in spatial and temporal filtered movie
mean_mov_mean = np.mean(movc) 

# Set a threshold for mov_norm
thresh_peak_norm = mean_mov_mean + 3*mean_mov_sd
thresh_act_pix = mean_mov_mean + threshold_scaling*mean_mov_sd

#%% Spatial Downsample of the normalized movie
# downsample movie at various spatial scales
mov_norm_down, grid_down = spatially_downsample(
    mov=mov_norm, n_scales=n_scales)
# xy dimensions original movie
_, Ly, Lx = mov.shape
# xy dimensions downsampled movies
Ly_down = [m.shape[-2] for m in mov_norm_down]
Lx_down = [m.shape[-1] for m in mov_norm_down]

# Spatial scale estimation
scale_pix, thresh_peak1, thresh_multiplier = set_scale_and_thresholds(
    mov_norm_down, grid_down, spatial_scale, threshold_scaling)

# test plot: Mean of the downsampled mov_norm at the appointed scale
mov_norm_down_a = mov_norm_down[spatial_scale]
plt.imshow(np.mean(mov_norm_down_a, axis=0), cmap='gray')
plt.title('Mean of Each Pixel Over Time [downsample_{}]\noverall mean={}'.format(spatial_scale, np.mean(mov_norm_down_a)))
# plt.axis('off') 
mean_mov_down_mean = np.mean(mov_norm_down_a)
#%% standard deviation of mov_norm_down
mov_norm_down_sd = standard_deviation_over_time(mov_norm_down[spatial_scale], batch_size=batch_size)
# plt.imshow(mov_norm_down_sd, cmap='viridis', vmin=np.percentile(mov_norm_down_sd, 1), vmax=np.percentile(mov_norm_down_sd,99))
plt.imshow(mov_norm_down_sd, cmap='viridis')
plt.title('Standard Deviation of Each Pixel Over Time [downsample_{}]'.format(spatial_scale))
plt.colorbar()
# plt.axis('off')

mean_mov_down_sd = np.mean(mov_norm_down_sd)
# Set a threshold for mov_norm
thresh_peak = mean_mov_down_mean + 3*mean_mov_down_sd

# a = np.mean(mov_norm_down_a, axis=0)>thresh_peak
#%% get std map of active pixels  (> thresh_peak) for ROI detection

# get standard deviation for pixels for all values > thresh_peak
mov_norm_sd_down = [
    threshold_reduce(m, thresh_peak) for m in mov_norm_down
]

thresh_peak_sd_down = np.mean(mov_norm_sd_down[spatial_scale]) + np.std(mov_norm_sd_down[spatial_scale])

# needed so that scipy.io.savemat doesn't fail in runpipeline with latest numpy (v1.24.3).
# dtype="object" is needed to have numpy array with elements having diff sizes
Vmap = np.asanyarray(mov_norm_sd_down, dtype="object").copy()

# test plot
n = len(mov_norm_sd_down)
ncols = 3  # Define the number of columns for your subplot grid
nrows = n // ncols + (n % ncols > 0)  # Calculate the required number of rows

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
for i, ax in enumerate(axs.flat):
    if i < n:
        im = ax.imshow(mov_norm_sd_down[i], cmap='viridis', aspect='auto')
        ax.set_title(f'mov_norm_sd_down {i+1}')
        plt.colorbar(im, ax=ax)
    else:
        ax.axis('off')  # Hide unused subplots

plt.tight_layout()
plt.show()