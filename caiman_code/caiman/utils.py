from pathlib import Path
import shutil
from tifffile import imread, imwrite, TiffFile
import numpy as np
from scipy.sparse import spdiags

import caiman as cm

def split_dual_channel_tif(p_tif, p_out):
    """Split scanimage dual-channel tif file into two separate files

    Writes new files ending with `<filename>_ch1.tif` and `<filename>_ch2.tif`

    May not work for scanimage recordings containing a single file.

    Parameters
    ----------
    p_tif : pathlike
        path to tif file
    p_out : pathlike
        folder to store the split tifs (e.g. tmp folder on local disk)
    """
    
    p_tif = Path(p_tif)
    p_out = Path(p_out)

    tif = TiffFile(p_tif)
    n_pages = len(tif.pages)

    # frame rate from scangimage metadata, rate includes both channels
    fps = tif.scanimage_metadata["FrameData"]["SI.hRoiManager.scanFrameRate"]

    ch1 = imread(p_tif, key=range(0, n_pages, 2))
    ch2 = imread(p_tif, key=range(1, n_pages, 2))

    p_tif_ch1 = p_out / f"{p_tif.stem}_ch1{p_tif.suffix}"
    p_tif_ch2 = p_out / f"{p_tif.stem}_ch2{p_tif.suffix}"

    imwrite(p_tif_ch1, ch1, metadata={"axes": "TYX", "fps": fps})
    imwrite(p_tif_ch2, ch2, metadata={"axes": "TYX", "fps": fps})


def load_bin(p_root, crop=False):

    p_data = p_root / 'data.bin'
    ops = np.load(p_root / 'ops.npy', allow_pickle=True).item()
    shape = ops["nframes"], ops["Ly"], ops["Lx"]
    data = np.memmap(p_data, mode='r', dtype='int16', shape=shape)

    if crop:
        x, y = ops["xrange"], ops["yrange"]
        data = data[:, slice(*y), slice(*x)]
        print(f"INFO: Cropped to x-range {x} and y-range {y}")
    
    bad_frames = ops['badframes']
    if n := bad_frames.sum():
        print(f'INFO: found {n} bad frames, but keeping them for now')

    return data

def save_data_as_mmap(p_ops, first_frame=0, last_frame=-1, crop=True):

    ops = np.load(p_ops, allow_pickle=True).item()
    p_data = p_ops.with_name("data.bin")
    p_memmap_base = p_data.with_name('memmap_')

    # set up memory-mapped files
    shape = ops["nframes"], ops["Ly"], ops["Lx"]
    data = np.memmap(p_data, mode='r', dtype='int16', shape=shape, order='C')

    # select frames
    data = data[first_frame:last_frame]

    # crop field of view
    if crop:
        x, y = ops["xrange"], ops["yrange"]
        data = data[:, slice(*y), slice(*x)]
        print(f"INFO: Cropped to x-range {x} and y-range {y}")

    p_memmap = cm.save_memmap(
        filenames=[data],
        base_name=str(p_memmap_base), # TODO test this basename
        order='C', border_to_0=0)
    
    return p_memmap

def load_ref_img(p_ops):
    ops = np.load(p_ops, allow_pickle=True).item()
    return ops['refImg']

def reshape(arr, dims, num_frames):
    return np.reshape(arr.T, [num_frames] + list(dims), order='F')

def check_range_int16(arr):
    min_val_int16 = np.iinfo(np.int16).min
    max_val_int16 = np.iinfo(np.int16).max
    return np.all((arr >= min_val_int16) & (arr <= max_val_int16))

def write_results_tifs(cnmf_estimates, orig, dims, p_out):

    num_frames = orig.shape[1]

    A, C, b, f = cnmf_estimates.A, cnmf_estimates.C, cnmf_estimates.b, cnmf_estimates.f,

    neural_activity = A.astype(np.float32) @ C.astype(np.float32)
    background = b.astype(np.float32) @ f.astype(np.float32)
    residual = orig.astype(np.float32) - neural_activity - background

    def write_tiff(p_tif, arr):
        arr = reshape(arr, dims, num_frames)
        if check_range_int16(arr):
            arr = arr.astype(np.int16)
        else:
            print(f"WARNING: Converting to int16 would result in data loss, keeping orignal dtype: {arr.dtype}")
        imwrite(p_tif, arr)

    write_tiff(p_out / 'neural_activity.tif', neural_activity)
    write_tiff(p_out / 'background.tif', background)
    write_tiff(p_out / 'residual.tif', residual)


def save_rois_imagej(cnmf_estimates, dims, perc, p_roi):

    from roifile import ImagejRoi
    from skimage import measure

    p_roi.unlink(missing_ok=True)

    for i in range(cnmf_estimates.A.shape[1]):
        img = np.reshape(cnmf_estimates.A[:, i], dims, order='F').toarray()
        thresh = np.percentile(img[img > 0], perc)
        xy = measure.find_contours(img, thresh)[0]
        roi = ImagejRoi.frompoints(list(zip(xy[:, 1], xy[:, 0])))
        roi.name = str(i)
        roi.tofile(p_roi)

def run_cnmf(images, parameter_dict, p_out):
    
    # start cluster
    _, clu, n_proc = cm.cluster.setup_cluster(backend='multiprocessing', n_processes=None, single_thread=False)

    # convert parameter dict to CNMFParams object
    parameters = cm.source_extraction.cnmf.params.CNMFParams(params_dict=parameter_dict) 
    
    # fit model
    cnmf_model = cm.source_extraction.cnmf.cnmf.CNMF(n_proc, params=parameters, dview=clu)
    cnmf_fit = cnmf_model.fit(images)

    # refit
    cnmf_refit = cnmf_fit.refit(images, dview=clu)

    # save
    cnmf_refit.save(str(p_out / 'cnmf_fit.hdf5'))

    # stop cluster
    cm.stop_server(dview=clu)

def load_cnmf(p_out):
    cnmf_refit = cm.source_extraction.cnmf.cnmf.load_CNMF(str(p_out / 'cnmf_fit.hdf5'))
    return cnmf_refit

def trace_per_roi(A, b, C, f, Yr):

    nA2 = np.ravel(np.power(A, 2).sum(0)) if isinstance(A, np.ndarray) else np.ravel(A.power(2).sum(0))
    nr, _ = C.shape

    Y_r = np.array(spdiags(1 / nA2, 0, nr, nr) *
                    (A.T * np.matrix(Yr) -
                    (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                    A.T.dot(A) * np.matrix(C)) + C)
    return Y_r

def create_mock_stat(A, dims, p_ops):

    ops = np.load(p_ops, allow_pickle=True).item()
    x0, y0 = ops['xrange'][0], ops['yrange'][0]

    all_keys = ['ypix', 'xpix', 'lam', 'med', 'footprint', 'mrs', 'mrs0', 'compact', 'solidity', 'npix', 'npix_soma', 'soma_crop',
                 'overlap', 'radius', 'aspect_ratio', 'npix_norm_no_crop', 'npix_norm', 'skew', 'std', 'neuropil_mask']

    A = A.toarray()
    A = np.reshape(A, [*dims] + [-1], order='F')
    A /= A.max()

    l = []
    for a in A.T:
        x, y = np.where(a > 0)
        lam = a[x, y]

        # back to original coordinates e.g. 512x512
        x += x0
        y += y0
        d = {'xpix': x, 'ypix': y, 'lam': lam,
            'med': (y.mean(), x.mean()),
            'radius': (x.max() - x.min())/2,
            'npix': len(x),
            }
        
        for k in all_keys:
            if k not in d:
                d[k] = 0

        l.append(d)

    return np.array(l)

def create_suite2p_files(cnmf_estimates, Yr, p_ops, p_s2p):

    # create folder
    p_s2p.mkdir(exist_ok=True)
    
    # copy ops file
    shutil.copy(p_ops, p_s2p / 'ops.npy')

    A = cnmf_estimates.A
    b = cnmf_estimates.b
    f = cnmf_estimates.f
    C = cnmf_estimates.C

    # traces in suite2p gui:
    # deconv: C
    # raw fluor: Y_r
    # neuropil: b_r
    np.save(p_s2p / 'spks.npy', C)


    Y_r = trace_per_roi(A, b, C, f, Yr)
    np.save(p_s2p / 'F.npy', Y_r)

    b_r = trace_per_roi(A, b, C, f, b @ f)
    np.save(p_s2p / 'Fneu.npy', b_r)

    # most properties in stat.npy are set to 0
    s = create_mock_stat(A, cnmf_estimates.dims, p_ops)
    np.save(p_s2p / 'stat.npy', s)



