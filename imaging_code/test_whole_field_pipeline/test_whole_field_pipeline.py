# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:33:41 2025

test cases for a whole-field pipeline to:
    1) remove green-to-red leaking ROIs
    2) calculate red-to-speed correlagram for thresholding individual sessions

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter


#%% functions 
def quadrant_mask(Ly, Lx, ny, nx, sT):
    mask = np.zeros((Ly, Lx), np.float32)
    mask[np.ix_(ny, nx)] = 1
    mask = gaussian_filter(mask, sT)
    return mask

def correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2):
    """
    annotated by Dinghao for better communication, 18 Feb 2025
    modified by Yingxue, originally from Suite2p function suite
    
    estimates and corrects bleedthrough from one channel into another using local regression.

    parameters:
    - Ly: image height
    - Lx: image width
    - nblks: number of blocks per dimension for local regression
    - mimg: source image (channel with bleedthrough)
    - mimg2: target image (channel affected by bleedthrough)

    returns:
    - mask_weight: (Ly, Lx) array of correction weights normalised between 0 and 1
    """
    sT = np.round((Ly + Lx) / (nblks * 2) * 0.25)  # std for gaussian masking later 
    
    # define block boundaries 
    yb = np.linspace(0, Ly, nblks + 1).astype(int)  # y border indices 
    xb = np.linspace(0, Lx, nblks + 1).astype(int)  # x border indices 
    
    # create masks and weights containers 
    mask = np.zeros((Ly, Lx, nblks, nblks), np.float32)
    weights = np.zeros((nblks, nblks), np.float32)
    
    for iy in range(nblks):
        for ix in range(nblks):
            ny = np.arange(yb[iy], yb[iy + 1]).astype(int)
            nx = np.arange(xb[ix], xb[ix + 1]).astype(int)
            mask[:, :, iy, ix] = quadrant_mask(Ly, Lx, ny, nx, sT)  # this provides smooth transitions between blocks, in order to minimise artefacts and maximise stability; added epsilon to resolve numerical instability
            x = mimg[np.ix_(ny, nx)].flatten() / np.max(mimg[np.ix_(ny, nx)] + 1e-6)  # normalised by local maxima
            x2 = mimg2[np.ix_(ny, nx)].flatten() / np.max(mimg2[np.ix_(ny, nx)] + 1e-6)  # same as above 
            a = (x * x2).sum() / ((x * x).sum() + 1e-6)  # linear regression, assuming no intercept term and strict linear predictability; potential numerical instability resolved by adding epsilon
            # weights[iy, ix] = min(a, 1.0)  # cap weights at 1.0
            weights[iy, ix] = np.clip(a, 0, 1)  # prevents overcorrection 
            
    mask /= mask.sum(axis=-1).sum(axis=-1)[:, :, np.newaxis, np.newaxis]  # ensures sum across all blocks at each pixel location is 1, preserving relative contributions
    mask *= weights
    mask_weight= mask.sum(axis=-1).sum(axis=-1)
    return mask_weight


#%% load reference 
ops = np.load(
    r'Z:\Jingyu\2P_Recording\AC926\AC926-20240305\02\suite2p-wang-lab_peakScaling=1_thresScaling=2.5\suite2p\plane0\ops.npy',
    allow_pickle=True
    ).item()

ming = ops['meanImg']
ming2 = ops['meanImg_chan2']


#%% get mask
Ly = Lx = 512
nblks = 256

mask_weight = correct_bleedthrough_gtor(Ly, Lx, nblks, ming, ming2)


#%% visual inspection 
fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(ming, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title='ref ch1 (from ops)')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(ming2, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title='ref ch2 (from ops)')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(mask_weight, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title=f'mask (nblks={nblks})')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(ming2-mask_weight*ming, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title=f'crd. ch2 (nblks={nblks})')


#%% get binary mask 
'''
strategy: eliminate all pixels with bleedthrough above the 95 percentile
caveat  : obviously this is quite arbitrary, so be careful with this 
'''
threshold = np.percentile(mask_weight, 99.9)

binary_mask = (mask_weight <= threshold).astype(np.uint8)

fig, ax = plt.subplots(figsize=(3.5,2.7))
ax.hist(mask_weight.flatten(), bins=500)
ax.axvspan(threshold, threshold+1e-3, color='red', linestyle='dashed')
ax.set(title='pxl. bleedthrough distr.',
       xlabel='bleedthrough (OLS coefficient)',
       ylabel='freq. occurrence')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(binary_mask, aspect='auto', cmap='Greys')
plt.colorbar(im, shrink=.5)
ax.set(title='binary mask')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(ming*binary_mask, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title='binarily masked ref ch1')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(ming2*binary_mask, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title='binarily masked ref ch2')


#%% overlap between binary mask and intensity mask 
threshold_intensity_ch1 = np.percentile(ming, 99.9)

intensity_mask = (ming >= threshold_intensity_ch1).astype(np.uint8)

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(intensity_mask, aspect='auto', cmap='Greys')
plt.colorbar(im, shrink=.5)
ax.set(title='intensity mask (ch1)')


#%% visualisation
fig, ax = plt.subplots(figsize=(3,3))

ax.scatter(ming.flatten(), mask_weight.flatten(), s=1)
ax.set(xlabel='ref ch1 intensity (pixel-wise)',
       ylabel='bleedthrough (alpha, pixel-wise)')


#%% dot product based
corr = (ming.flatten() * ming2.flatten()).reshape((512,512))

corr_mask = (corr <= np.percentile(corr, 99.9)).astype(np.uint8)

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(corr, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title='dot product map (ch1 . ch2)')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(corr_mask, aspect='auto', cmap='Greys')
plt.colorbar(im, shrink=.5)
ax.set(title='dot product-based mask')

fig, ax = plt.subplots(figsize=(3.5,3))
im = ax.imshow(corr_mask*ming2, aspect='auto')
plt.colorbar(im, shrink=.5)
ax.set(title='dot product-masked ref ch2')


#%% pixel-wise correlation 
tot_frames = ops['nframes']
shape = tot_frames, ops['Ly'], ops['Lx']
mov = np.memmap(
    r'Z:\Jingyu\2P_Recording\AC926\AC926-20240305\02\RegOnly\suite2p\plane0\data.bin', 
    mode='r', dtype='int16', shape=shape
    )
mov2 = np.memmap(
    r'Z:\Jingyu\2P_Recording\AC926\AC926-20240305\02\RegOnly\suite2p\plane0\data_chan2.bin', 
    mode='r', dtype='int16', shape=shape
    )

from time import time
from tqdm import tqdm
from datetime import timedelta

print('ch1 trace extraction starts')
t0 = time()  # timer

masked_trace = np.zeros(tot_frames)
for f in tqdm(range(tot_frames)):
    masked_trace[f] = sum(map(sum, mov[f,:,:] * corr_mask))
    
print('ch1 trace extraction complete ({})'.format(str(timedelta(seconds=int(time()-t0)))))


import pandas as pd 
beh = pd.read_parquet(
    r'Z:\Jingyu\2P_Recording\all_session_info\all_anm_behviour.parquet', 
    engine='pyarrow'
    ).loc['AC926-20240305-02']

frame_times = beh.frame_times[:-2]

speed_times = np.hstack(beh.speed_times)
raw_times = [s[0] for s in speed_times]
raw_speeds = [s[1] for s in speed_times]
start = raw_times[0]; end = raw_times[-1]
uni_times = np.arange(start, end, 20)

uni_speeds = np.interp(uni_times, raw_times, raw_speeds)
uni_trace = np.interp(uni_times, frame_times, masked_trace)

s1 = uni_trace - np.mean(uni_trace)
s2 = uni_speeds - np.mean(uni_speeds)
corr = np.correlate(s1, s2, mode='full')
lags = np.arange(-len(s1) + 1, len(s1))/50  # convert to seconds

# plotting 
fig, ax = plt.subplots(figsize=(3,3))

ax.plot(lags[int(len(lags)/2-1000):int(len(lags)/2+1000)], corr[int(len(lags)/2-1000):int(len(lags)/2+1000)])

ax.set(xlabel='lag (s)',
       ylabel='cross-correlation',
       title=f'peak @ {lags[np.argmax(corr)]}')


# fft
from numpy.fft import rfft, rfftfreq

# session-specific parameters 
samp_freq = 50
dt = 1/samp_freq
N = len(corr)
T = N * dt  # total time in seconds
df = samp_freq / N  # freq resolution 
fNQ = samp_freq / 2  # Nyquist 

faxis = rfftfreq(N, d=dt)

fft_whole_field = rfft(corr - np.mean(corr))
spectrum = np.abs(fft_whole_field) ** 2 / N

fig, ax = plt.subplots(figsize=(4, 2.8))
ax.plot(faxis, spectrum, color='darkgreen', lw=1)
ax.set(
    xlabel='frequency (Hz)',
    ylabel='power (a.u.)',
    ylim=(0, max(spectrum)),
    title='correlagram Fourier transform'
)
ax.set(xlim=(0.05, 0.35),
       ylim=(0, 1e22))

plt.show()



    speed = np.hstack(beh.speed_times)
    speed_times = [i[0] for i in speed]
    speeds = [i[1] for i in speed]
    

    # align speed with imaging frames
    start_value = min(speed_times, key=lambda x: abs(x-frame_times[0]))
    start_index = speed_times.index(start_value)
    end_value = min(speed_times, key=lambda x: abs(x-frame_times[-1]))
    end_index = speed_times.index(end_value)
    speeds = np.array(speeds[start_index:end_index], dtype='float32')
    speed_times = np.array(speed_times[start_index:end_index], dtype='float32')
    # interpolate speed to frames
    speed_frames = np.interp(frame_times, speed_times, speeds)

    ch1 = dFF_dlight[0, :]
    s1 = ch1 - np.mean(ch1)
    s2 = speed_frames - np.mean(speed_frames)
    corr = np.correlate(s1, s2, mode='full')
    lags = np.arange(-len(s1) + 1, len(s1))

    fig, ax = plt.subplots(figsize=(3,3), dpi=200)
    ax.plot(lags[int(len(lags)/2-600):int(len(lags)/2+600)], corr[int(len(lags)/2-600):int(len(lags)/2+600)],
            lw=1)
    ax.set(xlabel='lag',
           ylabel='cross-correlation_ch1_speed',
           title=f'peak @ {np.argmax(corr)-lags[-1]}')

    fig.tight_layout()
    fig.savefig(
        r"Z:\Jingyu\2P_Recording\dlight_analysis\dFF_whole_FOV\correlogram_ch1&ch2_speed\{}_corrlogram_ch1_speed.png"
        .format(session),
        dpi=300,
        bbox_inches='tight')
    plt.close(fig)

    ch2 = dFF_red[0, :]
    s1 = ch2 - np.mean(ch2)
    s2 = speed_frames - np.mean(speed_frames)
    corr = np.correlate(s1, s2, mode='full')
    lags = np.arange(-len(s1) + 1, len(s1))

    fig, ax = plt.subplots(figsize=(3,3), dpi=200)
    ax.plot(lags[int(len(lags)/2-600):int(len(lags)/2+600)], corr[int(len(lags)/2-600):int(len(lags)/2+600)],
            lw=1)
    ax.set(xlabel='lag',
           ylabel='cross-correlation_ch2_speed',
           title=f'peak @ {np.argmax(corr)-lags[-1]}')

    fig.tight_layout()
    fig.savefig(
        r"Z:\Jingyu\2P_Recording\dlight_analysis\dFF_whole_FOV\correlogram_ch1&ch2_speed\{}_corrlogram_ch2_speed.png"
        .format(session),
        dpi=300,
        bbox_inches='tight')
    plt.close(fig)
    
    
    
    
    
#%% 
def correct_bleedthrough_gtor(Ly, Lx, nblks, mimg, mimg2, threshold):
    """
    Subtract the bleedthrough of the green channel into the red channel using non-rigid regression.
    
    Parameters:
    - Ly, Lx: Dimensions of the image.
    - nblks: Number of blocks along each axis.
    - mimg: Green channel image (2D array).
    - mimg2: Red channel image (2D array).
    
    Note:
    The function uses an externally defined function `quadrant_mask` to generate quadrant masks.
    
    Returns:
    - mask_weight: The combined block weights from the spatial mask.
    - ming2_est: The estimated red channel bleedthrough correction.
    - ming2_est_thre: The estimated red channel values thresholded to highlight significant bleedthrough.
    """
    from tqdm import tqdm 
    
    # Initialize variables
    sT = np.round((Ly + Lx) / (nblks * 2) * 0.25)
    mask = np.zeros((Ly, Lx, nblks, nblks), dtype=np.float32)
    weights = np.zeros((nblks, nblks), dtype=np.float32)
    yb = np.linspace(0, Ly, nblks + 1).astype(int)
    xb = np.linspace(0, Lx, nblks + 1).astype(int)

    # Iterate through blocks
    for iy in tqdm(range(nblks)):
        for ix in range(nblks):
            # Block indices
            ny = np.arange(yb[iy], yb[iy + 1])
            nx = np.arange(xb[ix], xb[ix + 1])

            # Generate quadrant mask
            mask[:, :, iy, ix] = quadrant_mask(Ly, Lx, ny, nx, sT)

            # Flatten and normalize the green and red blocks
            block_green = mimg[np.ix_(ny, nx)].flatten()
            block_red = mimg2[np.ix_(ny, nx)].flatten()

            green_max = np.max(block_green) if np.max(block_green) > 0 else 1
            red_max = np.max(block_red) if np.max(block_red) > 0 else 1

            x = block_green / green_max
            x2 = block_red / red_max

            # Compute regression coefficient (estimating green with red)
            if (x * x).sum() > 0:
                a = (x * x2).sum() / (x * x).sum()
            else:
                a = 0

            # Clip weights to ensure they don't cause overcorrection
            weights[iy, ix] = np.clip(a, 0, 1)

    # Normalize the mask to ensure proper scaling
    normalization_factor = mask.sum(axis=-1).sum(axis=-1)[:, :, np.newaxis, np.newaxis]
    normalization_factor[normalization_factor == 0] = 1  # Avoid division by zero
    mask /= normalization_factor
    mask *= weights
    mask_weight = mask.sum(axis=-1).sum(axis=-1)
    
    # return mask_weight

    mimg = mimg [:, :, np.newaxis, np.newaxis]  #/ np.max(mimg.flatten())
    mask *= mimg
    ming2_est = mask.sum(axis=-1).sum(axis=-1)

    # Ensure mask_weight does not exceed the original pixel values
    ming2_est = np.minimum(ming2_est, mimg2)

    # Find pixels greater than the threshold
    mask1 = ming2_est > threshold

    # Get the intensity values of these pixels
    ming2_est_thre = ming2_est * mask1.astype(mask_weight.dtype)

    return mask_weight, ming2_est, ming2_est_thre