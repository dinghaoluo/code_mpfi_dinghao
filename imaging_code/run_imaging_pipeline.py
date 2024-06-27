# -*- coding: utf-8 -*-
"""
Created on Mon 13 May 17:13:42 2024
Modified on Tue 25 June 15:09:40 2024

This code combines grid_extract.py (Dinghao) and after_suite2p.py (Jingyu)

@authors: Dinghao Luo, Jingyu Cao
@modifiers: Dinghao Luo, Jingyu Cao
"""


#%% imports 
import sys

if (r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf
import imaging_pipeline_main_functions as ipmf


#%% suite2p ROIs or grid ROIs
roi_switch = input('process with...\n1: suite2p ROIs\n2: grid ROIs\n')
if roi_switch not in ['1', '2']:
    print('\nWARNING:\ninvalid input; halting processing')


#%% customise parameters
# plot reference or not 
plot_ref = True

# align to...
align_run = 1
align_rew = 1
align_cue = 0

# smoothing
smooth = 1

# how many seconds before and after each align_to landmark 
bef = 1; aft = 4  # in seconds 

if roi_switch=='1':
    # do we want to calculate dFF
    dFF = 1
    
if roi_switch=='2':
    # how many pixels x/y for each grid
    stride = 16
    # stride = 25
    # stride = 100
    border = 8  # ignore how many pixels at the border (1 side)
    # border = 6
    fit = ipf.check_stride_border(stride, border)
    
    # do we want to save the grid_traces if extracted 
    save_grids = 1

# print out 
if roi_switch=='1': roi_string='suite2p'
if roi_switch=='2': roi_string='grid'
printout = """
processing {} ROIs...
    align_run = {}
    align_rew = {}
    align_cue = {}
    bef = {}
    aft = {}
    stride = {}
    border = {}
    smooth = {}
    """.format(roi_string, align_run, align_rew, align_cue, bef, aft, stride, border, smooth)
print(printout)


#%% path
rec_path = r'Z:\Dinghao\2p_recording\A091i\A091i-20240613\A091i-20240613-01'

if 'Dinghao' in rec_path:
    reg_path = rec_path+r'\processed\suite2p\plane0'
    recname = rec_path[-17:]
    txt_path = r'Z:\Dinghao\MiceExp\ANMD{}\{}{}T.txt'.format(recname[1:4], recname[:4], recname[5:])
if 'Jingyu' in rec_path:
    reg_path = rec_path+r'\RegOnly\suite2p\plane0'
    recname = rec_path[-17:-3]+'-'+rec_path[-2:]
    txt_path = r'Z:\Jingyu\mice-expdata\{}\A{}T.txt'.format(rec_path[-23:-18], recname[2:])


#%% run 
if roi_switch=='1':
    ipmf.run_suite2p_pipeline
if roi_switch=='2':
    ipmf.run_grid_pipeline(rec_path, recname, reg_path, txt_path, 
                           stride, border, 
                           plot_ref, 
                           smooth, save_grids, 
                           bef, aft,
                           align_run, align_rew, align_cue)




















# opsfile = reg_path+r'\ops.npy'
# binfile = reg_path+r'\data.bin'
# bin2file = reg_path+r'\data_chan2.bin'

# extract_path = rec_path+r'_grid_extract'
# if not os.path.exists(extract_path):
#     os.makedirs(extract_path)

# # beh file
# print('\nreading behaviour file...')
# txt = ipf.process_txt(txt_path)


# #%% define gridmesh
# # parameters 
# grids = ipf.make_grid(stride=stride, dim=512, border=border)
# tot_grid = len(grids)**2

# # load data 
# ops = np.load(opsfile, allow_pickle=True).item()
# tot_frames = ops['nframes']
# shape = tot_frames, ops['Ly'], ops['Lx']
# mov = np.memmap(binfile, mode='r', dtype='int16', shape=shape)
# mov2 = np.memmap(bin2file, mode='r', dtype='int16', shape=shape)


# #%% plotting parameters
# dimension = len(grids)
# tot_plot = dimension**2
# plc.rcParams['figure.figsize'] = (dimension*2, dimension*2)


# #%% plot references (optional)
# plot_ref = True  # switch 
# if plot_ref:
#     if not os.path.exists(extract_path+r'\ref_ch1_{}.png'.format(stride)):
#         print('\ngenerating reference images...')
        
#         t0 = time()  # timer
#         ipf.plot_reference(mov, grids, stride, 512, 1, extract_path)
#         print('ref done ({})'.format(str(timedelta(seconds=int(time()-t0)))))
#         ipf.plot_reference(mov2, grids, stride, 512, 2, extract_path)
#         print('ref_ch2 done ({})'.format(str(timedelta(seconds=int(time()-t0)))))


# #%% do we need to extract traces? (channel 1)
# extract_trace = True  # default to True
# extract_file_path = extract_path+r'\grid_traces_{}.npy'.format(stride)
# if os.path.exists(extract_file_path):
#     extract_trace = False


# #%% extract traces (this is going to take the most time)
# if extract_trace:
#     print('\nch1 trace extraction starts')
#     t0 = time()  # timer
#     grid_traces = np.zeros((tot_grid, tot_frames))
    
#     for f in range(tot_frames):
#         # progress report 
#         for p in [.25, .5, .75]:
#             if f==int(tot_frames*p):
#                 print('{} ({}%) frames done ({})'.format(f, int(p*100), str(timedelta(seconds=int(time()-t0)))))
        
#         curr_frame = ipf.run_grid(mov[f,:,:], grids, tot_grid, stride)
#         for g in range(tot_grid):
#             grid_traces[g, f] = ipf.sum_mat(curr_frame[g])
#     print('ch1 trace extraction complete ({})'.format(str(timedelta(seconds=int(time()-t0)))))
    

# #%% save grid traces (optional but recommended)
# if extract_trace and save_grids:
#     np.save(extract_file_path, grid_traces)
#     print('ch1 traces saved to {}\n'.format(extract_file_path))

    
# #%% do we need to extract traces? (channel 2)
# extract_trace_ch2 = True  # default to True
# extract_file_path_ch2 = extract_path+r'\grid_traces_{}_ch2.npy'.format(stride)
# if os.path.exists(extract_file_path_ch2):
#     extract_trace_ch2 = False


# #%% extract traces (channel 2)
# if extract_trace_ch2:
#     print('ch2 trace extraction starts')
#     t0 = time()  # timee
#     grid_traces_ch2 = np.zeros((tot_grid, tot_frames))
    
#     for f in range(tot_frames):
#         # progress report 
#         for p in [.25, .5, .75]:
#             if f==int(tot_frames*p):
#                 print('{} ({}%) frames done ({})'.format(f, int(p*100), str(timedelta(seconds=int(time()-t0)))))
        
#         curr_frame = ipf.run_grid(mov2[f,:,:], grids, tot_grid, stride)
#         for g in range(tot_grid):
#             grid_traces_ch2[g, f] = ipf.sum_mat(curr_frame[g])
#     print('ch2 trace extraction complete ({})'.format(str(timedelta(seconds=int(time()-t0)))))


# #%% save grid traces (channel 2)
# if extract_trace_ch2 and save_grids:
#     np.save(extract_file_path_ch2, grid_traces_ch2)
#     print('ch2 traces saved to {}\n'.format(extract_file_path_ch2))


# #%% read grid traces (if exists)
# if extract_trace==False:
#     grid_traces = np.load(extract_file_path, allow_pickle=True)
#     grid_traces_ch2 = np.load(extract_file_path_ch2, allow_pickle=True)
#     print('traces read from {}\n'.format(extract_path))


# #%% timestamps
# print('\ndetermining behavioural timestamps...') 
# # pumps
# pump_times = txt['pump_times']

# # speed
# speed_times = txt['speed_times']

# # cues 
# movie_times = txt['movie_times']

# # frames
# frame_times = txt['frame_times']

# tot_trial = len(speed_times)
# tot_frame = len(frame_times)


# #%% correct overflow
# print('\ncorrecting overflow...')
# pump_times = ipf.correct_overflow(pump_times, 'pump')
# speed_times = ipf.correct_overflow(speed_times, 'speed')
# movie_times = ipf.correct_overflow(movie_times, 'movie')
# lick_times = ipf.correct_overflow(lick_times, 'lick')
# frame_times = ipf.correct_overflow(frame_times, 'frame')
# first_frame = frame_times[0]; last_frame = frame_times[-1]


# #%% get lick frames
# lick_frames = []
# for t in lick_times:
#     if t[0][0]>first_frame and t[-1][0]<last_frame:
#         licks = [ipf.find_nearest(lick[0], frame_times) for lick in t]
#         lick_frames.append(licks)


# #%% **fill in dropped $FM signals
# # since the 2P system does not always successfully transmit frame signals to
# # the behavioural recording system every time it acquires a frame, one needs to
# # manually interpolate the frame signals in between 2 frame signals that are 
# # further apart than 50 ms
# print('\nfilling in dropped $FM statements...')
# for i in range(len(frame_times)-1):
#     if frame_times[i+1]-frame_times[i]>50:
#         interp_fm = (frame_times[i+1]+frame_times[i])/2
#         frame_times.insert(i+1, interp_fm)
        
# # checks $FM against tot_frame
# if tot_frame<len(frame_times)-3 or tot_frames>len(frame_times):
#     print('\nWARNING:\ncheck $FM; halting processing for {}\n'.format(recname))
#     align_run==0; align_rew==0; align_cue==0  # if tot_frame is more than sync signals or fewer than syncs-3, then halt


# #%% run-onset aligned
# if align_run==1: 
#     print('\nplotting traces aligned to RUN...')
    
#     # find run-onsets
#     run_onsets = []
    
#     for trial in range(tot_trial):
#         times = [s[0] for s in speed_times[trial]]
#         speeds = [s[1] for s in speed_times[trial]]
#         uni_time = np.linspace(times[0], times[-1], int((times[-1] - times[0])))
#         uni_speed = np.interp(uni_time, times, speeds)  # interpolation for speed
        
#         run_onsets.append(ipf.get_onset(uni_speed, uni_time))
        
#     # filter run-onsets ([first_frame, last_frame])
#     run_onsets = [t for t in run_onsets if t>first_frame and t<last_frame]
    
#     # get aligned frame numbers
#     run_frames = []
#     for trial in range(len(run_onsets)):
#         if run_onsets!=-1:  # if there is a clear run-onset in this trial
#             rf = ipf.find_nearest(run_onsets[trial], frame_times)
#             if rf!=0:
#                 run_frames.append(rf)  # find the nearest frame
#         else:
#             run_frames.append(-1)
            
#     # align traces to run-onsets
#     tot_run = len(run_onsets)
#     run_aligned = np.zeros((tot_grid, tot_run-1, (bef+aft)*30))  # grid x trial x frame bin
#     run_aligned_ch2 = np.zeros((tot_grid, tot_run-1, (bef+aft)*30))
#     for i, r in enumerate(run_frames[:-1]):
#         run_aligned[:, i, :] = grid_traces[:, r-bef*30:r+aft*30]
#         run_aligned_ch2[:, i, :] = grid_traces_ch2[:, r-bef*30:r+aft*30]
    
#     print('plotting heatmaps...')
#     # heatmap chan1
#     fig = plt.figure(1)
#     for p in range(tot_plot):
#         curr_grid_trace = run_aligned[p, :, :]
#         curr_grid_map = np.zeros((tot_run-1, (bef+aft)*30))
#         for i in range(tot_run-1):
#             curr_grid_map[i, :] = normalise(curr_grid_trace[i, :])
            
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
#         ax.imshow(curr_grid_map, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
#     fig.suptitle('run_aligned')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_run_aligned.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
    
#     # heatmap chan2
#     fig = plt.figure(1)
#     for p in range(tot_plot):
#         curr_grid_trace_ch2 = run_aligned_ch2[p, :, :]
#         curr_grid_map_ch2 = np.zeros((tot_run-1, (bef+aft)*30))
#         for i in range(tot_run-1):
#             curr_grid_map_ch2[i, :] = normalise(curr_grid_trace_ch2[i, :])
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
#         ax.imshow(curr_grid_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_run], cmap='Greys')
#     fig.suptitle('run_aligned_ch2')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_run_aligned_ch2.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
    
#     print('plotting combined averaged traces...')
#     # average combined 
#     fig = plt.figure(1, figsize=(dimension*4.5, dimension*3))
#     for p in range(tot_plot):
#         curr_grid_trace = run_aligned[p, :, :]
#         curr_grid_trace_ch2 = run_aligned_ch2[p, :, :]
#         mean_trace = np.mean(curr_grid_trace, axis=0)
#         mean_trace_ch2 = np.mean(curr_grid_trace_ch2, axis=0)
#         sem_trace = sem(curr_grid_trace, axis=0)
#         sem_trace_ch2 = sem(curr_grid_trace_ch2, axis=0)
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='F', title='grid {}'.format(p))
#         ax.plot(xaxis, mean_trace, color='limegreen', linewidth=1)
#         ax.fill_between(xaxis, mean_trace+sem_trace,
#                                mean_trace-sem_trace,
#                         color='limegreen', edgecolor='none', alpha=.2)
#         ax2 = ax.twinx()
#         ax2.plot(xaxis, mean_trace_ch2, color='red', linewidth=1)
#         ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
#                                 mean_trace_ch2-sem_trace_ch2,
#                          color='red', edgecolor='none', alpha=.2)
#         ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
#     fig.suptitle('run_aligned_ch2')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_avg_run_aligned.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)


# #%% pump aligned
# if align_rew==1:
#     print('\nplotting traces aligned to REW...')
    
#     # filter pumps ([first_frame, last_frame])
#     pumps = [t for t in pump_times if t>first_frame and t<last_frame]
    
#     # get aligned frame numbers
#     pump_frames = []
#     for trial in range(len(pumps)):
#         pump_frames.append(ipf.find_nearest(pumps[trial], frame_times))

#     # align traces to pumps
#     tot_pump = len(pumps)
#     pump_aligned = np.zeros((tot_grid, (tot_pump-1), (bef+aft)*30))
#     pump_aligned_ch2 = np.zeros((tot_grid, (tot_pump-1), (bef+aft)*30))
#     for i, p in enumerate(pump_frames[:-1]):
#         pump_aligned[:, i, :] = grid_traces[:, p-bef*30:p+aft*30]
#         pump_aligned_ch2[:, i, :] = grid_traces_ch2[:, p-bef*30:p+aft*30]
        
#     print('plotting heatmaps...')
#     # heatmap ch1
#     fig = plt.figure(1)
#     for p in range(tot_plot):
#         curr_grid_trace = pump_aligned[p, :, :]
#         curr_grid_map = np.zeros((tot_pump-1, (bef+aft)*30))
#         for i in range(tot_pump-1):
#             curr_grid_map[i, :] = normalise(curr_grid_trace[i, :])
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
#         ax.imshow(curr_grid_map, aspect='auto', extent=[-bef,aft,1,tot_pump], cmap='Greys')

#     fig.suptitle('reward_aligned')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_rew_aligned.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
        
#     # heatmap ch2
#     fig = plt.figure(1)
#     for p in range(tot_plot):
#         curr_grid_trace_ch2 = pump_aligned_ch2[p, :, :]
#         curr_grid_map_ch2 = np.zeros((tot_pump-1, (bef+aft)*30))
#         for i in range(tot_pump-1):
#             curr_grid_map_ch2[i, :] = normalise(curr_grid_trace[i, :])
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
#         ax.imshow(curr_grid_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_pump], cmap='Greys')

#     fig.suptitle('reward_aligned_ch2')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_rew_aligned_ch2.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
    
#     print('plotting combined averaged traces...')
#     # average combined 
#     fig = plt.figure(1, figsize=(dimension*4.5, dimension*3))
#     for p in range(tot_plot):
#         curr_grid_trace = pump_aligned[p, :, :]
#         curr_grid_trace_ch2 = pump_aligned_ch2[p, :, :]
#         mean_trace = np.mean(curr_grid_trace, axis=0)
#         mean_trace_ch2 = np.mean(curr_grid_trace_ch2, axis=0)
#         sem_trace = sem(curr_grid_trace, axis=0)
#         sem_trace_ch2 = sem(curr_grid_trace_ch2, axis=0)
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='F', title='grid {}'.format(p))
#         ax.plot(xaxis, mean_trace, color='limegreen', linewidth=1)
#         ax.fill_between(xaxis, mean_trace+sem_trace,
#                                mean_trace-sem_trace,
#                         color='limegreen', edgecolor='none', alpha=.2)
#         ax2 = ax.twinx()
#         ax2.plot(xaxis, mean_trace_ch2, color='red', linewidth=1)
#         ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
#                                 mean_trace_ch2-sem_trace_ch2,
#                          color='red', edgecolor='none', alpha=.2)
#         ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
#     fig.suptitle('rew_aligned_ch2')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_avg_rew_aligned.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
        
        
# #%% cue-onset aligned
# if align_cue==1:
#     print('\nplotting traces aligned to CUE...')
    
#     # filter cues ([first_frame, last_frame])
#     cues = [t for t in cue_times if t>first_frame and t<last_frame]
    
#     # get aligned frame numbers
#     cue_frames = []
#     for trial in range(len(cues)):
#         cue_frames.append(ipf.find_nearest(cues[trial], frame_times))

#     # align traces to cues
#     tot_cue = len(cues)
#     cue_aligned = np.zeros((tot_grid, (tot_cue-1), (bef+aft)*30))
#     cue_aligned_ch2 = np.zeros((tot_grid, (tot_cue-1), (bef+aft)*30))
#     for i, c in enumerate(cue_frames[:-1]):
#         cue_aligned[:, i, :] = grid_traces[:, c-bef*30:c+aft*30]
#         cue_aligned_ch2[:, i, :] = grid_traces_ch2[:, c-bef*30:c+aft*30]
        
#     print('plotting heatmaps...')
#     # heatmap ch1
#     fig = plt.figure(1)
#     for p in range(tot_plot):
#         curr_grid_trace = cue_aligned[p, :, :]
#         curr_grid_map = np.zeros((tot_cue-1, (bef+aft)*30))
#         for i in range(tot_cue-1):
#             curr_grid_map[i, :] = normalise(curr_grid_trace[i, :])
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
#         ax.imshow(curr_grid_map, aspect='auto', extent=[-bef,aft,1,tot_cue], cmap='Greys')

#     fig.suptitle('cue_aligned')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_cue_aligned.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
        
#     # heatmap ch2
#     fig = plt.figure(1)
#     for p in range(tot_plot):
#         curr_grid_trace_ch2 = cue_aligned_ch2[p, :, :]
#         curr_grid_map_ch2 = np.zeros((tot_cue-1, (bef+aft)*30))
#         for i in range(tot_cue-1):
#             curr_grid_map_ch2[i, :] = normalise(curr_grid_trace[i, :])
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='trial #', title='grid {}'.format(p))
#         ax.imshow(curr_grid_map_ch2, aspect='auto', extent=[-bef,aft,1,tot_cue], cmap='Greys')

#     fig.suptitle('cue_aligned_ch2')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_cue_aligned_ch2.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
    
#     print('plotting combined averaged traces...')
#     # average combined 
#     fig = plt.figure(1, figsize=(dimension*4.5, dimension*3))
#     for p in range(tot_plot):
#         curr_grid_trace = cue_aligned[p, :, :]
#         curr_grid_trace_ch2 = cue_aligned_ch2[p, :, :]
#         mean_trace = np.mean(curr_grid_trace, axis=0)
#         mean_trace_ch2 = np.mean(curr_grid_trace_ch2, axis=0)
#         sem_trace = sem(curr_grid_trace, axis=0)
#         sem_trace_ch2 = sem(curr_grid_trace_ch2, axis=0)
        
#         ax = fig.add_subplot(dimension, dimension, p+1)
#         ax.set(xlabel='time (s)', ylabel='F', title='grid {}'.format(p))
#         ax.plot(xaxis, mean_trace, color='limegreen', linewidth=1)
#         ax.fill_between(xaxis, mean_trace+sem_trace,
#                                mean_trace-sem_trace,
#                         color='limegreen', edgecolor='none', alpha=.2)
#         ax2 = ax.twinx()
#         ax2.plot(xaxis, mean_trace_ch2, color='red', linewidth=1)
#         ax2.fill_between(xaxis, mean_trace_ch2+sem_trace_ch2,
#                                 mean_trace_ch2-sem_trace_ch2,
#                          color='red', edgecolor='none', alpha=.2)
#         ax.axvspan(0, 0, color='grey', alpha=.5, linestyle='dashed', linewidth=1)
#     fig.suptitle('rew_aligned_ch2')
#     fig.tight_layout()
#     fig.savefig('{}/grid_traces_{}_avg_cue_aligned.png'.format(extract_path, stride),
#                 dpi=300,
#                 bbox_inches='tight')
#     plt.show()
#     plt.close(fig)
    
    
# #%% end statement 
# print('processing for {} is finished\n\n\n'.format(recname))
#         cue_aligned[:, i*(bef+aft)*30:(i+1)*(bef+aft)*30] = grid_traces[:, c-bef*30:c+aft*30]