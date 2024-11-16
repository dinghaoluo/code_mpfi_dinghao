# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:03:35 2024

convert bin file to format of choice; GUI-equipped
modified to support targeted conversion, 12 Nov 2024, Dinghao 

@author: Dinghao Luo
"""


#%% imports
import numpy as np  
import sys 
import tifffile
import os
import cv2
from tqdm import tqdm
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import ttk, filedialog
from time import time
from datetime import timedelta
import threading  # to seperate the main thread from the processing thread (prevents GUI from freezing)

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
from imaging_utility_functions import gaussian_kernel_unity


#%% global variables
halt_flag = False


#%% functions
def create_gui():
    # create main window
    window = tk.Tk()
    window.title('tif-avi Conversion')
    
    # configure grid columns to allow for a new column on the right
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=0)
    window.grid_columnconfigure(2, weight=1)  # new column for frame_display_label

    # input file
    in_path_label = ttk.Label(window, text='select .tif file: ')
    in_path_label.grid(row=0, column=0, padx=10, pady=(5, 2), sticky='w')
    in_path_entry = ttk.Entry(window)
    in_path_entry.grid(row=1, column=0, padx=10, pady=(2, 5), sticky='ew')
    in_browse_button = ttk.Button(window, text='Browse', 
                                  command=lambda: in_path_entry.insert(0, filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])))
    in_browse_button.grid(row=1, column=1, padx=5, pady=(2, 5))

    # output path
    out_path_label = ttk.Label(window, text='select output path: ')
    out_path_label.grid(row=2, column=0, padx=10, pady=(5, 2), sticky='w')
    out_path_entry = ttk.Entry(window)
    out_path_entry.grid(row=3, column=0, padx=10, pady=(2, 5), sticky='ew')
    out_path_entry.insert(0, r'Z:\Dinghao\temp')
    out_browse_button = ttk.Button(window, text='Browse', 
                                   command=lambda: out_path_entry.insert(0, filedialog.askdirectory()))
    out_browse_button.grid(row=3, column=1, padx=5, pady=(2, 5))
        
    # parameter frame
    param_frame = ttk.Frame(window)
    param_frame.grid(row=4, rowspan=4, column=0, columnspan=2, pady=5, sticky='w')

    # seconds to convert
    seconds_label = ttk.Label(param_frame, text='seconds to convert:')
    seconds_label.grid(row=0, column=0, padx=10, pady=(3, 1), sticky='w')
    seconds_entry = ttk.Entry(param_frame, width=10)
    seconds_entry.grid(row=0, column=1, padx=5, pady=(3, 1), sticky='w')
    seconds_entry.insert(0, '3')
    
    # sampling frequency
    freq_label = ttk.Label(param_frame, text='sampling frequency (Hz):')
    freq_label.grid(row=1, column=0, padx=10, pady=1, sticky='w')
    freq_entry = ttk.Entry(param_frame, width=10)
    freq_entry.grid(row=1, column=1, padx=5, pady=1, sticky='w')
    freq_entry.insert(0, '30')

    # codec selection 
    codec_label = ttk.Label(param_frame, text='encoding codec:')
    codec_label.grid(row=2, column=0, padx=10, pady=1, sticky='w')
    codec_options = ['HFYU', 'XVID', 'MJPG']
    codec_var = tk.StringVar(value=codec_options[0])
    codec_dropdown = ttk.Combobox(param_frame, textvariable=codec_var, values=codec_options, state='readonly')
    codec_dropdown.grid(row=2, column=1, padx=10, pady=1)
    
    # end format selection 
    format_label = ttk.Label(param_frame, text='select output format:')
    format_label.grid(row=3, column=0, padx=10, pady=(1, 3), sticky='w')
    format_options = ['.avi', '.mov', '.mkv']
    format_var = tk.StringVar(value=format_options[0])
    format_dropdown = ttk.Combobox(param_frame, textvariable=format_var, values=format_options, state='readonly')
    format_dropdown.grid(row=3, column=1, padx=10, pady=(1, 3), sticky='ew')
        
    # terminal
    output_text = tk.Text(window, height=10, width=60, wrap=tk.WORD)
    output_text.grid(row=8, column=0, columnspan=2, padx=10, pady=(5, 10), sticky='ns')  # adjust sticky to north-south

    # move frame_display_label to the right column
    frame_display_label = ttk.Label(window)
    frame_display_label.grid(row=0, column=2, rowspan=9, padx=10, pady=10, sticky='ns')  # spans all rows, aligns vertically

    # button frame to contain the Run and Halt buttons
    button_frame = ttk.Frame(window)
    button_frame.grid(row=9, column=0, columnspan=2, pady=(5, 10), sticky='w')  

    # run button
    run_button = ttk.Button(button_frame, text='Run', 
                            command=lambda: start_processing_thread(in_path_entry.get(), 
                                                                    out_path_entry.get(), 
                                                                    codec_var.get(),
                                                                    format_var.get(),
                                                                    output_text, 
                                                                    frame_display_label,
                                                                    seconds_entry.get(),
                                                                    freq_entry.get()))
    run_button.grid(row=0, column=0, padx=10, pady=(5, 5))

    # halt button
    halt_button = ttk.Button(button_frame, text='Halt', command=halt_processing)
    halt_button.grid(row=0, column=1, pady=(5, 5))
    
    window.mainloop()


def create_mov(in_path_entry, out_path_entry, codec_var, format_var, 
               seconds_entry, freq_entry,
               write_func=None, display_func=None, 
               smooth=True, sigma=3):
    global halt_flag
    
    # load the .tif file 
    write_func(f'loading .tif from {in_path_entry}\n') if write_func else print(f'loading .tif file from {in_path_entry}\n')
    mov = tifffile.imread(in_path_entry)
    tot_frames, height, width = mov.shape
    
    # output path for the .avi file 
    tif_name = os.path.splitext(os.path.basename(in_path_entry))[0]
    out_path = os.path.join(out_path_entry, f'{tif_name}{format_var}')  # user chooses the end format, Dinghao 14 Nov 2024
    write_func(f'writing .avi to {out_path}\n') if write_func else print(f'writing .avi to {out_path}')
    
    if smooth:
        write_func('smoothing through time...\n') if write_func else print('smoothing through time...')
        t0 = time()
        kernel = gaussian_kernel_unity(sigma)
        pad_width = len(kernel) // 2
        mov_padded = np.pad(mov, ((pad_width, pad_width), (0, 0), (0, 0)), mode='reflect')
        mov = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 
                                  axis=0, 
                                  arr=mov_padded)[pad_width:-pad_width, :, :]
        if write_func:
            write_func(f'smoothing done ({timedelta(seconds=int(time()-t0))})\n')
        else:
            print(f'smoothing done ({timedelta(seconds=int(time()-t0))})')
    mov_min = mov.min()
    mov_max = mov.max()
    
    fourcc = cv2.VideoWriter_fourcc(*codec_var)
    out = cv2.VideoWriter(out_path, fourcc, int(freq_entry), (width, height), isColor=False)
    
    user_defined_frames = int(seconds_entry) * int(freq_entry)
    frames_to_write = min(user_defined_frames, tot_frames)
    if not out.isOpened():
        write_func('error: could not open video writer\n') if write_func else print('error: could not open video writer')
        return
    else:
        write_func(f'writing video to disk ({frames_to_write} f)...\n') if write_func else print(f'writing video to disk ({frames_to_write} f)...')
        for frame in tqdm(range(frames_to_write), file=sys.stdout, ncols=100):
            if halt_flag:
                write_func('process halted') if write_func else print('process halted')
                break
                
            frame_data = mov[frame, :, :]
            normalised_frame = ((frame_data - mov_min) / 
                                (mov_max - mov_min) * 255).astype('uint8')  # Normalize to 8-bit
            
            # Write to video
            out.write(normalised_frame)
            
            # Convert the frame to an Image format for display
            if display_func:
                pil_image = Image.fromarray(normalised_frame)
                display_func(pil_image)
    
        out.release()
        write_func('video saved successfully\n') if write_func else print('video saved successfully')
        

def halt_processing():
    global halt_flag
    halt_flag = True


def run_processing(in_path_entry, out_path_entry, codec_var, format_var, 
                   output_text, frame_display_label, 
                   seconds_entry, freq_entry):
    if in_path_entry[-3:]!='tif':
        output_text.insert(tk.END, '\nnot a .tif file\n')
        return None
        
    create_mov(in_path_entry, 
               out_path_entry,
               codec_var,
               format_var,
               seconds_entry,
               freq_entry,
               write_func=lambda msg: output_text.insert(tk.END, msg), 
               display_func=lambda pil_img: update_frame_display(pil_img, frame_display_label))
    
    output_text.insert(tk.END, 'done\n')
    output_text.yview(tk.END)
    

def start_processing_thread(in_path_entry, out_path_entry, codec_var, format_var, 
                            output_text, frame_display_label, 
                            seconds_entry, freq_entry):
    # run the processing function in a separate thread to avoid freezing the GUI
    processing_thread = threading.Thread(target=run_processing, args=(in_path_entry,
                                                                      out_path_entry,
                                                                      codec_var,
                                                                      format_var,
                                                                      output_text,
                                                                      frame_display_label,
                                                                      seconds_entry,
                                                                      freq_entry))
    processing_thread.start()
   
    
def update_frame_display(pil_image, label, brightness_factor=1.5, contrast_factor=1.5):
    """Update the image in the tkinter Label widget"""
    # enhance the image
    brightness_enhancer = ImageEnhance.Brightness(pil_image)
    pil_image_bright = brightness_enhancer.enhance(brightness_factor)
    contrast_enhancer = ImageEnhance.Contrast(pil_image_bright)
    pil_image_contrast = contrast_enhancer.enhance(contrast_factor)
    
    # convert to ImageTk format
    tk_image = ImageTk.PhotoImage(pil_image_contrast)
    
    # update the label with the new image
    label.config(image=tk_image)
    label.image = tk_image  # keep a reference to avoid garbage collection


#%% Run the GUI
if __name__ == '__main__':
    create_gui()