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
import threading  # to seperate the main thread from the processing thread (prevents GUI from freezing)

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
from imaging_utility_functions import gaussian_kernel_unity


#%% global variables
halt_flag = False
loaded_tif_data = None  # for sharing across functions
crop_rectangle = None  # for redrawing the rectangle through slider-adjustments
tif_max = None
tif_min = None


#%% functions
def create_gui():
    # create main window
    window = tk.Tk()
    window.title('tif-movie conversion')
    
    # configure grid columns to allow for a new column on the right
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=0)
    window.grid_columnconfigure(2, weight=1)  # new column for frame_display_label

    # input file
    in_path_label = ttk.Label(window, text='select .tif file (may take ~10 s to load large .tif files):')
    in_path_label.grid(row=0, column=0, padx=10, pady=(5, 1), sticky='w')
    in_path_entry = ttk.Entry(window)
    in_path_entry.grid(row=1, column=0, padx=10, pady=(1, 5), sticky='ew')
    in_browse_button = ttk.Button(window, text='Browse', 
                                  command=lambda: [in_path_entry.delete(0, tk.END),
                                                   in_path_entry.insert(0, filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])),
                                                   load_and_preview_image(in_path_entry.get(), canvas, frame_slider)])
    in_browse_button.grid(row=1, column=1, padx=5, pady=(1, 5))

    # output path
    out_path_label = ttk.Label(window, text='select output path: ')
    out_path_label.grid(row=2, column=0, padx=10, pady=(5, 1), sticky='w')
    out_path_entry = ttk.Entry(window)
    out_path_entry.grid(row=3, column=0, padx=10, pady=(1, 5), sticky='ew')
    out_path_entry.insert(0, r'Z:\Dinghao\temp')
    out_browse_button = ttk.Button(window, text='Browse', 
                                   command=lambda: out_path_entry.insert(0, filedialog.askdirectory()))
    out_browse_button.grid(row=3, column=1, padx=5, pady=(1, 5))
        
    # parameter frame
    param_frame = ttk.Frame(window)
    param_frame.grid(row=4, rowspan=3, column=0, columnspan=2, pady=5, sticky='w')
    
    # sampling frequency
    freq_label = ttk.Label(param_frame, text='sampling frequency (Hz):')
    freq_label.grid(row=0, column=0, padx=10, pady=1, sticky='w')
    freq_entry = ttk.Entry(param_frame, width=10)
    freq_entry.grid(row=0, column=1, padx=5, pady=1, sticky='w')
    freq_entry.insert(0, '30')

    # codec selection 
    codec_label = ttk.Label(param_frame, text='encoding codec:')
    codec_label.grid(row=1, column=0, padx=10, pady=1, sticky='w')
    codec_options = ['HFYU', 'XVID', 'MJPG']
    codec_var = tk.StringVar(value=codec_options[0])
    codec_dropdown = ttk.Combobox(param_frame, textvariable=codec_var, values=codec_options, state='readonly')
    codec_dropdown.grid(row=1, column=1, padx=10, pady=1)
    
    # end format selection 
    format_label = ttk.Label(param_frame, text='select output format:')
    format_label.grid(row=2, column=0, padx=10, pady=(1, 3), sticky='w')
    format_options = ['.avi', '.mov', '.mkv']
    format_var = tk.StringVar(value=format_options[0])
    format_dropdown = ttk.Combobox(param_frame, textvariable=format_var, values=format_options, state='readonly')
    format_dropdown.grid(row=2, column=1, padx=10, pady=(1, 3), sticky='ew')
        
    # terminal
    output_text = tk.Text(window, height=10, width=60, wrap=tk.WORD)
    output_text.grid(row=7, column=0, rowspan=3, columnspan=2, padx=10, pady=(5, 10), sticky='ns')  # adjust sticky to north-south
    
    # move frame_display_label to the right column
    frame_display_label = ttk.Label(window)
    frame_display_label.grid(row=0, column=2, rowspan=9, padx=10, pady=10, sticky='ns')  # spans all rows, aligns vertically

    # adding a canvas for displaying and interacting with the image
    canvas = tk.Canvas(window, width=400, height=400, bg='grey')
    canvas.grid(row=0, column=2, rowspan=9, padx=(10, 10), pady=(10, 0), sticky='nw')
        
    # button frame to contain the Run and Halt buttons
    button_frame = ttk.Frame(window)
    button_frame.grid(row=10, column=0, columnspan=2, pady=(5, 10), sticky='w')  

    # run button
    run_button = ttk.Button(button_frame, text='Run', 
                            command=lambda: start_processing_thread(in_path_entry.get(), 
                                                                    out_path_entry.get(), 
                                                                    codec_var.get(),
                                                                    format_var.get(),
                                                                    crop_entries,
                                                                    output_text,
                                                                    canvas,
                                                                    start_frame.get(), end_frame.get(),
                                                                    freq_entry.get()))
    run_button.grid(row=0, column=0, padx=10, pady=(5, 5))

    # halt button
    halt_button = ttk.Button(button_frame, text='Halt', command=halt_processing)
    halt_button.grid(row=0, column=1, pady=(5, 5))
    
    # cropping parameters
    crop_frame = ttk.Frame(window)
    crop_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky='w')
    
    crop_label = ttk.Label(crop_frame, text='crop (x, y, width, height):')
    crop_label.grid(row=0, column=0, padx=10, pady=(3, 1), sticky='w')
    
    # creating 4 entry fields for x, y, width, height
    crop_entries = [ttk.Entry(crop_frame, width=5) for _ in range(4)]
    for i, entry in enumerate(crop_entries):
        entry.grid(row=0, column=i+1, padx=(5 if i else 10, 5), pady=(3, 1), sticky='w')
    
    # default values for no cropping
    for entry, default_value in zip(crop_entries, ['0', '0', '0', '0']):  # initialise to no crop
        entry.insert(0, default_value)
    
    # enable interactive cropping by default
    enable_crop_interaction(canvas, frame_display_label, crop_entries)
    
    # frame to contain scrollbar and frame display 
    scroll_frame = ttk.Frame(window)
    scroll_frame.grid(row=9, column=2, padx=5, pady=(5,5), sticky='w')
    
    # frame navigation label (move to the right of the slider)
    frame_label = ttk.Label(scroll_frame, text='frame: 0')
    frame_label.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky='w')
    
    # horizontal scrollbar for frame navigation
    frame_slider = ttk.Scale(scroll_frame, from_=0, to=0, orient='horizontal', length=350)
    frame_slider.grid(row=0, column=1, padx=(5, 0), pady=(5, 5), sticky='w')
    
    # bind slider to frame update
    frame_slider.bind("<Motion>", lambda e: update_displayed_frame(canvas, frame_slider, frame_label))
    frame_slider.bind("<ButtonRelease-1>", lambda e: update_displayed_frame(canvas, frame_slider, frame_label))
    
    # frame selection variables
    start_frame = tk.IntVar(value=0)
    end_frame = tk.IntVar(value=0)
    
    # set start frame button
    set_start_button = ttk.Button(window, text='set start frame', 
                                  command=lambda: [start_frame.set(int(frame_slider.get())), 
                                                   frame_selection_label.config(text=f'start: {start_frame.get()} | end: {end_frame.get()}')])
    set_start_button.grid(row=10, column=2, padx=(10, 0), pady=(5, 10), sticky='w')
    
    # frame selection label (centered between the buttons)
    frame_selection_label = ttk.Label(window, text=f'start: {start_frame.get()} | end: {end_frame.get()}')
    frame_selection_label.grid(row=10, column=2, padx=(160, 160), pady=(5, 10), sticky='ew')
    
    # set end frame button
    set_end_button = ttk.Button(window, text='set end frame', 
                                command=lambda: [end_frame.set(int(frame_slider.get())), 
                                                 frame_selection_label.config(text=f'start: {start_frame.get()} | end: {end_frame.get()}')])
    set_end_button.grid(row=10, column=2, padx=(0, 10), pady=(5, 10), sticky='e')
    
    window.mainloop()

def create_mov(in_path_entry, out_path_entry, codec_var, format_var, 
               freq_entry, crop_params, start_frame, end_frame,
               write_func=None, display_func=None, 
               smooth=True, sigma=3, preloaded_data=None):
    global halt_flag

    # use preloaded data if available, otherwise load the .tif file
    if preloaded_data is not None:
        mov = preloaded_data
    else:
        write_func('loading .tif\n') if write_func else print('loading .tif file')
        mov = tifffile.imread(in_path_entry)
    
    tot_frames, height, width = mov.shape

    # output path for the video
    tif_name = os.path.splitext(os.path.basename(in_path_entry))[0]
    out_path = os.path.join(out_path_entry, f'{tif_name}{format_var}')  # use selected format
    write_func(f'writing video to {out_path}\n') if write_func else print(f'writing video to {out_path}')

    # apply cropping if crop_params are provided
    x, y, crop_width, crop_height = crop_params
    if all(crop_params):  # apply cropping if dimensions are valid
        if x + crop_width > width or y + crop_height > height:
            write_func('error: cropping dimensions exceed frame size\n') if write_func else print('error: cropping dimensions exceed frame size\n')
            return
        mov = mov[:, y:y+crop_height, x:x+crop_width]
        write_func(f'applied cropping: x={x}, y={y}, width={crop_width}, height={crop_height}\n') if write_func else print(f'applied cropping: x={x}, y={y}, width={crop_width}, height={crop_height}')
        height, width = crop_height, crop_width  # update dimensions after cropping
    else:
        write_func('no cropping applied\n') if write_func else print('no cropping applied')

    # apply smoothing if enabled
    if smooth:
        write_func('applying temporal smoothing...\n') if write_func else print('applying temporal smoothing...')
        kernel = gaussian_kernel_unity(sigma)
        pad_width = len(kernel) // 2
        mov_padded = np.pad(mov, ((pad_width, pad_width), (0, 0), (0, 0)), mode='reflect')
        mov = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=mov_padded)[pad_width:-pad_width, :, :]
        write_func('smoothing applied successfully\n') if write_func else print('smoothing applied successfully')

    # normalize frame values to 8-bit
    mov_min = mov.min()
    mov_max = mov.max()

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec_var)
    out = cv2.VideoWriter(out_path, fourcc, int(freq_entry), (width, height), isColor=False)

    if not out.isOpened():
        write_func('error: could not open video writer\n') if write_func else print('error: could not open video writer\n')
        return

    # process frames
    write_func(f'writing video (f{start_frame}-{end_frame})...\n') if write_func else print(f'writing video (f{start_frame}-{end_frame})...\n')

    for frame in tqdm(range(start_frame, end_frame), file=sys.stdout, ncols=50):
        if halt_flag:
            write_func('process halted\n') if write_func else print('process halted\n')
            break

        frame_data = mov[frame, :, :]
        normalised_frame = ((frame_data - mov_min) / (mov_max - mov_min) * 255).astype('uint8')  # normalize to 8-bit

        # write to video
        out.write(normalised_frame)

        # # display frame (optional)
        # if display_func:
        #     pil_image = Image.fromarray(normalised_frame)
        #     display_func(pil_image)

    # release video writer
    out.release()
    write_func('video saved successfully\n') if write_func else print('video saved successfully')

def display_frame_on_canvas(canvas, frame):
    """display a single frame on the canvas."""    
    normalised_frame = ((frame - tif_min) / (tif_max - tif_min) * 255).astype('uint8')
    pil_image = Image.fromarray(normalised_frame)
    display_image_on_canvas(canvas, pil_image)

def display_image_on_canvas(canvas, pil_image):
    """display the given PIL image on the canvas"""
    # convert PIL Image to ImageTk format
    tk_image = ImageTk.PhotoImage(pil_image)

    # update the canvas with the new image
    canvas.image = tk_image  # keep a reference to avoid garbage collection
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    if crop_rectangle:  # redraw the rectangle when creating an image on canvas
        x1, y1, x2, y2 = crop_rectangle
        canvas.create_rectangle(x1, y1, x2, y2, outline='red', tags='crop_rect')

def enable_crop_interaction(canvas, frame_display_label, crop_entries):
    """enable interactive cropping on the canvas"""
    global crop_rectangle
    crop_rect = [None]  # to store crop rectangle coordinates globally

    def start_crop(event):
        # record the starting point of the crop
        crop_rect[0] = (event.x, event.y)

    def update_crop(event):
        # draw a rectangle as the user drags the mouse
        canvas.delete('crop_rect')  # clear any previous rectangle
        canvas.create_rectangle(crop_rect[0][0], crop_rect[0][1],
                                event.x, event.y, outline='red', tags='crop_rect')

    def end_crop(event):
        # record the ending point of the crop and update the entries
        crop_rect.append((event.x, event.y))
        x1, y1 = crop_rect[0]
        x2, y2 = crop_rect[1]

        # ensure valid coordinates
        x, y = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        
        # update crop_rectangle coordinates 
        crop_rectangle = (x, y, x+width, y+height)

        # update cropping entry fields
        crop_entries[0].delete(0, tk.END)
        crop_entries[0].insert(0, str(x))
        crop_entries[1].delete(0, tk.END)
        crop_entries[1].insert(0, str(y))
        crop_entries[2].delete(0, tk.END)
        crop_entries[2].insert(0, str(width))
        crop_entries[3].delete(0, tk.END)
        crop_entries[3].insert(0, str(height))

    # bind mouse events to the canvas
    canvas.bind('<Button-1>', start_crop)  # left mouse button press
    canvas.bind('<B1-Motion>', update_crop)  # mouse drag
    canvas.bind('<ButtonRelease-1>', end_crop)  # left mouse button release

def halt_processing():
    global halt_flag
    halt_flag = True

def load_and_preview_image(tif_path, canvas, slider):
    """load the mean of the first 10 frames of the .tif file and display it on the canvas"""
    try:
        global loaded_tif_data  # declare as global to allow updates
        loaded_tif_data = tifffile.imread(tif_path)  # store the loaded .tif data
        
        global tif_max
        global tif_min
        tif_max = loaded_tif_data.max()
        tif_min = loaded_tif_data.min()
        
        # dynamically change the frame slider range 
        slider.config(from_=0, to=loaded_tif_data.shape[0]-1)

        # display the first frame initially
        display_frame_on_canvas(canvas, loaded_tif_data[0])

    except Exception as e:
        print(f"Error processing .tif file: {e}")

def run_processing(in_path_entry, out_path_entry, codec_var, format_var, crop_entries,
                   output_text, canvas, start_frame, end_frame, freq_entry):
    if os.path.splitext(in_path_entry)[-1].lower() not in ['.tif', '.tiff']:
        output_text.insert(tk.END, '\nnot a .tif file\n')
        return None
        
    crop_params = [int(entry.get()) if entry.get().isdigit() else 0 for entry in crop_entries]
    
    create_mov(in_path_entry, 
               out_path_entry,
               codec_var,
               format_var,
               freq_entry,
               crop_params,
               start_frame, end_frame,
               write_func=lambda msg: output_text.insert(tk.END, msg), 
               display_func=lambda pil_img: update_frame_display(pil_img, canvas),
               preloaded_data=loaded_tif_data)
    
    output_text.insert(tk.END, 'done\n')
    output_text.yview(tk.END)

def start_processing_thread(in_path_entry, out_path_entry, codec_var, format_var, crop_entries,
                            output_text, canvas, start_frame, end_frame, freq_entry):
    # run the processing function in a separate thread to avoid freezing the GUI
    processing_thread = threading.Thread(target=run_processing, args=(in_path_entry,
                                                                      out_path_entry,
                                                                      codec_var,
                                                                      format_var,
                                                                      crop_entries,
                                                                      output_text,
                                                                      canvas,
                                                                      start_frame,
                                                                      end_frame,
                                                                      freq_entry))
    processing_thread.start()

def update_displayed_frame(canvas, slider, label):
    """Update the canvas to display the frame corresponding to the slider value."""
    if loaded_tif_data is not None:
        frame_index = int(float(slider.get()))  # get the slider value as an integer
        label.config(text=f'frame: {frame_index}')  # update the frame label
        display_frame_on_canvas(canvas, loaded_tif_data[frame_index])  # update the canvas

def update_frame_display(pil_image, canvas, brightness_factor=1.5, contrast_factor=1.5):
    """Update the Canvas to display the movie frame."""
    # enhance the image
    brightness_enhancer = ImageEnhance.Brightness(pil_image)
    pil_image_bright = brightness_enhancer.enhance(brightness_factor)
    contrast_enhancer = ImageEnhance.Contrast(pil_image_bright)
    pil_image_contrast = contrast_enhancer.enhance(contrast_factor)
    
    # convert to ImageTk format
    tk_image = ImageTk.PhotoImage(pil_image_contrast)
    
    # update the canvas with the new image
    canvas.image = tk_image  # keep a reference to avoid garbage collection
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)


#%% Run the GUI
if __name__ == '__main__':
    create_gui()