# -*- coding: utf-8 -*-
"""
Created on 27 Nov 14:18:12 2024

run suite2p registration and ROI extraction on recording list of choice 

@author: Dinghao Luo
"""


#%% imports 
import sys
import os
import tkinter as tk
from tkinter import ttk
import threading

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import suite2p_functions as s2f

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE


#%% run all sessions
# for path in pathGRABNE:
#     sessname = path[-17:]
#     print('\n{}'.format(sessname))
    
#     reg_path = path+r'\processed'
#     if not os.path.exists(reg_path):  # if registration has not been performed
#         s2f.register(path)
#     else:
#         print('session already registered')
#     # if 'no tiffs' is raised, most likely it is due to typos in pathnames
    
#     roi_path = reg_path+r'\suite2p\plane0\stat.npy'
#     if not os.path.exists(roi_path):  # if roi extraction has not been performed 
#         s2f.run_roi_extraction(path)
#     else:
#         print('session ROIs already extracted')
        
        
#%% functions
def create_gui():
    # create main window
    window = tk.Tk()
    window.title('suite2p processing')
    
    # configure grid columns to allow for a new column on the right
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=0)

    # choose recording list
    reclist_label = ttk.Label(window, text = 'choose recording list: ')
    reclist_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
    reclist_options = ['axon-GCaMP LC', 'axon_GCaMP VTA']
    reclist_var = tk.StringVar(value=reclist_options[0])
    reclist_dropdown = ttk.Combobox(window, textvariable=reclist_var, values=reclist_options, state='readonly')
    reclist_dropdown.grid(row=1, column=1, padx=10, pady=5)
        
    # terminal
    output_text = tk.Text(window, height=10, width=60, wrap=tk.WORD)
    output_text.grid(row=2, column=0, rowspan=3, columnspan=2, padx=10, pady=(5, 10), sticky='ns')
    
    # button frame to contain the Run and Halt buttons
    button_frame = ttk.Frame(window)
    button_frame.grid(row=6, column=0, columnspan=2, pady=(5, 10), sticky='w')  

    # run button
    run_button = ttk.Button(button_frame, text='Run', 
                            command=lambda: start_processing_thread(reclist_var.get(),
                                                                    output_text))
    run_button.grid(row=0, column=0, padx=10, pady=(5, 5))

    # halt button
    halt_button = ttk.Button(button_frame, text='Halt', command=halt_processing)
    halt_button.grid(row=0, column=1, pady=(5, 5))
    
    window.mainloop()

def halt_processing():
    global halt_flag
    halt_flag = True

def run_processing(reclist_name, output_text):

    
    output_text.insert(tk.END, 'done\n')
    output_text.yview(tk.END)

def start_processing_thread(reclist_name, output_text):
    # run the processing function in a separate thread to avoid freezing the GUI
    processing_thread = threading.Thread(target=run_processing, args=(reclist_name, 
                                                                      output_text))
    processing_thread.start()


#%% Run the GUI
if __name__ == '__main__':
    create_gui()