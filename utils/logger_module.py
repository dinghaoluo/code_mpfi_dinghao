# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:10:26 2024

a logger module to help log each run of each script to a /log folder

@author: Dinghao Luo
"""


#%% imports 
import logging
import datetime
import os


#%% logging function
def log_run(script_name, params, results=None, errors=None, log_dir=r'Z:/Dinghao/code_mpfi_dinghao/logs'):
    """Logs the parameters, results, and errors to a timestamped log file."""
    
    # ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # generate a unique log filename based on the current date and time
    log_filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')+script_name
    log_filepath = '{}/{}.txt'.format(log_dir, log_filename)

    # configure logging
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # log the parameters
    logging.info('parameters: {}'.format(params))

    # log the results if provided
    if results is not None:
        logging.info('results: {}'.format(results))

    # log any errors if provided
    if errors is not None:
        logging.error('errors: {}'.format(errors))

    # indicate the end of the script
    logging.info('script run completed.')
    
    print('script run logged to {}'.format(log_filepath))