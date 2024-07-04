# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:49:23 2024

log temperature and humidity into a .txt file 

@author: Dinghao Luo
"""


#%% imports 
import serial 
import sys 
import glob
import os
import time
from datetime import datetime


#%% list ports 
# if sys.platform.startswith('win'):
#     ports = ['COM%s' % (i + 1) for i in range(256)]
# elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
#     # this excludes your current terminal "/dev/tty"
#     ports = glob.glob('/dev/tty[A-Za-z]*')
# elif sys.platform.startswith('darwin'):
#     ports = glob.glob('/dev/tty.*')
# else:
#     raise EnvironmentError('Unsupported platform')

# result = []
# for port in ports:
#     try:
#         s = serial.Serial(port)
#         s.close()
#         result.append(port)
#     except (OSError, serial.SerialException):
#         pass
# print('available serial ports: {}'.format(result))


#%% reading and writing 
# initialise
t1 = time.time() + 21  # 30 seconds 
serial_port = 'COM3';
baud_rate = 9600;  # Serial.begin(baud_rate)

# create log file 
log_name = r'log_{}'.format(str(datetime.now())[:10])
write_to_file_path = r'Z:/Dinghao/temp_humid_monitor/{}.txt'.format(log_name)

# label the txt file 
try: 
    with open(write_to_file_path, 'x') as file: 
        file.write('temperature and humidity log on {}\n\n\n'.format(str(datetime.now())[:10])) 
except FileExistsError: 
    print(r'{} already exists.'.format(write_to_file_path)) 

ser = serial.Serial(serial_port, baud_rate)
while str(datetime.now().time())[:5]!='22:00':  # run until 10 pm
    # wait for and parse data 
    data = str(ser.readline())  # ser.readline returns a byte sequence, convert to string without decoding
    temp = data[2:data.index('C')+1]
    humid = data[data.index('C')+1:data.index('%')+1]
    
    # get current time
    now = str(datetime.now())
    
    # monitor in terminal 
    print('\n'+now)
    print(temp)
    print(humid)
    
    # write to file
    f = open(write_to_file_path, 'a')
    f.write(now+'\n')
    f.write(temp+'\n')
    f.write(humid+'\n\n')
    f.close()

ser.close()  # stop accessing the COM port to allow access by others
