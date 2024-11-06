# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:33:52 2024

plot directory tree 

@author: Dinghao Luo
"""


import os

def scan_directory_tree(path, indent=''):
    output = ''
    if not os.path.isdir(path):
        return 'Invalid directory path.'

    dir_name = os.path.basename(path)
    output += f'{indent}{dir_name}\n'
    indent += '    '
    
    items = sorted(os.listdir(path))
    
    for i, item in enumerate(items):
        full_path = os.path.join(path, item)
        is_last = (i == len(items) - 1)
        prefix = '└── ' if is_last else '├── '

        if os.path.isdir(full_path):
            output += f'{indent}{prefix}{item}\n'
            output += scan_directory_tree(full_path, indent + ('    ' if is_last else '│   '))
        else:
            output += f'{indent}{prefix}{item}\n'
    
    return output

# Example usage:
path = 'Z:\\Dinghao\\code_mpfi_dinghao\\imaging_code'
print(scan_directory_tree(path))