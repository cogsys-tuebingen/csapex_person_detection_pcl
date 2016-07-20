#!/usr/bin/env python

import sys
import yaml

def main(input_filename, output_filename, window_size):
    with file(input_filename, 'r') as input_file:
        input = yaml.load(''.join(input_file.readlines()[1:]))
    
    with file(output_filename, 'w') as output_file:
        output_file.write('window_height: %s\n' % window_size[0])
        output_file.write('window_width: %s\n' % window_size[1])
        output_file.write('b: %s\n' % input['svm_rho'])  
        output_file.write('weights: %s\n' % input['svm_coeffs'])  

if __name__ == "__main__":
    if len(sys.argv) != 4 + 1:
        print("Usage: %s <input> <output> <window_width> <window_height>" % sys.argv[0])
        exit(1)
    else:
        main(sys.argv[1], sys.argv[2], (sys.argv[3], sys.argv[4]))
