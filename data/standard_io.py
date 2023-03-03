# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

import os
import csv
import pickle
import pandas as pd
import numpy as np

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def csv_read(file):
    f = open(file)
    csvReader = csv.reader(f)
    D = []
    for row in csvReader:
        D.append(row)
    return D

def csv_write(file, D):
    f = open(file,'w')
    csvWriter = csv.writer(f,lineterminator='\n')
    if np.ndim(D) == 1:
        csvWriter.writerow(D)
    elif np.ndim(D) == 2:
        for i in range(np.shape(D)[0]):
            line = D[i]
            csvWriter.writerow(line)
    f.close()

def pkl_read(file):
    f = open(file, 'rb')
    D = pickle.load(f)
    f.close()
    return D

def pkl_write(file, D):
    f = open(file,'wb')
    pickle.dump(D,f,protocol=4)
    f.close()
