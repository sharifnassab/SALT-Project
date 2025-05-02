import numpy as np
import os

def get_dir(name):
    if name=='ASH':
        return 'data'
    if name=='RSS':
        return 'data'

    
    
def data_loader(dataset_name):
    if 'RSS' in dataset_name:
        dataset_file_path = os.path.join(get_dir('RSS'), dataset_name)
    if 'ASH' in dataset_name:
        dataset_file_path = os.path.join(get_dir(('ASH')), dataset_name)
    with open(dataset_file_path, 'r') as file:
        data = np.loadtxt(dataset_file_path, skiprows=3)
    dim = data.shape[1]-1
    num_steps = data.shape[0]
    return env_laoded_from_CSV(data), num_steps, dim


class env_laoded_from_CSV():
    def __init__(self, data):
        self.X=data[:,1:]
        self.Y=data[:,0]
        self.iter=0

    def next(self):
        x=self.X[self.iter,:]
        y=self.Y[self.iter]
        self.iter+=1
        return x, y
    

