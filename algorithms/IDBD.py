import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime


alg_name =  'IDBD' 
alpha0 = 1e-7
meta_stepsize = 1e-5

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS6-20-1100000.txt' 
save_to='csvs/test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for bias in ['True']:
        for meta_stepsize in [1e-5, 1e-4, 1e-3, 1e-2]:
            for alpha0 in [1e-6]:
                sweeps.append({
                                'file_path':os.path.abspath(__file__),
                                'alpha0':alpha0,
                                'meta_stepsize':meta_stepsize,
                                'bias':bias,
                                })
    return sweeps
    #command_list = ['python3 IDBD.py ' + ' '.join([f'--{param}={config[param]}' for param in config]) for config in list_param_sweeps]
    #return command_list

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("utils/configurator.py").read())  # overrides from command line or config file

bias = False if bias in [False,'False',0,'0'] else True
alg_config={
            'dataset':dataset,
            'alg_name':alg_name,
            'params':{
                    'metastep':meta_stepsize,
                    'alpha0':alpha0,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    env, num_steps, d = data_loader(dataset)
    w = np.zeros([d+ (1 if bias else 0)])
    MSE_list = []

    h = 0.0
    beta = np.log(alpha0)
    for iteration in range(num_steps):
        x,z = env.next()
        if bias: 
            x=np.append(x,1)
        
        y = (w*x).sum()
        delta = z-y
        try:
            MSE_list.append(delta**2)
        except:
            break
        beta = beta + meta_stepsize *delta *x *h
        alpha = np.exp(beta)
        w = w + alpha *delta *x
        h = h * np.clip(1-alpha*x*x, a_min=0, a_max=None) + alpha *delta *x
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    