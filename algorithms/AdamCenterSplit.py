import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime

alg_name =  'AdamCenterSplit' 
alpha = 1e-3
momentum_param = 0.9
grad_normalizer_param = 0.999
input_centering_param = 0.9999 

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/test.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def param_sweeps():
    sweeps=[]
    for bias in ['True']: # ['False', 'True']
        for grad_normalizer_param in [.999]: # [.999, .99]
            for momentum_param in [0.9]: #[0.0, 0.9]
                for input_centering_param in [.999, .9999]:
                    for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                        sweeps.append({
                                        'file_path':os.path.abspath(__file__),
                                        'alpha':alpha,
                                        'momentum_param':momentum_param,
                                        'grad_normalizer_param':grad_normalizer_param,
                                        'input_centering_param':input_centering_param,
                                        'bias':bias,
                                        })
    return sweeps

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("utils/configurator.py").read())  # overrides from command line or config file

bias = False if bias in [False,'False',0,'0'] else True
alg_config={
            'dataset':dataset,
            'alg_name':alg_name,
            'params':{
                    'alpha':alpha,
                    'beta_m':momentum_param,
                    'beta_nu':grad_normalizer_param,
                    'beta_centering': input_centering_param,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------


class Adam():
    def __init__(self, alpha, momentum_param=0.9, grad_normalizer_param=0.999, w0=0.0):
        self.required_input = 'grad'
        self.alpha = alpha
        self.w = w0+0.0
        self.bet1 = momentum_param
        self.bet2 = grad_normalizer_param
        self.t=0
        self.norm_trace = 0.0
        self.m = 0.0

    def step(self, grad):
        self.t+=1
        self.m = self.bet1*self.m + (1-self.bet1)*grad
        self.norm_trace = self.bet2*self.norm_trace + (1-self.bet2)*(grad**2)
        m = self.m/ (1-self.bet1**self.t)
        v = (self.norm_trace/ (1-self.bet2**self.t))**.5 + 1e-10
        self.w = self.w - self.alpha * m/v
        return self.w
    
    def get_normalizer(self, grad):
        t = self.t+1
        v = self.bet2*self.norm_trace + (1-self.bet2)*(grad**2)
        v = (v / (1-self.bet2**t))**.5 + 1e-8
        return v



class input_cenetring_split():
    def __init__(self, input_centering_param=0.9999):
        self.bet_c = input_centering_param
        self.t=0
        self.center = 0.0

    def step(self, x):
        self.t+=1
        if self.t<=1:
            self.center = np.zeros_like(x)
        self.center = self.bet_c * self.center  +  (1.0-self.bet_c) * x
        centered_x = x-self.center
        return np.concatenate([centered_x, self.center], axis=0)
        

if __name__ == "__main__":
    env, num_steps, input_dim = data_loader(dataset)
    w = np.zeros([2*input_dim+ (1 if bias else 0)])
    optimizer = Adam(alpha, momentum_param, grad_normalizer_param, w0=w)
    centering = input_cenetring_split(input_centering_param)
    MSE_list = []


    for iteration in range(num_steps):
        if 1:
            x,z = env.next()
            x_tilde = centering.step(x)
            xb = np.append(x_tilde,1) if bias else x_tilde
            
            delta = (w*xb).sum() - z
            
            MSE_list.append(delta**2)
            grad = delta*xb
            w = optimizer.step(grad)
        else:
            break        

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    