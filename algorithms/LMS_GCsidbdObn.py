import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime
from copy import deepcopy


alg_name =  'LMS_GCsidbdObn'      # goal directed centering with sidbd

alpha = 1e-5
alpha_mu_0 = 1e-3 # stepsize of the first layer
meta_stepsize_mu = 1e-3
meta_post_norm_mu = 0.99


bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/_test3.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        for meta_stepsize_mu in [1e-3]:
            for meta_post_norm_mu in [
                                    0.99,
                                    ]: 
                sweeps.append({
                                'file_path':os.path.abspath(__file__),
                                'alpha':alpha,
                                'meta_stepsize_mu':meta_stepsize_mu,
                                'meta_post_norm_mu':meta_post_norm_mu,
                                'bias':'True'
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
            'alg_name':f'{alg_name}',
            'params':{
                    'alpha':alpha,
                    'meta_stepsize_mu':meta_stepsize_mu,
                    'meta_post_norm_mu':meta_post_norm_mu,
                    },
            'bias':'True',
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,0,None))


def softmax(x):
    exp_z = np.exp(x-np.median(x))  # median is substracted for numerical statibility and avoiding exp blow-up
    return exp_z/exp_z.sum()
def act_exp(x): 
    return np.exp(x)
def act_sigmoid(x): 
    return 1 / (1 + np.exp(-x))

class GCsidbdObn():
    def __init__(self, meta_stepsize_mu, meta_post_norm_mu, alpha_mu_0):
        self.t = 0
        self.mu = 0.0
        self.meta_stepsize_mu = meta_stepsize_mu
        self.meta_post_norm_mu = meta_post_norm_mu
        self.beta_mu = np.log(alpha_mu_0)

        self.trace_meta_post_norm_mu = 0.0
        self.h_mu = 0.0

    def step(self, x, z, w, b):
        self.t+=1
        x_tilde = x-self.mu
        y = (w*x_tilde).sum() + b
        delta = z  - y

        meta_beta_mu_increment = delta * (w*self.h_mu).sum() 
        self.trace_meta_post_norm_mu = self.trace_meta_post_norm_mu + (1-self.meta_post_norm_mu) * (meta_beta_mu_increment**2 - self.trace_meta_post_norm_mu)
        self.beta_mu = self.beta_mu + self.meta_stepsize_mu * meta_beta_mu_increment / np.clip(np.sqrt(self.trace_meta_post_norm_mu/(1-self.meta_post_norm_mu**self.t)), a_min=1e-8, a_max=None)
        alpha_mu = act_sigmoid(self.beta_mu)

        obn_coeff_for_mu = 1/((w*w).sum()+1e-8)
        self.mu = self.mu - alpha_mu * delta * w * obn_coeff_for_mu

        h_projection =  (w*self.h_mu).sum() * alpha_mu*w * obn_coeff_for_mu
        h_overshoor_bound = min([1, np.sqrt(np.sum(self.h_mu**2) / ((h_projection**2).sum()+1e-16))])
        self.h_mu = self.h_mu - h_overshoor_bound*h_projection + delta*w * obn_coeff_for_mu

        return x_tilde, self.mu, None   


if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)

    if alpha_mu_0=='alpha':
         alpha_mu_0 = alpha
    elif isinstance(alpha_mu_0,str) and alpha_mu_0.split('/')[0]=='alpha':
        alpha_mu_0 =alpha/float(alpha_mu_0  .split('/')[1])
         
    
    wb = np.zeros(d+1)
    input_normalizer = GCsidbdObn(meta_stepsize_mu=meta_stepsize_mu, meta_post_norm_mu=meta_post_norm_mu, alpha_mu_0=alpha_mu_0)
    MSE_list = []
    t = 0

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        try:
            x,z = env.next()
            x_tilde, mu, sigma = input_normalizer.step(x, z, wb[:d], b=wb[d] if bias else 0.0)
            
            xb = np.append(x_tilde,1)
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)
            
            #obn_coeff_for_w = min(1, 1/(alpha*xb*xb).sum())
            wb = wb + alpha *  delta * xb

            if np.isnan(wb).any(): break
        except:
            break
        
    #     if iteration>1_100_000-10:
    #          print(x[:5])
    # print()
    # print(init_c[:5])
    # print(c[:5])
    # print(wb[:5])
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    