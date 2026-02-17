import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader_fast import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime


alg_name =  'LMS_EAN2' 
alpha = 1e-4
eta = .999
eta_v = 0.99

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def param_sweeps():
    sweeps=[]
    for bias in ['True']:
            for eta in [.001]:
                for eta_v in [0.01]:
                    for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                        sweeps.append({
                                        'file_path':os.path.abspath(__file__),
                                        'alpha':alpha,
                                        'eta':eta,
                                        'eta_v':eta_v,
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
                    'eta':eta,
                    'eta_v':eta_v,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------



class EAN2():
    def __init__(self, eta, eta_v=None):
        self.var = 0.0
        self.t = 0
        self.epsilon = 1e-8
        self.eta = eta
        if eta_v is None:
            self.eta_v = eta
        else:
            self.eta_v = eta_v

    def step(self, x):
        self.t+=1
        if self.t==1:
            self.one_minus_eta_to_the_t = 1.0-self.eta
            self.one_minus_eta_v_to_the_t = 1.0-self.eta_v
            self.mu = x + 0.0
            mu = self.mu + 0.0
            sigma = self.epsilon * np.ones_like(x)
            x_tilde = np.zeros_like(x)
        else:
            coeff_var = self.eta_v/(1-self.one_minus_eta_v_to_the_t)
            self.one_minus_eta_v_to_the_t *= (1-self.eta_v)
            self.one_minus_eta_to_the_t *= (1-self.eta)
            coeff_mu = self.eta/(1-self.one_minus_eta_to_the_t)
            self.var = self.var + coeff_var*((x-self.mu)**2 - self.var) 
            mu = self.mu+0.0
            sigma = np.clip(np.sqrt(self.var), a_min=self.epsilon, a_max=None)
            x_tilde = (x-mu) /  sigma
            self.mu = self.mu + coeff_mu*(x-self.mu)
        return x_tilde, mu, sigma


if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)
    w = np.zeros([d]) #+ (1 if bias else 0)])
    b = 0.0
    dd = d+1 if bias else d
    input_normalizer = EAN2(eta=eta, eta_v=eta_v)
    MSE_list = []

    t = 0
    for iteration in range(num_steps):
        t+=1
        x,z = env.next()
        x_tilde, mu, sigma = input_normalizer.step(x)

        y = (w*x_tilde).sum() + b
        try:
            MSE_list.append((y-z)**2)
        except:
            break

        w = w + (alpha/d) * (z-y)*x_tilde
        if bias:
            b = b + (alpha/(d+1))*(z-y)
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    