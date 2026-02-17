import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime

alg_name =  'Newton' 
alpha = 1e-2
momentum_param = 0.0
hess_gamma = 0.999

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/test.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for bias in ['True']:
        for hess_gamma in [.999]:#[.99, .999, .9999]:
            for momentum_param in [0.0]:
                for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                    sweeps.append({
                                    'file_path':os.path.abspath(__file__),
                                    'alpha':alpha,
                                    'momentum_param':momentum_param,
                                    'hess_gamma':hess_gamma,
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
                    #'momentum_param':momentum_param,
                    'gamma':hess_gamma,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class Newtonm():
    def __init__(self, alpha, momentum_param=0.0, hess_gamma=0.999, w0=0.0):
        self.required_input = 'grad,hess'
        self.alpha = alpha
        self.w = w0+0.0
        self.bet1 = momentum_param
        self.gamma = hess_gamma
        self.t=0
        self.hess_trace = 0.0
        self.m = 0.0
        self.eps_I = 1e-8 * np.eye(self.w.size)

    def step(self, grad, x):
        x=x.reshape([-1,1])
        hess = x@ x.T
        self.t+=1
        self.m = self.bet1*self.m + (1-self.bet1)*grad
        self.hess_trace = self.gamma*self.hess_trace + (1-self.gamma)*hess
        m = self.m #/ (1-self.bet1**self.t)
        H = self.hess_trace/ (1-self.gamma**self.t) + self.eps_I
        self.w = self.w - self.alpha * np.linalg.solve(H,m).flatten()
        return self.w

        

class Newtonm_fast_but_less_stable():
    def __init__(self, alpha, momentum_param=0.0, hess_gamma=0.999, w0=0.0):
        self.required_input = 'grad,hess'
        self.alpha = alpha
        self.w = w0+0.0
        self.bet1 = momentum_param
        self.gamma = hess_gamma
        self.t=0
        self.m = 0.0
        self.hess_inv = np.eye(self.w.size)

    def step(self, grad, x):
        x=x.reshape([-1,1])
        self.t+=1
        self.m = self.bet1*self.m + (1-self.bet1)*grad
        m = self.m #/ (1-self.bet1**self.t)
        Hinv_x = self.hess_inv @ x
        gam_coeff = (1-self.gamma)/self.gamma
        self.hess_inv = (1/self.gamma)*  (self.hess_inv - (gam_coeff/(1+gam_coeff*(x.T@Hinv_x)))*(Hinv_x@Hinv_x.T))
        #self.hess_inv = (self.hess_inv + self.hess_inv.T)/2.0
        self.w = self.w - self.alpha * (self.hess_inv@x).flatten()
        return self.w


if __name__ == "__main__":
    env, num_steps, input_dim = data_loader(dataset)
    w = np.zeros([input_dim+ (1 if bias else 0)])
    optimizer = Newtonm(alpha, momentum_param, hess_gamma, w0=w)
    MSE_list = []


    for iteration in range(num_steps):
        try:
            x,z = env.next()
            if bias: 
                x=np.append(x,1)
            delta = (w*x).sum() - z
            MSE_list.append(delta**2)
            grad = delta*x
            w = optimizer.step(grad, x)
            if np.isnan(w).any(): break
        except:
            break
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    