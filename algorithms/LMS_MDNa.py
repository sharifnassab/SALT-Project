import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime



alg_name =  'LMS-MDNa' 
alpha = 1e-4
eta0 = .001
theta_MDN = 1e-3

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def param_sweeps():
    sweeps=[]
    for bias in ['True']:
            for eta0 in [.001]:
                for theta_MDN in [1e-3, 1e-2, 1e-1]:
                    for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                        sweeps.append({
                                        'file_path':os.path.abspath(__file__),
                                        'alpha':alpha,
                                        'eta0':eta0,
                                        'theta_MDN':theta_MDN,
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
                    'theta':theta_MDN,
                    'eta0':eta0,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------


class MDNa():
    def __init__(self, theta, eta0, activation='sigmoid'):
        if activation=='sigmoid':
            self.act = lambda b: 1/(1+np.exp(-b))
            act_inv = lambda a: -np.log(1/a -1)
            self.d_dbeta_given_eta = lambda eta: eta*(1.0-eta)
        self.theta = theta
        self.var = 0.0
        self.t = 0
        self.epsilon = 1e-8
        self.beta_mu = act_inv(eta0)
        self.beta_v = act_inv(eta0)
        self.h_mu = 0.0
        self.h_v = 0.0
        self.eta_mu_prod = 1.0
        self.eta_v_prod = 1.0

    def step(self, x):
        self.t+=1
        eta_mu = self.act(self.beta_mu)
        eta_v = self.act(self.beta_v)

        if self.t==1:
            self.mu = x
            self.eta_mu_prod *= 1.0-eta_mu 
            x_tilde = np.zeros_like(x)
            sigma = self.epsilon * np.ones_like(x)
            return x_tilde, self.mu, sigma

        self.eta_mu_prod *= 1.0-eta_mu  
        self.eta_v_prod *= 1.0-eta_v
        
        self.var = self.var + eta_v*((x-self.mu)**2 - self.var) / (1-self.eta_v_prod)# (1-self.eta_v_prod) is not present in the original MDN
        sigma = np.clip(np.sqrt(self.var), a_min=self.epsilon, a_max=None)
        x_tilde = (x-self.mu) /  sigma
        self.mu = self.mu + eta_mu*(x-self.mu) /(1-self.eta_mu_prod) # (1-self.eta_mu_prod) is not present in the original MDN

        self.beta_mu = self.beta_mu + self.theta *x_tilde *self.h_mu  * eta_mu*(1-eta_mu)
        self.beta_v = self.beta_v + self.theta *(x_tilde**2-1) *self.h_v  * eta_v*(1-eta_v)
        self.h_mu = (1-eta_mu)*self.h_mu + x_tilde 
        self.h_v = (1-eta_v)*self.h_v + (x_tilde**2-1)
        return x_tilde, self.mu, sigma



def run_once(dataset=dataset, alpha=alpha, eta0=eta0, theta_MDN=theta_MDN, bias=bias):
    env, num_steps, d = data_loader(dataset)
    w = np.zeros([d]) #+ (1 if bias else 0)])
    b = 0.0
    dd = d+1 if bias else d
    input_normalizer = MDNa(theta=theta_MDN, eta0=eta0)
    MSE_list = []

    for iteration in range(num_steps):
        x,z = env.next()
        x_tilde, mu, sigma = input_normalizer.step(x)

        y = (w*x_tilde).sum() + b
        try:
            MSE_list.append((y-z)**2)
        except:
            break

        w = w + (alpha/dd) * (z-y)*x_tilde
        if bias:
            b = b + (alpha/dd)*(z-y)
        
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    return MSE_list, steady_state_RMSE



if __name__ == "__main__":
    MSE_list, steady_state_RMSE = run_once()

    ###-----------------------
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    