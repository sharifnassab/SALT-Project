import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime



alg_name =  'LMS-MDNcenter' 
alpha = 1e-4
eta0 = .001
theta_MDN = 1e-4
transform = 'post:0.01'  # 'None' 'signW'  'pre:none,post:0.01'

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def param_sweeps():
    sweeps=[]
    for bias in ['True']:
            for eta0 in [.001]:
                for theta_MDN in [1e-3]:#, 1e-2]:
                    for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                        for transform in [
                                        #   'post:none', 
                                           'post:0.01', 
                                          ]:
                            sweeps.append({
                                            'file_path':os.path.abspath(__file__),
                                            'alpha':alpha,
                                            'eta0':eta0,
                                            'theta_MDN':theta_MDN,
                                            'bias':bias,
                                            'transform':transform,
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
                    'transform':transform,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------


class MDNcenter():
    def __init__(self, theta, eta0, transform='None', activation='sigmoid'):
        if activation=='sigmoid':
            self.act = lambda b: 1/(1+np.exp(-np.clip(b, -500, 500)))
            act_inv = lambda a: -np.log(1/a -1)
            self.d_dbeta_given_eta = lambda eta: eta*(1.0-eta)
        self.theta = theta
        # extract prenorm_type  from transform string if applicable
        if transform.split(':')[1].lower() in ['none']:
            self.post_norm_type = 'None'
        else:
            self.post_norm_type = transform.split(':')[1]
            self.postnorm_coeff_mu = float(self.post_norm_type)


        
        self.t = 0
        self.epsilon = 1e-8
        self.beta_mu = act_inv(eta0)
        self.h_mu = 0.0
        self.eta_mu_prod = 1.0
        self.postnorm_mu = 1.0
        self.update_trace = 1.0
        

    def step(self, x):
        self.t+=1
        eta_mu = self.act(self.beta_mu)
        sigma = 1.0

        if self.t==1:
            self.mu = x
            self.eta_mu_prod *= 1.0-eta_mu 
            x_tilde = np.zeros_like(x)
            return x_tilde, self.mu, sigma
        
        self.eta_mu_prod *= 1.0-eta_mu  
        
        x_tilde = (x-self.mu) 
        self.mu = self.mu + eta_mu*(x-self.mu) / (1-self.eta_mu_prod) 


        meta_update_mu = x_tilde * self.h_mu

        if self.post_norm_type=='None':
            post_norm_mu = 1
        else:
            self.postnorm_mu += self.postnorm_coeff_mu * (meta_update_mu**2 - self.postnorm_mu)
            post_norm_mu = 1/np.sqrt(self.postnorm_mu + 1e-8)
            

        self.beta_mu = self.beta_mu - self.theta * post_norm_mu  * meta_update_mu

        self.h_mu = (1-eta_mu)*self.h_mu + x_tilde 


        # if self.t%10000==0:
        #     print(
        #         f"{self.t}\n"
        #         f"{npfmt5(eta_mu)}\n"
        #         f"{npfmt5(self.h_mu)}\n"
        #         f"{npfmt5(x_tilde)}, {npfmt5(x)}\n"
        #         )
        return x_tilde, self.mu, sigma


def npfmt5(a):
    return np.array2string(a[:5], precision=4, floatmode='fixed', separator=' ')


def main_loop(dataset=dataset, alpha=alpha, eta0=eta0, theta_MDN=theta_MDN, bias=bias):
    env, num_steps, d = data_loader(dataset)
    w = np.zeros([d]) #+ (1 if bias else 0)])
    b = 0.0
    dd = d+1 if bias else d
    input_normalizer = MDNcenter(theta=theta_MDN, eta0=eta0, transform=transform)
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
    MSE_list, steady_state_RMSE = main_loop()

    ###-----------------------
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    