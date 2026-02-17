import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime



alg_name =  'LMS-GMDN' 
alpha = 1e-4
eta0 = .001
theta_MDN = 1e-4
transform = 'pre:none,post:v'  # 'None' 'signW' 'postnorm_v' 'postnorm_mu' 'postnorm_both'

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
                                        #   'pre:none,post:none', 
                                        #   'pre:signW,post:none', 
                                        #   'pre:none,post:v@0.01', 
                                        #   'pre:signW,post:v@0.01', 
                                        #   'pre:none,post:v@0.001', 
                                        #   'pre:signW,post:v@0.001', 
                                        #   'pre:none,post:both@0.01@0.01', 
                                           'pre:signW,post:both@0.01@0.01',
                                        #   'pre:signW,post:both@0.001@0.001',
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


class GMDN():
    def __init__(self, theta, eta0, transform='None', activation='sigmoid'):
        if activation=='sigmoid':
            self.act = lambda b: 1/(1+np.exp(-np.clip(b, -500, 500)))
            act_inv = lambda a: -np.log(1/a -1)
            self.d_dbeta_given_eta = lambda eta: eta*(1.0-eta)
        self.theta = theta
        self.var = 0.
        # extract prenorm_type  from transform string if applicable
        if transform.split(',')[0].split(':')[1].lower() in ['none']:
            self.prenorm_type = 'None'
        else:
            self.prenorm_type = transform.split(',')[0].split(':')[1]
        if transform.split(',')[1].split(':')[1].lower() in ['none']:
            self.post_norm_type = 'None'
        else:
            self.post_norm_type = 'postnorm_'+transform.split(',')[1].split(':')[1]

        
        self.t = 0
        self.epsilon = 1e-8
        self.beta_mu = act_inv(eta0)
        self.beta_v = act_inv(eta0)
        self.h_mu = 0.0
        self.h_v = 0.0
        self.eta_mu_prod = 1.0
        self.eta_v_prod = 1.0
        self.postnorm_v = 1.0
        self.postnorm_mu = 1.0
        self.update_trace = 1.0
        

    def step(self, x, z, w, b):
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
        
        #x_tilde_old = (x-self.mu) / np.clip(np.sqrt(self.var), a_min=self.epsilon, a_max=None)
        #var_old = self.var + 0.0

        self.var = self.var + eta_v*((x-self.mu)**2 - self.var) / (1-self.eta_v_prod) # (1-self.eta_v_prod) is not present in the original MDN
        sigma = np.clip(np.sqrt(self.var), a_min=self.epsilon, a_max=None)
        #x_centered = x-self.mu
        x_tilde = (x-self.mu) /  sigma
        self.mu = self.mu + eta_mu*(x-self.mu) / (1-self.eta_mu_prod) # (1-self.eta_v_prod) is not present in the original MDN

        delta = z  - ((w*x_tilde).sum() + b)

        #self.beta_mu = self.beta_mu + self.theta *x_tilde *self.h_mu 
        #self.beta_v = self.beta_v + self.theta *(x_tilde**2-1) *self.h_v  

        # apply normalization to the meta update
        if self.prenorm_type=='None':
            pre_norm_mu = 1
            pre_norm_v = 1
        elif self.prenorm_type=='signW':
            pre_norm_mu = 1/(np.abs(w)+1e-8)
            pre_norm_v = 1/(np.abs(w)+1e-8)

        meta_update_mu = pre_norm_mu  * delta * w * self.h_mu
        meta_update_v  = pre_norm_v   * delta * w * x_tilde * self.h_v

        postnorm_coeff_mu = 0.01
        postnorm_coeff_v = 0.01
        if self.post_norm_type=='None':
            post_norm_mu = 1
            post_norm_v = 1
        elif self.post_norm_type.split('@')[0]=='postnorm_mu':
            postnorm_coeff_mu = float(self.post_norm_type.split('@')[1]) if '@' in self.post_norm_type else postnorm_coeff_mu
            self.postnorm_mu += postnorm_coeff_mu * (meta_update_mu**2 - self.postnorm_mu)
            post_norm_mu = 1/np.sqrt(self.postnorm_mu + 1e-8)
            post_norm_v = 1
        elif self.post_norm_type.split('@')[0]=='postnorm_v':
            postnorm_coeff_v = float(self.post_norm_type.split('@')[1]) if '@' in self.post_norm_type else postnorm_coeff_v
            self.postnorm_v += postnorm_coeff_v * (meta_update_v**2 - self.postnorm_v)
            post_norm_mu = 1
            post_norm_v = 1/np.sqrt(self.postnorm_v)
        elif self.post_norm_type.split('@')[0]=='postnorm_both':
            postnorm_coeff_mu = float(self.post_norm_type.split('@')[1]) if '@' in self.post_norm_type else postnorm_coeff_mu
            postnorm_coeff_v = float(self.post_norm_type.split('@')[2]) if len(self.post_norm_type.split('@'))>2 else postnorm_coeff_v
            self.postnorm_mu += postnorm_coeff_mu * (meta_update_mu**2 - self.postnorm_mu)
            self.postnorm_v += postnorm_coeff_v * (meta_update_v**2 - self.postnorm_v)
            post_norm_mu = 1/np.sqrt(self.postnorm_mu + 1e-8)
            post_norm_v = 1/np.sqrt(self.postnorm_v + 1e-8)
            

        self.beta_mu = self.beta_mu - self.theta * post_norm_mu  * meta_update_mu
        self.beta_v  = self.beta_v  - self.theta * post_norm_v   * meta_update_v 

        self.h_mu = (1-eta_mu)*self.h_mu + x_tilde 
        self.h_v = (1-eta_v)*self.h_v + (x_tilde**2-1) 

        #self.h_mu = (1-eta_mu)*self.h_mu + x_centered 
        #self.h_v = (1-eta_v)*self.h_v + x_centered**2 - var_old



        # if self.t%10000==0:
        #     print(
        #         f"{self.t}\n"
        #         f"{npfmt5(eta_mu)}, {npfmt5(eta_v)}\n"
        #         f"{npfmt5(self.h_mu)}, {npfmt5(self.h_v)}\n"
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
    input_normalizer = GMDN(theta=theta_MDN, eta0=eta0, transform=transform)
    MSE_list = []

    for iteration in range(num_steps):
        x,z = env.next()
        x_tilde, mu, sigma = input_normalizer.step(x, z, w, b)

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
    