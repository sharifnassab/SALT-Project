from datetime import datetime
import os
import numpy as np
from utils.data_loader import get_dir
import multiprocessing
from algorithms.Adam import param_sweeps as Adam_params
from algorithms.Newtonm import param_sweeps as Newton_params
from algorithms.NAD_SAN import param_sweeps as NAD_SAN_params
from algorithms.NAD_EAN import param_sweeps as NAD_EAN_params
from algorithms.LMS import param_sweeps as LMS_params
from algorithms.LMS_SAN import param_sweeps as LMS_SAN_params
from algorithms.LMS_EAN import param_sweeps as LMS_EAN_params
from algorithms.LMS_MDN import param_sweeps as LMS_MDN_params
from algorithms.LMS_MDNa import param_sweeps as LMS_MDNa_params
from algorithms.LMS_MDNPN import param_sweeps as LMS_MDNPN_params
from algorithms.LMS_MDNPN_KC import param_sweeps as LMS_MDNPN_KC_params


num_parallel_processes = 4
#num_parallel_processes = multiprocessing.cpu_count()-1

ASH_dir = get_dir('ASH')
RSS_dir = get_dir('RSS')

def run_experiment(run_index, total_runs, command):
    print(f"{run_index}/{total_runs}")
    os.system(command)

def params_to_commands(alg_config):
    return f'python3 {alg_config['file_path']} ' + ' '.join([f'--{param}={alg_config[param]}' for param in alg_config if not param=='file_path'])

def replace_sweep_of_a_variable(all_configs, new_param_name, list_new_param):
    # Find unique combinations excluding the parameter we're sweeping over
    other_param_names = [k for k in all_configs[0] if k != new_param_name]
    unique_combinations = {tuple(item[k] for k in other_param_names): item for item in all_configs}

    # Now create all combinations with the new parameter sweep
    new_configs = []
    for comb in unique_combinations.values():
        for param_value in list_new_param:
            new_item = comb.copy()
            new_item[new_param_name] = param_value
            new_configs.append(new_item)
    return new_configs


if __name__=='__main__':
    UID = 'results_'
    
    #list_alpha = [x.item() for x in list(10**(np.linspace(start=-5, stop=0, num=int(5/1) +1)))]
    list_alpha = [x.item() for x in 
                  list(10**(np.linspace(start=-8, stop=.5, num=int(8.5/.5) +1)))]

    dataset_list = [
                    #'SALTdata-RSS1-20-1100000.txt',
                    'SALTdata-RSS5-20-1100000.txt',
                    'SALTdata-RSS6-20-1100000.txt',
                    'SALTdata-RSS7-20-1100000.txt',
                    'SALTdata-ASH5-16-1100000.txt',
                    'SALTdata-ASH6-20-1100000.txt',
                    'SALTdata-ASH7-30-1100000.txt',
                    'SALTdata-ASH9-1-1100000.txt',
                    'SALTdata-ASH10-5-1100000.txt',
                    'SALTdata-ASH11-4-1100000.txt',
                    'SALTdata-ASH12-5-1100000.txt'
                    ]

    command_list = []

    for alg_params in [
                        Adam_params(),
                        Newton_params(),
                        #NAD_SAN_params(),
                        #NAD_EAN_params(),
                        LMS_params(),
                        #LMS_SAN_params(),
                        LMS_EAN_params(),
                        #LMS_MDN_params(),
                        #LMS_MDNa_params(),
                        #LMS_MDNPN_params(),
                        LMS_MDNPN_KC_params(),
                        ]:
        alg_params = replace_sweep_of_a_variable(alg_params, new_param_name='alpha', list_new_param=list_alpha)
        for alg_config in alg_params:
            alg_command = params_to_commands(alg_config)
            for dataset in dataset_list:
                command_list.append(alg_command + ' ' +
                                    ' '.join([f'--save_to=csvs/{UID}.csv', 
                                            f'--dataset={dataset}'])
                                    )

    if False:
        print(len(command_list))
        assert 0
    
    if False: # remove existing runs
        from utils.remove_existing_runs import remove_existing_runs
        keys_for_config = ['dataset', 'alg_name', ['params',['alpha']], 'bias', 'input_normalization', 'input_norm_gamma', 'save_to']
        args_list = remove_existing_runs([x+['--alg_name=SGD'] for x in args_list], keys_for_config, csv_file_path)

    indexed_commands = [(i, len(command_list), cmd) for i, cmd in enumerate(command_list, start=1)]
    with multiprocessing.Pool(processes=num_parallel_processes) as pool:
        pool.starmap(run_experiment, indexed_commands, chunksize=1)

