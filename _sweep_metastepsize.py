from datetime import datetime
import os
import numpy as np
from utils.data_loader import get_dir
import multiprocessing
import json
from algorithms.IDBD import param_sweeps as IDBD_params
from algorithms.IDBD2 import param_sweeps as IDBD2_params
from algorithms.IDBD_MGEN import param_sweeps as IDBD_MGEN_params
from algorithms.Normalized_IDBD import param_sweeps as Normalized_IDBD_params


#num_parallel_processes = 7
num_parallel_processes = multiprocessing.cpu_count()-1

ASH_dir = get_dir('ASH')
RSS_dir = get_dir('RSS')

def run_experiment(run_index, total_runs, command):
    print(f"{run_index}/{total_runs}")
    os.system(command)

def params_to_commands(alg_config):
    return f'python3 {alg_config['file_path']} ' + ' '.join([f'--{param}='+ ('"'+f'{alg_config[param]}'+
                                                                             '"' if isinstance(alg_config[param],dict) 
                                                                                 else f'{alg_config[param]}') 
                                                                            for param in alg_config if not param=='file_path']) 
def replace_sweep_of_a_variable(all_configs, new_param_name, list_new_param):
    unique_keys = {
        json.dumps({k: v for k, v in cfg.items() if k != new_param_name}, sort_keys=True)
        for cfg in all_configs
    }

    return [
        {**json.loads(cfg_json), new_param_name: val}
        for cfg_json in unique_keys
        for val in list_new_param
    ]

if __name__=='__main__':
    UID = 'results_'
    list_meta_stepsize = [x.item() for x in 
                          #list(10**(np.linspace(start=-6, stop=-2, num=int(4/1) +1)))]
                          list(10**(np.linspace(start=-7, stop=.5, num=int(7.5/.5) +1)))]
    
    dataset_list = [
                    'SALTdata-RSS1-20-1100000.txt',
                    #'SALTdata-RSS5-20-1100000.txt',
                    #'SALTdata-RSS6-20-1100000.txt',
                    #'SALTdata-RSS7-20-1100000.txt',
                    #'SALTdata-ASH5-16-1100000.txt',
                    #'SALTdata-ASH6-20-1100000.txt',
                    #'SALTdata-ASH7-30-1100000.txt',
                    #'SALTdata-ASH9-1-1100000.txt',
                    #'SALTdata-ASH10-5-1100000.txt',
                    #'SALTdata-ASH11-4-1100000.txt',
                    #'SALTdata-ASH12-5-1100000.txt'
                    ]

    command_list = []
    for alg_params in [
                        IDBD_params(),
                        IDBD2_params(),
                        IDBD_MGEN_params(),
                        Normalized_IDBD_params(),
                        ]:
        alg_params = replace_sweep_of_a_variable(alg_params, new_param_name='meta_stepsize', list_new_param=list_meta_stepsize)
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

