import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
import ast
import itertools
import numpy as np
from adjustText import adjust_text
import csv
import json


csv_file = 'csvs/results_RSS1_.csv'
x_lim = None
#x_lim = [1e-4,1]


dataset_info = [
    {'name':'RSS1', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 22.5},
    #{'name':'RSS5', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 27.0},
    #{'name':'RSS6', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 28.0},
    #{'name':'RSS7', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 23.0},
    #{'name':'ASH5', "dim":16, 'From':'900k', "To":"1.1m", "ylim_max": 43.0},
    #{'name':'ASH6', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 40.0},
    #{'name':'ASH7', "dim":30, 'From':'900k', "To":"1.1m", "ylim_max": 50.0},
    #{'name':'ASH9', "dim":1, 'From':'900k', "To":"1.1m", "ylim_max": 24.0},
    ##{'name':'ASH10', "dim":5, 'From':'900k', "To":"1.1m"},
    ##{'name':'ASH11', "dim":4, 'From':'900k', "To":"1.1m", "ylim_max": 100.0},
    #{'name':'ASH12', "dim":5, 'From':'900k', "To":"1.1m", "ylim_max": 60.0},
]


experiments = [
    {"Alg":"Adam", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    {"Alg":"Newton", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"NAD-SAN", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"NAD-EAN", "Model":"affine", "Params":{"tau":100}, "x_axis":"alpha"},
    {"Alg":"LMS", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"LMS-SAN", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    {"Alg":"LMS-EAN", "Model":"affine", "Params":{"tau":100}, "x_axis":"alpha"},
    #{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.001}, "x_axis":"alpha"},
    #{"Alg":"LMS-MDNa", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha"},
    #{"Alg":"LMS-MDNPN", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha"},
    {"Alg":"LMS-MDNPN_KC", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha"},
    
    {"Alg":"IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"IDBD2", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    
    #{"Alg":"LMS^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_EAN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNa^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNPN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    {"Alg":"LMS_MDNPN_KC^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    
    #{"Alg":"LMS^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_EAN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNa^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNPN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    {"Alg":"LMS_MDNPN_KC^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    
]


## Do not change past this point


csv_column_numbers = {'Alg': 0, 'Model': 1, 'Params': 2, 'Dataset': 4, "Dim":5, "From":8, "To":9, 'Result': 16}



def remove_experiments_that_have_no_data_in_the_CSV_file(csv_file, dataset, experiments):
    # Read the entire CSV file once
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        all_rows = list(csv.reader(file))

    def does_csv_contain_this_experiment(experiment):
        for row in all_rows:
            if (row[csv_column_numbers['Dataset']] != str(dataset['name']) or
                row[csv_column_numbers['Dim']]     != str(dataset['dim']) or
                row[csv_column_numbers['From']]    != str(dataset['From']) or
                row[csv_column_numbers['To']]      != str(dataset['To']) or
                row[csv_column_numbers['Alg']]     != str(experiment['Alg']) or
                row[csv_column_numbers['Model']]   != str(experiment['Model'])):
                continue
            try:
                param_str = row[csv_column_numbers['Params']].replace('""', '"')
                params = json.loads(param_str)
            except:
                try:
                    params = ast.literal_eval(row[csv_column_numbers['Params']])
                except:
                    continue
            if experiment['x_axis'] not in params:
                continue
            if all(k in params and str(params[k]) == str(v) for k, v in experiment.get('Params', {}).items()):
                return True
        return False

    list_output_experiments = []
    for experiment in experiments:
        if does_csv_contain_this_experiment(experiment):
            list_output_experiments.append(experiment)
        else:
            raise ValueError(f'\nThe csv file contains no rows corresponding to the following experiment:\n {experiment}')
    return list_output_experiments
        

def distributed_legend(ax, curves, lines, y_max, legend_fontsize=13):
    x_all = np.concatenate([c['x'] for c in curves])
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    x_positions = np.linspace(x_min, x_max, len(curves) + 2)[1:-1]

    texts, out_of_bounds = [], []
    for line, xpos in zip(lines, x_positions):
        xdata, ydata = line.get_data()
        label = line.get_label().split(',')[0]
        valid = np.isfinite(ydata) & (ydata <= y_max if y_max else True)

        if np.any(valid):
            idx = np.abs(xdata[valid] - xpos).argmin()
            texts.append(plt.text(xdata[valid][idx], ydata[valid][idx], label,
                                  color=line.get_color(), fontsize=legend_fontsize, fontweight='bold'))
        else:
            out_of_bounds.append((label, line.get_color()))

    adjust_text(texts, avoid_self=True, avoid_points=True, expand_text=(1.2, 1.4))

    # Handle out-of-bounds labels
    if out_of_bounds:
        y_bottom, y_top = ax.ylim()
        x_left, _ = ax.xlim()
        vertical_spacing = (y_top - y_bottom) * 0.04

        for i, (label, color) in enumerate(out_of_bounds):
            ax.text(x_left, y_bottom + vertical_spacing * i, f"{label}: out of bounds",
                     fontsize=legend_fontsize, fontweight='bold', color=color, va='bottom')



def manage_outlier_segments(x, y, threshold, color, epsilon=.5):
    for i in range(len(x) - 1):
        if y[i+1] > threshold:
            dy,dx = (y[i+1] - y[i]) , np.abs(x[i+1] - x[i])
            y_segment = [y[i], y[i] + epsilon*dy/(np.abs(dy)+dx+1e-10)]
            x_segment = [x[i], x[i] + epsilon*dx/(np.abs(dy)+dx+1e-10)]
            plt.plot(x_segment, y_segment, linestyle=':', color=color)
    for i in range(1,len(x)):
        if y[i-1] > threshold:
            dy,dx = np.abs(y[i-1] - y[i]) , np.abs(x[i] - x[i-1])
            y_segment = [y[i] + epsilon*dy/(np.abs(dy)+dx+1e-10), y[i]]
            x_segment = [x[i] - epsilon*dx/(np.abs(dy)+dx+1e-10), x[i]]
            plt.plot(x_segment, y_segment, linestyle=':', color=color)



def build_curves(csv_file, experiments, csv_column_numbers, dataset):
    def flatten_dict(d, parent_key='', sep='.'):
        items = {}
        if not isinstance(d, dict): return {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict): items.update(flatten_dict(v, new_key, sep=sep))
            else: items[new_key] = v
        return items
    
    df = pd.read_csv(csv_file, header=None)

    # Rename columns based on provided indices
    col_indices = csv_column_numbers
    df.rename(columns={col_indices['Alg']: 'Alg', 
                       col_indices['Model']: 'Model', 
                       col_indices['Params']: 'Params',
                       col_indices['Dataset']: 'Dataset', 
                       col_indices['Result']: 'Result',
                       col_indices['Dim']: 'Dim',
                       col_indices['From']: 'From',
                       col_indices['To']: 'To'}, inplace=True)

    df['Params_dict'] = df['Params'].apply(ast.literal_eval).apply(flatten_dict)
    min_result = float('inf')

    base_filtered_df = df[(df['Dataset'] == dataset['name']) &
                          (df['Dim'] == dataset['dim']) &
                          (df['From'] == dataset['From']) &
                          (df['To'] == dataset['To'])]

    curves = []
    for exp in experiments:
        alg, model, params_filter, x_axis = exp["Alg"], exp["Model"], exp["Params"], exp["x_axis"]
        
        filtered_df = base_filtered_df[(base_filtered_df['Alg'] == alg) & (base_filtered_df['Model'] == model)]

        all_params = filtered_df['Params_dict'].apply(lambda d: set(d.keys())).explode().unique().tolist()
        other_params = [p for p in all_params if p not in params_filter and p != x_axis]

        param_values_combinations = {}
        for param in other_params:
            param_values_combinations[param] = filtered_df['Params_dict'].apply(lambda d: d.get(param)).dropna().unique().tolist()

        combinations = list(itertools.product(*param_values_combinations.values())) if param_values_combinations else [()]

        for comb in combinations:
            temp_df = filtered_df.copy()
            comb_dict = dict(zip(other_params, comb))
            for key, value in {**params_filter, **comb_dict}.items():
                temp_df = temp_df[temp_df['Params_dict'].apply(lambda p: p.get(key) == value)]

            temp_df = temp_df[temp_df['Params_dict'].apply(lambda p: x_axis in p)]
            temp_df['x_value'] = temp_df['Params_dict'].apply(lambda p: p[x_axis])

            temp_df = temp_df.sort_values(by='x_value')

            #if not temp_df['Result'].empty:
            #    min_result = min(min_result, temp_df['Result'].min())

            params_str = ', '.join(f'{k}={v}' for k, v in {**params_filter, **comb_dict}.items())
            label = f'{alg}, {model}, ({params_str}), '
            curves.append({'x':temp_df['x_value'], 'y':temp_df['Result'], 'label':label, 'linestyle':('--' if exp['x_axis']=='metastep' else '-' if  exp['x_axis']=='alpha' else 0/0)})
    for curve in curves:
        old_x = np.nan
        print('\n\n', curve['label'])
        for x,y in zip(list(curve['x']),list(curve['y'])):
            if not x==old_x:
                print(f'alpha=10^({np.log10(x)})\t  RMSE={y}')
            old_x = x
    return curves





def plot_curves(curves, dataset, ax=None):
    remove_outlier_segments_larger_than = 1.5*dataset['ylim_max'] if 'ylim_max' in dataset and dataset['ylim_max'] is not None else None
    fontsize = {
                "in_figure_legend":15,
                "detailed_legend":9,
                "axes":15,
                "axes_numbers":14,
                "title":16
                }
    ylim_min_epsilon = 1
    min_result = min(curve['y'].min() for curve in curves)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))

    lines = []
    for curve in curves:
        if remove_outlier_segments_larger_than is not None:
            y_with_naned_outliers = np.array(curve['y']) 
            y_with_naned_outliers[np.array(curve['y'])>remove_outlier_segments_larger_than] =np.nan
            line, = ax.plot(np.array(curve['x']), y_with_naned_outliers, marker='o', label=curve['label'], linestyle=curve['linestyle'])
            y_with_nan_to_num = np.nan_to_num(np.array(curve['y']) , nan=1e16)
            manage_outlier_segments(list(curve['x']), y_with_nan_to_num , epsilon=2, threshold=remove_outlier_segments_larger_than, color=line.get_color())
        else:
            line, = ax.plot(curve['x'], curve['y'], marker='o', label=curve['label'], linestyle=curve['linestyle'])
        lines.append(line)
    if dataset['ylim_max'] is not None:
        ax.set_ylim(min_result - ylim_min_epsilon, dataset['ylim_max'])
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_xscale("log")
    ax.grid(True)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=fontsize['detailed_legend'])
    distributed_legend(ax, curves=curves, lines=lines, y_max=dataset['ylim_max']-1, legend_fontsize=fontsize['in_figure_legend'])

    ax.set_xlabel(' / '.join(sorted(list(set([exp['x_axis'] for exp in experiments])))), fontsize=fontsize['axes'])
    ax.set_ylabel('RMSE', fontsize=fontsize['axes'])
    title = (f"{dataset['name']}-{dataset['dim']},   "
             f"({dataset['From']}--{dataset['To']})")
    ax.tick_params(axis='both',labelsize=fontsize['axes_numbers'])
    ax.set_title(title, fontsize=fontsize['title'])
    

def plt_plot():
    plt.tight_layout(rect=[0.03, 0.03, .97, .97])
    plt.savefig(f'plots/{dataset['name']}.png', dpi=300, bbox_inches='tight')
    plt.show()

for dataset in dataset_info:
    verified_experiments = remove_experiments_that_have_no_data_in_the_CSV_file(csv_file, dataset, experiments)
    curves = build_curves(csv_file, verified_experiments, csv_column_numbers, dataset=dataset)
    plot_curves(curves, dataset, ax=None)
    plt_plot()

