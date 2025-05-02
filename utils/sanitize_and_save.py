import numpy as np
from datetime import datetime
import os
import csv
import json

def sanitize_and_save(save_to, MSE_list, alg_config):
    
    custom_RMSE_intervals = [
                            #'last_200k',
                            #'last_500k',
                            #'last_50%',
                            #'last_10%',
                            #'first_10%',
                            [900_000, 1_100_000],
                            #[600_000, 1_100_000],
                           ]
    num_segments = 10
    precision = 3
    for interval in custom_RMSE_intervals:
        alg_dict = {
            'Algorithm':alg_config['alg_name']  , 
            'Model':alg_config['model'], 
            'Parameters':alg_config['params'], 
            'Author':'Arsalan', 
            'Date added':datetime.now().strftime("%Y-%m-%d"), 
        }
            
        dataset_length = int(sanitize_dataset_name(alg_config['dataset'])['dataset_length'])
        dataset_dict = {
            'Dataset':sanitize_dataset_name(alg_config['dataset'])['dataset_name'], 
            'Dimension':sanitize_dataset_name(alg_config['dataset'])['dataset_dim'], 
            'Length':dataset_length, 
            'Dataset link':'', 
            'From':sanitize_number_to_k_m(interval[0]), 
            'To':sanitize_number_to_k_m(interval[1]), 
        }

        len_segment = dataset_length//num_segments
        results_dict = {f'RMSE segment {i+1}':np.sqrt(MSE_list[i*len_segment:(i+1)*len_segment].mean()) for i in range(num_segments)}
        results_dict['RMSE custom range'] = np.sqrt(MSE_list[interval[0]:interval[1]].mean())
        results_dict = {key:np.round(results_dict[key],precision) for key in results_dict}

        info_dict = {**alg_dict, **dataset_dict, **results_dict}

        add_line_to_csv_file(info_dict, save_to)


def sanitize_dataset_name(dataset):
    dataset_link=''
    if 'RSS' in dataset:
        return {'dataset_name':dataset.split('.')[0].split('-')[1], 
                'dataset_dim':dataset.split('.')[0].split('-')[2],
                'dataset_length':dataset.split('.')[0].split('-')[3]
                }
    if 'ASH' in dataset:
        return {'dataset_name':dataset.split('.')[0].split('-')[1], 
                'dataset_dim':dataset.split('.')[0].split('-')[2],
                'dataset_length':dataset.split('.')[0].split('-')[3]
                }




def extract_chunk_data(MSE_list):
    RMSE_chunks = []
    chunk_length = 100_000
    for i in range(len(MSE_list)//chunk_length):
        srat_ind = i*chunk_length
        end_ind = (i+1)*chunk_length
        RMSE_chunks.append(np.sqrt(MSE_list[srat_ind:end_ind].mean()))
    return RMSE_chunks

def extract_segment_data(MSE_list, num_segments):
    RMSE_segments = []
    len_segment = len(MSE_list)//num_segments
    for i in range(num_segments):
        srat_ind = i*len_segment
        end_ind = len(MSE_list) if i==num_segments-1 else (i+1)*len_segment
        RMSE_segments.append(np.sqrt(MSE_list[srat_ind:end_ind].mean()))
    return RMSE_segments



def sanitize_number_to_k_m(x):
    if x >= 1_000_000:
        if int(x) % 1_000_000 == 0:
            return f'{int(x)//1_000_000}m'
        return f'{x/1_000_000.0}m'
    elif x >= 1_000:
        if int(x) % 1_000 == 0:
            return f'{int(x)//1_000}k'
        return f'{x/1_000.0}k'
    return f'{int(x)}'

def extract_RMSE_custom_interval(MSE_list, custom_interval):
    len_data = len(MSE_list)
    if isinstance(custom_interval, list):
        start_ind, end_ind = custom_interval
    elif isinstance(custom_interval, str):
        assert 0 # incomplete
        if custom_interval in ['last_50%']:
            start_ind = int(.5*len_data)
            end_ind = len_data
        elif custom_interval in ['last_10%']:
            start_ind = int(.9*len_data)
            end_ind = len_data
        elif custom_interval in ['first_10%']:
            start_ind = 0
            end_ind = int(.1*len_data)
        elif custom_interval in ['last_200k']:
            start_ind = -min(200_000,len_data)
            end_ind = len_data
        elif custom_interval in ['last_500k']:
            start_ind = -min(500_000,len_data)
            end_ind = len_data
    return np.sqrt(MSE_list[start_ind:end_ind].mean())



def add_line_to_csv_file(info_dict, save_to):
    column_labels = [
        'Algorithm', 'Model', 'Parameters', 'Author', 
        'Dataset', 'Dimension', 'Length', 'Dataset link', 'From', 'To', 
        'Date added', 'verified by', 'Comm_author', 'Comm_others', 'reserved1', 'reserved2', 
        'RMSE custom range', 
        'RMSE segment 1', 'RMSE segment 2', 'RMSE segment 3', 'RMSE segment 4', 'RMSE segment 5', 'RMSE segment 6', 'RMSE segment 7', 'RMSE segment 8', 'RMSE segment 9', 'RMSE segment 10'
    ]
    info_keys = [key for key in info_dict]
    info_keys_lower = [info_key.lower() for info_key in info_keys]
    processed_data = []
    for key in column_labels:
        if key.lower() in info_keys_lower:
            info_key = info_keys[info_keys_lower.index(key.lower())]
            item = info_dict[info_key]
        else:
            item = ''
        if isinstance(item, dict):
            item = json.dumps(item)
        processed_data.append(item)

    with open(save_to, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(processed_data)
    
    




def check_existences_of_alg_in_csv(input_line, csv_path):
    if not os.path.exists(csv_path):
        open(csv_path, 'w').close()
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    input_fields = input_line.strip().split(',')
    if len(input_fields) < 5:
        raise ValueError("Input line does not have at least 5 comma-separated fields.")
    input_first5 = input_fields[:5]

    alg_already_exists_in_csv = False
    consistent = True  

    for line in lines:
        line_stripped = line.rstrip('\n')
        if line_stripped.startswith("**"):
            continue
        fields = line_stripped.split(',')
        if len(fields) < 5:
            continue
        if fields[:5] == input_first5:
            alg_already_exists_in_csv = True
            if line_stripped != input_line.strip():
                consistent = False
    if alg_already_exists_in_csv:
        if consistent:
            return 'exsits - consistent'
        return 'exists - InConsistent'
    return 'new'  # the algorithm is new and unique
    

def add_line_below_the_header(line, csv_path, UID, exsitence_and_consistency):
    if not os.path.exists(csv_path):
        open(csv_path, 'w').close()
    
    headers = ['new', 'exists - InConsistent', 'exsits - consistent']
    type_index = headers.index(exsitence_and_consistency)
    headers_list = [UID + ('  - ' + x) if x != 'New' else '' for x in headers]
    
    if not line.endswith('\n'):    # Ensure the input line ends with a newline.
        line += '\n'

    with open(csv_path, 'r') as f:
        lines = f.readlines()
    sep_line = "**"+'-'.join('' for i in range(50))+"\n"
    sep_line_thick = "**"+'='.join('' for i in range(100))+"\n"

    # --- Ensure the first header exists ---
    if not any(l.rstrip('\n') == headers_list[0] for l in lines):
        header0_line = headers_list[0] if headers_list[0].endswith('\n') else headers_list[0] + '\n'
        lines = [sep_line_thick, header0_line] + lines

    # Try to locate a header line that exactly matches headers_list[type_index]
    target_header = headers_list[type_index]
    matching_header_indices = [i for i, l in enumerate(lines) if l.rstrip('\n') == target_header]

    if matching_header_indices:
        header_index = matching_header_indices[0]  # take the first occurrence
        insertion_index = None
        for i in range(header_index + 1, len(lines)):
            if lines[i].lstrip().startswith("**--") or lines[i].lstrip().startswith("**=="):
                insertion_index = i
                break
        if insertion_index is None:
            # No following separator line found; append at the end.
            insertion_index = len(lines)
        # Insert the input line before the found separator line.
        lines.insert(insertion_index, line)
    else:
        # No header matching headers_list[type_index] was found.
        # Look for the largest i for i < type_index for which headers_list[i] exists in the CSV.
        found_prev = None
        for i in range(type_index - 1, -1, -1):
            if any(l.rstrip('\n') == headers_list[i] for l in lines):
                found_prev = i
                break
        if found_prev is not None:
            # Find the first occurrence of that previous header.
            for idx, l in enumerate(lines):
                if l.rstrip('\n') == headers_list[found_prev]:
                    previous_header_index = idx
                    break
            else:
                previous_header_index = None

            # Then find the header separator line that follows the previous header.
            insertion_index = None
            if previous_header_index is not None:
                for i in range(previous_header_index + 1, len(lines)):
                    if lines[i].lstrip().startswith("**--") or lines[i].lstrip().startswith("**==") :
                        insertion_index = i
                        break
            if insertion_index is None:
                insertion_index = len(lines)
        else:
            # If no previous header exists, choose to insert at the beginning.
            insertion_index = 0

        new_header_line = target_header if target_header.endswith('\n') else target_header + '\n'
        lines.insert(insertion_index, sep_line)
        lines.insert(insertion_index + 1, new_header_line)
        lines.insert(insertion_index + 2, line)

    # Write the updated content back to the CSV file.
    with open(csv_path, 'w') as f:
        f.writelines(lines)



# === Example usage ===
if __name__ == '__main__':
    # Example input line
    line = 'SGDm,{"stepsize":1e-05},1,Arsalan,RSS1,,,,,,,17.611,17.217,15.837,15.837,14.18,14.745,13.929,16.224,18.185,18.748,18.187,15.535,17.217'
    UID = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_file = 'csvs/test.txt'
    save_to = f'{csv_file}++{UID}'
    add_line_to_csv_file(line, save_to, UID=None)
    #exsitence_and_consistency = check_existences_of_alg_in_csv(line, csv_file)
    #add_line_below_the_header(line, csv_file, UID, exsitence_and_consistency)
    #print('exsitence_and_consistency: ', exsitence_and_consistency)





