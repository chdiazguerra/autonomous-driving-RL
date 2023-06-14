import os
import pickle
from argparse import ArgumentParser

SENSOR_TYPE = {'CAMERA': 0, 'DEPTH': 1, 'SEMANTIC': 2, 'OBSTACLE': 3}
TYPE_SENSOR = {v:k for k,v in SENSOR_TYPE.items()}

def get_data(data, info, folder, np=False):
    data_dict = {}
    for d in data:
        frame_real = d[0]
        type_ = d[1]
        if frame_real-1 in data_dict:
            frame = frame_real-1
        elif frame_real+1 in data_dict:
            frame = frame_real+1
        else:
            frame = frame_real
        if frame not in data_dict:
            data_dict[frame] = {}
        if type_ == SENSOR_TYPE['OBSTACLE']:
            data_dict[frame][TYPE_SENSOR[type_]] = d[2]
        else:
            name = f'{frame_real:08d}.npy' if np else f'{frame_real:08d}.png'
            data_dict[frame][TYPE_SENSOR[type_]] = os.path.abspath(f'{folder}/{TYPE_SENSOR[type_].lower()}/{name}')

    for frame_real, v in info.items():
        distance, yaw_diff, speed, is_junction = v
        if frame_real-1 in data_dict:
            frame = frame_real-1
        elif frame_real+1 in data_dict:
            frame = frame_real+1
        else:
            frame = frame_real
        if frame not in data_dict:
            continue
        data_dict[frame]['DISTANCE'] = distance
        data_dict[frame]['YAW_DIFF'] = yaw_diff
        data_dict[frame]['SPEED'] = speed
        data_dict[frame]['IS_JUNCTION'] = is_junction

    return data_dict

def clean_data(data_dict, folder_idx):
    final_data = []
    for frame, v in data_dict.items():
        if 'OBSTACLE' not in v:
            v['OBSTACLE'] = 100
        if not all([k in v for k in ['CAMERA', 'DEPTH', 'SEMANTIC', 'OBSTACLE', 'DISTANCE', 'YAW_DIFF', 'IS_JUNCTION']]):
            print(f'Frame {frame} dropped. Not enough data.')
            continue
        if os.path.exists(v['CAMERA']) and os.path.exists(v['DEPTH']) and os.path.exists(v['SEMANTIC']):
            data = {}
            data['CAMERA'] = v['CAMERA']
            data['DEPTH'] = v['DEPTH']
            data['SEMANTIC'] = v['SEMANTIC']
            data['DATA'] = [v['DISTANCE'], v['YAW_DIFF'], v['OBSTACLE']]
            data['JUNCTION'] = int(v['IS_JUNCTION'])
            data['IDX'] = folder_idx
            final_data.append(data)
        else:
            print(f'Frame {frame} dropped. File does not exist.')
    
    return final_data


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_file', type=str, default='dataset.pkl', help='output file name')
    parser.add_argument('--folders', type=str, nargs='+', help='folders to be processed', required=True)
    parser.add_argument('--idxs', type=int, nargs='+', default=None,
                        help='folders class index (separate different weather conditions)')
    parser.add_argument('-np', action='store_true', help='Camera images saved as .npy')

    args = parser.parse_args()

    if not args.out_file.endswith('.pkl'):
        args.out_file += '.pkl'

    final_data = []
    idxs = args.idxs if args.idxs is not None else range(len(args.folders))
    for folder_idx, folder in zip(idxs, args.folders):
        if not os.path.exists(folder):
            raise ValueError(f'Folder {folder} does not exist')
        
        print(f'Processing {folder}...')

        data = pickle.load(open(f'{folder}/data.pkl', 'rb'))
        info = pickle.load(open(f'{folder}/info.pkl', 'rb'))

        data_dict = get_data(data, info, folder, args.np)
        final_data.extend(clean_data(data_dict, folder_idx))

    with open(args.out_file, 'wb') as f:
        pickle.dump(final_data, f)
