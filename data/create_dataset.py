import os
import pickle
from argparse import ArgumentParser

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_file', type=str, default='dataset.pkl', help='output file name')
    parser.add_argument('--folders', type=str, nargs='+', help='folders to be processed', required=True)
    parser.add_argument('--idxs', type=int, nargs='+', default=None,
                        help='folders class index (separate different weather conditions)')
    parser.add_argument('-np', action='store_true', help='Camera images saved as .npy')

    args = parser.parse_args()

    if args.idxs is None:
        args.idxs = list(range(len(args.folders)))
    else:
        assert len(args.folders) == len(args.idxs), 'Number of folders and idxs must be the same'

    if not args.out_file.endswith('.pkl'):
        args.out_file += '.pkl'

    final_data = []
    for idx, folder in zip(args.idxs, args.folders):
        if not os.path.exists(folder):
            raise ValueError(f'Folder {folder} does not exist')
        
        print(f'Processing {folder}...')

        folder = os.path.abspath(folder)
        camera = os.path.join(folder, 'camera')
        depth = os.path.join(folder, 'depth')
        semantic = os.path.join(folder, 'semantic')

        with open(os.path.join(folder, 'info.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        for frame, d in data.items():
            image = os.path.join(camera, f'{frame:08d}.png') if not args.np else os.path.join(camera, f'{frame:08d}.npy')
            depth_image = os.path.join(depth, f'{frame:08d}.png') if not args.np else os.path.join(depth, f'{frame:08d}.npy')
            semantic_image = os.path.join(semantic, f'{frame:08d}.png') if not args.np else os.path.join(semantic, f'{frame:08d}.npy')
            additional = d[:3]
            junction = d[4]
            final_data.append((idx, image, depth_image, semantic_image, additional, junction))

    with open(args.out_file, 'wb') as f:
        pickle.dump(final_data, f)
