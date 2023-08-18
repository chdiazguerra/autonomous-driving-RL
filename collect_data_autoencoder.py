from argparse import ArgumentParser

from data.collect_data import main
from configuration.config import *

if __name__=='__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--world-port', type=int, default=WORLD_PORT)
    argparser.add_argument('--host', type=str, default=WORLD_HOST)
    argparser.add_argument('--map', type=str, default=TRAIN_MAP, help="Load the map before starting the simulation")
    argparser.add_argument('--weather', type=str, default='ClearNoon',
                           choices=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset',
                                    'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset',
                                    'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon',
                                    'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset'],
                                    help='Weather preset')
    argparser.add_argument('--out_folder', type=str, default='./sensor_data', help="Output folder")
    argparser.add_argument('--nb_frames', type=int, default=300, help="Number of frames to record per route")
    argparser.add_argument('--nb_passes', type=int, default=7, help="Number of passes per route")
    argparser.add_argument('--freq_save', type=int, default=5, help="Frequency of saving data (in steps)")
    argparser.add_argument('-np', action='store_true', help='Save data as numpy arrays instead of images')
    argparser.add_argument('--begin', type=int, default=0, help='Begin at this episode (for resuming)')

    args = argparser.parse_args()

    main(args)