from argparse import ArgumentParser
import math
import pickle
import os

import torch

from reinforcement.carla_env import CarlaEnv, Route
from models.autoencoder import Autoencoder, AutoencoderSEM
from models.vae import VAE
from configuration import config

if __name__=='__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--world-port', type=int, default=config.WORLD_PORT)
    argparser.add_argument('--host', type=str, default=config.WORLD_HOST)
    argparser.add_argument('--weather', type=str, default='ClearNoon',
                           choices=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset',
                                    'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset',
                                    'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon',
                                    'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset'],
                                    help='Weather preset')
    argparser.add_argument('--cam_height', type=int, default=config.CAM_HEIGHT, help="Camera height")
    argparser.add_argument('--cam_width', type=int, default=config.CAM_WIDTH, help="Camera width")
    argparser.add_argument('--fov', type=int, default=config.CAM_FOV, help="Camera field of view")
    argparser.add_argument('--nb_episodes', type=int, default=90, help="Number of episodes to run")
    argparser.add_argument('--tick', type=float, default=config.TICK, help="Sensor tick length")

    argparser.add_argument('--out_folder', type=str, default=config.EXPERT_DATA_FOLDER, help="Output folder")
    argparser.add_argument('--model', type=str, default=config.AE_MODEL, help='model',
                           choices=['Autoencoder', 'AutoencoderSEM', 'VAE'])
    argparser.add_argument('--autoencoder_model', type=str, help="Autoencoder model path", default=config.AE_PRETRAINED)
    argparser.add_argument('--from_ep', type=int, default=0, help="Start episode number (for resuming)")
    argparser.add_argument('-no_exo_vehicles', help="Use exo vehicles", action='store_false')

    args = argparser.parse_args()

    if args.model=='AutoencoderSEM':
        autoencoder = AutoencoderSEM.load_from_checkpoint(args.autoencoder_model)
    elif args.model=='VAE':
        autoencoder = VAE.load_from_checkpoint(args.autoencoder_model)
    elif args.model=='Autoencoder':
        autoencoder = Autoencoder.load_from_checkpoint(args.autoencoder_model)
    else:
        raise ValueError(f"Unknown model {args.model}")

    autoencoder.freeze()
    autoencoder.eval()

    out_folder = os.path.join(args.out_folder, args.weather)

    os.makedirs(out_folder, exist_ok=True)

    env = CarlaEnv(autoencoder, args.world_port, args.host, config.TRAIN_MAP, args.weather,
                 args.cam_height, args.cam_width, args.fov, args.tick, 1000, exo_vehicles=args.no_exo_vehicles)
    
    spawn_points = env.map.get_spawn_points()
    possible_routes = Route.get_possibilities(config.TRAIN_MAP)
    routes = [[possible_routes[0]().end.location],
            [possible_routes[1]().end.location],
            [possible_routes[2]().end.location]]
    
    for i in range(len(routes)):
        os.makedirs(os.path.join(out_folder, f'Route_{i}'), exist_ok=True)

    try:    

        for episode in range(args.from_ep, args.nb_episodes):
            print("Episode {}".format(episode))

            route_id = episode % len(routes)

            route = routes[route_id]

            obs = env.reset(route_id)

            env.traffic_manager.ignore_lights_percentage(env.ego_vehicle, 100)
            env.traffic_manager.random_left_lanechange_percentage(env.ego_vehicle, 0)
            env.traffic_manager.random_right_lanechange_percentage(env.ego_vehicle, 0)
            env.traffic_manager.auto_lane_change(env.ego_vehicle, False)
            env.traffic_manager.set_desired_speed(env.ego_vehicle, 35.)
            env.traffic_manager.distance_to_leading_vehicle(env.ego_vehicle, 4.0)

            env.traffic_manager.set_path(env.ego_vehicle, route)

            env.ego_vehicle.set_autopilot(True)

            done = False
            data = []
            while not done:
                control = env.ego_vehicle.get_control()
                action = [control.steer, 0.0]
                if control.throttle > control.brake:
                    action[1] = control.throttle
                else:
                    action[1] = -control.brake
                obs_t1, reward, done, info = env.step(action)

                if info["speed"] < 1e-6 and action[1] <= 0.0: #Avoid getting stuck when learning
                    action[1] = -0.05

                data.append([obs, action, reward, obs_t1, done, info])

                obs = obs_t1

            out_file = os.path.join(out_folder, f"Route_{route_id}", f"{episode}.pkl")
            with open(out_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"End of episode {episode} - {len(data)} steps")
    finally:
        env.reset_settings()
