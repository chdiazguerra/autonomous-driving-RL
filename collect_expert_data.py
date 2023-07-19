from argparse import ArgumentParser
import math
import pickle
import os

import torch

from reinforcement.carla_env import CarlaEnv, Route
from models.autoencoder import Autoencoder, AutoencoderSEM

if __name__=='__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--world-port', type=int, default=2000)
    argparser.add_argument('--host', type=str, default='localhost')
    argparser.add_argument('--weather', type=str, default='ClearNoon',
                           choices=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset',
                                    'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset',
                                    'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon',
                                    'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset'],
                                    help='Weather preset')
    argparser.add_argument('--cam_height', type=int, default=256, help="Camera height")
    argparser.add_argument('--cam_width', type=int, default=256, help="Camera width")
    argparser.add_argument('--fov', type=int, default=100, help="Camera field of view")
    argparser.add_argument('--nb_vehicles', type=int, default=40, help="Number of vehicles in the simulation")
    argparser.add_argument('--nb_episodes', type=int, default=90, help="Number of episodes to run")
    argparser.add_argument('--tick', type=float, default=0.05, help="Sensor tick length")

    argparser.add_argument('-sem', action='store_true', help="Use semantic segmentation")
    argparser.add_argument('--autoencoder_model', type=str, help="Autoencoder model path", required=True)
    argparser.add_argument('--out_folder', type=str, default='./expert_data', help="Output folder")
    argparser.add_argument('-low_semantic', action='store_true', help="Use low resolution semantic segmentation")
    argparser.add_argument('--from_ep', type=int, default=0, help="Start episode number")

    args = argparser.parse_args()

    if args.sem:
        autoencoder = AutoencoderSEM.load_from_checkpoint(args.autoencoder_model)
    else:
        autoencoder = Autoencoder.load_from_checkpoint(args.autoencoder_model)

    autoencoder.freeze()
    encoder = autoencoder.encoder
    encoder.eval()

    os.makedirs(args.out_folder, exist_ok=True)

    env = CarlaEnv(args.world_port, args.host, 'Town01', args.weather,
                 args.cam_height, args.cam_width, args.fov, args.nb_vehicles, args.tick,
                 1000, args.low_semantic)
    
    spawn_points = env.map.get_spawn_points()
    possible_routes = Route.get_possibilities("Town01")
    routes = [[possible_routes[0]().end.location],
            [possible_routes[1]().end.location],
            [possible_routes[2]().end.location]]
    
    for i in range(len(routes)):
        os.makedirs(os.path.join(args.out_folder, f'Route_{i}'), exist_ok=True)

    try:    

        for episode in range(args.from_ep, args.nb_episodes):
            print("Episode {}".format(episode))

            route_id = episode % len(routes)

            route = routes[route_id]

            obs = env.reset(route_id)

            obs[0] = torch.from_numpy(obs[0]).float().unsqueeze(0)
            obs[0] = torch.permute(obs[0], (0, 3, 1, 2))
            obs[0] = encoder(obs[0]).numpy()

            prev_act = [0.0, 0.0]

            env.traffic_manager.ignore_lights_percentage(env.ego_vehicle, 100)
            env.traffic_manager.random_left_lanechange_percentage(env.ego_vehicle, 0)
            env.traffic_manager.random_right_lanechange_percentage(env.ego_vehicle, 0)
            env.traffic_manager.auto_lane_change(env.ego_vehicle, False)
            env.traffic_manager.set_desired_speed(env.ego_vehicle, 25.)
            env.traffic_manager.distance_to_leading_vehicle(env.ego_vehicle, 5.0)

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

                obs_t1[0] = torch.from_numpy(obs_t1[0]).float().unsqueeze(0)
                obs_t1[0] = torch.permute(obs_t1[0], (0, 3, 1, 2))
                obs_t1[0] = encoder(obs_t1[0]).numpy()

                if info["speed"] < 1e-6 and action[1] <= 0.0: #Avoid getting stuck when learning
                    action[1] = -0.05

                data.append([prev_act, obs, action, reward, obs_t1, done, info])

                obs = obs_t1
                prev_act = action

            out_file = os.path.join(args.out_folder, f"Route_{route_id}", f"{episode}.pkl")
            with open(out_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"End of episode {episode} - {len(data)} steps")
    finally:
        env.reset_settings()
