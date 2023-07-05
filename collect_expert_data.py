from argparse import ArgumentParser
import math
import pickle
import os

import torch

from reinforcement.carla_env import CarlaEnv
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
    argparser.add_argument('--nb_episodes', type=int, default=45, help="Number of episodes to run")
    argparser.add_argument('--tick', type=float, default=0.05, help="Sensor tick length")

    argparser.add_argument('-sem', action='store_true', help="Use semantic segmentation")
    argparser.add_argument('--autoencoder_model', type=str, help="Autoencoder model path", required=True)
    argparser.add_argument('--out_folder', type=str, default='./expert_data', help="Output folder")
    argparser.add_argument('-low_semantic', action='store_true', help="Use low resolution semantic segmentation")

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
                 math.inf, args.low_semantic)
    
    spawn_points = env.map.get_spawn_points()
    routes = [[149, 98, 197, 199, 110, 14, 115, 61],
            [222, 200, 77, 168, 19, 166, 164, 21, 161, 24, 159, 80, 46, 26, 157, 136, 155, 28, 151, 76, 104, 226, 228],
            [56, 53, 51, 83, 55, 169, 171, 173, 33, 11, 84, 137, 153, 27, 154]]

    for episode in range(args.nb_episodes):
        print("Episode {}".format(episode))

        route_id = episode % len(routes)

        spawn_point = spawn_points[routes[route_id][0]]
        route = []
        for i in routes[route_id][1:]:
            route.append(spawn_points[i].location)

        obs = env.reset(route_id)

        obs[0] = torch.from_numpy(obs[0]).float().unsqueeze(0)
        obs[0] = torch.permute(obs[0], (0, 3, 1, 2))
        obs[0] = encoder(obs[0]).numpy()

        env.ego_vehicle.set_transform(spawn_point)
        env.traffic_manager.ignore_lights_percentage(env.ego_vehicle, 100)
        env.traffic_manager.random_left_lanechange_percentage(env.ego_vehicle, 0)
        env.traffic_manager.random_right_lanechange_percentage(env.ego_vehicle, 0)
        env.traffic_manager.auto_lane_change(env.ego_vehicle, False)
        env.traffic_manager.set_desired_speed(env.ego_vehicle, 40.)

        env.traffic_manager.set_path(env.ego_vehicle, route)

        env.ego_vehicle.set_autopilot(True)

        done = False
        data = []
        while not done:
            control = env.ego_vehicle.get_control()
            action = [control.steer, control.throttle, control.brake]
            obs_t1, reward, done, info = env.step(action)

            obs_t1[0] = torch.from_numpy(obs_t1[0]).float().unsqueeze(0)
            obs_t1[0] = torch.permute(obs_t1[0], (0, 3, 1, 2))
            obs_t1[0] = encoder(obs_t1[0]).numpy()

            data.append([obs, action, reward, obs_t1, done, info])

            obs = obs_t1

        out_file = os.path.join(args.out_folder, f"{episode}.pkl")
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"End of episode {episode} - {len(data)} steps")
