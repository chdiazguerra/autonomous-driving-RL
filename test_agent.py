from argparse import ArgumentParser
import math
import pickle
import os

import torch
import numpy as np

from reinforcement.carla_env import CarlaEnv, Route
from models.autoencoder import Autoencoder, AutoencoderSEM
from models.vae import VAE
from configuration import config

def test_agent(env, weather_list, agent, route_id, nb_episodes):
    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    for episode in range(nb_episodes):
        weather = weather_list[episode%len(weather_list)]

        env.set_weather(weather)
        obs = env.reset(route_id)

        done = False
        episode_reward = 0
        nb_steps = 0

        while not done:
            act = agent.select_action(obs, noise=False)
            print(act)
            obs_t1, reward, done, _ = env.step(act)

            obs = obs_t1

            episode_reward += reward
            nb_steps += 1

            if done:
                if reward > 450:
                    success_rate += 1

                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                print('Evaluation episode %3d | Steps: %4d | Reward: %4d | Success: %r' % (episode + 1, nb_steps, episode_reward, reward>450))     
            
    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    success_rate /= nb_episodes
    avg_steps /= nb_episodes
    
    print('Average Reward: %.2f, Reward Deviation: %.2f | Average Steps: %.2f, Success Rate: %.2f' % (avg_reward, std_reward, avg_steps, success_rate))
    print('%.2f/%.2f/%.2f' % (success_rate, avg_reward, std_reward))
    return avg_reward, std_reward, success_rate

if __name__=='__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--world-port', type=int, default=config.WORLD_PORT)
    argparser.add_argument('--host', type=str, default=config.WORLD_HOST)
    argparser.add_argument('--cam_height', type=int, default=config.CAM_HEIGHT, help="Camera height")
    argparser.add_argument('--cam_width', type=int, default=config.CAM_WIDTH, help="Camera width")
    argparser.add_argument('--fov', type=int, default=config.CAM_FOV, help="Camera field of view")
    argparser.add_argument('--tick', type=float, default=config.TICK, help="Sensor tick length")

    argparser.add_argument('--model', type=str, default=config.AE_MODEL, help='model',
                           choices=['Autoencoder', 'AutoencoderSEM', 'VAE'])
    argparser.add_argument('--autoencoder_model', type=str, help="Autoencoder model path", default=config.AE_PRETRAINED)
    
    argparser.add_argument('--device', type=str, default='cpu', help="Device to use for testing", choices=['cuda', 'cpu'])
    argparser.add_argument('--nb_episodes', type=int, default=20, help="Number of episodes of testing")
    argparser.add_argument('--route_id', type=int, required=True, help="Route id to use for testing")
    argparser.add_argument('--agent_model', type=str, required=True, help="Trained agent file")
    argparser.add_argument('--type', type=str, default='test', choices=['test', 'train'], help="Type of evaluation")
    argparser.add_argument('-exo_vehicles', action='store_true', help="Use exo vehicles")

    args = argparser.parse_args()

    if not os.path.exists(args.autoencoder_model):
        raise Exception('Autoencoder model not found')
    
    if not os.path.exists(args.agent_model):
        raise Exception('Agent model not found')

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

    map_name = config.TRAIN_MAP if args.type=='train' else config.TEST_MAP
    print(map_name)

    env = CarlaEnv(autoencoder, args.world_port, args.host, map_name, 'ClearNoon',
                 args.cam_height, args.cam_width, args.fov, args.tick, 500, exo_vehicles=args.exo_vehicles)
    
    num_routes = len(Route.get_possibilities(config.TRAIN_MAP))
    weather_list = config.TRAIN_WEATHER if args.type=='train' else config.TEST_WEATHER

   
    with open(args.agent_model, 'rb') as f:
        agent = pickle.load(f)   

    try:
        test_agent(env, weather_list, agent, args.route_id, args.nb_episodes)
    finally:
        env.reset_settings()