from argparse import ArgumentParser
import math
import pickle
import os

import torch
import numpy as np

from reinforcement.carla_env import CarlaEnv, Route
from reinforcement.ddpg_agent import DDPGAgent
from models.autoencoder import Autoencoder, AutoencoderSEM
from models.vae import VAE
from configuration import config

def train_agent(env, weather_list, agent, nb_training_episodes, save_folder, route_id, nb_updates=250, episode_skip=10):

    if (agent.episode_nb+1)==1:
        avg_reward, std_reward, success_rate = test_agent(env, weather_list, agent, route_id)
        agent.tr_steps_vec.append(agent.tr_step+1)
        agent.avg_reward_vec.append(avg_reward)
        agent.std_reward_vec.append(std_reward)
        agent.success_rate_vec.append(success_rate)
        agent.save_actor(os.path.join(save_folder, f"actor_ep_{agent.episode_nb+1}.pt"))

    max_steps = 200
    noise = True
    for agent.episode_nb in range(agent.episode_nb, nb_training_episodes):
        if agent.episode_nb > episode_skip*2:
            max_steps = 1000

        done = False
        episode_reward = 0
        episode_steps = 0

        agent.reset_noise()
        weather = weather_list[agent.episode_nb%len(weather_list)]
        env.set_weather(weather)

        obs = env.reset(route_id)

        transitions = []
        while not done and episode_steps < max_steps:
            act = agent.select_action(obs, noise=noise)
            obs_t1, reward, done, info = env.step(act)

            transitions.append((obs, act, reward, obs_t1, done))

            obs = obs_t1
            
            episode_reward += reward
            episode_steps += 1
            agent.tr_step += 1

        
        print('Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4d | Success: %5r' % \
                    (agent.tr_step + 1, agent.episode_nb + 1, episode_steps, episode_reward, reward>=450))
        
        if info['collision']:
            print("Collision")
            for transition in transitions[-50:]:
                agent.store_transition_collision(*transition)
            for transition in transitions[:-50]:
                agent.store_transition(*transition)

        else:
            for transition in transitions:
                agent.store_transition(*transition)
        
        if agent.episode_nb+1 > episode_skip:
            for _ in range(nb_updates):
                agent.update()

        if (agent.episode_nb+1)%20==0 and (agent.episode_nb+1)>episode_skip:
            avg_reward, std_reward, success_rate = test_agent(env, weather_list, agent, route_id)
            agent.tr_steps_vec.append(agent.tr_step+1)
            agent.avg_reward_vec.append(avg_reward)
            agent.std_reward_vec.append(std_reward)
            agent.success_rate_vec.append(success_rate)
            agent.save_actor(os.path.join(save_folder, f"actor_ep_{agent.episode_nb+1}.pt"))

        agent.save(os.path.join(save_folder, "agent.pkl"))

def test_agent(env, weather_list, agent, route_id):
    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    nb_episodes =3*len(weather_list)

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
    
    argparser.add_argument('--device', type=str, default='cpu', help="Device to use for training", choices=['cuda', 'cpu'])
    argparser.add_argument('--nb_episodes', type=int, default=config.TRAIN_EPISODES, help="Number of episodes of training")
    argparser.add_argument('--save_folder', type=str, default=config.AGENT_FOLDER, help="Path to save the agent and data")
    argparser.add_argument('--route_id', type=int, default=config.ROUTE_ID, help="Route id to use for training")
    argparser.add_argument('--nb_updates', type=int, default=config.DDPG_NB_UPDATES, help="Number of updates per episode")

    args = argparser.parse_args()

    if not os.path.exists(args.autoencoder_model):
        raise Exception('Autoencoder model not found')

    os.makedirs(args.save_folder, exist_ok=True)
    save_agent_path = os.path.join(args.save_folder, 'agent.pkl')

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

    env = CarlaEnv(autoencoder, args.world_port, args.host, config.TRAIN_MAP, 'ClearNoon',
                 args.cam_height, args.cam_width, args.fov, args.tick, 500, exo_vehicles=config.USE_EXO_VEHICLES)
    
    num_routes = len(Route.get_possibilities(config.TRAIN_MAP))
    weather_list = config.TRAIN_WEATHER

    if os.path.exists(save_agent_path):
        with open(save_agent_path, 'rb') as f:
            agent = pickle.load(f)
    else:
        agent = DDPGAgent(obs_dim=260, nb_actions=2, device='cpu', lr_actor=1e-4, lr_critic=1e-3,
                 batch_size=config.DDPG_BATCH_SIZE, gamma=0.95, tau=0.005, clip_norm=5e-3, buffer_size=20000, action_clip=(-1,1),
                 collision_percentage=0.2, noise_sigma=config.DDPG_NOISE_SIGMA, noise_decay=1/300, sch_gamma = 0.9,
                 sch_steps=config.DDPG_SCH_STEPS, use_expert_data=config.DDPG_USE_EXPERT_DATA, expert_percentage=0.25,
                 lambda_bc=0.5, use_env_model=config.DDPG_USE_ENV_MODEL, lambda_env=0.2,
                 env_steps=config.DDPG_ENV_STEPS)
        
        if config.DDPG_USE_EXPERT_DATA:
            agent.load_expert_data(config.DDPG_EXPERT_DATA_FILE)

            print("Pretraining...")
            for _ in range(config.DDPG_PRETRAIN_STEPS):
                agent.pretrain_update()     

    try:
        train_agent(env, weather_list, agent, args.nb_episodes, args.save_folder, args.route_id, args.nb_updates)
    finally:
        env.reset_settings()