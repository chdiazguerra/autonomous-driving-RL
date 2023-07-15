from argparse import ArgumentParser
import math
import pickle
import os

import torch
import numpy as np

from reinforcement.carla_env import CarlaEnv, Route
from reinforcement.agent import TD3ColDeductiveAgent
from models.autoencoder import Autoencoder, AutoencoderSEM

def train_agent(env, encoder, num_routes, weather_list, agent, nb_pretraining_steps, nb_training_steps):
    print(f"Pretraining... {nb_pretraining_steps} steps")
    for i in range(nb_pretraining_steps):
        agent.update_step(i, True)
    agent.save()
    print("Pretraining done")

    print(f"Training... {nb_training_steps} steps")
    agent.change_opt_lr(1e-4, 1e-3)

    tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec = [], [], [], []
    evaluate = False

    # avg_reward, std_reward, success_rate = test_agent(env, encoder, num_routes, weather_list, agent)
    # tr_steps_vec.append(0)
    # avg_reward_vec.append(avg_reward)
    # std_reward_vec.append(std_reward)
    # success_rate_vec.append(success_rate)

    done = False
    episode_nb = 0
    episode_reward = 0
    episode_steps = 0

    route_id = episode_nb%num_routes
    weather = weather_list[episode_nb//num_routes]

    env.set_weather(weather)
    obs = env.reset(route_id)
    obs[0] = torch.from_numpy(obs[0]).float().unsqueeze(0)
    obs[0] = torch.permute(obs[0], (0, 3, 1, 2))
    obs[0] = encoder(obs[0]).numpy()
    obs[1] = obs[1]
    prev_act = [0.,0.]

    without_noise = False
    updates = 0

    for tr_step in range(nb_training_steps):

        if (tr_step+1)% (nb_training_steps / 20) == 0:
            evaluate = True

        if (tr_step+1)%200 == 0:
            without_noise = not without_noise
            print("Without Noise" if without_noise else "With Noise")

        act = agent.select_action(obs, prev_act, eval=without_noise)
        obs_t1, reward, done, _ = env.step(act)

        obs_t1[0] = torch.from_numpy(obs_t1[0]).float().unsqueeze(0)
        obs_t1[0] = torch.permute(obs_t1[0], (0, 3, 1, 2))
        obs_t1[0] = encoder(obs_t1[0]).numpy()

        agent.store_transition(prev_act, obs, act, reward, obs_t1, done)

        if tr_step > 100 and tr_step%5 == 0:
            agent.update_step(updates, False)
            updates += 1

        obs = obs_t1
        
        episode_reward += reward
        episode_steps += 1

        if done:
            print('Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4d | Success: %5r' % \
                        (tr_step + 1, episode_nb + 1, episode_steps, episode_reward, reward>=450))
            episode_nb += 1
            done = False
            episode_reward = 0
            episode_steps = 0

            agent.save()
            with open('train_data.pkl', 'wb') as f:
                pickle.dump((tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec), f)

            if evaluate:
                avg_reward, std_reward, success_rate = test_agent(env, encoder, num_routes, weather_list, agent)
                tr_steps_vec.append(tr_step+1)
                avg_reward_vec.append(avg_reward)
                std_reward_vec.append(std_reward)
                success_rate_vec.append(success_rate)

                done = False
                episode_reward = 0
                episode_steps = 0

                evaluate = False

            route_id = episode_nb%num_routes
            weather = weather_list[(episode_nb%(num_routes*len(weather_list)))//num_routes]

            env.set_weather(weather)
            obs = env.reset(route_id)
            obs[0] = torch.from_numpy(obs[0]).float().unsqueeze(0)
            obs[0] = torch.permute(obs[0], (0, 3, 1, 2))
            obs[0] = encoder(obs[0]).numpy()
            obs[1] = obs[1]
            prev_act = [0.,0.]
            agent.reset_noise()

def test_agent(env, encoder, num_routes, weather_list, agent):
    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    nb_episodes = num_routes*len(weather_list)

    for episode in range(nb_episodes):
        route_id = episode%num_routes
        weather = weather_list[episode//num_routes]

        env.set_weather(weather)
        obs = env.reset(route_id)
        obs[0] = torch.from_numpy(obs[0]).float().unsqueeze(0)
        obs[0] = torch.permute(obs[0], (0, 3, 1, 2))
        obs[0] = encoder(obs[0]).numpy()

        done = False
        episode_reward = 0
        nb_steps = 0
        prev_act = [0.,0.]

        while not done:
            act = agent.select_action(obs, prev_act, eval=True)
            obs_t1, reward, done, _ = env.step(act)

            obs_t1[0] = torch.from_numpy(obs_t1[0]).float().unsqueeze(0)
            obs_t1[0] = torch.permute(obs_t1[0], (0, 3, 1, 2))
            obs_t1[0] = encoder(obs_t1[0]).numpy()

            obs = obs_t1

            episode_reward += reward
            nb_steps += 1

            prev_act = act

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
    argparser.add_argument('--world-port', type=int, default=2000)
    argparser.add_argument('--host', type=str, default='localhost')
    argparser.add_argument('--cam_height', type=int, default=256, help="Camera height")
    argparser.add_argument('--cam_width', type=int, default=256, help="Camera width")
    argparser.add_argument('--fov', type=int, default=100, help="Camera field of view")
    argparser.add_argument('--nb_vehicles', type=int, default=40, help="Number of vehicles in the simulation")
    argparser.add_argument('--nb_episodes', type=int, default=45, help="Number of episodes to run")
    argparser.add_argument('--tick', type=float, default=0.05, help="Sensor tick length")

    argparser.add_argument('-sem', action='store_true', help="Use semantic segmentation")
    argparser.add_argument('--autoencoder_model', type=str, help="Autoencoder model path", required=True)
    
    argparser.add_argument('--device', type=str, default='cuda', help="Device to use for training", choices=['cuda', 'cpu'])
    argparser.add_argument('--pre_steps', type=int, default=10000, help="Number of steps of pretraining")
    argparser.add_argument('--steps', type=int, default=300000, help="Number of steps of training")
    argparser.add_argument('--save_path', type=str, default='agent.pkl', help="Path to save the agent")
    argparser.add_argument('--exp_data', type=str, default='exp_data.pkl', help="Path of the expert data")

    args = argparser.parse_args()

    if args.sem:
        autoencoder = AutoencoderSEM.load_from_checkpoint(args.autoencoder_model)
    else:
        autoencoder = Autoencoder.load_from_checkpoint(args.autoencoder_model)

    autoencoder.freeze()
    encoder = autoencoder.encoder
    encoder.eval()

    env = CarlaEnv(args.world_port, args.host, 'Town01', 'ClearNoon',
                 args.cam_height, args.cam_width, args.fov, args.nb_vehicles, args.tick,
                 nb_frames_max=2000)
    
    num_routes = len(Route.get_possibilities('Town01'))
    weather_list = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']

    if os.path.exists(args.save_path):
        with open(args.save_path, 'rb') as f:
            agent = pickle.load(f)
    else:
        agent = TD3ColDeductiveAgent(obs_size=256, device=args.device, actor_lr=1e-3, critic_lr=1e-3,
                    pol_freq_update=2, policy_noise=0.2, noise_clip=0.5, act_noise=0.2, gamma=0.99,
                    tau=0.005, l2_reg=1e-5, env_steps=8, env_w=0.2, lambda_bc=0.1, lambda_a=0.9, lambda_q=1.0,
                    exp_buff_size=100000, actor_buffer_size=5000, exp_prop=0.25, batch_size=64,
                    save_path=args.save_path)
        
        with open(args.exp_data, 'rb') as f:
            exp_data = pickle.load(f)

        agent.load_exp_buffer(exp_data)

    train_agent(env, encoder, num_routes, weather_list, agent, args.pre_steps, args.steps)
    
    