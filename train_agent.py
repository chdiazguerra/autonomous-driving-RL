from argparse import ArgumentParser
import math
import pickle
import os

import torch
import numpy as np

from reinforcement.carla_env import CarlaEnv, Route
from reinforcement.agent import TD3ColDeductiveAgent
from models.autoencoder import Autoencoder, AutoencoderSEM

def train_agent(env, encoder, weather_list, agent, nb_pretraining_steps, nb_training_steps, save_folder, freq_eval, route_id):
    print(f"Pretraining... {nb_pretraining_steps-agent.pre_tr_step} steps")
    for agent.pre_tr_step in range(agent.pre_tr_step, nb_pretraining_steps):
        agent.update_step(agent.pre_tr_step, True)
    agent.save(os.path.join(save_folder, "agent.pkl"))
    print("Pretraining done")

    print(f"Training... {nb_training_steps} steps")
    if agent.change_lr:
        agent.change_opt_lr(1e-4, 1e-3)
        agent.change_lr = False

    evaluate = False

    avg_reward, std_reward, success_rate = test_agent(env, encoder, weather_list, agent, route_id)
    agent.tr_steps_vec.append(0)
    agent.avg_reward_vec.append(avg_reward)
    agent.std_reward_vec.append(std_reward)
    agent.success_rate_vec.append(success_rate)

    done = False
    episode_reward = 0
    episode_steps = 0

    weather = weather_list[agent.episode_nb%len(weather_list)]

    env.set_weather(weather)
    obs = env.reset(route_id)
    obs[0] = torch.from_numpy(obs[0]).float().unsqueeze(0)
    obs[0] = torch.permute(obs[0], (0, 3, 1, 2))
    obs[0] = encoder(obs[0]).numpy()
    obs[1] = obs[1]
    prev_act = [0.,0.]

    without_noise = False
    updates = 0

    for agent.tr_step in range(agent.tr_step, nb_training_steps):

        if (agent.tr_step+1)% freq_eval == 0:
            evaluate = True

        if (agent.tr_step+1)%1000 == 0 and (agent.tr_step+1) < 300000:
            without_noise = not without_noise
            print("Without Noise" if without_noise else "With Noise")
        if (agent.tr_step+1) >= 300000:
            without_noise = True

        act = agent.select_action(obs, prev_act, eval=without_noise)
        obs_t1, reward, done, _ = env.step(act)

        obs_t1[0] = torch.from_numpy(obs_t1[0]).float().unsqueeze(0)
        obs_t1[0] = torch.permute(obs_t1[0], (0, 3, 1, 2))
        obs_t1[0] = encoder(obs_t1[0]).numpy()

        agent.store_transition(prev_act, obs, act, reward, obs_t1, done)

        if agent.tr_step > 256 and agent.tr_step%2 == 0:
            agent.update_step(updates, False)
            updates += 1

        obs = obs_t1
        
        episode_reward += reward
        episode_steps += 1

        if done:
            print('Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4d | Success: %5r' % \
                        (agent.tr_step + 1, agent.episode_nb + 1, episode_steps, episode_reward, reward>=450))
            agent.episode_nb += 1
            done = False
            episode_reward = 0
            episode_steps = 0

            agent.save(os.path.join(save_folder, "agent.pkl"))
            with open(os.path.join(save_folder, "data.pkl"), 'wb') as f:
                pickle.dump((agent.tr_steps_vec, agent.avg_reward_vec, agent.std_reward_vec, agent.success_rate_vec), f)

            if evaluate:
                avg_reward, std_reward, success_rate = test_agent(env, encoder, weather_list, agent, route_id)
                agent.tr_steps_vec.append(agent.tr_step+1)
                agent.avg_reward_vec.append(avg_reward)
                agent.std_reward_vec.append(std_reward)
                agent.success_rate_vec.append(success_rate)

                done = False
                episode_reward = 0
                episode_steps = 0

                evaluate = False
                agent.save(os.path.join(save_folder, f"agent_{agent.tr_step+1}_steps.pkl"))

            weather = weather_list[(agent.episode_nb%len(weather_list))]

            env.set_weather(weather)
            obs = env.reset(route_id)
            obs[0] = torch.from_numpy(obs[0]).float().unsqueeze(0)
            obs[0] = torch.permute(obs[0], (0, 3, 1, 2))
            obs[0] = encoder(obs[0]).numpy()
            obs[1] = obs[1]
            prev_act = [0.,0.]
            agent.reset_noise()

def test_agent(env, encoder, weather_list, agent, route_id):
    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    nb_episodes = 3*len(weather_list)

    for episode in range(nb_episodes):
        weather = weather_list[episode%len(weather_list)]

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
    argparser.add_argument('--tick', type=float, default=0.05, help="Sensor tick length")

    argparser.add_argument('-sem', action='store_true', help="Use semantic segmentation")
    argparser.add_argument('--autoencoder_model', type=str, help="Autoencoder model path", required=True)
    
    argparser.add_argument('--device', type=str, default='cuda', help="Device to use for training", choices=['cuda', 'cpu'])
    argparser.add_argument('--pre_steps', type=int, default=2000, help="Number of steps of pretraining")
    argparser.add_argument('--steps', type=int, default=1000000, help="Number of steps of training")
    argparser.add_argument('--save_folder', type=str, default='./agent', help="Path to save the agent and data")
    argparser.add_argument('--exp_data', type=str, default='exp_data.pkl', help="Path of the expert data")
    argparser.add_argument('--freq_eval', type=int, default=10000, help="Frequency of evaluation")
    argparser.add_argument('--route_id', type=int, default=0, help="Route id to use for training")

    args = argparser.parse_args()

    if not os.path.exists(args.exp_data):
        raise Exception('Expert data not found')
    if not os.path.exists(args.autoencoder_model):
        raise Exception('Autoencoder model not found')

    os.makedirs(args.save_folder, exist_ok=True)
    save_agent_path = os.path.join(args.save_folder, 'agent.pkl')

    if args.sem:
        autoencoder = AutoencoderSEM.load_from_checkpoint(args.autoencoder_model)
    else:
        autoencoder = Autoencoder.load_from_checkpoint(args.autoencoder_model)

    autoencoder.freeze()
    encoder = autoencoder.encoder
    encoder.eval()

    env = CarlaEnv(args.world_port, args.host, 'Town01', 'ClearNoon',
                 args.cam_height, args.cam_width, args.fov, args.nb_vehicles, args.tick,
                 nb_frames_max=1000)
    
    num_routes = len(Route.get_possibilities('Town01'))
    weather_list = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']

    if os.path.exists(save_agent_path):
        with open(save_agent_path, 'rb') as f:
            agent = pickle.load(f)
    else:
        agent = TD3ColDeductiveAgent(obs_size=256, device=args.device, actor_lr=1e-3, critic_lr=1e-3,
                    pol_freq_update=2, policy_noise=0.2, noise_clip=0.5, act_noise=0.1, gamma=0.99,
                    tau=0.005, l2_reg=1e-5, env_steps=10, env_w=0.2, lambda_bc=0.1, lambda_a=0.9, lambda_q=1.0,
                    exp_buff_size=20000, actor_buffer_size=20000, exp_prop=0.25, batch_size=64,
                    scheduler_step_size=350, scheduler_gamma=0.9)
        
        with open(args.exp_data, 'rb') as f:
            exp_data = pickle.load(f)

        agent.load_exp_buffer(exp_data)

    train_agent(env, encoder, weather_list, agent, args.pre_steps, args.steps, args.save_folder, args.freq_eval, args.route_id)
