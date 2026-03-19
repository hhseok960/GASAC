import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import time

import SAC
import SAC_NF
from gasac.wasserstein import WGASACAgent
from gasac.vanilla import GASACAgent


def agent_test(agent, env_name, file_name, seed, test_episodes=10):
    print(f">>> Env_name: {env_name} - ", end="")
    env4test = gym.make(env_name)

    max_a = float(env4test.action_space.high[0])

    total_rewards = np.zeros(test_episodes)
    for epi in range(test_episodes):
        s, d = env4test.reset(seed=seed + 100 * epi)[0], False
        s = torch.unsqueeze(torch.FloatTensor(s), dim=0)

        while not d:
            a = agent.decide_action(s)
            s, r, termi, trunc, _ = env4test.step(a.squeeze().to("cpu").numpy() * max_a)
            s = torch.unsqueeze(torch.FloatTensor(s), dim=0)
            d = termi or trunc
            total_rewards[epi] += r

    if not os.path.exists("./results"):
        os.mkdir("./results")
    np.save(f"./results/{file_name}", total_rewards)
    return total_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Ant-v4")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=923, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timestep", default=int(25e3), type=int)  # Time steps initial random policy is used
    parser.add_argument("--test_freq", default=5000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timestep", default=int(1e6), type=int)  # Max time steps to run environment

    parser.add_argument("--agent", default="WGASAC")
    parser.add_argument("--nf_archi", default="SD_NVP")
    parser.add_argument("--lr_policy", default=0.0003, type=float)
    parser.add_argument("--lr_q_func", default=0.0003, type=float)
    parser.add_argument("--lr_disc", default=0.0003, type=float)
    parser.add_argument("--lr_alpha", default=0.0003, type=float)
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--balance", default=1., type=float)

    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    env = gym.make(args.env)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {"state_size": state_dim,
              "action_size": action_dim,
              "nf_archi": args.nf_archi,
              "batch_size": args.batch_size,
              "tau": args.tau,
              "lr_policy": args.lr_policy,
              "lr_q_func": args.lr_q_func,
              "lr_alpha": args.lr_alpha,
              "device": DEVICE}

    if "WGASAC" == args.agent:
        print("WGASAC Initialize")
        kwargs["lr_disc"] = args.lr_disc
        kwargs["balance"] = args.balance
        agent = WGASACAgent(**kwargs)
    elif "SAC_NF" == args.agent:
        print("SAC_NF Initialize")
        agent = SAC_NF.SACAgent(**kwargs)

    state = torch.FloatTensor(env.reset()[0])
    state = torch.unsqueeze(state, dim=0)
    score = 0.0

    test_score_list = []

    for step in range(args.max_timestep):
        if step < args.start_timestep:
            action = np.random.uniform(-1.0, 1.0, action_dim)
            state_next, reward, terminated, truncated, info = env.step(action * max_action)
            action = torch.FloatTensor(action.reshape(1, -1)).to(DEVICE)
        else:
            action = agent.decide_action(state)
            if torch.any(torch.isnan(action)):
                print("There is Nan in Action Vector")
                break
            state_next, reward, terminated, truncated, info = env.step(action.squeeze().to("cpu").numpy() * max_action)

        state_next = torch.unsqueeze(torch.FloatTensor(state_next), 0)

        score += reward
        reward = torch.FloatTensor([reward])

        agent.memorize(state, action, reward, state_next, terminated)

        state = state_next

        if (step + 1) >= args.start_timestep:
            agent.update_network()
            if (step + 1) % args.test_freq == 0:
                score_file_name = f"{args.env}_{args.agent}-{args.nf_archi}_{step + 1:09d}"
                agent.save_model(file_name=score_file_name)
                test_score = agent_test(agent, args.env, score_file_name, args.seed)
                test_score_list.append(test_score)
                print(f"#{step + 1} Result of agent test - score: {np.mean(test_score):.4f}")

        if terminated or truncated:
            state = torch.unsqueeze(torch.FloatTensor(env.reset()[0]), dim=0)
            score = 0.0

    df = pd.DataFrame(test_score_list)
    df["Avg Score"] = df.mean(axis=1)
    df["Step"] = np.arange(0, df.shape[0]) * args.test_freq + args.start_timestep
    score_df_file_name = f"Test_Score_{args.agent}-{args.nf_archi}_{args.env}.xlsx"

    df.to_excel(score_df_file_name, header=True, index=False)
    env.close()