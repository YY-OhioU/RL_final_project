import argparse
import random
import time
from collections import deque
from pathlib import Path

import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch
import torch.optim as optim
import logging
import pickle
from policy import Policy

LOG_FILE = 'log.txt'
SAVE_MODEL_PATH = 'policy_trained.pkl'
RECORD_FILE = 'statics_{id}.pkl'


def get_logger(isfile):
    lg = logging.getLogger('PG')
    lg.setLevel(logging.DEBUG)
    if isfile:
        handler = logging.FileHandler(LOG_FILE)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    lg.addHandler(handler)
    return lg


def save_statics(obj):
    run_id = 0
    pkl_file = Path(RECORD_FILE.format(id=run_id))
    while pkl_file.exists():
        run_id += 1
        pkl_file = Path(RECORD_FILE.format(id=run_id))
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    episode_reward_hist = deque(maxlen=print_every)
    reward_average_window = deque(maxlen=5)
    scores = []
    epi_reward_hist_total = []
    max_reward = -100
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]
        action_record = {
            0: 0,
            1: 0
        }
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            action_record[action] += 1
            saved_log_probs.append(log_prob)
            state, reward, done, _, info = env.step(action)
            rewards.append(reward)
            if done:
                # print(t)
                break
        # scores_deque.append(sum(rewards))
        episode_reward = np.array(rewards).sum()
        scores.append(info['score'])
        episode_reward_hist.append(episode_reward)
        reward_average_window.append(episode_reward)
        epi_reward_hist_total.append(episode_reward)

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])

            ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        avg_performance = np.mean(reward_average_window)
        if i_episode > 3 and avg_performance > max_reward:
            torch.save(policy, SAVE_MODEL_PATH)
            max_reward = avg_performance

        if i_episode % print_every == 0:
            logger.info('Episode {}\tAverage reward: {:.5f}. flap: {} hold:{}'.format(i_episode,
                                                                                      sum(episode_reward_hist) / print_every,
                                                                                      action_record[1],
                                                                                      action_record[0]))

    return scores, epi_reward_hist_total


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        state = state[0]
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            # action = random.choice([0, 1])
            new_state, reward, done, _, info = env.step(action)
            time.sleep(1 / 30)
            env.render()
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
        logger.info(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Policy Gradient Training')
    parser.add_argument('--train', action='store_true', help='Train the model or not')
    parser.add_argument('--noeval', action='store_true', help='Skip evaluation')
    parser.add_argument('--pretrained', help='path to pretrained model')
    parser.add_argument('--log', action='store_true', help='log message to file')
    args = parser.parse_args()
    logger = get_logger(args.log)

    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(device)
    env = gymnasium.make("FlappyBird-v0")

    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    pixelcopter_hyperparameters = {"h_size": [60, 300, 600], "n_training_episodes": 30000, "n_evaluation_episodes": 10,
                                   "max_t": 2000, "gamma": 0.99, "lr": 1e-4, "state_space": s_size,
                                   "action_space": a_size, }

    # Create policy and place it to the device
    if args.pretrained:
        pixelcopter_policy = torch.load(args.pretrained)
        logger.info("loaded pretrained model")
    else:
        pixelcopter_policy = Policy(pixelcopter_hyperparameters["state_space"],
                                    pixelcopter_hyperparameters["action_space"],
                                    pixelcopter_hyperparameters["h_size"], device=device).to(device)
    pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

    if args.train:
        train_records = reinforce(pixelcopter_policy, pixelcopter_optimizer,
                                  pixelcopter_hyperparameters["n_training_episodes"],
                                  pixelcopter_hyperparameters["max_t"],
                                  pixelcopter_hyperparameters["gamma"], 500)
        save_statics(train_records)

    if not args.noeval:
        evaluate_agent(env, 2000, 5,
                       pixelcopter_policy)
