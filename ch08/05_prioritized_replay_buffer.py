import gym
import ptan
import argparse
import random

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common, dqn_extra

NAME = "05_prio_replay"
PRIO_REPLAY_ALPHA = 0.6

def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"): # has batch weight!!
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    
    batch_weights_v = torch.tensor(batch_weights).to(device)
    
    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v).squeeze(-1)
    
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_s_vals = tgt_net(next_states_v).max(1)[0]
        next_s_vals[done_mask] = 0.0
        exp_s_a_vals = next_s_vals.detach() * gamma + rewards_v
        
    l = (state_action_vals - exp_s_a_vals) ** 2 # not summed
    losses_v = l * batch_weights_v
    return losses_v.mean(), (losses_v +1e-5).data.cpu().numpy() # why is the second arg needed -> to set priority


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong'] # get hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--faster", default=False, action="store_true", help="Use faster buffer")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params) # this automatically tracks epsilon?
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)
    
    if args.faster:
        BETA_START = 0.4
        BETA_FRAMES = 100000
        beta = BETA_START

        buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    else:
        buffer = dqn_extra.PrioReplayBuffer(exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
        
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    
    
    def process_batch(engine, batch_data):        
        batch, batch_indices, batch_weights = batch_data # got from buffer's sample
        optimizer.zero_grad()
        loss_v, sample_prios = calc_loss(batch, batch_weights, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss_v.backward() # loss_v is float, sample_prios is list before sum()
        optimizer.step()
        
        buffer.update_priorities(batch_indices, sample_prios)# very important!!
        epsilon_tracker.frame(engine.state.iteration)
        
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        
        if args.faster:
            v = BETA_START + engine.state.iteration * (1.0 - BETA_START) / BETA_FRAMES
            beta = min(1.0, v)
            return {
                "loss": loss_v.item(),
                "epsilon": selector.epsilon,
                "beta": beta # important to update beta!!
            }
        else:
            return {
                "loss": loss_v.item(),
                "epsilon": selector.epsilon,
                "beta": buffer.update_beta(engine.state.iteration) # important to update beta!!
            }
    
    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME) # exp_source is needed to check episode info
    if args.faster:
        engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size, beta))
    else:
        engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
