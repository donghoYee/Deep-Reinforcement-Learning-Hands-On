import gym
import ptan

import ptan.ignite as ptan_ignite
from datetime import datetime, timedelta

import argparse
import random
import warnings

import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from lib import dqn_model, common

NAME = "02_n_envs"

def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size: int, steps: int):
    buffer.populate(initial)
    while True:
        buffer.populate(steps) # performs several steps instead of just 1
        yield buffer.sample(batch_size)
        
        
if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning) #get rid of missing metrics warning
    
    params = common.HYPERPARAMS["pong"]
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--envs", type=int, default=3, help="Amount of environments to run in parallel")
    
    args = parser.parse_args()
    device  = torch.device("cuda" if args.cuda else "cpu")
    
    envs = [] # make a list of environments!
    for _ in range(args.envs):
        env = gym.make(params.env_name)
        env = ptan.common.wrappers.wrap_dqn(env)
        envs.append(env)
        
    params.batch_size *= args.envs # proportionally scale batch-size!!
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device) #data will come as a batch
    
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    
    agent = ptan.agent.DQNAgent(net, selector, device=device) # this doesn't need to know about the env size
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=params.gamma) # put a list of envs for parallezation
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration * args.envs) # scale the epsilon change speed accordingly!!
        
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        
        return {
            "loss":loss_v.item(),
            "epsilon":selector.epsilon,
        }
    
    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(exp_source, bound_avg_reward=17.0).attach(engine)
    ptan_ignite.EpisodeFPSHandler(fps_mul=args.envs).attach(engine) #mul fps
    
    
    @engine.on(ptan.ignite.EpisodeEvents.EPISODE_COMPLETED) # episode event
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('fps', 0) ,# default is 0
            timedelta(seconds=trainer.state.metrics.get('time_passed'))))
        
    
    @engine.on(ptan.ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics["time_passed"]),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate=True
        
    
    
    logdir=f"runs/{datetime.now().isoformat(timespec='minutes')}-{params.run_name}-{NAME}={args.envs}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")
    
    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    
    ptan_ignite.PeriodicEvents().attach(engine) # write every 100 iters
    handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss', 'avg_fps'], output_transform=lambda a:a)
    
    tb.attach(engine, log_handler=handler, event_name=ptan_ignite.PeriodEvents.ITERS_100_COMPLETED)
    
    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size, args.envs)) # put env count also!!
