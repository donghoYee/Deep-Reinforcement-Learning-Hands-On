import gym
import ptan
import ptan.ignite as ptan_ignite
from datetime import datetime, timedelta
import argparse
import collections
import warnings
from typing import List, Tuple

import torch
import torch.optim as optim
import torch.multiprocessing as mp # multiprocessing

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from lib import dqn_model, common

BATCH_MUL = 4
NAME= "03_parallel"

EpisodeEnded = collections.namedtuple(
    'EpisodeEnded', field_names=("reward", "steps", "epsilon")) # memory to put information about episode in


def play_func(params, net, cuda, exp_queue):
    env = gym.make(params.env_name) # make it every single time?
    env = ptan.common.wrappers.wrap_dqn(env)
    device = torch.device("cuda" if cuda else "cpu")
    
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma)
    
    for frame_idx, exp in enumerate(exp_source): # infinite dataset
        epsilon_tracker.frame(frame_idx/BATCH_MUL) # why divide?
        exp_queue.put(exp) # put experience into que!!
        for reward, steps in exp_source.pop_rewards_steps(): # what is that?
            exp_queue.put(EpisodeEnded(reward, steps, selector.epsilon)) # this has element when episode has ended!
            


class BatchGenerator: # wrapps ExperienceSourceFirstLast
    def __init__(self, buffer: ptan.experience.ExperienceReplayBuffer, exp_queue: mp.Queue,
                 fps_handler: ptan_ignite.EpisodeFPSHandler,
                 initial: int, batch_size: int):
        self.buffer = buffer
        self.exp_queue = exp_queue
        self.fps_handler = fps_handler
        self.initial = initial
        self.batch_size = batch_size
        self._rewards_steps = []
        self.epsilon=None
        pass
    
    def pop_rewards_steps(self) -> List[Tuple[float, int]]: # give information about the episode's ending
        res = list(self._rewards_steps)
        self._rewards_steps.clear() # clear memory buffer
        return res
    
    def __iter__(self): # called by the enumerate method? -? should be used by the put method?
        while True:
            while exp_queue.qsize() > 0: # if there is data
                exp = exp_queue.get()
                if isinstance(exp, EpisodeEnded):
                    self._rewards_steps.append((exp.reward, exp.steps))
                    self.epsilon = exp.epsilon
                else:
                    self.buffer._add(exp)
                    self.fps_handler.step() # there is date
                    
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size * BATCH_MUL)
            
            

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    
    mp.set_start_method("spawn")
    params = common.HYPERPARAMS["pong"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    env = gym.make(params.env_name) # is this necessary?
    env = ptan.common.wrappers.wrap_dqn(env)
    
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    
    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.replay_size) # do not put in anything as source as it will be handled manually
    
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    
    # start subprocess and experience que
    exp_queue = mp.Queue(maxsize=BATCH_MUL * 4)
    
    play_proc1 = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue)) # put in shared arguments like that!!
    play_proc1.start()
    
    play_proc2 = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue)) # put in shared arguments like that!!
    play_proc2.start()
    
    
    fps_handler = ptan.ignite.EpisodeFPSHandler()
    
    batch_generator = BatchGenerator(buffer, exp_queue, fps_handler, params.replay_initial, params.batch_size)
    
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        
        loss_v.backward()
        optimizer.step()
        
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
            
        return {
            "loss": loss_v.item(),
            "epsilon": batch_generator.epsilon,
        }
    
    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(batch_generator, bound_avg_reward=17.0).attach(engine) # called every time episode is over
    
    fps_handler.attach(engine, manual_step=True) #use manual_step!!
    
    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=trainer.state.metrics.get('time_passed', 0))))

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics['time_passed']),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{params.run_name}-{NAME}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    
    ptan_ignite.PeriodicEvents().attach(engine)
    handler = tb_logger.OutputHandler(tag="train", metric_names=["avg_loss", "avg_fps"], output_transform=lambda a: a)
    
    tb.attach(engine, log_handler=handler, event_name=ptan.ignite.PeriodEvents.ITERS_100_COMPLETED)
    
    engine.run(batch_generator) # calling iter() checks the que
    play_proc1.kill()
    play_proc1.join()
    play_proc2.kill()
    play_proc2.join()