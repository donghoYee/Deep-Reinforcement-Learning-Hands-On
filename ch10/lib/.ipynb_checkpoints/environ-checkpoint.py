import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2
    
class State:
    def __init__(self, bars_count, commision_perc, 
                 reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commision_perc, float)
        assert commision_perc >= 0.0
        assert isinstance(reset_on_close,bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commision_perc = commision_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        
    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count -1
        self.have_position=False
        self.open_price = 0.0
        self._prices = prices
        self._offset=offset
        
    @property #allows the call like .shape
    def shape(self):
        if self.volumes:
            return 4*self.bars_count +1 +1,
        else:
            return 3*self.bars_count+1+1,
    
    def encode(self): # current state into numpy array
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            ofs = self._offset + bar_idx
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position: # did not sell
            res[shift] = 0.0
        else: # sold
            res[shift] = self._cur_close() / self.open_price - 1.0 # positive when close price
            # is bigger than open price
        return res
    
    def _cur_close(self):
        # return real close price for the current bar
        open_price = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset] # relative price
        return open_price * (1.0 + rel_close)
    
    def step(self, action):
        """ Perform one step in price -> returns reward, done"""
        assert isinstance(action, Actions)
        reward = 0.0
        done=False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commision_perc
        elif action == Actions.Close and self.have_position: # close the share
            reward -= self.commision_perc
            done |= self.reset_on_close # end if reset_on_close is true
            if self.reward_on_close:
                reward += 100.0*(close/self.open_price - 1.0) # in percentage
            self.have_position = False
            self.open_price = 0.0
        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0] - 1 # when it reaches the end
    
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)
            
        return reward, done
    
class State1D(State): # inherit state!
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)
        
    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        start = self._offset - (self.bars_count -1)
        stop - self._offset+1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = self._cur_close() / self.open_price - 1.0
        return res
        
    
    
class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("StocksEnv-v0")
    
    def __init__(self, 
                 prices, # DATA DIRECTORY
                 bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, # tax?
                 reset_on_close=True, 
                 state_1d=False, 
                 random_ofs_on_reset=True,
                 reward_on_close=False, 
                 volumes=False
                ):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes) # this gives the obs?
        else:
            self._state=State(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()
        
    def reset(self):
        self._instrument = self.np_random.choice(
            list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choce(prices.high.shape[0]-bars*10)+bars
        else:
            offset=bars
        self._state.reset(prices, offset)
        return self._state.encode()
        
    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action) # _state does all the action for us
        obs = self._state.encode() # returns observation
        info = {
            "instrument":self._instrument,
            "offset": self._state._offset
        }
        return obs, reward, done, info
    
    def render(self, mode="human", close=False):
        pass
    def close(self):
        pass
    
    def seed(self, seed=None): # give same seed for each env concurrently
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1+1)%2**31
        return [seed1, seed2]
    
        
    @classmethod # initialize class from dir
    def from_dir(cls, data_dir, **kwargs):
        prices = {
            file: data.load_relative(file) for file in data.price_files(data_dir)
        }
        return StocksEnv(prices, **kwargs) # now that calls init!
        
        
