import ptan
import pathlib
import argparse
import gym.wrappers
import numpy as np

import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.contrib.handlers import tensorboard_logger as tb_logger


from lib import environ, data, models, common, validation

SAVES_DIR = pathlib.Path("saves")
STOCKS = "data/YNDX_160101_161231.csv"
VAL_STOCKS = "data/YNDX_150101_151231.csv" # default data


BATCH_SIZE = 32
BARS_COUNT = 10

EPS_START = 10
EPS_FINAL = 0.1
EPS_STPES = 1000000

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="Enable cuda", default=False,action="store_true")
    parser.add_argument("--data", default=STOCKS, help=f"Stocks file of dir, default={STOCKS}")
    parser.add_argument("--year", type=int, help="Year to train on, overrides --data")
    parser.add_argument("--val", default=VAL_STOCKS, help="Validation data, default="+VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    saves_path = SAVES_DIR / f"simple-{args.run}"
    saves_path.mkdir(parents=True, exist_ok = True) # make directories like this!
    
    data_path = pathlib.Path(args.data)
    val_path = pathlib.Path(args.val)
    
    if args.year is not None or data_path.is_file():
        if args.year is not None:
            stock_data = data.load_year_data(args.year) # load yearly data using lib/data
        else:
            stock_data = {"YNDX": data.load_relative(data_path)}
        
        env = environ.StocksEnv(stock_data, bars_count = BARS_COUNT) # make environment like this
        
        
    
    
    