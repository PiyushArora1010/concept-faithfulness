import argparse
from module.arguments import train_args
from tasks.train_engine import TrainEngine

if __name__ == '__main__':
    args = train_args()
    engine = TrainEngine(args)
    # breakpoint()
    engine._checking_reward_function([0,1,2])