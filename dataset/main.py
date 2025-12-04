import json
import argparse
import time

from name import *
import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=B4EAddress, help='Dataset used')
parser.add_argument('--trainsz', type=float, default=0.01, help='train size')
parser.add_argument('--testsz', type=float, default=0.66, help='test size')
parser.add_argument('--seed', type=int, default=111, help='seed')
args = parser.parse_args()

print("Model info:")
print(json.dumps(args.__dict__, indent='\t'))

data = args.data
trainsz = args.trainsz
testsz = args.testsz
seed = args.seed

train_test_split.set_seed(seed)


print("Generating data...")
s = time.time()
y = train_test_split.load_data(data)
train_test_split.split_train_val_test(data, y, trainsz, testsz)
e = time.time()
print("Generating successfully! Time cost: {}".format(e - s))