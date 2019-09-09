import argparse
import os
import sys
from os.path import join, dirname, abspath

import fasttext

cwd = dirname(abspath(__file__))
sys.path.append(dirname(dirname(cwd)))

parser = argparse.ArgumentParser("train.py")
parser.add_argument("--train", help="train data path", required=True)
parser.add_argument("-s", "--serialization-dir", help="directory in which to save the model and its logs",
                    required=True)
args = parser.parse_args()

train_path = os.path.abspath(join(cwd, args.train))
serialization_dir = os.path.abspath(join(cwd, args.serialization_dir))

model = fasttext.train_supervised(input=train_path, epoch=30, lr=0.1, wordNgrams=2, dim=50)
model.save_model('{}/model.bin'.format(serialization_dir))
print("Done!!!")
