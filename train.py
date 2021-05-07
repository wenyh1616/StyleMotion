"""Train script.

Usage:
    train.py <hparams> <dataset>
"""
import os
import motion
import numpy as np
import datetime

from docopt import docopt
from torch.utils.data import DataLoader, Dataset
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig

if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    assert dataset in motion.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, motion.Datasets.keys()))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)
    dataset = motion.Datasets[dataset]
    
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
		
    print("log_dir:" + str(log_dir))
    data = dataset(hparams)
    x_channels, cond_channels = data.get_train_dataset().n_channels()

    # build graph

    if hparams.Infer.pre_trained == "":
        built = build(x_channels, cond_channels, hparams, True)
    else:
        built = build(x_channels, cond_channels, hparams, False)


    # build trainer
    trainer = Trainer(**built, data=data, log_dir=log_dir, hparams=hparams)
    if hparams.Infer.pre_trained == "":
        trainer.train()
    else:
        test_path = './data/xia_test/'

        trainer.test_pair(test_path + 'angry_01_000.bvh', test_path + 'childlike_01_000.bvh', eps_std=1, frames=1, counter=0)



