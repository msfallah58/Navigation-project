import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger
import shutil


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    dataset = os.listdir(data_dir + '/training_and_validation')
    random.shuffle(dataset)
    
    dataset_size = len(dataset)
    training_size = dataset_size * 0.8
    validation_size = dataset_size * 0.2
    
    for i, tfrecord in enumerate(dataset):
        if i<training_size:
            shutil.move(data_dir + '/training_and_validation/' + tfrecord, data_dir + '/train')
            
        else:
            shutil.move(data_dir + '/training_and_validation/' + tfrecord, data_dir + '/val')
            
         

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)