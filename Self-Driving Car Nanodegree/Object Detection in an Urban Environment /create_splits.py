import argparse
import glob
import os
import random
import shutil
import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    dataset = os.listdir(data_dir + 'training_and_validation')
    np.random.shuffle(dataset)
    train_set, val_set = np.split(dataset, [int(0.8 * len(dataset))])
    
    for i, data in enumerate(dataset):
        if i < int(0.75 * len(dataset)):
            shutil.move(data_dir+'/training_and_validation/'+ data, data_dir + '/train')
        else:
            shutil.move(data_dir + '/training_and_validation/'+ data, data_dir + '/val')
            
    print('The dataset was splitted as follows: Training: {} | Validation {}'.format(int(0.8 * len(dataset)), int(0.2 * len(dataset))))
    return
               
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()
    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)