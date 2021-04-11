import argparse
import re
from model_wrappers import *
import time
import os

models = {
    'PPGN' : PPGNWrapper,
    'GIN' : GINWrapper,
    'DGCNN' : DGCNNWrapper,
    'DiffPool' : DiffPoolWrapper,
    'PPGNNew' : PPGNNewDataFormatWrapper
}

datasets = [
    'MUTAG', 
    'QM9',
    'COLLAB',
    'NCI1',
    'NCI109',
    'IMDB-BINARY',
    'IMDB-MULTI',
    'PROTEINS',
    'PTC_FM',
    'AIDS',
    'BZR_MD'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PPGN',
                        help=f'which model to use (default: PPGN) out of {list(models.keys())}')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--config', type=dict_load, default={},
                        help='config params for model in double quotes: e.g. "{lr: 0.001, epochs: 100}"')
    
    args = parser.parse_args()
    
    if args.model not in models:
        print(f'The model you have provided ({args.model}) is not a valid model.\
                  \nPlease choose one of the following:\
                  \n{list(models.keys())}')
        return
    
    if args.dataset not in datasets:
        print(f'The dataset you have provided ({args.dataset}) is not a valid model.\
                  \nPlease choose one of the following:\
                  \n{datasets}')
        return
    
    model_wrapper = models[args.model](args.dataset, args.config)
    accuracy = model_wrapper.run()
    print('\n\nRUN COMPLETED')
    print(f'Accuracy of {args.model} on {args.dataset} is: {accuracy}')
    if not os.path.exists('log/'):
        os.mkdir('log')
    with open(f'log/{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
        f.write(f'model: {args.model}\ndataset: {args.dataset}\nconfig: {args.config}\naccuracy: {accuracy}')

# Convert string to dict    
def dict_load(s):
    s = s[1:-1]
    return dict([re.split('\W*:\W*', pair) for pair in re.split('\W*,\W*', s)])
    
if __name__ == '__main__':
    main()