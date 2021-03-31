import argparse
from model_wrappers.PPGN_wrapper import PPGNWrapper

models = {
    'PPGN' : PPGNWrapper
}

datasets = [
    'MUTAG'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PPGN',
                        help=f'which model to use (default: PPGN) out of {models.keys()}')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--config', type=dict, default={},
                        help='config params for model: e.g. \'{lr: 0.001, epochs: 100}\'')
    
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
    
if __name__ == '__main__':
    main()