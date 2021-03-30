import argparse

models = {
    'PPGN' : None
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
                  \n{models.keys()}')
        return
    
    if args.dataset not in datasets:
        print(f'The dataset you have provided ({args.dataset}) is not a valid model.\
                  \nPlease choose one of the following:\
                  \n{datasets}')
        return
    
    models[args.model].run(args.dataset, args.config)
    
if __name__ == '__main__':
    main()