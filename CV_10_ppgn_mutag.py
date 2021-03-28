from model_utils import CV_10
from models.ppgn import PPGN
from dataset_loaders import load_dataset

ppgn = PPGN()
mutag = load_dataset("MUTAG", True)
CV_10(ppgn, mutag, 10)
