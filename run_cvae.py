import argparse
import torch
import tqdm
from pprint import pprint

from models.cvae import CVAE
from train import train
from utils import test_utils as tut
from utils import prob_utils as ut

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z', type=int, default=1, help="Number of latent dimensions")
parser.add_argument('--iter_max', type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run', type=int, default=0, help="Run ID")
parser.add_argument('--train', type=int, default=1, help="Flag for training")
parser.add_argument('--version', type=str, default='v2', help="Version of model")

args = parser.parse_args()

layout = [
    ('model={:s}', 'cvae'),
    ('z={:02d}', args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader = ut.get_data_loaders()
cvae = CVAE(z_dim=args.z, name=model_name, version=args.version).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)

    train(model=cvae,
          train_loader=train_loader,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)

    tut.evaluate_lower_bound(cvae, valid_loader)

else:
    tut.load_model_by_name(cvae, global_step=args.iter_max, device=device)
    tut.evaluate_lower_bound(cvae, valid_loader)