"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
from torch import nn
import numpy as np

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from few_shot.stn import STNv0, STNv1
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=20, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--suffix', default='', type=str)

# STN params
parser.add_argument('--stn', default=0, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--stn_reg_coeff', default=10, type=float)
parser.add_argument('--stn_hid_dim', default=32, type=int)
parser.add_argument('--stnlr', default=1e-3, type=float)
parser.add_argument('--stnweightdecay', default=1e-5, type=float)

# STNv1 params
parser.add_argument('--scalediff', default=0.1, type=float)
parser.add_argument('--theta', default=180, type=float)
parser.add_argument('--t', default=0.1, type=float)
parser.add_argument('--fliphoriz', default=0.5, type=float)

# Add more params
parser.add_argument('--targetonly', default=0, type=int)

args = parser.parse_args()
args.theta = args.theta / 180.0 * np.pi

### Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

evaluation_episodes = 1000
episodes_per_epoch = 100

if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 80
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 40
else:
    raise(ValueError, 'Unsupported dataset')

param_str = '{}_nt={}_kt={}_qt={}_'.format(args.dataset, args.n_train, args.k_train, args.q_train) + \
            'nv={}_kv={}_qv={}'.format(args.n_test, args.k_test, args.q_test) + \
            '_{}'.format(args.seed)
if args.stn:
    param_str += '_stn_{}'.format(args.stn_reg_coeff)

if args.suffix != '':
    param_str += '_{}'.format(args.suffix)
print(param_str)

###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)

#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)
model = nn.DataParallel(model)

stnmodel = None
stnoptim = None
print(args)
if args.stn:
    if args.dataset == 'miniImageNet':
        if args.stn == 1:
            stnmodel = STNv0((3, 84, 84), args)
        elif args.stn == 2:
            stnmodel = STNv1((3, 84, 84), args)
            args.stn_reg_coeff = 0
        else:
            raise NotImplementedError
    elif args.dataset == 'omniglot':
        if args.stn == 1:
            stnmodel = STNv0((1, 28, 28), args)
        elif args.stn == 2:
            stnmodel = STNv1((1, 28, 28), args)
            args.stn_reg_coeff = 0
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    stnmodel.to(device, dtype=torch.double)
    stnmodel = nn.DataParallel(stnmodel)
    # Get optimizer
    stnoptim = Adam(stnmodel.parameters(), lr=args.stnlr,
            weight_decay=args.stnweightdecay)

############
# Training #
############
print('Training Prototypical network on {}...'.format(args.dataset))
if args.stn:
    print('Training with STN')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()

def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr

callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath='/serverdata/rohit/models/proto_nets/{}.pth'.format(param_str),
        monitor='val_{}-shot_{}-way_acc'.format(args.n_test, args.k_test)
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + '/logs/proto_nets/{}.csv'.format(param_str)),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    stnmodel=stnmodel,
    stnoptim=stnoptim,
    args=args,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance},
)
