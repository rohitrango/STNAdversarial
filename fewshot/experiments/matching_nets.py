"""
Reproduce Matching Network results of Vinyals et al
"""
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot
from few_shot.matching import matching_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from few_shot.stn import STNv0, STNv1
from torch import nn
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
parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't')  # Quick hack to extract boolean
parser.add_argument('--distance', default='cosine')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--lstm-layers', default=1, type=int)
parser.add_argument('--unrolling-steps', default=2, type=int)


parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--suffix', default='', type=str)

# STN params
parser.add_argument('--stn', default=0, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--stn_reg_coeff', default=10, type=float)
parser.add_argument('--stn_hid_dim', default=32, type=int)
parser.add_argument('--stnlr', default=3e-4, type=float)
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

print(args)

evaluation_episodes = 1000
episodes_per_epoch = 100

if args.dataset == 'omniglot':
    n_epochs = 100
    dataset_class = OmniglotDataset
    num_input_channels = 1
    lstm_input_size = 64
elif args.dataset == 'miniImageNet':
    n_epochs = 200
    dataset_class = MiniImageNet
    num_input_channels = 3
    lstm_input_size = 1600
else:
    raise(ValueError, 'Unsupported dataset')

param_str = '{}_n={}_k={}_q={}_nv={}_kv={}_qv={}_dist={}_fce={}'.format(args.dataset, args.n_train, args.k_train, args.q_train, args.n_test, args.k_test, args.q_test, args.distance, args.fce) \
                + '_{}'.format(args.seed)
if args.stn:
    param_str += '_stn_{}'.format(args.stn_reg_coeff)

if args.suffix != '':
    param_str += '_{}'.format(args.suffix)
print(param_str)


#########
# Model #
#########
from few_shot.models import MatchingNetwork
model = MatchingNetwork(args.n_train, args.k_train, args.q_train, args.fce, num_input_channels,
                        lstm_layers=args.lstm_layers,
                        lstm_input_size=lstm_input_size,
                        unrolling_steps=args.unrolling_steps,
                        device=device)
model.to(device, dtype=torch.double)

stnmodel = None
stnoptim = None
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

############
# Training #
############
print('Training Matching Network on {}...'.format(args.dataset))
if args.stn:
    print('Training with STN')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


callbacks = [
    EvaluateFewShot(
        eval_fn=matching_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        fce=args.fce,
        args=args,
        stnmodel=None,
        stnoptim=None,
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + '/models/matching_nets/{}.pth'.format(param_str),
        monitor='val_{}-shot_{}-way_acc'.format(args.n_test, args.k_test),
        # monitor=f'val_loss',
    ),
    ReduceLROnPlateau(patience=20, factor=0.5, monitor='val_{}-shot_{}-way_acc'.format(args.n_test, args.k_test)),
    CSVLogger(PATH + '/logs/matching_nets/{}.csv'.format(param_str)),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    stnmodel=stnmodel,
    stnoptim=stnoptim,
    args=args,
    metrics=['categorical_accuracy'],
    fit_function=matching_net_episode,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'fce': args.fce, 'distance': args.distance}
)
