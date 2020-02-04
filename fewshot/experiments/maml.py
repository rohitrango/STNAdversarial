"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from few_shot.models import FewShotClassifier
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
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch-len', default=100, type=int)
parser.add_argument('--eval-batches', default=20, type=int)

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

args = parser.parse_args()

### Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    fc_layer_size = 64
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    fc_layer_size = 1600
    num_input_channels = 3
else:
    raise(ValueError('Unsupported dataset'))

param_str = '{}_order={}_n={}_k={}_metabatch={}_train_steps={}_val_steps={}_{}'.format(args.dataset, args.order, args.n, args.k, args.meta_batch_size, args.inner_train_steps, args.inner_val_steps, args.seed)
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
    batch_sampler=NShotTaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)


#########
# Model #
#########
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
    stnoptim = torch.optim.Adam(stnmodel.parameters(), lr=args.stnlr, weight_decay=args.stnweightdecay)

############
# Training #
############
print('Training MAML on {}...'.format(args.dataset))
meta_model = FewShotClassifier(num_input_channels, args.k, fc_layer_size).to(device, dtype=torch.double)
meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
loss_fn = nn.CrossEntropyLoss().to(device)


def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_


callbacks = [
    EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
    ),
    ModelCheckpoint(
        filepath=PATH + '/models/maml/{}.pth'.format(param_str),
        monitor='val_{}-shot_{}-way_acc'.format(args.n, args.k)
    ),
    ReduceLROnPlateau(patience=10, factor=0.5, monitor='val_loss'),
    CSVLogger(PATH + '/logs/maml/{}.csv'.format(param_str)),
]


fit(
    meta_model,
    meta_optimiser,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
    callbacks=callbacks,
    stnmodel=stnmodel,
    stnoptim=stnoptim,
    args=args,
    metrics=['categorical_accuracy'],
    fit_function=meta_gradient_step,
    fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                         'train': True,
                         'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                         'inner_lr': args.inner_lr},
)
