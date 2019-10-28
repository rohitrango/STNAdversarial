import os
import json
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

from protonets.engine import Engine, STNEngine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils

def main(opt):
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(opt['seed'])
    if opt['data.cuda']:
        torch.cuda.manual_seed(opt['seed'])

    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    model = model_utils.load(opt)
    # If concat is true, then dont dropout anything
    if opt['train.concat_stn']:
        opt['model.stn_dropout'] = 0

    stn_model = model_utils.load(opt, 'stn_model')
    aux_loss_fn = model_utils.load(opt, 'stn_loss')

    if opt['data.cuda']:
        model.cuda()
        stn_model.cuda()

    if opt['run_stn']:
        engine = STNEngine()
    else:
        engine = Engine()


    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }

    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=opt['train.gamma'])
        if 'stn_optimizer' in state.keys():
            state['stn_scheduler'] = lr_scheduler.StepLR(state['stn_optimizer'], opt['train.decay_every'], gamma=opt['train.gamma'])


    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()
        if 'stn_scheduler' in state.keys():
            state['stn_scheduler'].step()

    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                None,
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
                if 'stn_model' in state.keys():
                    state['stn_model'].cpu()
                    torch.save(state['stn_model'], os.path.join(opt['log.exp_dir'], 'best_model_stn.pt'))
                if opt['data.cuda']:
                    state['model'].cuda()
                    if 'stn_model' in state.keys():
                        state['stn_model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()
            if 'stn_model' in state.keys():
                state['stn_model'].cpu()
                torch.save(state['stn_model'], os.path.join(opt['log.exp_dir'], 'best_model_stn.pt'))

            torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
            if opt['data.cuda']:
                state['model'].cuda()
                if 'stn_model' in state.keys():
                    state['stn_model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, { })

    engine.train(
        model = model,
        stn_model = stn_model,
        loader = train_loader,
        aux_loss_fn = aux_loss_fn,
        stn_loss_params = None,
        concat_stn = opt['train.concat_stn'],
        optim_method = getattr(optim, opt['train.optim_method']),
        optim_config = { 'lr': opt['train.learning_rate'],
                         'weight_decay': opt['train.weight_decay'] },
        stn_optim_method = getattr(optim, opt['train.stn_optim_method']),
        stn_optim_config = {'lr': opt['train.stn_learning_rate'],
                          'weight_decay': opt['train.stn_weight_decay']},
        max_epoch = opt['train.epochs']
    )
