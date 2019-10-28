import os
import json
import math
from tqdm import tqdm

import torch
import torchnet as tnt

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils

def main(opt, augment_stn):
    # load model
    model = torch.load(opt['model.model_path'])
    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')
    if model_opt['run_stn']:
        print("Loading STN here")
        stn_model = torch.load(opt['model.model_path'].replace('.pt', '_stn.pt'))
        stn_model.eval()
    else:
        stn_model = None

    # Augment overrides it anyway
    if not augment_stn:
        stn_model = None

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(opt['seed'])
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(opt['seed'])

    data = data_utils.load(data_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()
        if stn_model is not None:
            stn_model.cuda()

    meters = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }

    model_utils.evaluate(model, stn_model, data['test'], meters, desc="test")

    for field,meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))
