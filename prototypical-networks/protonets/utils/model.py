from tqdm import tqdm
import torch

from protonets.utils import filter_opt
from protonets.models import get_model

def load(opt, key='model_name'):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt[key]

    del model_opt[key]
    return get_model(model_name, model_opt)


def evaluate(model, stn_model, data_loader, meters, desc=None):
    model.eval()
    if stn_model is not None:
        stn_model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        # modify the samples with STN if STN is present
        if stn_model is not None:
            sampled = stn_model(sample)[0]
            sample['xs'] = torch.cat([sample['xs'], sampled['xs']], 1)
        _, output = model.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
