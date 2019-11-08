from tqdm import tqdm
import torch
from protonets.models.losses import kl_divergence

class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = { }
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False,
            # STN specific params
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['model'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)

'''
Define STN engine for handling STN loss and our normal loss
'''
class STNEngine(Engine):

    """this STN engine basically has the normal loop plus the STN"""

    def __init__(self):
        """TODO: to be defined. """
        Engine.__init__(self)

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False,

            # STN specific params
            'stn_model': kwargs['stn_model'],
            'stn_optim_method': kwargs['stn_optim_method'],
            'stn_optim_config': kwargs['stn_optim_config'],
            'concat_stn': kwargs['concat_stn'],
            'kl_div_coeff': kwargs['kl_div_coeff'],

            # Extra parameters
            'aux_loss_fn': kwargs['aux_loss_fn'],
            'stn_loss_params': kwargs['stn_loss_params'],
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])
        state['stn_optimizer'] = state['stn_optim_method'](state['stn_model'].parameters(), **state['stn_optim_config'])
        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            # Go to training mode
            state['model'].train()
            state['stn_model'].train()

            self.hooks['on_start_epoch'](state)
            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                # Zero grad both the STN and model
                state['model'].zero_grad()
                state['stn_model'].zero_grad()

                # Sample and pass through STN first
                state['sample'] = sample
                self.hooks['on_sample'](state)
                state_sampled, thetas, info = state['stn_model'](state['sample'])
                # Concat them if asked to
                if state['concat_stn']:
                    for key in ['xs']:
                        state_sampled[key] = torch.cat([state_sampled[key], state['sample'][key]], 1)

                # get loss from model
                loss, _ = state['model'].loss(state_sampled)
                # This part is for STN VAE loss
                loss = -loss + state['aux_loss_fn'](thetas, state['stn_loss_params'])
                if 'mean' in info.keys():
                    #print(state['kl_div_coeff']*kl_divergence(info))
                    loss += state['kl_div_coeff']*kl_divergence(info)
                loss.backward()

                #print(thetas[0])
                state['stn_optimizer'].step()
                # detach the modules
                for k in ['xs', 'xq']:
                    state_sampled[k] = state_sampled[k].detach()

                if state['concat_stn']:
                    for key in ['xs']:
                        state_sampled[key] = torch.cat([state_sampled[key], state['sample'][key]], 1)

                # Run the actual model
                state['model'].zero_grad()
                loss, state['output'] = state['model'].loss(state_sampled)
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)
                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)

