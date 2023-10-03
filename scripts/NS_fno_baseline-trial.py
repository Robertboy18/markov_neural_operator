import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from neuralop import Trainer
from neuralop.models import FNO
import wandb
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO, FNO2d
from neuralop import Trainer
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

import sys
sys.path.append('../')
from utilities import *

sys.path.append('../models')
from models.fno_2d import *

from timeit import default_timer
import scipy.io
import argparse  # Import the argparse library

torch.manual_seed(0)
np.random.seed(0)

wandb.login(key='0d28fab247b1d30084a6ea7af891401bb5d1c20e')

wandb.init(
    entity='research-pino_ifno',
    project='re5000',
    name='resolution-128-final-check-cyclicLRsweep'
)

# Create an ArgumentParser object
# Read the configuration
config_name = 'default'
pipe = ConfigPipeline([YamlConfig('./incremental.yaml', config_name='default', config_folder='../scripts/config'),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder='../scripts/config')
                      ])
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Make sure we only print information when needed
config.verbose = config.verbose

#Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

# Main
ntrain = 90
ntest = 10

width = 128

in_dim = 1
out_dim = 1

batch_size = 10
epochs = 50
learning_rate = 0.001
scheduler_step = 10
scheduler_gamma = 0.5

loss_k = 0 # H0 Sobolev loss = L2 loss
loss_group = True

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

sub = 1 # spatial subsample
S = 128

T_in = 100 # skip first 100 seconds of each trajectory to let trajectory reach attractor
T = 400 # seconds to extract from each trajectory in data
T_out = T_in + T
step = 1 # Seconds to learn solution operator

t1 = default_timer()
data = np.load('/ngc_workspace/jiawei/datasets/2D_NS_Re5000.npy?download=1')
#data = np.load('/ngc_workspace/jiawei/datasets/NS_Re5000_256')
#data = np.load('/home/robert/data/2D_NS_Re5000.npy?download=1')
data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]

train_a = data[:ntrain,T_in-1:T_out-1].reshape(ntrain*T, S, S)
train_u = data[:ntrain,T_in:T_out].reshape(ntrain*T, S, S)

test_a = data[-ntest:,T_in-1:T_out-1].reshape(ntest*T, S, S)
test_u = data[-ntest:,T_in:T_out].reshape(ntest*T, S, S)

assert (S == train_u.shape[2])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
output_encoder = None

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

# Model
#model = Net2d(in_dim, out_dim, S, modes, width).cuda()
model = FNO(n_modes=(128, 128), hidden_channels=width, in_channels=1, out_channels=1)
#model = FNO2d(n_modes_height=modes, n_modes_width=modes, hidden_channels=width, in_channels=1, out_channels=1)
model.to(device)
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,step_size_down=config.opt.step_size_down,base_lr=config.opt.base_lr,max_lr=config.opt.max_lr,step_size_up=config.opt.step_size_up,mode=config.opt.mode,last_epoch=-1,cycle_momentum=False)

lploss = LpLoss(size_average=False)
h1loss = HsLoss(k=1, group=False, size_average=False)
h2loss = HsLoss(k=2, group=False, size_average=False)
myloss = HsLoss(k=loss_k, group=loss_group, size_average=False)

#Log parameter count
n_params = count_params(model)

if config.verbose:
    print(f'\nn_params: {n_params}')
    sys.stdout.flush()

if config.wandb.log:
    to_log = {'n_params': n_params}
    if config.n_params_baseline is not None:
        to_log['n_params_baseline'] = config.n_params_baseline,
        to_log['compression_ratio'] = config.n_params_baseline/n_params,
        to_log['space_savings'] = 1 - (n_params/config.n_params_baseline)
    wandb.log(to_log)
    wandb.watch(model)

train_loss=myloss
eval_losses={'h1': h1loss, 'l2': lploss, 'h2': h2loss}
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()
trainer = Trainer(model, n_epochs=config.opt.n_epochs,
                  device=device,
                  mg_patching_levels=config.patching.levels,
                  mg_patching_padding=config.patching.padding,
                  mg_patching_stitching=config.patching.stitching,
                  wandb_log=config.wandb.log,
                  log_test_interval=config.wandb.log_test_interval,
                  log_output=config.wandb.log_output,
                  use_distributed=config.distributed.use_distributed,
                  verbose=config.verbose, incremental = config.incremental.incremental_grad.use, 
                  incremental_loss_gap=config.incremental.incremental_loss_gap.use, 
                  incremental_resolution=config.incremental.incremental_resolution.use, dataset_name="Re5000", save_interval=config.checkpoint.interval, model_save_dir=config.checkpoint.directory + config.checkpoint.name)


trainer.train(train_loader, test_loader,
              output_encoder,
              model,
              optimizer,
              scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)

if config.wandb.log:
    wandb.finish()
