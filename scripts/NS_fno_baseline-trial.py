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

import sys
sys.path.append('../')
from utilities import *

sys.path.append('../models')
from models.fno_2d import *

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

wandb.init(
    entity='research-pino_ifno',
    project='re5000',
    name='baseline'
)
# Main
ntrain = 90
ntest = 10

modes = 20
width = 128

in_dim = 1
out_dim = 1

batch_size = 50
epochs = 50
learning_rate = 0.0005
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
data = np.load('/home/robert/data/2D_NS_Re5000.npy?download=1')
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
model = FNO(n_modes=(64, 64), hidden_channels=width, in_channels=1, out_channels=1)
#model = FNO2d(n_modes_height=modes, n_modes_width=modes, hidden_channels=width, in_channels=1, out_channels=1)
model.to(device)
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

lploss = LpLoss(size_average=False)
h1loss = HsLoss(k=1, group=False, size_average=False)
h2loss = HsLoss(k=2, group=False, size_average=False)
myloss = HsLoss(k=loss_k, group=loss_group, size_average=False)

train_loss=myloss
eval_losses={'h1': h1loss, 'l2': lploss, 'h2': h2loss}
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


trainer = Trainer(model, n_epochs=20,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=True,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True, dataset_name='Re5000')


trainer.train(train_loader, test_loader,
              output_encoder,
              model,
              optimizer,
              scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)

wandb.finish()

