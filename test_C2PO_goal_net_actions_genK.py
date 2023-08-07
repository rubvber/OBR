import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from C2PO import C2PO
import pickle
import sys

sys.path.append('../AttentionExperiments/src/')
from active_3dsprites import active_3dsprites_dataset

pl.seed_everything(1234)

model = C2PO().load_from_checkpoint('/home/rubber/C2PO/snellius_checkpoints/version_65/checkpoints/last.ckpt')
model.maxF=12
model.val_predict=0
model.interactive=True
model.action_frames=(4, 5, 6, 7, 8, 9, 10, 11)
model.lambda_a = 0.1
model.planning_horizon=3
model.action_generation_type = 'goal'

for K in range(2,6):
    active_dsprites_test = active_3dsprites_dataset({
            'N': 512,
            'interactive': True,            
            'action_frames': [],            
            'gpus': 9,
            'with_rotation': False,
            'scale_min': 1.5,
            'scale_max': 1.50001,
            'rand_seed0': 50000+10000+1234+4343,
            'bgcolor': 127,
        })

    model.K = K+1
    test_loader = DataLoader(active_dsprites_test, batch_size=8 if K<5 else 4, num_workers=4, persistent_workers=False, drop_last=True)
    trainer = pl.Trainer(devices=(9,), accelerator="gpu", strategy='ddp', precision=16)
    test_data = trainer.predict(model, dataloaders=test_loader)

    with open('/home/rubber/C2PO/results/C2PO_snellius-v65_goal-net-actions_lambda-0.1_unit-prec_af_4_5_6_7_8_9_10_11-num_sprites_{}.pkl'.format(K), 'wb') as fh:
        pickle.dump(test_data, fh)


