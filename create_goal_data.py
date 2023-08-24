from active_dsprites import active_dsprites
from C2PO import C2PO
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.distributed import all_gather
import torch, pickle, sys

import sys

sys.path.append('../AttentionExperiments/src/')
from active_3dsprites import active_3dsprites_dataset


model = C2PO.load_from_checkpoint('/home/rubber/C2PO/snellius_checkpoints/version_82/checkpoints/last.ckpt', init_percept_net_path=None)                            
model.return_images = False 
model.init_tau=0.8
model.val_predict=0

gpus = (0,1,2,3,4,5,6,7)

for i in range(1):
    trainer = pl.Trainer(devices=gpus, accelerator="gpu", strategy='ddp', precision=16)
    active_3dsprites_train = active_3dsprites_dataset({
                'N': 50000,            
                'action_frames': [2,4,6,8],            
                'episode_length': 12,             
                'gpus': gpus[trainer.global_rank],               
                'with_rotation': False,
                'scale_min': 1.5,
                'scale_max': 1.50001,
                'rand_seed0': 1234+i*10000,
                'bgcolor': 127,
                'rule_goal': 'IfHalfTorus',
                'v_sd': 0,            
                'goal_frames': 3,
            })

    train_loader = DataLoader(active_3dsprites_train, 8, num_workers=4, drop_last=True)
    train_data = trainer.predict(model, dataloaders=train_loader)
    with open('/home/rubber/C2PO/goaltraindata_batch{}_rank{}.pkl'.format(i,trainer.global_rank), 'wb') as fh:
        pickle.dump(train_data, fh)
    del trainer, train_loader, train_data

trainer = pl.Trainer(devices=gpus, accelerator="gpu", strategy='ddp', precision=16)

active_3dsprites_val = active_3dsprites_dataset({
            'N': 10000,            
            'action_frames': [2,4,6,8],            
            'episode_length': 12,             
            'gpus': gpus[trainer.global_rank],               
            'with_rotation': False,
            'scale_min': 1.5,
            'scale_max': 1.50001,
            'rand_seed0': 50000+10000+1234,
            'bgcolor': 127,
            'rule_goal': 'IfHalfTorus',
            'v_sd': 0,            
            'goal_frames': 3,
        })


val_loader = DataLoader(active_3dsprites_val, 8, num_workers=2, drop_last=True)
val_data = trainer.predict(model, dataloaders=val_loader)
with open('/home/rubber/C2PO/goalvaldata{}.pkl'.format(trainer.global_rank), 'wb') as fh:
    pickle.dump(val_data, fh)

if trainer.global_rank==0:
    for s in ('train', 'val'):
        data_list = []        
        N = 1 if s=='train' else 1
        for j in range(N):
            for i in range(len(gpus)):            
                if s=='train':
                    fpath = '/home/rubber/C2PO/goal{}data_batch{}_rank{}.pkl'.format(s, j, i)
                else:
                    fpath = '/home/rubber/C2PO/goal{}data{}.pkl'.format(s, i)
                with open(fpath, 'rb') as fh:
                    foo = pickle.load(fh)
                    data_list.append(torch.cat([x['final_lambda'] for x in foo],0))            

        final_lambda = torch.cat(data_list,0)
        with open('/home/rubber/C2PO/goal_data/{}_data_IfHalfTorus_vsd0_snellius-v82_12frames.pkl'.format(s), 'wb') as fh:
            pickle.dump(final_lambda, fh)


    

