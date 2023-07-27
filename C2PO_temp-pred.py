import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from C2PO import C2PO
import pickle, sys

sys.path.append('../AttentionExperiments/src/')
from active_3dsprites import active_3dsprites_dataset

pl.seed_everything(1234)

active_dsprites_test = active_3dsprites_dataset({
            'N': 16,
            'episode_length': 4,
            'action_frames': (2,),
            'interactive': False,
            'gpus': (0,),
            'with_rotation': False,
            'bounding_actions': False,
            'scale_min': 1.5,
            'scale_max': 1.50001,
            'bgcolor': 127,
            'include_bgd_action': True,
            'no_depth_motion': False,
            'rand_seed0': 50000+10000+1234+4243
        })

# model = C2PO().load_from_checkpoint('/home/rubber/C2PO/C2PO_logs/lightning_logs/version_30/checkpoints/last.ckpt')
# model = C2PO().load_from_checkpoint('/home/rubber/C2PO/snellius_checkpoints/version_58/checkpoints/last.ckpt')
model = C2PO().load_from_checkpoint('/home/rubber/C2PO/snellius_checkpoints/version_59/checkpoints/last.ckpt')

model.maxF=4
model.val_predict=0

test_loader = DataLoader(active_dsprites_test, 16, num_workers=2, persistent_workers=False, drop_last=True)
trainer = pl.Trainer(devices=(0,), accelerator="gpu", strategy='ddp', precision=16)
test_data = trainer.predict(model, dataloaders=test_loader)

device='cuda:0'
model.to(device)

final_lambda = test_data[0]['final_lambda']
mu, mup, *_ = torch.chunk(final_lambda, 4, -1)
N,_,K,_ = mu.shape[:]

rec, mask = model.decoder(mu.view(-1, model.n_latent).to(device).detach())
rec = rec.view((N,4,K)+rec.shape[-3:])
mask = mask.view((N,4,K)+mask.shape[-3:])
_, mask_cat = mask.max(2,keepdim=True) #Categorical index of maximum mask value per pixel
mask_cat_oh = (mask_cat==(torch.arange(K, device=device).view(1,1,K,1,1,1)))*1 #one-hot version of mask index (could also do sth like a==max(a), but that doesn't break ties)
rec_comb = (mask_cat_oh*rec.detach()).sum(2) #Reconstruction obtained by hard-masking the reconstructions from the different slots

this_grid = make_grid(rec_comb.view(N*4,3,64,64), nrow=4)
Image.fromarray((this_grid.permute(1,2,0)*255).to(torch.uint8).cpu().numpy()).save('./results/temp-pred_snellius-v59_rec.png')
Image.fromarray((make_grid(test_data[0]['ims'].view(16*4,3,64,64), nrow=4).permute(1,2,0)*255).to(torch.uint8).cpu().numpy()).save('./results/temp-pred_snellius-v59_orig.png')


mu_pred = mup[:,(-1,)] * torch.arange(1,4+1).view(1,4,1,1) + mu[:,(-1,)]
rec, mask = model.decoder(mu_pred.view(-1, model.n_latent).to(device).detach())
rec = rec.view((N,4,K)+rec.shape[-3:])
mask = mask.view((N,4,K)+mask.shape[-3:])
_, mask_cat = mask.max(2,keepdim=True) #Categorical index of maximum mask value per pixel
mask_cat_oh = (mask_cat==(torch.arange(K, device=device).view(1,1,K,1,1,1)))*1 #one-hot version of mask index (could also do sth like a==max(a), but that doesn't break ties)
rec_comb_pred = (mask_cat_oh*rec.detach()).sum(2) #Reconstruction obtained by hard-masking the reconstructions from the different slots

rec_all = torch.cat((rec_comb,rec_comb_pred),1)
this_grid = make_grid(rec_all.view(N*8,3,64,64), nrow=8)
Image.fromarray((this_grid.permute(1,2,0)*255).to(torch.uint8).cpu().numpy()).save('./results/temp-pred_snellius-v59_rec+pred.png')


