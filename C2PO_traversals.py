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

model = C2PO().load_from_checkpoint('/home/rubber/C2PO/C2PO_logs/lightning_logs/version_30/checkpoints/last.ckpt')

model.maxF=4
model.val_predict=0

test_loader = DataLoader(active_dsprites_test, 16, num_workers=2, persistent_workers=False, drop_last=True)
trainer = pl.Trainer(devices=(0,), accelerator="gpu", strategy='ddp', precision=16)
test_data = trainer.predict(model, dataloaders=test_loader)

device='cuda:0'
model.to(device)

final_lambda = test_data[0]['final_lambda'][:,-1]
mu, *_ = torch.chunk(final_lambda, 4, -1)
N,K,_ = mu.shape[:]
# trav_range = torch.linspace(-1,1,20, device=device)
trav_range = torch.linspace(-2,2,20, device=device)
T = trav_range.shape[0]
for i in range(model.n_latent):
    trav = torch.zeros(trav_range.shape[0], model.n_latent, device=device)
    trav[:,i] = trav_range
    mu_trav = mu.unsqueeze(2).to(device).detach() + trav.view(1,1,-1,model.n_latent)
    rec, mask = model.decoder(mu_trav.view(-1,model.n_latent))
    rec = rec.view((N,K,T)+rec.shape[-3:])
    mask = mask.view((N,K,T)+mask.shape[-3:])

    _, mask_cat = mask.max(1,keepdim=True) #Categorical index of maximum mask value per pixel
    mask_cat_oh = (mask_cat==(torch.arange(K, device=device).view(1,K,1,1,1,1)))*1 #one-hot version of mask index (could also do sth like a==max(a), but that doesn't break ties)
    rec_comb = (mask_cat_oh*rec.detach()).sum(1) #Reconstruction obtained by hard-masking the reconstructions from the different slots
    this_grid = make_grid(rec_comb.view(N*20,3,64,64), nrow=20)
    Image.fromarray((this_grid.permute(1,2,0)*255).to(torch.uint8).cpu().numpy()).save('./results/traversal_v30_dim{}.png'.format(i))





