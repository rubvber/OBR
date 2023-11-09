import torch, os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from C2PO import C2PO
from active_3dsprites import active_3dsprites_dataset, active_3dsprites_vecenv, active_3dsprites_env
from math import floor, ceil
import numpy as np
from tqdm import tqdm

def run_demo(K=3, threeD=True, batch_size=4):
    data = demo_run_inf(K,threeD, batch_size)
    demo_plot(data, threeD)

def demo_run_inf(K=3, threeD=True, batch_size=4):
    if threeD:
        ckpt_path = 'threeD.ckpt'
        if not os.path.exists(ckpt_path):
            ckpt_path = 'C2PO/threeD.ckpt'
        if not os.path.exists(ckpt_path):
            raise Exception('Checkpoint path not found')


    model = C2PO().load_from_checkpoint(ckpt_path, init_percept_net_path = None)
    model.maxF=12
    model.val_predict=0
    model.interactive=True
    model.action_frames=(4, 5, 6, 7, 8, 9, 10, 11)
    model.lambda_a = 0.5
    model.planning_horizon=3
    model.action_generation_type = 'goal'
    model.ignore_goal_vel = False
    model.action_placement_method = 'hedge'

    active_dsprites_test = active_3dsprites_dataset({
            'N': 16,
            'interactive': True,            
            'action_frames': [],            
            'gpus': 0,
            'with_rotation': False,
            'scale_min': 1.5,
            'scale_max': 1.50001,
            'rand_seed0': 50000+10000+1234+4343,
            'bgcolor': 127,            
            'v_sd': 1/64*10,
        })

    model.K = K+1
    test_loader = DataLoader(active_dsprites_test, batch_size=batch_size, num_workers=2, persistent_workers=False, drop_last=True)
    trainer = pl.Trainer(devices=(0,), accelerator="gpu", precision=16)
    test_data = trainer.predict(model, dataloaders=test_loader)

    return test_data

def save_grid_image(x, name):  
    Image.fromarray((make_grid(x, nrow=4)*255).permute(1,2,0).to(torch.uint8).cpu().numpy()).save(name)
    return


def save_grid_gif(x, name):
    ims = [Image.fromarray((make_grid(foo,nrow=4)*255).permute(1,2,0).to(torch.uint8).cpu().numpy()).quantize(dither=Image.NONE) for foo in x]
    ims[0].save(name, 'gif', save_all=True, append_images=ims[1:], loop=0, duration=[33,]*(len(x)-1)+[4000,])
    return

def demo_plot(data, threeD=True):
    N,F,_,Z = data[0]['true_states'].shape
    render_size=128
    
    frames = torch.zeros(N,len(data),(F-1)*10+1,3,render_size, render_size)
    # frames = []
    goal_ims = torch.zeros(N,len(data), 3, render_size, render_size)
    for j,d in enumerate(tqdm(data, desc='Processing results batches')):
        if threeD:
            rule_goal='IfHalfTorus'
            ad = active_3dsprites_vecenv(init_data=(d['true_states'][:,0], d['true_bgc']), ctx = {'rule_goal': rule_goal, 'im_size': render_size})
        for i,t in enumerate(tqdm(np.arange(0,11.00001,0.1), desc='Rendering frames')):
            
            t0 = floor(t)
            t1 = ceil(t)
            td = t%1
            interp_state = (1-td)*d['true_states'][:,t0] + td*d['true_states'][:,t1]
            
            if threeD:                                
                if i>0: ad.set_obj_poses(interp_state)                
                ims, _ = ad.render(keep_render_env=True)
                frames[:,j,i] = ims
                # frames.append(ims)

        if threeD:            
            goal = ad.get_goal_states()
            ad.destroy_render_envs()
            del ad
            ad_goal = active_3dsprites_vecenv(init_data=(goal, d['true_bgc']), ctx={'im_size': render_size})
            ims, _ = ad_goal.render()
            goal_ims[:,j] = ims

    frames = frames.view(N*len(data),frames.shape[2],3,render_size,render_size)
    goal_ims = goal_ims.view(N*len(data),3,render_size,render_size)
    # frames = [[frames[i,f] for f in range(frames.shape[1])] for i in range(frames.shape[0])]
    frames = [frames[:,f] for f in range(frames.shape[1])]
    
    print('Saving images...')
    save_grid_gif(frames, 'zz_gif.gif')        
    save_grid_image(goal_ims, 'zz_goal.gif')

    pass


if __name__ == "__main__":
    run_demo()
