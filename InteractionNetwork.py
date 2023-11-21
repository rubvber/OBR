import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from active_dsprites import active_dsprites
from matplotlib import pyplot as plt
from C2PO import STEFunction
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# class ads_ground_truth(DataSet):
#     def __init__(self, N):
#         super().__init()
#         self.N=N

#     def __getitem__(self, idx):
#         self.env = active_dsprites(interactive=True, rand_seed0=idx)
#         self.

#     def __len__(self):
#         return self.N

class PairPredNet(nn.Module):
    # This will get as input the action-perturbed linear object predictions.
    def __init__(self, in_size, out_size, hidden_size=64, K=4, gate='ReTanh', pred_gate_type='single'):
        super().__init__()
        self.hidden_size=hidden_size
        self.pred_gate_type = pred_gate_type        
        self.in_size = in_size
        self.out_size = out_size
        # self.encoder = nn.Sequential(
        #     nn.Linear(in_size, hidden_size),
        #     nn.ELU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ELU(),
        # )
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ELU(),            
        )

        self.self_function = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size*2)
        )

        self.pair_function = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size*2),
        )

        if self.pred_gate_type=='single':
            self.pred_gate_size = 1
        elif self.pred_gate_type=='per_latent':
            self.pred_gate_size = out_size

        self.comb = nn.Sequential(
            # nn.Linear(hidden_size*2+in_size, hidden_size),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, out_size+self.pred_gate_size)
        )
        # self.comb = nn.Linear(hidden_size*2+in_size, out_size+self.pred_gate_size)

        self.pairs = []
        for k in range(K):
            rem_list = list(range(K))
            rem_list.remove(k)
            for l in rem_list:
                self.pairs.append([k, l])

        if gate=='HeavySide':
            self.gate = lambda x: STEFunction.apply(x)
        elif gate=='Sigmoid':
            self.gate = torch.nn.Sigmoid()
        elif gate=='ReTanh':
            self.gate = lambda x: torch.max(torch.zeros(1, device=x.device), x.tanh())

    def forward(self, z):
        N, F, K, L = z.shape

        # z0 = z.clone()
        z1 = self.encoder(z)

        z_self = self.self_function(z1)
        z_self = z_self[:,:,:,:self.hidden_size] * torch.sigmoid(z_self[:,:,:,self.hidden_size:])

        #Assemble tensor of object pairs
        z_pairs_concat = torch.ones(N,F,K*(K-1),self.hidden_size*2,device=z.device)*torch.nan
        cnt = 0
        for k in range(K):
            rem_list = list(range(K))
            rem_list.remove(k)
            for l in rem_list:
                z_pairs_concat[:,:,cnt,:] = torch.cat((z1[:,:,k,:], z1[:,:,l,:]),-1)
                cnt += 1

        z = self.pair_function(z_pairs_concat.view(N*F*K*(K-1),self.hidden_size*2))
        z = (z[:,:self.hidden_size] * torch.sigmoid(z[:,self.hidden_size:])).view(N,F,K*(K-1),self.hidden_size)

        z_interact = torch.zeros(N,F,K,K,self.hidden_size, device=z.device)
        for i, pair in enumerate(self.pairs):
            z_interact[:,:,pair[0],pair[1],:] = z[:,:,i,:]

        z_interact = z_interact.sum(-2)
        # z = self.comb(torch.cat((z0,z1,z_interact),-1).view(N*F*K,-1))
        z = self.comb(torch.cat((z_self,z_interact),-1).view(N*F*K,-1))
        
        z = torch.cat((z[:,:-self.pred_gate_size], self.gate(z[:,-self.pred_gate_size:])), -1)

        return z.view(N,F,K,-1)
        

                        



class InteractionNetwork(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, pred_horizon=1):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.pred_horizon = pred_horizon
        # self.pair_pred_net = PairPredNet(pred_gate_type='per_latent', n_latent=6, K=3)
        self.pair_pred_net = PairPredNet(pred_gate_type='per_latent', in_size=12*2, out_size=12, K=3)
        # self.pair_pred_net = PairPredNet(pred_gate_type='per_latent', in_size=12*2, out_size=12, K=3, hidden_size=128)

    def pred(self, x):
        
        colors, shapes, scales, orientations, positions, velocities = torch.split(x, [3,1,1,1,2,2], -1)
        shapes = shapes==torch.arange(1,4,device=x.device).view(1,1,1,3).to(x.dtype)
        x = torch.cat((colors,shapes,scales,orientations,positions,velocities),-1)
        lin_pred = torch.cat((colors, shapes, scales, orientations, positions+velocities, velocities),-1)
        
        L = x.shape[-1]

        # nonlin_pred = self.pair_pred_net(lin_pred)        
        nonlin_pred = self.pair_pred_net(torch.cat((lin_pred, x),-1))        
        nonlin_pred, nonlin_pred_gate = torch.split(nonlin_pred, [L,L],-1)
        
        pred = nonlin_pred_gate*nonlin_pred + (1-nonlin_pred_gate)*lin_pred
        # pred = nonlin_pred

        colors, shapes, scales, orientations, positions, velocities = torch.split(pred, [3,3,1,1,2,2], -1)
        shapes = (shapes * torch.arange(1,4,device=x.device).view(1,1,1,3)).sum(-1,True)
        pred = torch.cat((colors, shapes, scales, orientations, positions, velocities), -1)

        return pred, nonlin_pred_gate


    def forward(self, x):
        #x has dimensions N x F x K x L
        #function returns the predicted state 1 step ahead             
        
        all_pred = torch.ones((self.pred_horizon,) + x.shape, device=x.device)*torch.nan
        all_nonlin_pred_gate = torch.ones((self.pred_horizon,) + x.shape[:-1]+(12,), device=x.device)*torch.nan

        
        for i in range(self.pred_horizon):
            pred, nonlin_pred_gate = self.pred(x)
            all_pred[i] = pred
            all_nonlin_pred_gate[i] = nonlin_pred_gate
            x = pred
        
        return all_pred, all_nonlin_pred_gate
    
    def eval_step(self, x):        
        pred, gate = self.forward(x)
        pred_loss = torch.zeros(1, device=x.device)

        for i in range(self.pred_horizon):
            this_pred_loss = ((x[:,(i+1):,:,6:] - pred[i,:,:-(i+1),:,6:])**2).sum((2,3)).mean()
            pred_loss = pred_loss + this_pred_loss
        
        # pred_loss = ((x[:,1:,:,6:]-pred[:,:-1,:,6:])**2).sum((2,3)).mean()        
        gate_loss = STEFunction.apply(gate).sum((3,4)).mean()
        gate_mean = gate.mean()
        loss = pred_loss + 0*gate_loss

        return {'loss': loss, 'pred_loss': pred_loss, 'gate_loss': gate_loss, 'gate_mean': gate_mean, 'pred': pred}
    
    def training_step(self, x, batch_idx):
        loss_dict = self.eval_step(x)

        self.log_dict({
            'train_loss': loss_dict['loss'],
            'train_pred_loss': loss_dict['pred_loss'],
            'train_gate_loss': loss_dict['gate_loss'],
            'train_mean_gate': loss_dict['gate_mean'],
        }, sync_dist=True)

        return loss_dict['loss']
    
    def validation_step(self, x, batch_idx):
        loss_dict = self.eval_step(x)

        self.log_dict({
            'val_loss': loss_dict['loss'],
            'val_pred_loss': loss_dict['pred_loss'],
            'val_gate_loss': loss_dict['gate_loss'],
            'val_mean_gate': loss_dict['gate_mean'],
        }, sync_dist=True)

        if batch_idx==0 and self.global_rank==0:
            colors = [[1.0,0,0],[0,1.0,0],[0,0,1.0]]
            F = x.shape[1]
            
            for p in range(self.pred_horizon):
                fh, ax = plt.subplots(16,4, figsize=(15,40))
                ax = ax.flatten()

                fh2, ax2 = plt.subplots(16,4, figsize=(15,40))
                ax2 = ax2.flatten()
                for i in range(16*4):
                    for k in range(x.shape[2]):
                        ax[i].plot(x[i,:,k,6].cpu(), x[i,:,k,7].cpu(), color=colors[k])
                        ax[i].plot(loss_dict['pred'][p,i,:,k,6].cpu(), loss_dict['pred'][p,i,:,k,7].cpu(), color=colors[k], linestyle='--')
                        ax[i].set_xlim(0.0,1.0)
                        ax[i].set_ylim(0.0,1.0)
                        ax[i].set_aspect('equal', adjustable='box')

                        ax2[i].plot(torch.arange(0,F), x[i,:,k,6].cpu(), color=colors[k])
                        ax2[i].plot(torch.arange(0,F), x[i,:,k,7].cpu(), color=colors[k])
                        ax2[i].plot(torch.arange(p+1, F+p+1), loss_dict['pred'][p,i,:,k,6].cpu(), color=colors[k], linestyle='--')
                        ax2[i].plot(torch.arange(p+1, F+p+1), loss_dict['pred'][p,i,:,k,7].cpu(), color=colors[k], linestyle='--')
                        

                self.logger.experiment.add_figure('predictions_{}'.format(p), fh, self.current_epoch) 
                self.logger.experiment.add_figure('predictions_time_{}'.format(p), fh2, self.current_epoch) 
                plt.close(fh)
                plt.close(fh2)
            

        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.pair_pred_net.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.333333, patience=5, min_lr=1e-5)

        return ({'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}})
    
        # return optimizer

if __name__=="__main__":

    # model = InteractionNetwork(learning_rate=3e-4).load_from_checkpoint('/home/rubber/C2PO/InteractionNetworkLogs/lightning_logs/version_15/checkpoints/last.ckpt')
    # model.learning_rate=3e-4
    # model.pred_horizon=4

    model = InteractionNetwork(learning_rate=3e-4, pred_horizon=4)

    train_dataset = active_dsprites(
        N=int(5e4),
        with_collisions=True,
        pos_smp_stats=(0.2,0.8),
        include_prior_pref=False,
        rand_seed0=1234,
        return_sprite_data=True,
        interactive=False,
        num_frames=12,
        action_frames=[15,],
        scale_min = 1/4,
        direct_actions=True,
        include_bgd_action=False,
    )   

    val_dataset = active_dsprites(
        N=int(1e4),
        with_collisions=True,
        pos_smp_stats=(0.2,0.8),
        include_prior_pref=False,
        rand_seed0=1234+int(5e4),
        return_sprite_data=True,
        interactive=False,
        num_frames=12,
        action_frames=[15,],
        scale_min = 1/4,
        direct_actions=True,
        include_bgd_action=False,
    )

    train_loader = DataLoader(train_dataset, 128, num_workers=8, drop_last=True, persistent_workers=True)
    val_loader  = DataLoader(val_dataset, 256, num_workers=8, drop_last=True, persistent_workers=True)

    checkpoint_callback = ModelCheckpoint(
        # monitor="val_loss_cumul",        
        monitor="val_loss",
        save_top_k=1, 
        mode="min",
        save_last=True         
    )    
    
    callbacks = [checkpoint_callback]

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    trainer = pl.Trainer(devices=(5,), accelerator='gpu', logger=pl.loggers.TensorBoardLogger('./InteractionNetworkLogs/'), 
                         callbacks=callbacks, strategy='ddp')
    
    # trainer = pl.Trainer(devices=(1,), accelerator='gpu', logger=pl.loggers.TensorBoardLogger('./InteractionNetworkLogs/'), 
    #                      callbacks=callbacks, strategy='ddp', resume_from_checkpoint='/home/rubber/C2PO/InteractionNetworkLogs/lightning_logs/version_14/checkpoints/last.ckpt')
    

    trainer.fit(model, train_loader, val_loader)

    #v