from math import log, sqrt
import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np

from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt

from active_dsprites import active_dsprites



def ismember(d, k):
  return [1 if (i in k) else 0 for i in d]

def gumbel_sample(logits, dim=0, n=1, tau=1.0, include_sample_dim=False):
    '''
    Uses the Gumbel-softmax trick for (approximately) sampling values of categorical random variables in a differentiable 
    manner. The output will be vectors that are close to one-hot - how close depends on the temperature parameter 'tau'. 
    Smaller values of tau correspond to higher temperatures and a closer approximation, but may also result in numerical 
    instability (e.g. underflow) and vanishing gradients. 

    The temperature may be increased (i.e. tau reduced) according to an annealing schedule across trainig epochs. 

    More information found here:   
    https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    '''

    shape = logits.shape + (n,) if n>1 or include_sample_dim else logits.shape
    if n==1 and include_sample_dim: 
        logits = logits.unsqueeze(-1)
    elif n>1:
        logits = torch.cat((logits.unsqueeze(-1), logits.detach().unsqueeze(-1).expand(logits.shape + (n-1,))), -1)        

    g = -(-torch.rand(shape, device=logits.device).log()).log()
    z = (logits+g)/tau
    z = z - z.amax(dim=dim,keepdim=True) #Subtract max for numerical stability
    z = z.exp() + 1e-10 #Add a slight offset for numerical stability    
    z = z/z.sum(dim=dim, keepdim=True)

    
    return z

def FARI(targ, pred):
    '''
    Computes the (foreground-adjusted) Rand Index between a prediction 'pred' and the ground-truth 'targ'. 
    Both should be image-sized tensors of corresponding sizes, with dimensions [batch x H x W] (any non-
    batch dimensions in desired input tensors can be "folded into" the batch dimension beforehand"). 

    Input tensors should contain integer values indicating the cluster assignment of each pixel. Importantly,
    pixels belonging to the background in the ground truth should be assigned a value of '0' in 'targ'. 
    Otherwise, you are free to use whatever assignment values you want, and the values used in 'pred' are
    fully arbitrary (e.g. no special value for the background)
    '''

    assert targ.shape==pred.shape, 'Tensor shapes must match (found {} and {})'.format(targ.shape, pred.shape)

    N = targ.shape[0]

    FARI_scores = torch.ones(N, device=targ.device)*float('nan')
    ARI_scores = torch.ones_like(FARI_scores)*float('nan')

    targ, pred = [foo.view(N,-1) for foo in (targ, pred)]
    fg = targ!=0
    for i in range(N):
        ARI_scores[i]  = adjusted_rand_score(targ[i], pred[i]) 
        FARI_scores[i] = adjusted_rand_score(targ[i,fg[i,:]], pred[i,fg[i,:]]) #Only considers pixels that are foreground in the ground-truth


    return ARI_scores, FARI_scores
    
    


        
class GoalNet(nn.Module):
    def __init__(self, n_latent, hidden_size=64):        
        super().__init__()    
        
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
                nn.Linear(n_latent*4, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ELU()
        )                    
        self.ctx_layer = nn.Linear(hidden_size, hidden_size)        
        self.decoder = nn.Sequential(    
                nn.ELU(),
                nn.Linear(hidden_size*2, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, n_latent*4)
        )
            
    def forward(self, x):
        #x is batch x slot x input_dim (where frames may be folded into the batch dimension)
        N, K, n_latent = x.shape #NOTE: n_latent here is the total length of a latent vector, not the number of distinct latent variables (which is smaller by a factor 4)       
        
        x0 = x
        x = self.encoder(x.view(N*K, n_latent))
        
        ctx = self.ctx_layer(x)
        ctx = ctx.view(-1, K, self.hidden_size).mean(1, keepdim=True) #Unfold slots and average over them
        ctx = ctx.expand(N, K, n_latent)            
        x = x.view(-1, K, self.hidden_size) #Unfold slots so we can concatenate
        x = torch.cat((x,ctx), -1)
        x = x.view(-1, self.hidden_size*2)
        
        x = self.decoder(x)
        x_lat_dim = n_latent        
        x = x.view(N, K, x_lat_dim) #Unfold one more time        
        
        x = x0 + x
        
            
        return x


class SBD(nn.Module):
    def __init__(self, in_channels=32+2, out_channels=4, im_size=64, config='simple'):
        super().__init__()        
        self.im_size=im_size                
        
        if config=='simple':
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, 32, 5, padding=2),
                nn.ELU(),
                nn.ConvTranspose2d(32, 32, 5, padding=2),
                nn.ELU(),
                nn.ConvTranspose2d(32, 32, 5, padding=2),
                nn.ELU(),
                nn.ConvTranspose2d(32, 32, 5, padding=2),
                nn.ELU(),
                nn.ConvTranspose2d(32, out_channels, 5, padding=2),
            )
        elif config=='greffCLEVR':
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, 64, 3, padding=1),
                nn.ELU(),                
                nn.ConvTranspose2d(64, 64, 3, padding=1),
                nn.ELU(),
                nn.ConvTranspose2d(64, 64, 3, padding=1),                
                nn.ELU(),
                nn.ConvTranspose2d(64, 64, 3, padding=1),                             
                nn.ELU(),                
                nn.ConvTranspose2d(64, out_channels, 3, padding=1),
            )
        
        gx, gy = torch.meshgrid([torch.linspace(-1,1,im_size)]*2)        
        gg = torch.stack((gx, gy)).unsqueeze(0) # Creates size 1 x 2 x im_size x im_size
        self.register_buffer('xy_grid', gg)        

    
    def forward(self,z):        
        N = z.shape[0]
        z = z.view(z.shape + (1,1)).expand(-1,-1,*[self.im_size]*2) #Expands z to include H & W dimensions
        gg = self.xy_grid.expand(N, *[-1]*3)
        z = torch.cat((z, gg), 1) #Concatenate z in channel dimension with coordinate grid    
        conv_out = self.conv(z)                
        mu_x, mask_logits = torch.split_with_sizes(conv_out, (3,1), 1)        
        mu_x = torch.sigmoid(mu_x) 
        return mu_x, mask_logits

class RefineNet(nn.Module):
    def __init__(self, tot_n_latent, vec_input_size, im_input_size, config='simple'):
        super().__init__()
        #tot_n_latent here should mean the total dimensionality of the latent space, including mu and logsd, derivatives and actions
        if config=='simple':
            self.conv = nn.Sequential(
                nn.Conv2d(im_input_size, 32, 5, stride=2),
                nn.ELU(),
                nn.Conv2d(32, 32, 5, stride=2),
                nn.ELU(),
                nn.Conv2d(32, 32, 5, stride=2),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(32*5**2, 128), #5x5 is the size of the final conv map (not the kernel size)
                nn.ELU()
            )        
                    
            self.lstm = nn.LSTMCell(128+vec_input_size, 128) #Times tot_n_latent by two here because the inputs we get as input the latents and their gradients        
            self.fc_out = nn.Linear(128, tot_n_latent)        
        elif config=='greffCLEVR':
            self.conv = nn.Sequential(
                nn.Conv2d(im_input_size, 64, 3, stride=2),
                nn.ELU(),
                nn.Conv2d(64, 64, 3, stride=2),
                nn.ELU(),
                nn.Conv2d(64, 64, 3, stride=2),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(64*7**2, 256), #7x7 is the size of the final conv map
                nn.ELU()
            )  
            self.lstm = nn.LSTMCell(256+vec_input_size, 256) #Times tot_n_latent by two here because the inputs we get as input the latents and their gradients        
            self.fc_out = nn.Linear(256, tot_n_latent)     
        
    def forward(self, x, h, c):
        conv_out = self.conv(x['img'])        
        h,c = self.lstm(torch.cat((conv_out, x['vec']), 1), (h,c))
        return self.fc_out(h), h, c 

class ActionNet(nn.Module):
    def __init__(self, vec_input_size=4+4+2, out_size=4):
        super().__init__()

        self.lstm = nn.LSTMCell(vec_input_size, 32) 
        self.fc_out = nn.Linear(32, out_size)

    def forward(self, x, h, c):
        
        h,c = self.lstm(x,(h,c))
        return self.fc_out(h), h, c
        


    
def stable_softmax(x, dim=-1):
    x = (x-x.amax(dim, keepdim=True)).exp()
    x = x+1e-12    
    x = x/x.sum(dim,True)

    return x



class C2PO(pl.LightningModule):
    def __init__(self, 
            n_latent=16, 
            K=4, 
            learning_rate=3e-4,             
            im_var=0.3**2,             
            beta=5.0, 
            sigma_s=0.1, 
            val_predict=0,            
            reduceLR_factor=0.0,
            reduceLR_patience=10,
            reduceLR_minlr=0.0,            
            train_num_iter=12,
            train_iter_per_timestep=4,                        
            sigma_chi=0.1,                        
            init_percept_net_path=None,            
            freeze_percept_net=False,            
            init_tau=1.0,
            ar_tau=3e-5, 
            min_tau=0.5,                       
            D_init_sd=0.02,            
            train_win_size=3,            
            interactive=False,
            maxF=None,
            action_generation_type='random', #Only relevant for interactive mode. Can be 'random' or 'goal-driven'.
            action_frames = (2,), #Only relevant for interactive mode. Can be tuple, list or range.             
            num_mask_samples=1,                        
            lambda_a = 0.1,                        
            with_goal_net = False,            
            init_goal_net_path = None,            
            heart_becomes_square=0,
            threeD = False,
            with_rotation = False,
            network_config = 'simple',
            new_first_action_inf = False,
            reg_D_lambda = (0.0,0.0)
        ):            

            
        super().__init__()
        
        self.n_latent = n_latent #This is the number of latent dimensions before extending to generalized coordinates (so e.g. with state and first derivative, this number will be doubled)
        self.K = K                
        self.im_var = im_var        
        self.im_prec = self.im_var**-1 
        self.sigma_s = sigma_s
        self.learning_rate = learning_rate                
        self.beta=beta        
        self.val_predict=val_predict        
        self.reduceLR_factor = reduceLR_factor
        self.reduceLR_patience = reduceLR_patience
        self.reduceLR_minlr = reduceLR_minlr        
        self.train_num_iter=train_num_iter
        self.train_iter_per_timestep=train_iter_per_timestep                
        self.init_percept_net_path = init_percept_net_path
        self.freeze_percept_net = freeze_percept_net        
        self.sigma_chi = sigma_chi
        self.init_tau = init_tau
        self.ar_tau = ar_tau                
        self.min_tau = min_tau
        self.D_init_sd = D_init_sd
        self.train_win_size = train_win_size        
        self.interactive = interactive
        self.maxF=maxF
        self.action_generation_type = action_generation_type
        self.action_frames=action_frames        
        self.num_mask_samples=num_mask_samples                
        self.lambda_a = lambda_a        
        self.with_goal_net = with_goal_net        
        self.init_goal_net_path = init_goal_net_path        
        self.heart_becomes_square = heart_becomes_square
        self.threeD = threeD
        self.with_rotation = with_rotation
        self.network_config = network_config
        self.new_first_action_inf = new_first_action_inf
        self.reg_D_lambda = reg_D_lambda
        
        self.save_hyperparameters()
        
        self.im_prec = self.im_var**-1 
        if network_config=='simple':
            self.register_buffer('h0', torch.zeros(1, 128))       
        elif network_config=='greffCLEVR':
            self.register_buffer('h0', torch.zeros(1, 256))       

        self.tot_n_latent = n_latent*2*2 #Total number of latent entries: n_latent times two because we have state and derivative, and then whole thing times two again because we have mu and logsd for all (or times three if we also output autocorrs)

        if threeD:
            self.action_dim = 6 if with_rotation else 3
        else:
            assert not with_rotation, 'Combination 2D + rotations not yet implemented'
            self.action_dim = 2 

        assert not (threeD and interactive), "Combination interactive + threeD not yet implemented"

        if self.init_percept_net_path is not None:            
            '''
            We don't use the freeze() method because this causes problems when using autograd later to compute the 
            gradients for inference. I guess autograd requires all intermediate tensors to have requires_grad=True, even if the
            gradient w.r.t. a tensor isn't necessary to compute the gradients of the non-freezed tensors.             
            '''            
            percept_net = C2PO.load_from_checkpoint(init_percept_net_path, init_percept_net_path=None, init_goal_net_path=None, foo='blah')             
            self.decoder = percept_net.decoder
            self.refine_net = percept_net.refine_net
            self.layer_norms = percept_net.layer_norms
            self.lambda0 = percept_net.lambda0            
            self.action_net = percept_net.action_net

            if hasattr(percept_net, 'D'):
                self.D = percept_net.D                                        
                                    

            if not self.init_percept_net_path==self.init_goal_net_path:
                del percept_net
        else:
            self.decoder = SBD(in_channels=n_latent+2, config=self.network_config)            
            self.refine_net = RefineNet(self.tot_n_latent, vec_input_size=self.tot_n_latent*2, im_input_size = 16, config=self.network_config)
                        
            #mu and logsd for derivatives make sense to be 0 (logsd=0 means sd=1) as there's no reason to initialize the derivatives to anything else, and we might risk initializing them far away from the actual speed
            self.lambda0 = nn.Parameter(torch.cat((
                torch.randn(1, self.n_latent), #mu for states should be random
                torch.zeros(1, self.n_latent), #mu for derivative of states should be 0
                torch.randn(1, self.n_latent), #logsd for states should be random 
                torch.zeros(1, self.n_latent), #logsd for derivative of states should be 0                                
            ), dim=1)
            )
                                
            self.layer_norms = nn.ModuleDict({
                'rec_grad': nn.LayerNorm((3,64,64), elementwise_affine=False),
                'mask_grad': nn.LayerNorm((1,64,64), elementwise_affine=False),
                'post_grad': nn.LayerNorm((self.tot_n_latent), elementwise_affine=False), 
                'pix_like': nn.LayerNorm((1,64,64), elementwise_affine=False),
                'action_grad': nn.LayerNorm((self.action_dim*2), elementwise_affine=False),  
            })            
                    
            self.D = nn.Parameter(torch.randn(self.action_dim,self.n_latent)*self.D_init_sd)                                              
            self.action_net = ActionNet(vec_input_size=5*self.action_dim, out_size=2*self.action_dim)                                
                
        self.gumbel_tau={
                'curr_tau': 1.0,
                'tau0': init_tau,
                'min': min_tau,
                'anneal_rate': ar_tau
            }
                
        if with_goal_net:
            if self.init_goal_net_path is not None:
                if self.init_percept_net_path==self.init_goal_net_path:
                    #In this case we don't need to load again
                    init_goal_net = percept_net                    
                else:
                    init_goal_net = self.load_from_checkpoint(init_goal_net_path, init_percept_net_path=None, init_action_net_path=None, init_goal_net_path=None, foo='blah')                                             
                self.goal_net = init_goal_net.goal_net
                del init_goal_net
            else:             
                hidden_size=64
                self.goal_net = GoalNet(n_latent, hidden_size)


    def comb_rec(self, rec, mask):
        K = rec.shape[2]
        mask = mask.detach()
        _, mask_cat = mask.max(2,keepdim=True) #Categorical index of maximum mask value per pixel
        mask_cat_oh = (mask_cat==(torch.arange(K, device=rec.device).view(1,1,K,1,1,1)))*1 #one-hot version of mask index (could also do sth like a==max(a), but that doesn't break ties)
        rec_comb = (mask_cat_oh*rec.detach()).sum(2) #Reconstruction obtained by hard-masking the reconstructions from the different slots

        return rec_comb, mask_cat

        
    def decode(self, this_lambda, do_sample=True):
        N,F,K,_ = this_lambda.shape[:]
        mu, _, logsd, _ = torch.split(this_lambda[:,:,:,:self.n_latent*4], self.n_latent, dim=-1)
        eps = torch.randn_like(mu)            
        z = mu+eps*logsd.exp() if do_sample else mu
        z = torch.clamp(z, -1000, 1000) #This is for stability - since we really expect things to be mostly in a range of [-3, 3], it's quite a liberal restriction.
        rec, mask_logits = self.decoder(z.view(N*F*K,self.n_latent)) 
        _,C,H,W = rec.shape
        rec, mask_logits = rec.view(N,F,K,C,H,W), mask_logits.view(N,F,K,1,H,W)
        mask = stable_softmax(mask_logits, 2)
        mask_logits = mask.log() #This ensures that the logits reflect the probabilities generated by stable_softmax, and (thus) may prevent negative inf's

        return rec, mask_logits, mask

    def forward(self, x, 
            num_inf_steps=4,
            is_train=False,             
            num_predict=0, #Number of frames to predict into the future. We assume that the entire input is to be used for inference, so any frames that aren't to be included in this should be omitted when passing input to forward()            
            action_fields=None,
            prior_pref=None,            
            win_size=3, #Size of inference window            
            maxF=None, #If we run in interactive mode, then we need to specify the maximum number of frames            
            ):

        '''
        This needs to do a few things. In the outermost loop, we're sliding the inference window over
        the data. If we're running in interactive mode, then this loop also may involve selecting an action,
        sending this to the environment, updating the environment state and obtaining a new observation from it.

        If we're not running in interactive mode, then the size of the input is set and that's all we do 
        on. The inference window starts with just the first available input, then expands (one frame at a time)
        to its full size, and after that it moves by one frame after each inference period. The number of inference
        steps we spend before advancing one frame is given by num_inf_steps.

        If we are in interactive mode, the above still holds, but whenever we advance one frame we have to get
        new data from the environment. We may also generate an action to send to the environment first, if the 
        newest frame is equal to or greater than t_begin_action.

        For a given position of the inference window, we perfor num_inf_steps iterations of inference on it.         
        '''


        def predict(prev_lambda, curr_lambda=None, curr_action_lambda=None, action_fields=None, first_frame_overall=False):         
            '''
         
            The output is the prediction of the current lamdba (based on the prev lambda and the curr action).

            s'_t = s'_t-1 + D*a_t + noise
            s_t  = s_t-1  + s'_t  + noise            

            '''

            assert not(curr_action_lambda is not None and action_fields is not None), 'Cannot provide both action lambda and action fields'
            
            mu_state_prev, mu_prime_prev, logsd_state_prev, logsd_prime_prev = torch.split(prev_lambda, self.n_latent, dim=-1)
            if curr_lambda is not None:            
                _, mu_prime_curr, _, logsd_prime_curr = torch.split(curr_lambda, self.n_latent, dim=-1)
            var_prime_prev = (2*logsd_prime_prev).exp()
            mu_prime_pred = mu_prime_prev
            var_prime_pred = var_prime_prev
            
            if curr_action_lambda is not None:
                if self.new_first_action_inf:
                    if first_frame_overall:                    
                        # In this case there cannot be an action lambda so we set it to 0 mu and 0 var
                        N,_,K,_ = curr_action_lambda.shape[:]
                        # foo = torch.cat((torch.zeros(N,1,K,self.action_dim, device=self.device), torch.ones(N,1,K,self.action_dim,device=self.device)*-1000),-1)
                        foo = torch.zeros(N,1,K,self.action_dim*2, device=self.device)
                        # foo = torch.cat((torch.zeros(N,1,K,self.action_dim, device=self.device), torch.ones(N,1,K,self.action_dim,device=self.device)*-torch.inf),-1)
                        curr_action_lambda = torch.cat((foo, curr_action_lambda[:,1:]),1)
                mu_action, logsd_action = torch.chunk(curr_action_lambda, 2, dim=-1) #Here 2 is appropriate (rather than self.action_dim) as we're just splitting the tensor in 2            
                
                mu_prime_pred = mu_prime_pred + (self.D.view(1,1,1,self.action_dim,-1)*mu_action.unsqueeze(-1)).sum(-2)             
                var_prime_pred = var_prime_pred + (self.D.view(1,1,1,self.action_dim,-1)**2*logsd_action.exp().unsqueeze(-1)**2).sum(-2)                
        
            if curr_lambda is None:
                # If no curr_lamdba was supplied, then we'll use the predicted derivative to produce the state prediction
                mu_state_pred = mu_state_prev + mu_prime_pred
                var_state_pred = (2*logsd_state_prev).exp() + var_prime_pred
            else:
                mu_state_pred = mu_state_prev + mu_prime_curr
                var_state_pred = (2*logsd_state_prev).exp() + (2*logsd_prime_curr).exp()

            if mu_state_pred.ndim==var_state_pred.ndim+1:
                var_state_pred=var_state_pred.unsqueeze(-1).expand_as(mu_state_pred)
                var_prime_pred=var_prime_pred.unsqueeze(-1).expand_as(mu_prime_pred)
                return torch.cat((mu_state_pred, mu_prime_pred, var_state_pred.log()*0.5, var_prime_pred.log()*0.5), -2)
            else:
                return torch.cat((mu_state_pred, mu_prime_pred, var_state_pred.log()*0.5, var_prime_pred.log()*0.5), -1)

        
        def compute_loss(in_dict):
            x = in_dict['x']
            rec = in_dict['rec']
            mask = in_dict['mask']
            prev_mask_logits = in_dict['prev_mask_logits']
            curr_lambda = in_dict['curr_lambda']
            pred_lambda = in_dict['pred_lambda']
            action_fields = in_dict['action_fields']
            curr_action_lambda = in_dict['curr_action_lambda']
            sigma_s = in_dict['sigma_s']

            N,F,C,H,W = x.shape[:]  
            K = rec.shape[2]

            use_frames = range(0,F)             
                            
            if in_dict['first_frame_overall']:
                '''
                This means the first frame was the very first and so there is no previous one, and we shouldn't compute an 
                action loss for the first frame (as there is no previous mask to apply it to).                        
                '''
                use_frames = range(1,F)   



            mu_curr, logsd_curr = torch.split(curr_lambda, self.n_latent*2, dim=3)
            mu_pred, logsd_pred = torch.split(pred_lambda, self.n_latent*2, dim=3)
            if curr_action_lambda is None:
                logsd_action = torch.tensor([], device=x.device)
            else:
                mu_action, logsd_action = torch.chunk(curr_action_lambda, 2, dim=-1)

            rec_err_sq = (x.unsqueeze(2)-rec)**2
            pix_ll = (-0.5*(np.log(2*np.pi*self.im_var) + self.im_prec*rec_err_sq)).sum(3) #sum over RGB channels (same as product in probability domain)

            pix_like_weighted = mask.squeeze(3)*pix_ll.exp() #squeeze mask in the channel dimension as it's only a single channel (but don't squeeze in the frame dimension!)
            pix_like = pix_like_weighted.sum(2) + 1e-40 #Sum over slots to get the overal pixel likelihood, add a tiny offset for numerical stability

            del pix_like_weighted

            rec_loss = -pix_like.log().sum((2,3))         
            
            if self.new_first_action_inf and not 0 in use_frames:                
                logsd_action = torch.cat((torch.ones(N,1,self.K,self.action_dim, device=x.device), logsd_action[:,1:]), 1)

            gaussian_entropy = 0.5*(1.0 + log(2*np.pi) + 2*torch.cat((logsd_curr, logsd_action), -1)).sum((2,3)) 
            
            if mu_pred.ndim==mu_curr.ndim+1:
                mu_curr, logsd_curr, sigma_s = map(lambda x: x.unsqueeze(-1), (mu_curr, logsd_curr, sigma_s))
                                    
               
            prediction_CE = (0.5*log(2*np.pi) + sigma_s.log() + 0.5*sigma_s**-2*((mu_curr-mu_pred)**2 + logsd_curr.exp()**2 + logsd_pred.exp()**2)).sum(3) #Don't sum over slots yet      

            if prediction_CE.ndim==4: 
                prediction_CE=prediction_CE.sum(2).mean(-1)   
            else:
                prediction_CE=prediction_CE.sum(2)

            del mu_curr, mu_pred, logsd_curr, logsd_pred     
     
            
            mask_sample = gumbel_sample(prev_mask_logits[:,use_frames], dim=2, tau=self.gumbel_tau['curr_tau'], n=self.num_mask_samples, include_sample_dim=True)
            #mask_logits is already conditioned on sample of s gathered previously                                
            action_pred = (mask_sample*action_fields[:,use_frames].unsqueeze(-1)).sum((4,5))
            del mask_sample

            action_loss = (0.5*log(2*np.pi) + log(self.sigma_chi) + 0.5*self.sigma_chi**-2*((mu_action[:,use_frames].unsqueeze(-1) - action_pred)**2 + logsd_action[:,use_frames].unsqueeze(-1).exp()**2)).sum((2,3)).mean(-1)                
        
            if not 0 in use_frames:
                action_loss = torch.cat((torch.zeros(N,1, device=x.device), action_loss),1)
                        
                
        
            loss_per_im = -gaussian_entropy + prediction_CE + self.beta*rec_loss + action_loss
            loss = loss_per_im.mean(0) #Mean rather than sum so that it doesn't depend on batch size                        
            
            out_dict = {
                'loss': loss,
                'rec_loss': rec_loss.mean(0),
                'pred_loss': prediction_CE.mean(0),
                'negent_loss': -gaussian_entropy.mean(0),
                'action_loss': action_loss.mean(0),
                'pix_ll': pix_ll,
                'pix_like': pix_like,
                'loss_per_im': loss_per_im,
                }
                        
            return out_dict

        
        def generate_inf_inputs(x):
            ims = x['ims']
            loss = x['loss']
            rec = x['rec']
            rec_loss = x['rec_loss'].sum()
            mask = x['mask']
            mask_logits = x['mask_logits']
            pix_ll = x['pix_ll']
            pix_like = x['pix_like']
            curr_lambda = x['curr_lambda']
            curr_action_lambda = x['curr_action_lambda']            

            N,F,K,_,H,W = rec.shape[:]

            rec_grad = torch.autograd.grad(rec_loss, rec, retain_graph=True, only_inputs=True)[0] #rec only affects rec_loss (not other loss components), and scaling by beta doesn't matter here                
            mask_grad = torch.autograd.grad(loss, mask, retain_graph=True, only_inputs=True)[0]             
            action_grad = torch.autograd.grad(loss, curr_action_lambda, retain_graph=True, only_inputs=True)[0]            
            post_grad = torch.autograd.grad(loss, curr_lambda, retain_graph=True, only_inputs=True)[0] #Always retain graph since it's possible we're going to do more iterations of the current frame(s) later
  
            pix_like = pix_like.view(N,F,1,1,H,W).expand(N,F,K,1,H,W)

            rec_grad = self.layer_norms['rec_grad'](rec_grad).detach()
            mask_grad = self.layer_norms['mask_grad'](mask_grad).detach()
            post_grad = self.layer_norms['post_grad'](post_grad).detach()
            pix_like = self.layer_norms['pix_like'](pix_like).detach()                        
            action_grad = self.layer_norms['action_grad'](action_grad).detach()
                                            
            post_mask = stable_softmax(pix_ll,1).view(N,F,K,1,H,W)

            '''
            Shouldn't all these inputs be detached (not just the grad's & likelihoods)? Right now, the rec's and masks are not detached, which means the decoder is being trained
            not just to generate good outputs w.r.t. reconstruction loss, but also to provide good inputs to the refinement net, which isn't necessarily desirable, and also leads 
            to a much deeper graph to backpropagate through.    
            '''
            
            im_inputs = torch.cat((
                ims.unsqueeze(2).expand(N,F,K,C,H,W),
                rec,
                mask,
                mask_logits,
                rec_grad,
                mask_grad,
                post_mask,
                pix_like,                                                
                self.decoder.xy_grid.view(1,1,1,2,H,W).expand(N,F,K,2,H,W),

            ), 3).view(N*F*K,-1,H,W)
            vec_inputs = torch.cat((
                curr_lambda,
                post_grad,                                
            ),-1).view(N*F*K, -1) 

            if not(im_inputs.isfinite().all().item() and vec_inputs.isfinite().all().item()):                    
                print('\nNan or inf refine inputs found\n')
                print('\nNan or inf im_inputs:\n')
                print(np.argwhere(im_inputs.isfinite().logical_not().cpu()))
                
                print('\nNan or inf vec_inputs:\n')
                print(np.argwhere(vec_inputs.isfinite().logical_not().cpu()))
            

            return im_inputs, vec_inputs, action_grad

        
        def iterate(x, init_lambda, init_h, init_c, action_fields=None, num_steps=4, first_prev_lambda=None, x_diff=None):


            '''
            -x is expected to be [batch x frame x C x H x W], and should (only) cover all frames we want to do
            inference on
            -init_lambda is expected to be [batch x frame x slot x latents]            
            -If first_pred_lambda is specified, then it gives the predicted latents for the first frame in
            the inference window. Otherwise, we assume the first frame in the window is the first frame overall
            and only constrained by a static prior.

            Steps:
            -Decode sample to get loss and auxiliary inputs
            -Perform inference given input and auxiliary inputs

            We keep everything in the format of [batch x frame x slot x ...] as much as possible, and only 
            collapse frames and slots into the batch dimension when necessary
            '''

            curr_lambda, *curr_action_lambda = torch.split(init_lambda, self.tot_n_latent,-1) #curr_lambda may include autocorrs                                  
            curr_h, *curr_action_h = torch.split(init_h, self.h0.shape[-1],-1)
            curr_c, *curr_action_c = torch.split(init_c, self.h0.shape[-1],-1)                       
            curr_action_lambda, curr_action_h, curr_action_c = map(lambda x: x[0], (curr_action_lambda, curr_action_h, curr_action_c))

            N,F,C,H,W = x.shape[:]            

            losses = torch.zeros((F,num_steps+1), device=x.device)
            losses_per_im = torch.zeros((N,F,num_steps+1), device=x.device)
            rec_losses = torch.zeros((F,num_steps+1), device=x.device)
            pred_losses = torch.zeros((F,num_steps+1), device=x.device)
            negent_losses = torch.zeros((F,num_steps+1), device=x.device)
            action_losses = torch.zeros((F,num_steps+1), device=x.device)

            if first_prev_lambda is None:
                #This means we didn't receive a previous lambda and will use the prior instead for the first frame
                first_frame_overall = True
                sigma_s = torch.cat((
                    torch.ones((1,1,1,1), device=x.device),
                    torch.ones((1,F-1,1,1), device=x.device)*self.sigma_s
                ), 1)
                #Prior has mean 0 and sd 1.0, however this s.d. is captured in sigma_s. There's no previous state that we're uncertain about, so the variance here is 0 (and thus logsd=-inf).
                first_prev_lambda = torch.zeros((N,1,K,curr_lambda.shape[-1]), device=x.device) 
                first_prev_lambda[:,:,:,self.n_latent*2:] = -torch.inf
            else:
                first_frame_overall = False
                sigma_s = torch.ones((1,F,1,1), device=x.device)*self.sigma_s	            

            for t in range(num_steps+1):
                         
                if first_frame_overall:
                    #This means the first frame in the input is the first frame overall and so there is no previous mask
                    rec, mask_logits, mask = self.decode(curr_lambda)                        
                    first_prev_mask = torch.zeros(N,1,K,1,H,W, device=x.device)
                    # first_prev_mask_logits = torch.ones(N,1,K,1,H,W, device=x.device)*-torch.inf
                    first_prev_mask_logits = torch.ones(N,1,K,1,H,W, device=x.device) #This makes it so the sampling is random                    
                else:
                    # Sample from q(s) and decode sample                                 
                    rec, mask_logits, mask = self.decode(torch.cat((first_prev_lambda,curr_lambda),1))                
                    _, rec = torch.split(rec, (1,F), dim=1)                        

                    first_prev_mask_logits, mask_logits = torch.split(mask_logits, (1,F), dim=1)
                    first_prev_mask, mask = torch.split(mask, (1,F), dim=1)

                prev_mask = torch.cat((first_prev_mask, mask[:,:-1]),1)
                prev_mask_logits = torch.cat((first_prev_mask_logits, mask_logits[:,:-1]),1)
                              

                '''
                Generate predictions. We now generate them for the *current* time points, based on the *previous* state lambdas and *current*
                action lambdas. This means that we *always* need to (re-)generxate predictions, and there is no need to generate a prediction
                for a new-to-be-added time point if we have reached the end of this iteration sequence.               
                '''

                prev_lambda = torch.cat((first_prev_lambda, curr_lambda[:,:-1]), 1)                     
                pred_lambda = predict(prev_lambda, curr_lambda=curr_lambda, curr_action_lambda=curr_action_lambda, first_frame_overall=first_frame_overall)
                
                #Compute loss
                cl_dict = compute_loss({
                    'x': x,
                    'rec': rec,
                    'mask': mask,
                    'prev_mask_logits': prev_mask_logits,
                    'mask_logits': mask_logits,
                    'curr_lambda': curr_lambda,
                    'pred_lambda': pred_lambda,
                    'prev_lambda': prev_lambda,
                    'action_fields': action_fields,
                    'curr_action_lambda': curr_action_lambda,
                    'sigma_s': sigma_s,                    
                    'first_frame_overall': first_frame_overall,
                })                 
                losses[:,t] = cl_dict['loss']
                rec_losses[:,t] = cl_dict['rec_loss']
                pred_losses[:,t] = cl_dict['pred_loss']
                action_losses[:,t] = cl_dict['action_loss']
                negent_losses[:,t] = cl_dict['negent_loss']
                losses_per_im[:,:,t] = cl_dict['loss_per_im']

                #Generate inputs for refinement net update
                if t < num_steps:                        
                    im_inputs, vec_inputs, action_grad = generate_inf_inputs({
                        'ims': x,
                        'loss': cl_dict['loss'].sum(),
                        'mask': mask,
                        'mask_logits': mask_logits,
                        'rec': rec,
                        'rec_loss': cl_dict['rec_loss'],
                        'pix_ll': cl_dict['pix_ll'],
                        'pix_like': cl_dict['pix_like'],
                        'curr_lambda': curr_lambda,
                        'curr_action_lambda': curr_action_lambda,
                        'pred_loss': cl_dict['pred_loss'].sum(),
                        'prev_lambda': prev_lambda,
                        'x_diff': x_diff,
                    })
                                        
                
                    lambda_update, h_new, c_new = self.refine_net({'img': im_inputs, 'vec': vec_inputs}, curr_h.view(N*F*K,-1), curr_c.view(N*F*K,-1))                    
                    if not lambda_update.isfinite().all().item():
                        print('\nNan or inf lambda update found')                        
                    else:
                        curr_lambda = curr_lambda + lambda_update.view(N,F,K,-1)
                        curr_h, curr_c = h_new.view(N,F,K,-1), c_new.view(N,F,K,-1)
                    
                    del lambda_update, h_new, c_new
                    
                    exp_action = (prev_mask*action_fields).sum((4,5))
                    action_vec_inputs = torch.cat((
                        curr_action_lambda,
                        action_grad,
                        exp_action
                    ),-1).view(N*F*K,-1)
                    action_lambda_update, action_h_new, action_c_new = self.action_net(action_vec_inputs, curr_action_h.view(N*F*K,-1), curr_action_c.view(N*F*K,-1))

                    if not action_lambda_update.isfinite().all().item():
                        print('\nNan or inf action_lambda update found')
                    else:
                        curr_action_lambda = curr_action_lambda + action_lambda_update.view(N,F,K,-1)
                        curr_action_h, curr_action_c = action_h_new.view(N,F,K,-1), action_c_new.view(N,F,K,-1)

                    del action_lambda_update, action_h_new, action_c_new


            rec = rec.detach()
            rec_comb, mask_cat = self.comb_rec(rec, mask.detach())            

            mask = {
                'prob': mask,
                'cat' : mask_cat
            }
                        
            out_dict = {
                'losses': losses,               
                'rec_losses': rec_losses.detach(),
                'pred_losses': pred_losses.detach(),
                'negent_losses': negent_losses.detach(),
                'action_losses': action_losses.detach(),
                'losses_per_im': losses_per_im.detach(),
                'final_h': curr_h,
                'final_c': curr_c,
                'final_lambda': curr_lambda,
                'final_action_h': curr_action_h,
                'final_action_c': curr_action_c,
                'final_action_lambda': curr_action_lambda,                
                'rec': rec_comb,
                'mask': mask,                
            }
            

            return out_dict

        K = self.K
        
        if self.interactive:
            N = len(x[0])
            env = active_dsprites(data=x, num_sprites=x[0].shape[2])
            x, true_masks, prior_pref = env.render()                        
            _,C,H,W = x.shape[:]
            x=x.view(N,1,C,H,W)
            true_masks=true_masks.view(N,1,H,W)
            action_fields = torch.zeros(N,1,1,self.action_dim,H,W, device=x.device)
            prior_pref = prior_pref.view(N,1)        

            true_states = env.sprite_data.unsqueeze(1)    
            true_bgc = env.bgcolor
        else:             
            N,maxF,C,H,W = x.shape[:] #Note that we never pass to-be-predicted frames to forward(), so F is always the full number of frames to be inferred. (To-be predicted frames are split off from the batch outside this function.)

        
        assert action_fields is not None, 'no action fields were supplied'
        if self.interactive and self.action_generation_type=='goal':
            '''
            Pre-computing some stuff here that we'll need whenever we plan goal-directed actions. Could even be done more efficiently 
            by just doing it once and setting it as a fixed model variable. However, it does depend on the planning horizon, so even if D
            doesn't change any more, if we want to plan with variable horizons, then this has to be re-computed too.                
            '''
            if not hasattr(self, 'planning_horizon'):
                self.planning_horizon=5
            onevec = torch.ones(self.planning_horizon, device=x.device)    
            d = torch.arange(1, self.planning_horizon+1, device=x.device)
            row_idx, col_idx = torch.meshgrid(torch.arange(1, self.planning_horizon*self.action_dim+1, device=x.device),torch.arange(1, self.planning_horizon+1, device=x.device), indexing='ij')
            omega_d = torch.max(torch.tensor(0, device=x.device), (row_idx+1)/2-col_idx+1).to(torch.int)*(row_idx%2) + (col_idx<=(row_idx/2))*(1-row_idx%2)                        
            Ww = torch.kron(omega_d, self.D.T.contiguous()) #Just calling this Ww since we already have a W                
   
        if action_fields is not None and not self.interactive:        
            action_fields = action_fields.view(N,maxF,1,self.action_dim,H,W) #Insert the slot dimension 

        
        init_action_lambda = torch.zeros(N,1,K,2*self.action_dim, device=x.device, requires_grad=True)
        init_action_h = torch.zeros(N,1,K,32, device=x.device)
        init_action_c = torch.zeros_like(init_action_h)


        all_losses = [list() for _ in range(maxF)]
        all_rec_losses = [list() for _ in range(maxF)]
        all_pred_losses = [list() for _ in range(maxF)]
        all_negent_losses = [list() for _ in range(maxF)]
        all_action_losses = [list() for _ in range(maxF)]
        all_losses_per_im = [torch.tensor([], device=x.device) for _ in range(maxF)]
        all_recs = torch.ones((N,maxF,C,H,W), device=x.device)*torch.nan        

        all_masks = {'prob': torch.ones((N,maxF,K,H,W), device=x.device)*torch.nan, 'cat': torch.ones((N,maxF,H,W), device=x.device)*torch.nan} if not is_train else None
        
        all_lambdas = torch.ones((N,maxF,K,self.tot_n_latent), device=x.device)*torch.nan
        if self.with_goal_net:
            all_pref_lambdas = torch.zeros((N,maxF,K,self.tot_n_latent), device=x.device)        
        
        max_win_pos = maxF-1 + win_size-1
        # win_pos is defined as the position of the leading frame.
        # We keep moving the window until it's fully out of the input frames, so the last window has only the final frame inside it.
        
        for win_pos in range(max_win_pos+1): 
            win_start = max(0,win_pos-win_size+1) #First frame in inference window (inclusive)
            win_end = min(win_pos+1, maxF) #First frame not in inference window (i.e. window end, exclusive)'            
            win = range(win_start, win_end)
            this_x = x[:,win]            
            this_action_fields = action_fields[:,win] 

            if win_pos==0:
                '''
                If this is the first window, comprising just the first frame, then everything is initialized to default values
                '''
                init_lambda = self.lambda0.view(1,1,1,-1).expand(N,1,K,-1)
                init_h = self.h0.view(1,1,1,-1).expand(N,1,K,-1)
                init_c = torch.zeros((N,1,K,self.h0.shape[-1]),device=x.device)
            else:
                '''
                If this is a later window, then we want to initialize the frames we've already done inference on,
                to their final values from the previous inference. Frames we have yet to do inference on, we want to
                initialize to their predicted values (for lambda) or else to the default initial settings (for h and c).
                
                NOTE: this code assumes that the window always advances by 1 frame, which is the only behavior that is 
                currently implemented.                 
                ''' 
                prev_win_pos = win_pos-1
                prev_win_start = max(0,prev_win_pos-win_size+1) #First frame in inference window (inclusive)
                prev_win_end = min(prev_win_pos+1, maxF) #First frame not in inference window (i.e. window end, exclusive)'            
                prev_win = range(prev_win_start, prev_win_end)
                
                logical_idx = np.isin(np.array(prev_win),np.array(win))
                new_win_idx_in_old_win = np.where(logical_idx)[0] #Indexes in old window of time points that are in new window                
                assert np.logical_not(logical_idx).sum()<=1, 'Discrepancy with previous window is more than 1 frame - something is wrong!'                                
                      
                if prev_win_end < win_end:                             
                    exp_action = (iter_out_dict['mask']['prob'][:,(-1,)]*this_action_fields[:,(-1,)]).sum((4,5))
                    new_init_action_lambda = torch.cat((exp_action, torch.zeros((N,1,K,self.action_dim), device=x.device, requires_grad=True)),-1)
                    init_action_lambda = torch.cat((iter_out_dict['final_action_lambda'][:,new_win_idx_in_old_win], new_init_action_lambda),1)
                    init_action_h = torch.cat((iter_out_dict['final_action_h'][:,new_win_idx_in_old_win], torch.zeros(N,1,K,32, device=x.device)),1)
                    init_action_c = torch.cat((iter_out_dict['final_action_c'][:,new_win_idx_in_old_win], torch.zeros(N,1,K,32, device=x.device)),1)
                else:
                    init_action_lambda = iter_out_dict['final_action_lambda'][:,new_win_idx_in_old_win]
                    init_action_h = iter_out_dict['final_action_h'][:,new_win_idx_in_old_win]
                    init_action_c = iter_out_dict['final_action_c'][:,new_win_idx_in_old_win]
                                        
                if prev_win_end < win_end:          
                    next_pred_lambda = predict(iter_out_dict['final_lambda'][:,(-1,),:,:self.n_latent*4], curr_action_lambda=new_init_action_lambda)
                    init_lambda = torch.cat((iter_out_dict['final_lambda'][:,new_win_idx_in_old_win], next_pred_lambda),1)
                    init_h = torch.cat((iter_out_dict['final_h'][:,new_win_idx_in_old_win], self.h0.view(1,1,1,-1).expand(N,1,K,-1)),1)
                    init_c = torch.cat((iter_out_dict['final_c'][:,new_win_idx_in_old_win], torch.zeros((N,1,K,self.h0.shape[-1]),device=x.device)),1)
                else:
                    init_lambda = iter_out_dict['final_lambda'][:,new_win_idx_in_old_win]
                    init_h = iter_out_dict['final_h'][:,new_win_idx_in_old_win]
                    init_c = iter_out_dict['final_c'][:,new_win_idx_in_old_win]
                        
            init_lambda = torch.cat((init_lambda, init_action_lambda),-1)        
            init_h = torch.cat((init_h, init_action_h),-1)       
            init_c = torch.cat((init_c, init_action_c),-1)       
            

            if win_start==0:
                first_prev_lambda=None
            else:
                '''
                If the first frame in the window is not the first overall, then we need to use the final lambda we got from the first frame of the 
                last window.                
                '''
                first_prev_lambda=iter_out_dict['final_lambda'][:,(0,)]


            iter_out_dict = iterate(this_x, init_lambda, init_h, init_c, this_action_fields, first_prev_lambda=first_prev_lambda, num_steps=num_inf_steps)
                                    

            if win_pos>=win_size-1:
                '''
                If we've reached a window position where we've visited the lagging-edge frame for the final time, then we need
                to save its reconstruction. We could just gather all the reconstructions in a (frame x iteration) list, like we do for
                losses, but since the rec's are image-sized this would waste a lot of memory, and it's not necessary since we're only
                interested (for now) in the final reconstructions
                '''                
                all_recs[:,win_start] = iter_out_dict['rec'][:,0]             
                if not is_train:   
                    all_masks['prob'][:,win_start] = iter_out_dict['mask']['prob'][:,0].squeeze().detach()                
                    all_masks['cat'][:,win_start] = iter_out_dict['mask']['cat'][:,0].squeeze()                
                all_lambdas[:,win_start] = iter_out_dict['final_lambda'][:,0]
                

            if win_pos==max_win_pos:
                '''
                If we've reached the final window, then we additionally need to save reconstructions for the last frames as well.
                '''
                all_recs[:,win_start+1:] = iter_out_dict['rec'][:,1:]
                if not is_train:   
                    all_masks['prob'][:,win_start+1:] = iter_out_dict['mask']['prob'][:,1:].squeeze(-3).detach() #Squeeze out the channel dimension (if we don't pass the dimension, there is an edge case where the last window contains only two frames, and then after dropping the first frame we're left with an additional singleton dimension (frames) which we don't want to squeeze.)                
                    all_masks['cat'][:,win_start+1:] = iter_out_dict['mask']['cat'][:,1:].squeeze(-3).squeeze(-3) #Here we additinally squeeze out the slot dimension, and unfortunately pytorch doesn't let us squeeze two dimensions at once              
                all_lambdas[:,win_start+1:] = iter_out_dict['final_lambda'][:,1:]
                
            elif self.interactive and win_pos<(maxF-1): 
                '''
                If we're running in interactive mode, we need to optionally generate an action and then advance the environment by one step.

                Note that we only do this if the current window position is such that the next window includes a new frame. 
                For instance, if maxF==12, then the index of the final frame is 11, which means that win_pos==10 is the final position where
                a new frame needs to be generated.  
                '''
                if win_pos+1 in self.action_frames:
                    if self.action_generation_type=='random':
                        action_field = env.get_random_action(env.a_sd)
                    elif self.action_generation_type=='goal':                 
                        action_field = torch.zeros(N,2,H*W,device=x.device)                                                     
                       
                        mu_curr, _ = torch.split(iter_out_dict['final_lambda'][:,-1], self.n_latent*2, dim=-1)
                        _, mu_curr_prime = torch.chunk(mu_curr, 2, -1)                            
                        if self.with_goal_net:
                            obj_actions = torch.zeros(N,K,self.planning_horizon*self.action_dim, device=x.device)
                            goal_lambda = self.goal_net(iter_out_dict['final_lambda'][:,-1].contiguous())                                                            
                            all_pref_lambdas[:,win_pos] = goal_lambda

                            mu_pref, logsd_pref = torch.chunk(goal_lambda, 2, -1)
                            prec_pref = (logsd_pref*-2).exp()/100 
                            prec_pref = torch.ones_like(prec_pref)
                            logsd_pref = prec_pref.log()
                            for i in range(N):
                                for k in range(self.K):
                                    L = torch.diag(torch.kron(onevec,(logsd_pref[i,k]*-2).exp()))
                                    WtL = Ww.T@L
                                    WLWiWL = torch.inverse((WtL@Ww + self.lambda_a*torch.eye(self.planning_horizon*self.action_dim,device=Ww.device)).to(torch.float32))@WtL 
                                    obj_actions[i,k] = (WLWiWL.view(1,1,self.planning_horizon*self.action_dim,-1)* \
                                        (torch.kron(onevec.view(1,1,-1), mu_pref[i,k]-mu_curr[i,k]) - \
                                        torch.kron(d.view(1,1,-1), torch.cat((mu_curr_prime[i,k], torch.zeros_like(mu_curr_prime[i,k])),-1))).unsqueeze(-2)).sum(-1)

                        # Note that these actions are in units of pixels, whereas env.step() expects actions in units of fractions of the image size.
                        curr_mask = iter_out_dict['mask']['prob'][:,-1].squeeze().view(N,K,-1) #N x K x H*W
                        for i in range(N):
                            for k in range(K):
                                _, idx = torch.sort(curr_mask[i,k], descending=True)
                                for j in range(len(idx)):
                                    if action_field[i,0,idx[j]]==0:
                                        action_field[i,:,idx[j]] = obj_actions[i,k,:self.action_dim]/H #Take the first action in the planned sequence. Divide by image size because it is currently in pixels and env.step() expects it as a fraction of image size
                                        break
                        action_field = action_field.view(N,self.action_dim,H,W)

                else:
                    action_field = torch.zeros(N,self.action_dim,H,W,device=x.device)
                                
                env.step(action_field.detach())
                if self.heart_becomes_square>1 and win_pos+1==self.heart_becomes_square:                    
                    sd = env.sprite_data
                    for i in range(N):
                        is_heart = sd[i,:,3]==3
                        sd[i,is_heart,3] = 1
                        env.sprite_data = sd
                    
                im,true_mask,this_pref = env.render()                
                x = torch.cat((x,im.unsqueeze(1)), 1)
                true_states = torch.cat((true_states, env.sprite_data.unsqueeze(1)), 1)
                action_fields = torch.cat((action_fields,action_field.view(N,1,1,self.action_dim,H,W)*H), 1) #Multiply by image size because we work in pixels for legacy reasons. Should probably make everything consistent in future.
                prior_pref = torch.cat((prior_pref, this_pref.view(N,1)), 1)
                '''
                The action fields we get from get_random_action are in coordinates of [0,1] rather than the pixel coordinates we get from getitem. Ideally we would
                choose one fixed system for this, but for now let's just handle it like this. Most important thing is that the action passed to env.step() is *not* in pixel
                coordinates, as it expects the [0,1] range ones so otherwise the sprites get teleported out of the image frame.               
                '''
                true_masks = torch.cat((true_masks,true_mask.unsqueeze(1)),1)
                
            

            
            for f in range(iter_out_dict['losses'].shape[0]):
                '''
                We gather losses in a list of lists (frame x iteration), so that we can later compute things like the final loss, weighted mean loss, etc.
                '''
                frame_idx = win_start + f
                num_iter = iter_out_dict['losses'][f].shape[-1]

                final_frame_visit=False
                if frame_idx==win_start and win_pos>=(win_size-1): final_frame_visit=True #If this is the lagging frame of a window that isn't still growing into the video, then this is the final time we have visited this frame.
                if win_pos==max_win_pos: final_frame_visit=True #If this is the final window position, then any frame inside it has also been visited for the last time.
                

                include_iters = range(num_iter) if final_frame_visit else range(num_iter-1)
                
                '''
                The line below was actually bugged when the window size didn't match the video size.
                Instead of 'win_end-1' it should read 'win_size-1', since if is the index within the window, 
                not within the video. This means that the final loss of the final frame actually doesn't contribute
                to the training loss. 

                Moreover, this was handling the problem backwards. It was always (meant to be) the final loss, but only
                including the intial loss for frames that had just been initialized. It's actually better to always include
                the initial loss, but only take the final loss when the frame is visited for the last time, because that way,
                you include the loss just after a future frame is initialized, which contributes to the prediction loss of 
                past frames.  
                '''
                                
                all_losses[frame_idx].extend(list(iter_out_dict['losses'][f,include_iters])) 
                all_rec_losses[frame_idx].extend(list(iter_out_dict['rec_losses'][f,include_iters])) 
                all_pred_losses[frame_idx].extend(list(iter_out_dict['pred_losses'][f,include_iters])) 
                all_negent_losses[frame_idx].extend(list(iter_out_dict['negent_losses'][f,include_iters])) 
                all_action_losses[frame_idx].extend(list(iter_out_dict['action_losses'][f,include_iters])) 
                all_losses_per_im[frame_idx] = torch.cat((all_losses_per_im[frame_idx], iter_out_dict['losses_per_im'][:,f,include_iters]), -1)
                
        iter_lengths = [len(foo) for foo in all_losses]
        total_loss = 0.0
        frame_final_losses = torch.zeros(maxF, device=x.device)
        all_losses_tensor = torch.ones((maxF,max(iter_lengths)), device=x.device)*torch.nan                
        all_losses_per_im_tensor = torch.ones((N,maxF,max(iter_lengths)), device=x.device)*torch.nan                
        for f in range(maxF):                        
            w = torch.linspace(max(iter_lengths)-iter_lengths[f]+1, max(iter_lengths), iter_lengths[f], device=x.device)/max(iter_lengths)                        
            for i in range(len(w)):
                total_loss = total_loss + (all_losses[f][i]*w[i]) #This is a bit clunky but more elegant solutions all resulted in the loss not getting a grad_fn, thus messing with autograd
            frame_final_losses[f] = all_losses[f][-1]
            all_losses_tensor[f, 0:iter_lengths[f]]=torch.tensor(all_losses[f])
            all_losses_per_im_tensor[:,f,0:iter_lengths[f]] = all_losses_per_im[f]
            
        loss_steps = all_losses_tensor.nanmean(0)
        
        if num_predict>0:
            with torch.no_grad():
                pred_lambda = torch.zeros((N,num_predict,K,self.n_latent*4), device=x.device)                
                pred_lambda[:,0] = predict(iter_out_dict['final_lambda'][:,-1,:,:self.n_latent*4]) 
                for i in range(1,num_predict):
                    pred_lambda[:,i] = predict(pred_lambda[:,i-1,:,:self.n_latent*4])
                    

                pred_rec,pred_mask,_ = self.decode(pred_lambda, do_sample=False)
                pred_rec,_ = self.comb_rec(pred_rec, pred_mask)
                all_recs = torch.cat((all_recs, pred_rec), 1)
                if self.interactive:
                    for i in range(num_predict):
                        env.step()
                        im,true_mask,pref = env.render()
                        x = torch.cat((x,im.unsqueeze(1)), 1)
                        true_masks = torch.cat((true_masks,true_mask.unsqueeze(1)), 1)

        if not is_train:
            total_loss, frame_final_losses, loss_steps = map(lambda x: x.detach(), (total_loss, frame_final_losses, loss_steps))
        else:
            if not total_loss.isfinite(): total_loss = None

        out_dict = {
            'total_loss': total_loss,        
            'frame_losses': frame_final_losses,
            'loss_steps': loss_steps,    
            'loss_rec': torch.tensor([foo[-1] for foo in all_rec_losses]).mean(),
            'loss_pred': torch.tensor([foo[-1] for foo in all_pred_losses]).mean(),
            'loss_negent': torch.tensor([foo[-1] for foo in all_negent_losses]).mean(),
            'loss_action': torch.tensor([foo[-1] for foo in all_action_losses]).mean(),
            'rec': all_recs,            
            'mask': all_masks,            
            'final_lambda': all_lambdas,            
            'ims': x,
            'true_masks': true_masks if self.interactive else None,
            'true_states': true_states if self.interactive else None,
            'true_bgc': true_bgc if self.interactive else None,
            'loss_steps_per_im': all_losses_per_im_tensor,
            'prior_pref': prior_pref,            
            'pref_lambda': all_pref_lambdas if self.with_goal_net and self.interactive else None,
            'action_fields': action_fields,
        }

        return out_dict


    def training_step(self, batch, batch_idx):        
        
        self.gumbel_tau['curr_tau'] = np.maximum(self.gumbel_tau['tau0']*np.exp(-self.gumbel_tau['anneal_rate']*self.global_step), self.gumbel_tau['min'])

        if not self.interactive:
            if isinstance(batch, list):
                ims, _, action_fields = batch[:]
         
            out_dict = self.forward(
                ims,
                action_fields=action_fields,                
                is_train=True,
                num_inf_steps=self.train_iter_per_timestep,
                win_size=self.train_win_size)              
        else:
            out_dict = self.forward(
                batch, 
                is_train=True,
                num_inf_steps=self.train_iter_per_timestep,
                win_size=self.train_win_size,
                maxF=self.maxF)       

        if self.with_goal_net:
            N,F,K,_ = out_dict['final_lambda'].shape
            pred_goal = self.goal_net(out_dict['final_lambda'].view(N*F, K, -1)).view(N,F,K,-1)
            targ_goal = out_dict['final_lambda'][:,(-1,)]
            targ_mu, targ_logsd = torch.chunk(targ_goal, 2, -1)
            pred_mu, pred_logsd = torch.chunk(pred_goal, 2, -1)

            goal_loss = (-pred_logsd.sum((1,2,3)) + 0.5*((-2*targ_logsd).exp()*((targ_mu-pred_mu)**2 + (2*pred_logsd).exp())).sum((1,2,3))).mean(0)            
            out_dict['total_loss'] = out_dict['total_loss'] + goal_loss
            self.log('train_loss_goal', goal_loss)

        if self.reg_D_lambda != (0.0,0.0):
            out_dict['total_loss'] = out_dict['total_loss'] + self.reg_D_lambda[0]*self.D.abs().sum() + self.reg_D_lambda[1]*(self.D**2).sum()

        self.log_dict({
            'train_loss_final': out_dict['frame_losses'].sum(),
            'train_loss_pred': out_dict['loss_pred'],            
            'train_loss_rec': out_dict['loss_rec'],
            'train_loss_negent': out_dict['loss_negent'],
            'train_loss_action': out_dict['loss_action'],
            'curr_tau': self.gumbel_tau['curr_tau'],  
        }, sync_dist=True)            
        
        if out_dict['total_loss'] is not None: self.log('train_loss_cumul', out_dict['total_loss'], sync_dist=True)
        if hasattr(self, 'log_sigma_chi'): self.log('sigma_chi', self.sigma_chi, sync_dist=True)
            
        return {'loss': out_dict['total_loss']}
  
    def validation_step(self, batch, batch_idx,):        
        torch.set_grad_enabled(True) #Pytorch lightning turns this off by default during validation, but we need it here for the internal gradient computation in forward()        

        if not self.interactive:
            
            ims, true_masks, action_fields = batch[:]

            if self.val_predict==0:
                end_idx = ims.shape[1]
            else:
                end_idx = -self.val_predict
            F_infer = ims.shape[1]-self.val_predict

            if action_fields is None:
                pass_action_fields=None
            else:             
                pass_action_fields=action_fields[:,:end_idx]
                   
            out_dict = self.forward(
                ims[:,:end_idx],
                action_fields=pass_action_fields,              
                num_predict=self.val_predict,
                num_inf_steps=self.train_iter_per_timestep,
                win_size=self.train_win_size)
  
        else:
            #Interactive
            out_dict = self.forward(
                batch,
                num_predict=self.val_predict,
                num_inf_steps=self.train_iter_per_timestep,
                win_size=self.train_win_size,
                maxF=self.maxF)
            F_infer = batch.shape[1]-self.val_predict
            ims = out_dict['ims']            
            true_masks = out_dict['true_masks']            
            end_idx = self.maxF
            

        
        torch.set_grad_enabled(False) #Not sure if this is necessary        
        N,_,_,H,W = ims.shape
        ARI_scores, FARI_scores = FARI(true_masks[:,:end_idx].reshape(N*F_infer,H,W).cpu(), out_dict['mask']['cat'].reshape(N*F_infer,H,W).cpu())

        
        if self.with_goal_net:
            N,F,K,_ = out_dict['final_lambda'].shape
            pred_goal = self.goal_net(out_dict['final_lambda'].view(N*F, K, -1)).view(N,F,K,-1)
            targ_goal = out_dict['final_lambda'][:,(-1,)]
            targ_mu, targ_logsd = torch.chunk(targ_goal, 2, -1)
            pred_mu, pred_logsd = torch.chunk(pred_goal, 2, -1)

            goal_loss_per_latent = (-pred_logsd.sum((1,2)) + 0.5*((-2*targ_logsd).exp()*((targ_mu-pred_mu)**2 + (2*pred_logsd).exp())).sum((1,2))).mean(0)            
            goal_loss = goal_loss_per_latent.sum()                
            goal_mse  = ((targ_mu-pred_mu)**2).mean()
            out_dict['total_loss'] = out_dict['total_loss'] + goal_loss                
            self.log_dict({'val_loss_goal': goal_loss,
                'val_loss_final': out_dict['frame_losses'].sum()+goal_loss, 
                'val_mse_goal': goal_mse},
                sync_dist=True)                    
            
            if batch_idx==0:
                pred_rec, _, pred_mask = self.decode(pred_goal, do_sample=False)
                pred_goal_ims, _ = self.comb_rec(pred_rec, pred_mask) 
        else:
            self.log('val_loss_final', out_dict['frame_losses'].sum(), sync_dist=True)

        if out_dict['total_loss'] is not None: self.log('val_loss_cumul', out_dict['total_loss'], sync_dist=True)
       
        
        self.log('val_loss_action', out_dict['loss_action'], sync_dist=True)      

        mse_recon = ((ims[:,:end_idx]-out_dict['rec'][:,:end_idx])**2).mean()
        mse_pred  = ((ims[:,end_idx:]-out_dict['rec'][:,end_idx:])**2).mean()
        _, logsd = torch.split(out_dict['final_lambda'], self.n_latent*2, dim=-1)

        self.log_dict({            
            'val_loss_pred': out_dict['loss_pred'], 
            'val_loss_rec': out_dict['loss_rec'],
            'val_loss_negent': out_dict['loss_negent'],  
            'mse_recon': mse_recon, 
            'mse_pred': mse_pred,      
            'ARI': ARI_scores.mean(),
            'F-ARI': FARI_scores.mean(),
            'latent_sd_avg': (2*logsd.exp()).sqrt().mean(),
        }, sync_dist=True) 

        for i in range(self.D.shape[0]):
            for j in range(self.D.shape[1]):
                self.log('D/{}-{}'.format(i,j), self.D[i,j], sync_dist=True)
       
        val_out_dict = {                
                'total_loss': out_dict['total_loss'],
                'loss_steps': out_dict['loss_steps'],              
                'loss_steps_per_im': out_dict['loss_steps_per_im'],  
                }
    
        if self.with_goal_net:                
            val_out_dict['goal_loss_per_latent'] = goal_loss_per_latent
        if batch_idx==0:
            val_out_dict['rec'] = out_dict['rec']
            val_out_dict['mask'] = out_dict['mask']['prob']
            val_out_dict['orig_ims'] = ims            
            if self.with_goal_net:
                val_out_dict['pred_goal_ims'] = pred_goal_ims                    

        return val_out_dict
   

    def validation_epoch_end(self, outputs) -> None:
        K = outputs[0]['mask'].shape[2]        
        N, F, C, H, W = outputs[0]['orig_ims'].shape

        if not self.logger==None:                
            
            F_inf = F-self.val_predict        

            orig_vids = outputs[0]['orig_ims']            
            rec_vids = outputs[0]['rec']
            mask_vids = outputs[0]['mask'].transpose(1,2).reshape(N,F_inf*K,1,H,W).expand(-1,-1,3,-1,-1)
            vids = torch.cat((orig_vids[:,:F_inf], rec_vids[:,:F_inf], mask_vids), 1)
            
            rec_vid_grid = make_grid(vids.view(N*F_inf*(2+K),C,H,W), nrow=F_inf)
            
            pred_vids = torch.cat((orig_vids, rec_vids), 1)
            pred_vid_grid = make_grid(pred_vids.view(N*F*2,C,H,W), nrow=F)
                        
            self.logger.experiment.add_image('recons', rec_vid_grid.cpu(), self.current_epoch)        
            self.logger.experiment.add_image('predictions', pred_vid_grid.cpu(), self.current_epoch)        

            if self.with_goal_net:
                pred_goal_ims = outputs[0]['pred_goal_ims']
                goal_grid = torch.cat((orig_vids[:,:F_inf], rec_vids[:,:F_inf], pred_goal_ims), 1)
                goal_grid = make_grid(goal_grid.view(N*F_inf*3,C,H,W), nrow=F_inf)
                self.logger.experiment.add_image('goal_pred', goal_grid.cpu(), self.current_epoch)

                goal_loss_per_latent = torch.stack([foo['goal_loss_per_latent'] for foo in outputs]).to(torch.float32)
                mean_plot = goal_loss_per_latent.mean(0).detach().cpu()
                err_plot  = goal_loss_per_latent.std(0).detach().cpu()/sqrt(goal_loss_per_latent.shape[0])
                fh = plt.figure()
                plt.errorbar(np.arange(mean_plot.shape[0]), mean_plot, err_plot, linestyle='', marker='o')
                plt.xlabel('Latent #')
                plt.ylabel('Goal loss')              
                
                self.logger.experiment.add_figure('goal_loss_per_latent', fh, self.current_epoch, close=True)                
                plt.close(fh)
              

            loss_steps = torch.stack([foo['loss_steps'] for foo in outputs]).to(torch.float32)
            mean_plot = loss_steps.mean(0).detach().cpu()
            err_plot  = loss_steps.std(0).detach().cpu()/sqrt(loss_steps.shape[0])
            fh = plt.figure()
            plt.errorbar(np.arange(mean_plot.shape[0]), mean_plot, err_plot)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')

            self.logger.experiment.add_figure('loss_steps', fh, self.current_epoch, close=True)  
            plt.close(fh)
                
            fig = plt.figure(figsize=(10,10))                
            pos = plt.imshow(self.D.t().detach().cpu())                
            fig.colorbar(pos)
            self.logger.experiment.add_figure('D', fig, self.current_epoch, close=True)                 
            plt.close(fig)
        
        return super().validation_epoch_end(outputs)

    def predict_step(self, batch, batch_idx):
        with torch.inference_mode(False):
            
            torch.set_grad_enabled(True) #Pytorch lightning turns this off by default during prediction, but we need it here for the internal gradient computation in forward()        

            if not self.interactive:
                
                ims, _, action_fields = batch[:]

                if self.val_predict==0:
                    end_idx = ims.shape[1]
                else:
                    end_idx = -self.val_predict
                
                if action_fields is None:
                    pass_action_fields=None
                else:                    
                    pass_action_fields=action_fields[:,:end_idx]
                         
                out_dict = self.forward(
                    ims[:,:end_idx],
                    action_fields=pass_action_fields,        
                    num_predict=self.val_predict,
                    num_inf_steps=self.train_iter_per_timestep,
                    win_size=self.train_win_size)

            else:
                #Interactive              
                out_dict = self.forward(
                    batch,
                    num_predict=self.val_predict,
                    num_inf_steps=self.train_iter_per_timestep,
                    win_size=self.train_win_size,
                    maxF=self.maxF)                
                end_idx = self.maxF                    

                if self.with_goal_net:
                    pref_rec,pref_mask,_ = self.decode(out_dict['pref_lambda'], do_sample=False)
                    pref_rec,_ = self.comb_rec(pref_rec, pref_mask)
                    out_dict['pref_rec'] = pref_rec

        torch.set_grad_enabled(False) 

        return out_dict
    
    def configure_optimizers(self):        

        pars = []
        if not self.freeze_percept_net:
            pars.extend([*self.refine_net.parameters(), *self.decoder.parameters(), self.D, *self.action_net.parameters(), self.lambda0])            
        
        if self.with_goal_net:
            pars.extend([*self.goal_net.parameters()])
        
        if len(pars)==0:
            raise Exception('No trainable parameters')
            
        optimizer = torch.optim.Adam(pars, lr=self.learning_rate)
      
        if self.reduceLR_factor>0:
            scheduler = ReduceLROnPlateau(optimizer, factor=self.reduceLR_factor, patience=self.reduceLR_patience, min_lr=self.reduceLR_minlr)            
            return ({'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss_final'}})
        else:
            return optimizer

        
if __name__ == "__main__":

    pass
    

