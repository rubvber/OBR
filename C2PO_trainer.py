import argparse
from tkinter import Y
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from C2PO import C2PO 
from torch.utils.data import DataLoader
from active_dsprites import active_dsprites
from argparse import ArgumentTypeError
import re



def main(args):
    # pl.seed_everything(1235)
    pl.seed_everything(args.random_seed)    


    if isinstance(args.gpus, list):
        if len(args.gpus)==1 and args.gpus[0]<0:
            gpus = abs(args.gpus[0])
        else:
            gpus = tuple(args.gpus)
    else:
        if args.gpus<0:
            gpus = abs(args.gpus)
        else:
            gpus = (args.gpus,)

    active_dsprites_train = active_dsprites(        
        N=256 if args.debug_run else 50000,
        num_frames=args.ad_num_frames,
        action_frames=args.action_frames,
        interactive=args.interactive,
        pos_smp_stats=(0.2,0.8),            
        include_bgd_action=args.include_bgd_action,        
        bounding_actions=args.ad_bounding_actions,
        rule_goal = args.ad_rule_goal,
        rule_goal_actions= args.ad_rule_goal_actions,        
        )
    active_dsprites_val = active_dsprites(
        include_masks=True,         
        N=64 if args.debug_run else 10000,
        interactive=args.interactive,
        rand_seed0=50000+1234+4242, #As we want to avoid duplicating any indices from the training set, we need to add its randseed0 and its size (plus some safety margin)
        num_frames=args.ad_val_num_frames,
        action_frames=args.action_frames,
        pos_smp_stats=(0.2,0.8),         
        include_bgd_action=args.include_bgd_action,                
        bounding_actions=args.ad_bounding_actions if args.val_predict==0 else False,
        rule_goal = args.ad_rule_goal,
        rule_goal_actions= args.ad_rule_goal_actions,        
        ) 

    
    if args.val_batch_size==None:
        val_batch_size=args.batch_size
    else:
        val_batch_size=args.val_batch_size

    train_loader = DataLoader(active_dsprites_train, args.batch_size, num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True) 
    val_loader = DataLoader(active_dsprites_val, val_batch_size, num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True)
    
    pass_args = {        
        'K': 4,        
        'learning_rate': args.learning_rate,
        'im_var': args.im_sd**2,         
        'beta': args.beta, 
        'sigma_s': args.pred_sd, 
        'val_predict': args.val_predict,        
        'reduceLR_factor': args.reduceLR_factor, 
        'reduceLR_patience': args.reduceLR_patience, 
        'reduceLR_minlr': args.reduceLR_minlr, 
        'train_iter_per_timestep': args.train_iter_per_timestep, 
        'n_latent': args.n_latent,         
        'sigma_chi': args.sigma_chi,                
        'init_percept_net_path': args.init_percept_net_path,
        'init_action_net_path': args.init_action_net_path,
        'freeze_percept_net': args.freeze_percept_net,
        'freeze_action_net': args.freeze_action_net,        
        'init_tau': args.init_tau,
        'ar_tau': args.ar_tau,
        'min_tau': args.min_tau,                
        'D_init_sd': args.D_init_sd,        
        'train_win_size': args.train_win_size,        
        'interactive': args.interactive,
        'maxF': args.maxF,
        'action_generation_type': args.action_generation_type,
        'action_frames': args.action_frames,        
        'num_mask_samples': args.num_mask_samples,                
        'with_goal_net': args.with_goal_net,                    
        'init_goal_net_path': args.init_goal_net_path,        
    }


    assert not ((args.resume_from_checkpoint or args.load_model_path is not None) and args.init_percept_net_path is not None), 'Cannot combine init_percept_path with resume_from_checkpoint or load_model_path'
    
    if (args.load_model_path is not None) or args.resume_overrideLR:      
        print('\n-----EXISTING MODEL WILL BE LOADED BUT OPTIMIZER STATE AND CERTAIN HYPERPARAMETERS OVERWRITTEN-----\n')  
        if args.load_model_path is not None:
            assert not args.resume_overrideLR, 'Can''t use both load_model_path and resume_overrideLR - the two are mutually exclusive'            
            load_path=args.load_model_path
        elif args.resume_overrideLR:      
            print('\n-----RESUMING TRAINING WITH NEW LEARNING RATE SETTINGS-----\n')      
            pass_args = {
                'learning_rate': args.learning_rate,
                'reduceLR_factor': args.reduceLR_factor,
                'reduceLR_patience': args.reduceLR_patience,
                'reduceLR_minlr': args.reduceLR_minlr
            }
            load_path=args.resume_from_checkpoint        
        model = C2PO.load_from_checkpoint(load_path, **pass_args)
        ckpt_path=None
    else:
        model = C2PO(**pass_args)
        ckpt_path=args.resume_from_checkpoint   

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_cumul",
        save_top_k=1, 
        mode="min",
        save_last=True         
    )    
    callbacks = [checkpoint_callback]
    if args.reduceLR_factor>0:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

    model.all_calling_args = args

    trainer = pl.Trainer(devices=gpus, accelerator="gpu", strategy='ddp' if len(gpus)>1 else None, precision=args.precision, max_epochs=args.max_epochs, callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger('./C2PO_logs/'), gradient_clip_val=args.gradient_clip_val, gradient_clip_algorithm='norm', resume_from_checkpoint=ckpt_path,
        accumulate_grad_batches= args.accumulate_grad_batches, track_grad_norm=2, num_nodes=1)
    
    trainer.fit(model, train_loader, val_loader)    
    

def str2bool(v):
    # code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parseNumList(string):
    # code from: https://stackoverflow.com/a/6512463/2660885
    m = re.match(r'(\d+)(?:-(\d+))?$', string)    
    if not m:
        raise ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IODINE")

    parser.add_argument('--max_epochs', default=200, type=int, "Maximum number of epochs to train for.")
    parser.add_argument('--K', default=4, type=int, help="Number of slots in the model")
    parser.add_argument('--data_str', default='3sprites-3frames', type=str, help='Identifier string to find (dsprites) data')    
    parser.add_argument('--resume_from_checkpoint', default=None, help='Checkpoint from which to resume training')
    parser.add_argument('--gpus', default=0, nargs='+', type=int, help='Indices of GPU devices to use')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for each invidual GPU (if using multiple GPUs, effective batch size is this number multiplied by number of GPUs)')
    parser.add_argument('--learning_rate', default=3e-4, type=float, help='Learning rate (automatically scaled so this is the effective learning rate given the effective batch size)')
    parser.add_argument('--im_sd', default=0.3, type=float, help='Image noise variance hyperparameter')
    parser.add_argument('--pred_sd', default=0.1, type=float, help='SD of prediction errors in the dynamics')    
    parser.add_argument('--beta', default=1.0, type=float, help='Beta coefficient that multiplies the reconstruction (neg ll) term in the loss')    
    parser.add_argument('--precision', default=16, type=int, help='Precision of tensors')
    parser.add_argument('--val_batch_size', default=32, type=int, help='Batch size for validation - if not set this defaults to the value specified in batch_size')
    parser.add_argument('--val_predict', default=4, type=int, help='Number of frames to predict ahead at validation (output will be logged as images). If set to 0 no prediction is generated at all.')    
    parser.add_argument('--update_vars', default=0.0, type=float, help='Coefficient by which image and prediction variances are updated after each batch. Coefficient is the weight on the current batch. If 0, then no update is done.')
    parser.add_argument('--reduceLR_factor', default=0.0, type=float, help='Reduce LR by this factor when it hits a plateau in the validation loss (0 means no LR reduction)')
    parser.add_argument('--reduceLR_patience', default=10, type=int, help='Reduce LR after validatoin loss hasn''t improved for this many epochs. Ignored if reduceLR_factor=0.')
    parser.add_argument('--reduceLR_minlr', default=3e-6, type=float, help='Lower bound on (absolute) learning rate for LR scheduler. Ignored if reduceLR_factor=0.')        
    parser.add_argument('--train_iter_per_timestep', default=4, type=int, help='How many refinement iterations to complete before adding another frame to the inference time window')    
    parser.add_argument('--n_latent', default=16, type=int, help='Number of variables in the latent space')
    parser.add_argument('--resume_overrideLR', default=False, type=str2bool, help='When resuming from checkpoint, load model weights and hyperparameters but override learning rate settings to those specified here.')        
    parser.add_argument('--sigma_chi', default=0.3, type=float, help='Value of sigma_chi hyperparameter')
    parser.add_argument('--load_model_path', default=None, type=str, help='Path of a model to load weights from. This is distinct from resume_from_checkpoint which also loads all hyperparameters and optimizer states.' + 
        'By contrast, load_model_path does not load hyperparameters or optimizer states. Hyperparameters will be set to their specified or default values.')        
    parser.add_argument('--init_tau', default=2.0, type=float, help='Value to initialize the tau parameter to, if action_net=True')
    parser.add_argument('--ar_tau', default=3e-6, type=float, help='Anneal rate for tau parameter, if action_net=True')
    parser.add_argument('--min_tau', default=0.5, type=float, help='Minimum value of tau parameter, if action_net=True')
    parser.add_argument('--init_percept_net_path', default=None, type=str, help='Path to perception net, if None then decoder and refinement networks are initialized randomly. Do not use together with load_model_path or resume_from_checkpoint')
    parser.add_argument('--init_action_net_path', default=None, type=str, help='Path to action net - see help for init_percept_net_path.')
    parser.add_argument('--freeze_percept_net', default=False, type=str2bool, help='Freeze parameters of the refine net and decoder')
    parser.add_argument('--freeze_action_net', default=False, type=str2bool, help='Freeze parameters of the action net')    
    parser.add_argument('--random_seed', default=1235, type=int, help='Random seed')    
    parser.add_argument('--D_init_sd', default=0.02, type=float, help='SD of Normal distribution used to initialize D matrix')    
    parser.add_argument('--train_win_size', default=2, type=int, help='Size of the sliding inference window')
    parser.add_argument('--ad_num_frames', default=4, type=int, help='Number of frames to simulate in active-dsprites')
    parser.add_argument('--ad_val_num_frames', default=8, type=int, help='Number of frames to simulate in active-dsprites, for validation data')    
    parser.add_argument('--gradient_clip_val', default=5.0, type=float, help='Gradient clipping value')
    parser.add_argument('--interactive', default=False, type=str2bool, help='Run in interactive mode?')
    parser.add_argument('--maxF', default=4, type=int, help='If running in interactive mode, the maximum numnber of frames to simulate')
    parser.add_argument('--action_generation_type', default='random', type=str, help='If running in interactive mode, the method to generate actions onine (\'random\' or \'goal-directed\')')
    parser.add_argument('--action_frames', default=(2,), type = int, nargs='+', help='The frames in which to perform actions.')
    parser.add_argument('--num_mask_samples', default=1, type=int, help='How many mask samples to take in order to compute the action loss')    
    parser.add_argument('--include_bgd_action', default=True, type=str2bool, help='Include action on background when sampling actions')        
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)        
    parser.add_argument('--debug_run', default=False, type=str2bool, help='Do quick debug run with fewer train & val batches')        
    parser.add_argument('--ad_bounding_actions', default=False, type=str2bool, help='Use bounding actions in active dsprites')
    parser.add_argument('--with_goal_net', default=False, type=str2bool, help='Include goal net to learn transform from current to target states')
    parser.add_argument('--ad_rule_goal', default=None, type=str, help='Rule-based goal to sample from in active-dsprites')        
    parser.add_argument('--ad_rule_goal_actions', default=False, type=str2bool, help='Whether to generate actions towards goals in active-dsprites')
    parser.add_argument('--init_goal_net_path', default=None, type=str, help='Path to checkpoint file containing an IODINE network state from which to load the goal net')
    


    args = parser.parse_args()
      
    main(args)

