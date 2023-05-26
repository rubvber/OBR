import cairo, torch, random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def heart(t):
    # Copied from https://github.com/deepmind/dsprites-dataset/issues/2
    x = 16*np.sin(t)**3.
    y = 13*np.cos(t) - 5*np.cos(2*t) -2*np.cos(3*t) - np.cos(4*t)
    return x, -y #Returning -y makes it so the heart is upright by default


class active_dsprites(Dataset):
    '''
    Active dsprites environment. Can also behave like a pytorch Dataset object. 
    
    If interactive = False, then the getitem function pre-renders an episode from the environment,
    with a fixed length and a fixed random policy. 

    If interactive = True, then the getitem function returns the initial (randomly generated) state of
    the environment. When running the network we can then use this to instantiate a new environment object.

    In both cases, the getitem method only returns a single item, which is then batched with other items
    by the dataloader. The difference is that, in interactive mode, the batch of environment states can 
    then be used to instantiate a multi-item environment object, which includes a non-singleton batch dimension.
    
    '''
    def __init__(self,
        im_size=64,
        device='cpu',        
        num_sprites= 3,        
        scale_min = 1.0/6, #Minimum object scale (1.0 is image size)
        scale_max = 1.0/3, #Maximum objec scale
        v_sd = 4/64, #SD of velocity distribution
        a_sd = 4/64, #SD of action distribution        
        interactive = True, #Use the class in interactive mode
        N = 50000, #Size of dataset
        rand_seed0 = 1234, #Offset for idx-dependent random seeds from which data are generated (for reproducibility)
        num_frames = None, #If not running interactive, this is how many frames will be simulated
        action_frames = None, #If not running interactive, sample action at the end of this frame
        data = None, #If not running interactive, we can initialize an environment with specified data (a tuple of (sprite_data, bgcolor)). If left to None, data will be generated randomly
        include_masks = True,
        include_action_fields = True,        
        pos_smp_method = 'uniform', #How to sample positions (can be 'uniform' or 'normal')
        pos_smp_stats  = (0.0, 1.0), #Statistics of distribution to sample positions from. For uniform, this is (min, max). For normal, this is (mean, sd).
        use_legacy_action_timing = False,
        sample_central_action_location = False, #Sample action locations near the center of the (visible) object
        include_bgd_action = True, #Include action on bgd pixel when sampling actions        
        bounding_actions = False,        
        rule_goal = None, #This can be set to a string that sets a rule-based goal that will be achieved by the end of the sequence
        rule_goal_actions = False, #Perform goal-directed actions (by oracle agent)?
        ):

        self.im_size=im_size
        self.device=device        
        self.num_sprites=num_sprites
        self.v_sd = v_sd
        self.a_sd = a_sd                
        self.interactive = interactive
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_range = scale_max-scale_min
        self.N = N
        self.rand_seed0 = rand_seed0
        self.num_frames = num_frames
        self.action_frames = action_frames
        self.include_masks = include_masks
        self.include_action_fields = include_action_fields        
        self.pos_smp_method = pos_smp_method
        self.pos_smp_stats  = pos_smp_stats
        self.use_legacy_action_timing = use_legacy_action_timing
        self.sample_central_action_location = sample_central_action_location
        self.include_bgd_action = include_bgd_action        
        self.bounding_actions = bounding_actions
        self.rule_goal = rule_goal 
        self.rule_goal_actions = rule_goal_actions
        
        self.curr_masks=None

        # Initialize heart shape        
        heart_c = np.array(heart(np.linspace(0,2*np.pi, 100))).T 
        heart_c -= heart_c.min(0) #Normalize to range of [0, 1] (width and height)
        heart_c /= heart_c.max(0)
        heart_c -= 0.5 #Shift to [-0.5, 0.5]
        self.heart_c = heart_c
        
        if interactive and data is not None:              
            assert isinstance(data, tuple) or isinstance(data, list), 'Input data should be tuple or list of (sprite_data, bgcolor)'
            self.sprite_data = data[0].view(-1,self.num_sprites,10)  
            self.bgcolor = data[1].view(-1)            
            self.device=self.sprite_data.device

        if self.sample_central_action_location:
            self.xy_grid = torch.stack(torch.meshgrid(torch.arange(1,im_size+1, device=self.device), torch.arange(1,im_size+1, device=self.device), indexing='ij'),-1)

        


    def step(self, actions=None, do_not_update=False): 
        '''
        Actions are supplied as actions fields with an acceleration at each pixel. 

        Can work either in batched or unbatched mode. In the latter case we just temporarily insert a dummy batch dimension
        of size 1, and then remove it again later.

        '''
        
        if actions is not None:            
            
            assert actions.ndim==4, 'Action fields have incorrect dimensions'            

            if self.curr_masks==None:
                '''
                Normally we rely on there already being masks available from a previous render step.
                '''
                _, self.curr_masks, _ = self.render()
                        
            actions = torch.einsum('bijk,bljk->bil', self.curr_masks*1.0, actions)            
            
            new_sprite_data=self.sprite_data.clone()            
            new_sprite_data[:,:,-2:] += actions #Increment velocity by acceleration

        new_sprite_data[:,:,6:8] += new_sprite_data[:,:,8:] #Increment position by velocity
        
        if do_not_update:
            return new_sprite_data
        else:
            self.sprite_data = new_sprite_data
            self.curr_masks = None #As a precaution, clear the current masks if there were any, since they will no longer be accurate (this prevents that we would call step() again before re-rendering the environment, and thus simulate an action based on outdated masks)
        
    def render(self):        
        '''
        We use pycairo only to draw individual object masks, which we then combine manually. This is convenient since we need a 
        separate mask per object anyway, so it doesn't make sense to draw the objects sequentially in pycairo only to then have to
        reconstruct the separate masks somehow. Also, this way, we know exactly how masks are defined and combined.

        If the environment's batch dimension is singleton, then it gets removed when returning observations, masks and
        preferences.
        '''        
        
        batch_size = self.sprite_data.shape[0]
        masks = [self.draw_obj(*foo[0,3:8]) for foo in torch.split(self.sprite_data.view(-1, self.sprite_data.shape[-1]),1)]              
        masks = torch.stack(masks, 0).view(batch_size, self.num_sprites, *(self.im_size,)*2)
        fg_mask = masks.any(1, True)

        masks *= torch.arange(1,self.num_sprites+1, dtype=torch.int64, device=self.sprite_data.device).view(1, self.num_sprites,1,1)
        masks = torch.logical_and(masks==masks.amax(1,True), fg_mask)

        ims = (masks.unsqueeze(2)*self.sprite_data[:,:,:3].view(batch_size, self.num_sprites, 3, 1, 1)).sum(1) #Multiply masks by RGB color triplets and sum over sprites to get color ims with black bgd        
        ims += (torch.logical_not(fg_mask)*self.bgcolor.view(batch_size, 1, 1, 1))
        ims /= 255

        self.curr_masks = masks #These masks are binary

        masks = (masks*torch.arange(1,self.num_sprites+1, dtype=torch.int64, device=self.sprite_data.device).view(self.num_sprites,1,1)).sum(1) #These masks are categorical
        
        return ims.squeeze(0), masks.squeeze(0)

    def draw_obj(self, shape, scale=1.0, angle=0.0, x=0.0, y=0.0):
        '''
        Draws a single object and returns it as a binary tensor. 

        No matter the object, the logic is the same: we first translate, rotate and scale the drawing
        canvas with respect to the image. We then draw the object in its intrinsic coordinate frame with 
        no transformations. The transformation of the canvas takes care of all the transformations we want
        to apply. 

        For each object, the size and position are determined in reference to the smallest square frame that 
        fits the object (in its intrinsic coordinate frame). For squares this is obviously trivial. For ellipses, 
        the size is therefore the length along the major axis, i.e. an ellipse of size 0.8 is as long as the width
        of a 0.8-sized square. Hearts are made to be "square" by default, i.e. their largest vertical extent equals 
        their largest horizontal extent, and so they snugly fit into a square frame. 

        This means that, for squares and ellipses, their position is actually equal to the object's center of mass,
        due to their symmetry. Hearts are not vertically symmetric, however, and therefore their vertical coordinate
        is not equal to the vertical position of the center of mass (instead, it is equal to the center of the square 
        frame in which the heart fits). 

        We use this approach because (1) networks can learn an arbitary position encoding anyway and (2) this is simpler
        to define than it is to calculate the center of mass of complex objects like hearts.      

        Objects are *not* anti-aliased - the reason being that we want the true object masks to be binary. That is, a pixel
        is either occupied or not occupied. Anti-alising would muddy this and thus violate the assumptions of our generativne 
        model, as pixels would end up blending colors around object borders. 
        '''

        if not isinstance(scale, tuple): scale = (scale,)*2
        
        surface = cairo.ImageSurface(cairo.FORMAT_A8, *(self.im_size,)*2)  
        cr = cairo.Context(surface)
        cr.scale(*(self.im_size,)*2)
        cr.set_antialias(cairo.ANTIALIAS_NONE)

        cr.translate(x,y)
        cr.rotate(angle)
        cr.scale(*scale)

        #Order in dsprites is 1: square, 2: ellipse, 3: heart
        if shape==1:
            cr.rectangle(-.5, -.5, 1.0, 1.0)
        elif shape==2:        
            cr.scale(1.0, 0.5) #W:H ratio of the ellipse is 2:1, and default orientation (angle=0) is horizontal (i.e. longest axis aligned with horizontal)
            # This second scaling is baked-in and just changes the ratio. 
            cr.arc(0.0, 0.0, 1.0/2, 0.0, np.pi*2)

        elif shape==3:
            cr.move_to(*self.heart_c[0])    
            for i in range(1,self.heart_c.shape[0]):      
                cr.line_to(*self.heart_c[i])
            cr.close_path()
        else:
            raise Exception('Invalid shape requested')    

        cr.set_source_rgb(1,1,1)
        cr.fill_preserve()
        t = torch.tensor(surface.get_data(), device=self.device).view(*(self.im_size,)*2)
        t = torch.div(t, 255, rounding_mode='floor')        
        
        return t

    def obj_actions_to_action_fields(self, obj_actions):
        '''
        Take a tensor of object actions, N x K x 2, and convert it to a tensor of action fields,
        N x H x W. If K is 1 larger than self.num_sprites, then the last entry is assumed to encode
        the action on the background. 
        '''

        N,K = obj_actions.shape[:2]

        assert K == self.num_sprites or K==self.num_sprites+1, 'Size of obj_actions is incompatible with number of sprites'

        include_bgd_action = True if K==self.num_sprites+1 else False
        if include_bgd_action: bg_masks = torch.logical_not(self.curr_masks.any(1))                

        if hasattr(torch, 'argwhere'):
            aw_fun = torch.argwhere
        else:
            aw_fun = lambda x: np.argwhere(x.cpu()).T

        action_fields = torch.zeros(N, 2, *(self.im_size,)*2, device=self.device)                

        for i in range(N):
            for k in range(K):
                if (obj_actions[i,k]==0).all().item(): continue
                if k < self.num_sprites:
                    this_mask = self.curr_masks[i,k] 
                else:
                    this_mask = bg_masks[i]

                if not this_mask.any().item(): continue
                
                indices = aw_fun(this_mask)

                if self.sample_central_action_location and k<self.num_sprites:
                    num_pix = (this_mask*1.0).sum()
                    masked_xy = (this_mask.unsqueeze(-1)*1.0*self.xy_grid)
                    mean_pos = masked_xy.sum((0,1),True)/num_pix
                    dist_from_mean = ((self.xy_grid-mean_pos)**2).sum(-1).sqrt() 
                    _, order = torch.sort(dist_from_mean[indices[:,0],indices[:,1]])
                    index = indices[order[0]]                    
                else:
                    indices = aw_fun(this_mask)
                    pick = random.randint(0, indices.shape[0]-1)
                    index = indices[pick]

                action_fields[i, :, index[0], index[1]] = obj_actions[i,k]

        return action_fields

    def get_goal_state(self):
        if self.rule_goal == 'HeartLR+TMB':            
            goal = self.sprite_data.clone()
            goal[:,:,8:] = 0 #Target velocity is always 0            
            for j in range(goal.shape[0]):
                heart_idx = self.sprite_data[j,:,3]==3                            
                if heart_idx.any():
                    goal[j,:,6]=0.2
                else:
                    goal[j,:,6]=0.8            
                goal_ypos = [0.2, 0.5, 0.8]
                for s_idx in range(1,4):
                    this_idx = self.sprite_data[j,:,3]==s_idx
                    goal[j,this_idx,7] = goal_ypos[s_idx-1]
        elif self.rule_goal == 'HeartXORSquareLR+TMB':
            goal = self.sprite_data.clone()
            goal[:,:,8:] = 0 #Target velocity is always 0  
            for j in range(goal.shape[0]):
                heart_idx = self.sprite_data[j,:,3]==3                            
                square_idx = self.sprite_data[j,:,3]==1                            
                if heart_idx.any() ^ square_idx.any(): # ^ is XOR
                    goal[j,:,6]=0.2
                else:
                    goal[j,:,6]=0.8            
                goal_ypos = [0.2, 0.5, 0.8]
                for s_idx in range(1,4):
                    this_idx = self.sprite_data[j,:,3]==s_idx
                    goal[j,this_idx,7] = goal_ypos[s_idx-1]         

        return goal
                
    def get_random_action(self, include_bgd_action=True, action_sd=4.0/64, return_action_fields=True):
        '''
        Sample a random action for each sprite, using the true masks of those sprites. 
        '''
        batch_size = self.sprite_data.shape[0]
        
       
        actions_per_frame = self.num_sprites
        if include_bgd_action: actions_per_frame+=1

        obj_actions = torch.zeros(batch_size, self.num_sprites+include_bgd_action, 2, device=self.device)
        for i in range(batch_size):
            action_objects = list(range(actions_per_frame))
            if actions_per_frame < (self.num_sprites+include_bgd_action):
                random.shuffle(action_objects)
                action_objects = action_objects[:actions_per_frame]                             
            for j in action_objects:
                obj_actions[i,j] = torch.randn(2,device=self.device)*action_sd

        if return_action_fields:
            action_fields = self.obj_actions_to_action_fields(obj_actions)
            return action_fields
        else:
            return obj_actions


    def __len__(self):        
        return self.N

    def __getitem__(self, idx):
        '''
        Regardless of whether we're running interactive or not, we need to sample an initial state for the environment.
        To do this reproducibly, we'll temporarily create a random number stream that depends on the index we received
        from the dataloader. 

        '''        
        
        rs = torch.get_rng_state() #We don't want to disturb the rest of the RNG, so we'll store the state we entered with and restore it later
        self.sprite_data, self.bgcolor = self.initialize_environment(idx+self.rand_seed0)
        
        if (self.rule_goal is not None and not self.interactive) and self.rule_goal_actions:
            self.goal = self.get_goal_state()            

        if self.interactive:            
            torch.set_rng_state(rs) #Restore RNG state we entered with
            return self.sprite_data, self.bgcolor
        else:
            # frame x RGB x H x W   
            ims = torch.zeros((self.num_frames, 3, *(self.im_size,)*2), device=self.device)
            if self.include_masks:
                masks = torch.zeros((self.num_frames, *(self.im_size,)*2), device=self.device, dtype = torch.uint8)
            else:
                masks = torch.empty(0, device=self.device)
            if self.include_action_fields:
                action_fields = torch.zeros((self.num_frames, 2, *(self.im_size,)*2), device=self.device)
            else:
                action_fields = torch.empty(0, device=self.device)
                        

            for f in range(self.num_frames):
                ims[f], mask = self.render()
                if self.include_masks: masks[f] = mask                 
                
                if f < (self.num_frames-1): #If this isn't the final frame, then we need to advance the environment by one step (and perhaps simulate an action)                                        
                    action_field = torch.zeros((1, 2, *(self.im_size,)*2), device=self.device)  
                                           
                    obj_actions = torch.zeros(1, self.num_sprites+self.include_bgd_action, 2)      
                    this_do_action=False                                     
                    plan_end_actions=False
                    if f+1 in self.action_frames: this_do_action=True                           
                    if self.rule_goal and self.rule_goal_actions:
                        if f+1 >= self.num_frames-2: 
                            this_do_action=False
                            plan_end_actions=True
                    if this_do_action:
                        obj_actions = self.get_random_action(action_sd=self.a_sd, include_bgd_action=self.include_bgd_action, return_action_fields=False)                        
                        action_field = self.obj_actions_to_action_fields(obj_actions)                    
                                
                    
                    if self.bounding_actions and not plan_end_actions:
                        '''
                        Check whether next frame would see the positions of the objects (nearly) leave the frame. If so, take an action to 
                        prevent this. This action overwrites any otherwise scheduled actions.
                        '''                        
                        pred_sprite_data = self.step(action_field, do_not_update=True)                        
                        too_high = pred_sprite_data[:,:,6:8]>0.95
                        too_low  = pred_sprite_data[:,:,6:8]<0.05
                        oob = torch.logical_or(too_high, too_low)
                        if oob.any():
                            velocity = self.sprite_data[:,:,8:]                        
                            desired_velocity = velocity.clone()
                            desired_velocity[oob] = -velocity[oob]                        
                            req_actions = desired_velocity-velocity
                            if self.include_bgd_action:
                                req_actions=torch.cat((req_actions, torch.zeros(1,1,2, device=self.device)),1)
                            obj_actions[req_actions!=0] = req_actions[req_actions!=0]
                            action_field = self.obj_actions_to_action_fields(obj_actions)                    
                    

                    if plan_end_actions:
                        '''
                        In this case, in the last remaining 2 frames, we just perform 2 actions that get us to our goal state. The first action
                        achieves the target position, and the second cancels out the velocity so that it becomes 0.
                        '''
                        if f+1 == self.num_frames-2:
                            goal_pos    = self.goal[:,:,6:8]
                            curr_pos    = self.sprite_data[:,:,6:8]
                            curr_v      = self.sprite_data[:,:,8:]                            
                            need_v      = goal_pos-curr_pos                                
                            obj_actions = need_v-curr_v
                        elif f+1 == self.num_frames-1:
                            curr_v      = self.sprite_data[:,:,8:]                            
                            obj_actions = -curr_v
                        else:
                            raise Exception('This should not be possible')
                        action_field = self.obj_actions_to_action_fields(obj_actions)                 

                    if self.include_action_fields: action_fields[f+1] = action_field*self.im_size #Multiply by image-size so it's in units of pixels, for backwards compatibility
                    self.step(action_field)                    
            
            torch.set_rng_state(rs) #Restore RNG state we entered with
            return ims, masks, action_fields
        
    
    def initialize_environment(self, randseed=None, batch_size=1):
        if randseed is not None: torch.manual_seed(randseed)

        cval = torch.tensor([0, 63, 127, 191, 255], device=self.device)
        bgcolor = cval[torch.randint(0, len(cval), (1,), device=self.device)]
        
        
        scales = torch.rand((batch_size, self.num_sprites,1), device=self.device)*self.scale_range + self.scale_min
        orientations = torch.rand((batch_size, self.num_sprites,1), device=self.device)*torch.pi*2            
        velocities = torch.randn((batch_size, self.num_sprites, 2), device=self.device)*self.v_sd
        colors = cval[torch.randint(0, len(cval), (batch_size, self.num_sprites, 3), device=self.device)]
        

        if self.rule_goal=='HeartLeftRight' or self.rule_goal=='HeartLR+TMB':
            #In this case we want to balance the number of trials that have a heart
            #Shapes are encoded as 1: square, 2: ellipse, 3: heart                    
            shapes = torch.zeros(batch_size, self.num_sprites, 1, device=self.device)                   
            has_heart = (torch.rand(batch_size,device=self.device)>0.5)                 
            num_has_heart = has_heart.sum()
            shapes[has_heart.logical_not()] = torch.randint(1, 3, (batch_size-num_has_heart, self.num_sprites,1), device=self.device).to(shapes.dtype) #This sampling excludes hearts entirelyice)>0.5)             
            shapes[has_heart] = torch.randint(1, 4, (num_has_heart, self.num_sprites,1), device=self.device).to(shapes.dtype) #For the remaining trials, we first sample normally
            shapes[has_heart, torch.randint(self.num_sprites, (num_has_heart,))] = 3.0 #and then set a random shape to be a heart (which it might have been already), thus guaranteeing at least 1 heart in the scene                                
        elif self.rule_goal=='HeartXORSquareLR+TMB':
            #In this case we want to balance four options: (0) Neither squares nor hearts, (1) Square but no heart, (2) Heart but no square, (3) Both heart and square 
            cond = torch.randint(4, (batch_size,), device=self.device) 
            shapes = torch.zeros(batch_size, self.num_sprites, 1, device=self.device)
            shapes[cond==0] = 2 #All ellipses
            shapes[cond==1] = torch.randint(1, 3, ((cond==1).sum(), self.num_sprites,1), device=self.device).to(shapes.dtype) #This sampling excludes hearts entirely                
            shapes[cond==1, torch.randint(self.num_sprites, ((cond==1).sum(),))] = 1 #Then we set one random index in each trial to a square (if it wasn't already)
            shapes[cond==2] = torch.randint(2, 4, ((cond==2).sum(), self.num_sprites,1), device=self.device).to(shapes.dtype) #This sampling excludes squares entirely                
            shapes[cond==2, torch.randint(self.num_sprites, ((cond==2).sum(),))] = 3 #Then we set one random index in each trial to a heart (if it wasn't already)
            if (cond==3).any():
                shapes[cond==3] = torch.stack([torch.randperm(3, device=self.device) for _ in range((cond==3).sum())], dim=0).unsqueeze(-1) + 1.0 #This also always includes an ellipse
        else:            
            shapes = torch.randint(1, 4, (batch_size, self.num_sprites,1), device=self.device)

        if self.pos_smp_method=='uniform':
            positions = torch.rand((batch_size, self.num_sprites, 2), device=self.device)*(self.pos_smp_stats[1]-self.pos_smp_stats[0]) + self.pos_smp_stats[0]
        elif self.pos_smp_method=='normal':
            positions = torch.randn((batch_size, self.num_sprites, 2), device=self.device)*self.pos_smp_stats[1] + self.pos_smp_stats[0]
        else:
            raise Exception("Unknown position-sampling method '" + self.pos_smp_method + "'")

        sprite_data = torch.cat((colors, shapes, scales, orientations, positions, velocities), -1)

        return sprite_data, bgcolor


if __name__ =="__main__":
    
    # # Non-interactive mode (dataset/dataloader behavior):
    # ad_dataset = active_dsprites(        
    #     interactive=False,
    #     num_frames=12,
    #     action_frames=(2,),         
    #     bounding_actions=True                
    # )
         
    # train_loader = DataLoader(ad_dataset, batch_size=32)
    # dataiter = iter(train_loader)
    # ims, masks, action_fields, _ = dataiter.next()

    # # Interactive mode (environment behavior):
    # ad_dataset = active_dsprites(interactive=True)        
    # train_loader = DataLoader(ad_dataset, batch_size=32)
    # dataiter = iter(train_loader)
    # env_data = dataiter.next()

    # env = active_dsprites(data=env_data, include_prior_pref=False)
    # frame0_ims, frame0_masks, _ = env.render()
    # action_fields = env.get_random_action()
    # env.step(action_fields)
    # frame1_ims, frame1_masks, _ = env.render()

    
    # # Interactive mode (environment behavior):
    # ad_dataset = active_dsprites(interactive=True)
    # train_loader = DataLoader(ad_dataset, batch_size=32)
    # dataiter = iter(train_loader)
    # env_data = dataiter.next()

    # env = active_dsprites(data=env_data, bounding_events=True, include_prior_pref=False)
    # frame0_ims, frame0_masks, _ = env.render()
    # for i in range(100):
    #     action_fields = env.get_random_action()
    #     env.step(action_fields)
    #     frame1_ims, frame1_masks, _ = env.render()


    # Non-interactive mode (dataset/dataloader behavior):
    ad_dataset = active_dsprites(        
        interactive=False,
        num_frames=12,
        action_frames=(2,4,6,8),         
        bounding_actions=True,
        rule_goal='HeartXORSquareLR+TMB',
        rule_goal_actions='end',
        rand_seed0=1234
    )

    ad_dataset.initialize_environment(batch_size=16)
         
    train_loader = DataLoader(ad_dataset, batch_size=32)
    dataiter = iter(train_loader)
    ims, masks, action_fields, _ = dataiter.next()

    from PIL import Image
    from torchvision.utils import make_grid
    
    Image.fromarray((make_grid(ims.view(32*12,3,64,64), nrow=12).permute((1,2,0))*255).to(torch.uint8).numpy()).save('foo.png')


