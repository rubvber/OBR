import cairo, torch, random
import numpy as np
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader
from math import ceil, floor
import time
from torch.nn import functional
from shapely.geometry import Polygon, LineString
from shapely import affinity
from shapely.ops import nearest_points
import glob, re
from matplotlib import pyplot as plt
import pymunk

# def heart(t):
#     # Copied from https://github.com/deepmind/dsprites-dataset/issues/2
#     x = 16*np.sin(t)**3.
#     y = 13*np.cos(t) - 5*np.cos(2*t) -2*np.cos(3*t) - np.cos(4*t)
#     return x, -y #Returning -y makes it so the heart is upright by default

def heart(t):
    # Copied from https://github.com/deepmind/dsprites-dataset/issues/2
    
    x = 16*np.sin(t)**3.
    y = 13*np.cos(t) - 5*np.cos(2*t) -2*np.cos(3*t) - np.cos(4*t)
    
    y = -y
    x -= x.min() #Normalize to range of [0, 1] (width and height)
    y -= y.min()
    x /= x.max()
    y /= y.max()
    x -= 0.5
    y -= 0.5

    return x, y #Returning -y makes it so the heart is upright by default

def ellipse(t):
    x = 0.5*np.cos(t)
    y = 0.25*np.sin(t)

    return x,y

def collision_shape(shape_fun, dx=0, dy=0, scale=1.0, rot=0):
    t = np.linspace(0, np.pi*2, 50)
    x,y = shape_fun(t)

    if dx==0 and dy==0 and scale==1.0 and rot==0:
        return x, y

    if rot !=0:      
        new_x = x * np.cos(rot) - y * np.sin(rot)
        new_y = x * np.sin(rot) + y * np.cos(rot)
        x = new_x
        y = new_y
        
    x = x*scale + dx
    y = y*scale + dy    

    return x,y

def square(t):
    x = np.array([.0, 0, 1, 1])-0.5
    y = np.array([.0, 1, 1, 0])-0.5
   
    return x,y


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
        # batch_size = 8,
        num_sprites= 3,        
        scale_min = 1.0/6, #Minimum object scale (1.0 is image size)
        scale_max = 1.0/3, #Maximum objec scale
        v_sd = 4/64, #SD of velocity distribution
        a_sd = 4/64, #SD of action distribution
        # pref_mean = (0.5,0.5), #Mean of position preference distribution
        pref_mean = (0.15,0.15), #Mean of position preference distribution
        pref_sd = (0.1,0.1), #SD of position preference distribution        
        interactive = True, #Use the class in interactive mode
        N = 50000, #Size of dataset
        rand_seed0 = 1234, #Offset for idx-dependent random seeds from which data are generated (for reproducibility)
        num_frames = None, #If not running interactive, this is how many frames will be simulated
        action_frames = None, #If not running interactive, sample action at the end of this frame
        data = None, #If not running interactive, we can initialize an environment with specified data (a tuple of (sprite_data, bgcolor)). If left to None, data will be generated randomly
        include_masks = True,
        include_action_fields = True,
        include_prior_pref = True,
        pos_smp_method = 'uniform', #How to sample positions (can be 'uniform' or 'normal')
        pos_smp_stats  = (0.0, 1.0), #Statistics of distribution to sample positions from. For uniform, this is (min, max). For normal, this is (mean, sd).
        use_legacy_action_timing = False,
        sample_central_action_location = False, #Sample action locations near the center of the (visible) object
        include_bgd_action = True, #Include action on bgd pixel when sampling actions
        actions_per_frame = -1,
        pref_type = 'single',
        sample_from_pref = False,
        bounding_actions = False,
        bounding_events = False,
        rule_goal = None, #This can be set to a string that sets a rule-based goal that will be achieved by the end of the sequence
        rule_goal_actions = None,
        withhold_combs = None,
        scenario = None,            
        with_collisions = False,
        verbose = False,
        collision_threshold = 0.005,
        collision_debug_images = False,
        ):

        self.im_size=im_size
        self.device=device
        # self.batch_size=batch_size
        self.num_sprites=num_sprites
        self.v_sd = v_sd
        self.a_sd = a_sd
        
        self.curr_masks=None
        self.pref_mean = pref_mean
        self.pref_sd = pref_sd
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
        self.include_prior_pref = include_prior_pref

        self.pos_smp_method = pos_smp_method
        self.pos_smp_stats  = pos_smp_stats

        self.use_legacy_action_timing = use_legacy_action_timing
        self.sample_central_action_location = sample_central_action_location

        self.include_bgd_action = include_bgd_action
        self.actions_per_frame = actions_per_frame

        self.pref_type=pref_type
        self.sample_from_pref = sample_from_pref

        self.bounding_actions = bounding_actions
        self.bounding_events  = bounding_events

        self.rule_goal = rule_goal 
        self.rule_goal_actions = rule_goal_actions
        self.withhold_combs = withhold_combs

        self.scenario=scenario

        self.with_collisions = with_collisions
        self.verbose = verbose
        self.collision_threshold = collision_threshold
        self.collision_debug_images = collision_debug_images
        
        # assert not ((rule_goal is not None) and interactive), 'Cannot have rule_goal and interactive at the same time' 

        if self.sample_from_pref:
            self.v_sd=0
            self.a_sd=0


        # Initialize heart shape        
        self.heart_c = np.array(heart(np.linspace(0,2*np.pi, 100))).T 
        # heart_c = np.array(heart(np.linspace(0,2*np.pi, 100))).T torch
        # heart_c -= heart_c.min(0) #Normalize to range of [0, 1] (width and height)
        # heart_c /= heart_c.max(0)
        # heart_c -= 0.5 #Shift to [-0.5, 0.5]
        # self.heart_c = heart_c
        
        if interactive and data is not None:              
            assert isinstance(data, tuple) or isinstance(data, list), 'Input data should be tuple or list of (sprite_data, bgcolor)'
            self.sprite_data = data[0].view(-1,self.num_sprites,10)  
            self.bgcolor = data[1].view(-1)            
            self.device=self.sprite_data.device

        if self.sample_central_action_location:
            self.xy_grid = torch.stack(torch.meshgrid(torch.arange(1,im_size+1, device=self.device), torch.arange(1,im_size+1, device=self.device), indexing='ij'),-1)

    #@profile
    def is_colliding(self, obj1, obj2, pos1=None, pos2=None, scale1=None, scale2=None):
        #Input positions should be not the shape centroids but the positions from sprite_data, as these are the centroids of the bounding squares
        # if obj2.type=='LineString':
        #     pass
        # else:
        #     if not any((pos1 is None, pos2 is None, scale1 is None, scale2 is None)):
        #         d = ((pos2-pos1)**2).sum().sqrt()
        #         if d < (scale1/2 + scale2/2 + self.collision_threshold*2):
        #             return False

        
        if obj1.intersects(obj2):
            return True
        
        # cp = nearest_points(obj1, obj2) 
        # if cp[0].distance(cp[1]) < self.collision_threshold:
        #     return True
        
        return False

    #@profile        
    def make_collision_models(self, shapes, scales, angles, pos):
        objs = []
        K = shapes.shape[0]
        true_pos = torch.zeros_like(pos)
        t = np.linspace(0, 2*np.pi,100)
        for k in range(K):
            if shapes[k]==1:                        
                obj = Polygon([(-0.5,-0.5),(-0.5, 0.5),(0.5,0.5),(0.5,-0.5)])                        
            elif shapes[k]==2:
                x,y = ellipse(t)
                obj = Polygon(np.stack((x,y),-1))
            elif shapes[k]==3:
                x,y = heart(t)
                obj = Polygon(np.stack((x,y),-1))

            obj = affinity.scale(obj, xfact=scales[k], yfact=scales[k])
            obj = obj.buffer(self.collision_threshold/2)
            obj = affinity.rotate(obj, angle=angles[k].item(), use_radians=True)
            obj = affinity.translate(obj, xoff=pos[k,0], yoff=pos[k,1])
            objs.append(obj)
            true_pos[k] = torch.tensor([obj.centroid.x, obj.centroid.y]) #We need the true centers of mass to compute the collision dynamics
        
        return objs, true_pos

    def make_collision_sim(self, sprite_data):
        #Makes collision simulation for a single environment instance, i.e. sprite_data should be 2-D with first dimension equal to K

        space = pymunk.Space()
        space.gravity = 0,0
        borders = (
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
            ((1, 0), (1, 1)),
        )

        for border in borders:
            body = pymunk.Body(pymunk.Body.STATIC)    
            shape = pymunk.Segment(body, *border, 0.0)
            shape.elasticity = 1.0
            shape.mass = 1e40
            space.add(body, shape)

        shape_funs = (square, ellipse, heart)

        shapes, scales, angles, pos, vel = torch.split_with_sizes(sprite_data[:,3:], [1,1,1,2,2], dim=-1)
                
        for k in range(self.num_sprites):
            body = pymunk.Body() 
            shape_vert = collision_shape(shape_funs[int(shapes[k].item())-1], pos[k,0].item(), pos[k,1].item(), scales[k].item(), angles[k].item())
            shape_vert = [tuple(x) for x in np.array(shape_vert).T]
            shape = pymunk.Poly(body, shape_vert)                    
            shape.mass = 1.0
            shape.elasticity = 1.0
            body.velocity = vel[k,0].item(), vel[k,1].item()
            space.add(body, shape)
            body.moment = float('inf') #Infinite moment of inertia to (effectively) prevent rotations - have to set this after adding shape to body as moment gets automatically calculated at that point
        
        return space


    #@profile
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
            # actions = torch.einsum('bijk,bjkl->bil', self.curr_masks*1.0, actions)

            new_sprite_data=self.sprite_data.clone()            
            new_sprite_data[:,:,-2:] += actions #Increment velocity by acceleration

        if not self.with_collisions:
            new_sprite_data[:,:,6:8] += new_sprite_data[:,:,8:] #Increment position by velocity
        else:
            #Simulate in micro-time
            N = self.sprite_data.shape[0]            
            N_steps = 50

            for j in range(N):
                
                shapes, scales, angles, pos, vel = torch.split_with_sizes(new_sprite_data[j,:,3:], [1,1,1,2,2], dim=-1)
                space = self.make_collision_sim(new_sprite_data[j])
             
                for _ in range(N_steps):           
                    space.step(1/N_steps)            

                displacement = torch.tensor([b.position for b in space.bodies[4:]], device=self.device) #Body positions always start out at 0,0 (regardless of collision shape positions) so position at the end of the simulation is equal to the accumulated displacement                
                pos += displacement
                vel = torch.tensor([b.velocity for b in space.bodies[4:]], device=self.device)

            new_sprite_data[j,:,3:] = torch.cat((shapes, scales, angles, pos, vel),dim=-1)        

            del space            
            
 
        if self.bounding_events:
            new_pos = new_sprite_data[:,:,6:8]
            too_high = new_pos>0.95
            too_low  = new_pos<0.05
            oob = torch.logical_or(too_high, too_low)
            if oob.any():
                velocity = new_sprite_data[:,:,8:]                        
                desired_velocity = velocity.clone()
                desired_velocity[oob] = -velocity[oob] #Invert velocities                       
                new_sprite_data[:,:,8:] = desired_velocity

            new_pos[too_high] = 0.95-(new_pos[too_high]-0.95) #Reflect positions in the boundary
            new_pos[too_low]  = 0.05+(0.05-new_pos[too_low])
            new_sprite_data[:,:,6:8] = new_pos
            
                       
        
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

        masks = (masks*torch.arange(1,self.num_sprites+1, dtype=torch.int64, device=self.sprite_data.device).view(1,self.num_sprites,1,1)).sum(1) #These masks are categorical

        if self.include_prior_pref:
            if self.sample_from_pref:
                prefs = torch.ones(batch_size, device=self.device)
            else:                
                if self.pref_type=='single':
                    prefs = norm.pdf(self.sprite_data[:,:,6:].cpu(), loc=self.pref_mean+(0.0,0.0), scale=self.pref_sd*2).prod((1,2))                 
                elif self.pref_type=='mix':                    
                    raise Exception('Mixture preference computation is not correctly implemented')
                    shapes = self.sprite_data[:,:,3].to(torch.long).squeeze()-1
                    pref_means = torch.tensor(self.pref_mean)[shapes]
                    if pref_means.ndim<3: pref_means=pref_means.unsqueeze(0)
                    pref_means = torch.cat((pref_means, torch.zeros((pref_means.shape[:2]+ (2,)))),-1)
                    prefs = norm.pdf(self.sprite_data[:,:,6:].cpu()-pref_means, loc=0.0, scale=self.pref_sd*2).prod((1,2)) #THIS IS WRONG - should be computed under the marginal dist rather than conditioning on the selected shape
                prefs = torch.tensor(prefs, device=self.device) 
        else:
            prefs = torch.tensor([], device=self.device)
        
        return ims.squeeze(0), masks.squeeze(0), prefs.squeeze(0)        

    
    def draw_obj(self, shape, scale=1.0, angle=0.0, x=0.0, y=0.0, render_size=None):
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

        if render_size==None: render_size=self.im_size
        
        surface = cairo.ImageSurface(cairo.FORMAT_A8, *(render_size,)*2)  
        cr = cairo.Context(surface)
        cr.scale(*(render_size,)*2)
        cr.set_antialias(cairo.ANTIALIAS_NONE)        
        # cr.set_antialias(cairo.ANTIALIAS_BEST)


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
        t = torch.tensor(surface.get_data(), device=self.device).view(*(render_size,)*2)
        t = torch.div(t, 255, rounding_mode='floor')
        # t //= 255 #// means integer division

        del surface, cr
        
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
        if self.rule_goal == 'HeartLeftRight':
            heart_idx = self.sprite_data[0,:,3]==3
            goal = self.sprite_data.clone()
            goal[:,:,8:] = 0 #Target velocity is always 0
            if heart_idx.any():
                goal[:,heart_idx.logical_not(),6] = 0.1 #All non-hearts go to the left                
            else:   
                goal[:,heart_idx.logical_not(),6] = 0.9 #All non-hearts go to the right
        elif self.rule_goal == 'TopMiddleBottom':
            goal = self.sprite_data.clone()
            goal[:,:,8:] = 0 #Target velocity is always 0
            goal[:,:,6] = 0.5 #Preferred x-coordinate is always 0.5
            goal_ypos = [0.15, 0.5, 0.85]
            for s_idx in range(1,4):
                this_idx = self.sprite_data[0,:,3]==s_idx
                goal[:,this_idx,7] = goal_ypos[s_idx-1]
        elif self.rule_goal == 'HeartLR+TMB':            
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
                
    def get_random_action(self, include_bgd_action=True, action_sd=4.0/64, actions_per_frame=-1, return_action_fields=True):
        '''
        Sample a random action for each sprite, using the true masks of those sprites. 
        '''
        batch_size = self.sprite_data.shape[0]
        # action_field = torch.zeros(batch_size, 2, *(self.im_size,)*2, device=self.device)        
        if actions_per_frame==-1:
            actions_per_frame = self.num_sprites
            if include_bgd_action: actions_per_frame+=1
            
        if hasattr(torch, 'argwhere'):
            aw_fun = torch.argwhere
        else:
            aw_fun = lambda x: np.argwhere(x.cpu()).T
            
        if include_bgd_action: bg_mask = torch.logical_not(self.curr_masks.any(1))   

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
        
        if (self.rule_goal is not None and not self.interactive) and self.rule_goal_actions is not None:
            self.goal = self.get_goal_state()            
                    
            if self.rule_goal_actions == 'lsq':
                onevec = torch.ones(self.num_frames-1, device=self.device)    
                d = torch.arange(1, self.num_frames, device=self.device)
                row_idx, col_idx = torch.meshgrid(torch.arange(1, (self.num_frames-1)*2+1, device=self.device),torch.arange(1, self.num_frames, device=self.device), indexing='ij')
                omega_d = torch.max(torch.tensor(0, device=self.device), (row_idx+1)/2-col_idx+1).to(torch.int)*(row_idx%2) + (col_idx<=(row_idx/2))*(1-row_idx%2)                        
                Ww = torch.kron(omega_d, torch.eye(2, device=self.device)) 

                foo = torch.kron(onevec, self.goal[:,:,6:]-self.sprite_data[:,:,6:]) - torch.kron(d.view(1,1,-1), torch.cat((self.sprite_data[:,:,8:], torch.zeros_like(self.sprite_data[:,:,8:])), -1))

                obj_actions = (foo.squeeze()@(torch.inverse(Ww.T@Ww + torch.eye(Ww.shape[1], device=self.device)*50)@Ww.T).T).view(1,self.num_sprites, -1, 2)
            elif self.rule_goal_actions == 'end':
                '''
                In this case we'll just do two goal-directed actions: one to instantly move the objects to their desired locations, 
                and one to stop them in their tracks. This behavior will be incorporated into the per-frame action selection, so that 
                it can be combined with random actions and bounding actions.
                '''


        if self.interactive:
            # ims, _ = self.render()
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
            if self.include_prior_pref:
                prefs = torch.zeros((self.num_frames, ), device=self.device)
            else:
                prefs = torch.empty(0, device=self.device)
            

            for f in range(self.num_frames):
                ims[f], mask, pref = self.render()
                if self.include_masks: masks[f] = mask 
                if self.include_prior_pref: prefs[f] = pref
                
                if f < (self.num_frames-1): #If this isn't the final frame, then we need to advance the environment by one step (and perhaps simulate an action)                                        
                    action_field = torch.zeros((1, 2, *(self.im_size,)*2), device=self.device)  
                    if self.rule_goal and not self.rule_goal_actions=='end':                        
                        action_field = self.obj_actions_to_action_fields(obj_actions[:,:,f])
                    else:                            
                        obj_actions = torch.zeros(1, self.num_sprites+self.include_bgd_action, 2)      
                        this_do_action=False                                     
                        plan_end_actions=False
                        if f+1-self.use_legacy_action_timing in self.action_frames: this_do_action=True                           
                        if self.rule_goal and self.rule_goal_actions=='end':
                            if f+1-self.use_legacy_action_timing >= self.num_frames-2: 
                                this_do_action=False
                                plan_end_actions=True
                        if this_do_action:
                            obj_actions = self.get_random_action(action_sd=self.a_sd, include_bgd_action=self.include_bgd_action, actions_per_frame=self.actions_per_frame, return_action_fields=False)                        
                            action_field = self.obj_actions_to_action_fields(obj_actions)                    
                            '''
                            Previously, we defined the actions as happening at the end of frame 'f' which would then alter the velocities in the next frame, f+1.
                            Now, we instead define the action as happening at the *start* of a frame, and affecting the velocities in that frame. That is, we basically
                            just shift the 'border' between discrete time points a little - the actual dynamics and update equations dont't really change. To 
                            accommodate this change, we need to store the action we've just generated at index f+1. To maintain backwards compatibility, we'll 
                            add a legacy mode, which shifts it back to f.                         
                            '''                                                
                        
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
                            if f+1-self.use_legacy_action_timing == self.num_frames-2:
                                goal_pos    = self.goal[:,:,6:8]
                                curr_pos    = self.sprite_data[:,:,6:8]
                                curr_v      = self.sprite_data[:,:,8:]                            
                                need_v      = goal_pos-curr_pos                                
                                obj_actions = need_v-curr_v
                            elif f+1-self.use_legacy_action_timing == self.num_frames-1:
                                curr_v      = self.sprite_data[:,:,8:]                            
                                obj_actions = -curr_v
                            else:
                                raise Exception('This should not be possible')
                            action_field = self.obj_actions_to_action_fields(obj_actions)

                    # else:
                        # action_field=None              
                        #In this case the entry in the action_fields tensor will stay all 0s          

                    if self.include_action_fields: action_fields[f+1-self.use_legacy_action_timing*1] = action_field*self.im_size #Multiply by image-size so it's in units of pixels, for backwards compatibility
                    self.step(action_field)                    
            
            torch.set_rng_state(rs) #Restore RNG state we entered with
            return ims, masks, action_fields, prefs

        
    
    def initialize_environment(self, randseed=None, batch_size=1):
        if randseed is not None: torch.manual_seed(randseed)

        cval = torch.tensor([0, 63, 127, 191, 255], device=self.device)
        bgcolor = cval[torch.randint(0, len(cval), (1,), device=self.device)]
        
        
        scales = torch.rand((batch_size, self.num_sprites,1), device=self.device)*self.scale_range + self.scale_min
        orientations = torch.rand((batch_size, self.num_sprites,1), device=self.device)*torch.pi*2            
        
        if self.sample_from_pref:
            if self.pref_type=='mix' or self.pref_type=='copy':           
                assert self.num_sprites==3, 'Sampling from mixture pref distribution only works with 3 sprites currently (future implementations could include K<3 - with K>3 you get heavy occlusions)'                     
                _, shapes = torch.sort(torch.rand(batch_size,3, device=self.device),-1)
                shapes = (shapes+1).view(batch_size, 3)
                
                pref_means = torch.tensor(self.pref_mean)[shapes-1]
                if pref_means.ndim<3: pref_means=pref_means.unsqueeze(0)                
                positions = torch.randn((batch_size, self.num_sprites, 2), device=self.device)*torch.tensor(self.pref_sd,device=self.device).view(1,1,-1) + pref_means
                shapes = shapes.unsqueeze(-1)
            else:
                raise Exception('Sampling from pref not implemented for options other than mixture distirbution')
        else:
            if self.rule_goal=='HeartLeftRight' or self.rule_goal=='HeartLR+TMB':
                #In this case we want to balance the number of trials that have a heart
                #Shapes are encoded as 1: square, 2: ellipse, 3: heart                    
                shapes = torch.zeros(batch_size, self.num_sprites, 1, device=self.device)                   
                has_heart = (torch.rand(batch_size,device=self.device)>0.5)                 
                num_has_heart = has_heart.sum()
                shapes[has_heart.logical_not()] = torch.randint(1, 3, (batch_size-num_has_heart, self.num_sprites,1), device=self.device).to(shapes.dtype) #This sampling excludes hearts entirelyice)>0.5) 
                if self.withhold_combs=='heart+square':
                    shapes[has_heart] = torch.randint(2, 4, (num_has_heart, self.num_sprites,1)).to(shapes.dtype) #In this case there is never a square present if there is also a heart
                elif self.withhold_combs=='square+ellipse':    
                    square_trials = torch.rand(batch_size,device=self.device)>0.5
                    for j in range(batch_size):
                        pr = torch.tensor([0, square_trials[j], square_trials[j].logical_not(), has_heart[j]])*1.0
                        shapes[j,:] = torch.multinomial(pr, num_samples=self.num_sprites, replacement=True).to(shapes.dtype).to(self.device).view(self.num_sprites,1)
                        # print(pr)
                    # if num_has_heart>0:
                        
                    #     if has_square.any():
                    #         shapes[has_square & has_heart] = torch.multinomial(torch.tensor([1.0, 0, 1.0]), (has_square & has_heart).sum()*self.num_sprites, 
                    #                                                     replacement=True).view((has_square & has_heart).sum(), self.num_sprites, 1).to(shapes.dtype).to(self.device)
                    #     if not has_square.all():
                    #         shapes[has_square.logical_not() & has_heart] = torch.multinomial(torch.tensor([0, 1.0, 1.0]), (has_square.logical_not()& has_heart).sum()*self.num_sprites, 
                    #                                                     replacement=True).view((has_square.logical_not() & has_heart).sum(), self.num_sprites, 1).to(shapes.dtype).to(self.device)
                elif self.withhold_combs=='square+ellipse_only':
                    assert self.num_sprites==3, 'square+ellipse_only not implemented for num_sprites other than 3'
                    for j in range(batch_size):
                        if has_heart[j]:
                            shapes[j,:] = (torch.randperm(3)+1).view(1,self.num_sprites,1).to(shapes.dtype).to(self.device) #If there is a heart, then we just want all three shapes
                        else:
                            shapes[j,:] = torch.cat((torch.randperm(2)+1, torch.randint(1,3, (1,)))).view(1,self.num_sprites,1).to(shapes.dtype).to(self.device) #If there is no heart, then we want at least one square and one ellipse
                else:
                    shapes[has_heart] = torch.randint(1, 4, (num_has_heart, self.num_sprites,1), device=self.device).to(shapes.dtype) #For the remaining trials, we first sample normally
                
                if not self.withhold_combs=='square+ellipse_only':
                    shapes[has_heart, torch.randint(self.num_sprites, (num_has_heart,))] = 3.0 #and then set a random shape to be a heart (which it might have been already), thus guaranteeing at least 1 heart in the scene                                
            elif self.rule_goal=='HeartXORSquareLR+TMB':
                assert self.num_sprites>1, 'HeartXORSquareLR+TMB rule shape sampling is incompatible with fewer than 2 sprites'
                #In this case we want to balance four options: (0) Neither squares nor hearts, (1) Square but no heart, (2) Heart but no square, (3) Both heart and square 
                cond = torch.randint(4, (batch_size,), device=self.device) 
                shapes = torch.zeros(batch_size, self.num_sprites, 1, device=self.device)
                shapes[cond==0] = 2 #All ellipses
                shapes[cond==1] = torch.randint(1, 3, ((cond==1).sum(), self.num_sprites,1), device=self.device).to(shapes.dtype) #This sampling excludes hearts entirely                
                shapes[cond==1, torch.randint(self.num_sprites, ((cond==1).sum(),))] = 1 #Then we set one random index in each trial to a square (if it wasn't already)
                shapes[cond==2] = torch.randint(2, 4, ((cond==2).sum(), self.num_sprites,1), device=self.device).to(shapes.dtype) #This sampling excludes squares entirely                
                shapes[cond==2, torch.randint(self.num_sprites, ((cond==2).sum(),))] = 3 #Then we set one random index in each trial to a heart (if it wasn't already)
                if (cond==3).any():                    
                    cond3_shapes = torch.randint(1,4,((cond==3).sum(), self.num_sprites, 1),device=self.device).to(shapes.dtype) 
                    foo = torch.stack([torch.randperm(self.num_sprites, device=self.device) for _ in range((cond==3).sum())], dim=0).unsqueeze(-1) #Sample a random order of sprite indices
                    for j in range(cond3_shapes.shape[0]):
                        cond3_shapes[j,foo[j,0]] = 1 #The first randomly selected sprite becomes a square
                        cond3_shapes[j,foo[j,1]] = 3 #The second randomly selected sprite becomes a heart. 
                        #The rest of the shapes remain as they were.
                    shapes[cond==3] = cond3_shapes
            else:
                if self.withhold_combs=='heart+square': raise Exception('Not implemented')
                shapes = torch.randint(1, 4, (batch_size, self.num_sprites,1), device=self.device)

            if self.pos_smp_method=='uniform':
                positions = torch.rand((batch_size, self.num_sprites, 2), device=self.device)*(self.pos_smp_stats[1]-self.pos_smp_stats[0]) + self.pos_smp_stats[0]
            elif self.pos_smp_method=='normal':
                positions = torch.randn((batch_size, self.num_sprites, 2), device=self.device)*self.pos_smp_stats[1] + self.pos_smp_stats[0]
            else:
                raise Exception("Unknown position-sampling method '" + self.pos_smp_method + "'")
        velocities = torch.randn((batch_size, self.num_sprites, 2), device=self.device)*self.v_sd

        if self.withhold_combs=='heart+lights':
            has_heart = (shapes==3).any(1).squeeze()
            colors = cval[torch.randint(0, len(cval), (batch_size, self.num_sprites, 3), device=self.device)]    
            if has_heart.any():            
                colors[has_heart] = cval[torch.randint(0, ceil(len(cval)/2), (has_heart.sum(), self.num_sprites, 3), device=self.device)]            
        elif self.withhold_combs=='heart+lights_only':
            #The opposite of the above: sample only arrangements with at least one heart and all light colors
            if self.rule_goal is not None:
                assert not ('Heart' in self.rule_goal), 'Illegal combination'            
            shapes[:, torch.randint(3, (batch_size,))] = 3.0 #Set a random shape to be a heart (which it might have been already), thus guaranteeing at least 1 heart in the scene                                
            colors = cval[torch.randint(ceil(len(cval)/2), len(cval), (batch_size, self.num_sprites, 3), device=self.device)]     
        elif self.withhold_combs=='lights':
            colors = cval[torch.randint(0, ceil(len(cval)/2), (batch_size, self.num_sprites, 3), device=self.device)] #Colors sampled from darks only    
        elif self.withhold_combs=='lights_only':
            colors = cval[torch.randint(ceil(len(cval)/2), len(cval), (batch_size, self.num_sprites, 3), device=self.device)] #Colors sampled from lights only    
        else:
            colors = cval[torch.randint(0, len(cval), (batch_size, self.num_sprites, 3), device=self.device)]           

        if self.scenario=='HeartBehindSquare':
            '''
            In this case we want to sample configurations where a heart is behind a square, and the other object is not a heart.
            This scenario will override shape, size, position and velocity values that may have been sampled earlier.
            '''

            #First sample a square and a heart, with the heart behind the square, and then a random other non-heart object
            shapes = torch.zeros(batch_size, self.num_sprites, 1, device=self.device)                   
            shapes[:,-1] = 1 #Square (objects get drawn from first to last index, so the last one drawn is always in front)
            shapes[:,-2] = 3 #Heart
            shapes[:,:-2] = torch.randint(1,3, (batch_size, self.num_sprites-2, 1), device=self.device)

            positions[:,-2] = positions[:,-1] #Set position of heart equal to that of the square
            velocities[:,-2] = velocities[:,-1] #Set velocity of heart equal to that of the square
            sq_scale = scales[:,-1]
            for i in range(batch_size): scales[i,-2] = torch.rand(1)*(sq_scale[i]-self.scale_min) + self.scale_min #Sample a heart size that is smaller than the square in front of it                                      
        elif self.scenario=='OneHeartInFront':
            # Sample a collection of objects with exactly one heart present that is in front.
            shapes = torch.zeros(batch_size, self.num_sprites, 1, device=self.device)                   
            shapes[:,-1] = 3 #Heart (objects get drawn from first to last index, so the last one drawn is always in front)            
            shapes[:,:-1] = torch.randint(1,3, (batch_size, self.num_sprites-2, 1), device=self.device) #Not hearts           

        if self.with_collisions:
            for j in range(batch_size):       
                for _ in range(100): #This just ensures that we cannot get stuck in this loop indefinitely 
                    # objs, _ = self.make_collision_models(shapes[j], scales[j], orientations[j], positions[j])
                    
                    collision_detected = False
                    space = self.make_collision_sim(torch.cat((colors[j], shapes[j], scales[j], orientations[j], positions[j], velocities[j]), -1))                  
                    
                    for k, shape in enumerate(space.shapes[4:]):
                        query_info = space.shape_query(shape)
                        if len(query_info)>0:
                            collision_detected=True
                            #If a collision was detected for this object, we'll just resample its position (no need to resample the ones it collides with)                            
                            if self.pos_smp_method=='uniform':
                                positions[j,k] = torch.rand(2, device=self.device)*(self.pos_smp_stats[1]-self.pos_smp_stats[0]) + self.pos_smp_stats[0]
                            elif self.pos_smp_method=='normal':
                                positions[j,k] = torch.randn(2, device=self.device)*self.pos_smp_stats[1] + self.pos_smp_stats[0]

                        if collision_detected: break #If a collision was detected, we adjusted an object position so we need to check again

                    if not collision_detected: break                                

                del space


        sprite_data = torch.cat((colors, shapes, scales, orientations, positions, velocities), -1)
        

        return sprite_data, bgcolor


if __name__ =="__main__":
    
    # Non-interactive mode (dataset/dataloader behavior):
    # ad_dataset = active_dsprites(        
    #     interactive=False,  
    #     im_size=64*4,
    #     v_sd=0.25/64,
    #     num_frames=64*4,
    #     action_frames=(),         
    #     with_collisions=True,
    #     collision_threshold=0.0001,
    #     # bounding_actions=True,
    #     pos_smp_stats=(0.2,0.8),
    #     # rand_seed0=1234,
    #     rand_seed0=1234+8,
    #     verbose=True,
    #     collision_debug_images=True,
    # )

    ad_dataset = active_dsprites(        
        interactive=False,  
        im_size=64,
        v_sd=4/64,
        num_frames=12,
        action_frames=(2,),         
        with_collisions=True,
        collision_threshold=0.0001,
        # bounding_actions=True,
        pos_smp_stats=(0.2,0.8),
        # rand_seed0=1234,
        rand_seed0=1234+8,
        verbose=True,
        collision_debug_images=True,
    )
         
    train_loader = DataLoader(ad_dataset, batch_size=16, num_workers=0)
    dataiter = iter(train_loader)
    t1 = time.perf_counter()
    ims, masks, action_fields, _ = dataiter.next()
    t2 = time.perf_counter()
    print(t2-t1)

    from PIL import Image
    from torchvision.utils import make_grid
    import imageio
    
    Image.fromarray((make_grid(ims.view(train_loader.batch_size*ad_dataset.num_frames,3,ad_dataset.im_size,ad_dataset.im_size), nrow=ad_dataset.num_frames).permute((1,2,0))*255).to(torch.uint8).numpy()).save('foo_collisions_pymunk.png')

    # for i in range(train_loader.batch_size):
    #     im_list=torch.split(ims[i].squeeze(), 1, dim=0)
    #     im_list=[Image.fromarray((x.squeeze().permute((1,2,0))*255).to(torch.uint8).numpy()) for x in im_list]
    #     imageio.mimsave('foo_collisions_movie_pymunk{}.gif'.format(i), im_list, 'GIF-PIL', fps=10, loop=0)



    # ad_dataset = active_dsprites(        
    #     interactive=False,  
    #     im_size=256,
    #     v_sd=4/64,
    #     num_frames=6,
    #     action_frames=[],
    #     with_collisions=False,
    #     bounding_actions=True,
    #     pos_smp_stats=(0.2,0.8),
    #     rand_seed0=1234+8
    # )
         
    # train_loader = DataLoader(ad_dataset, batch_size=32, num_workers=4)
    # dataiter = iter(train_loader)
    # t1 = time.perf_counter()
    # ims, masks, action_fields, _ = dataiter.next()
    # t2 = time.perf_counter()
    # print(t2-t1)

    # from PIL import Image
    # from torchvision.utils import make_grid	

    
    # Image.fromarray((make_grid(ims.view(train_loader.batch_size*ad_dataset.num_frames,3,ad_dataset.im_size,ad_dataset.im_size), nrow=ad_dataset.num_frames).permute((1,2,0))*255).to(torch.uint8).numpy()).save('ad_examples_4Pablo.png')




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


    

    # ad_dataset = active_dsprites(        
    #     interactive=True,
    #     N=640,
    #     num_frames=4,
    #     action_frames=(2,),         
    #     bounding_actions=True,
    #     rule_goal='HeartLR+TMB',
    #     withhold_combs='square+ellipse_only',
    #     rand_seed0=2000,
    # )

    # train_loader = DataLoader(ad_dataset, batch_size=640, num_workers=4, persistent_workers=True)
    # dataiter = iter(train_loader)
    # a,b = dataiter.next()

    # print('foo')

    # t1 = time.perf_counter()
    # for i in range(10):
    #     ims, masks, action_fields, _ = dataiter.next()
    # t2 = time.perf_counter()
    # print((t2-t1)/10)

    # from PIL import Image
    # from torchvision.utils import make_grid
    
    # Image.fromarray((make_grid(ims.view(16*12,3,64,64), nrow=12).permute((1,2,0))*255).to(torch.uint8).numpy()).save('foo_collisions.png')


