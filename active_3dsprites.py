from panda3d.core import GraphicsPipe, FrameBufferProperties, WindowProperties
from panda3d.core import PointLight, AmbientLight, VBase4, Material, Texture, Vec4
from panda3d.core import NodePath, VBase3, GraphicsOutput, Camera, loadPrcFileData
from panda3d.core import GraphicsPipeSelection, GraphicsEngine, AntialiasAttrib, ModelPool, RenderState, TexturePool
from panda3d.core import Loader as PandaLoader
import numpy as np
from PIL import Image
import torch, os, gc


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, InterpolationMode
from torchvision.utils import make_grid

class active_3dsprites_dataset(Dataset):
    def __init__(self, ctx: dict = {}):
        
        def_args = {
            'im_size': 64,  
            'num_objs': 3,                       
            'render_size': None,
            'depth_min': 20,
            'depth_max': 30,
            'pos_max_at_min_depth': 4,
            'pos_max_at_max_depth': 6.5,
            'scale_min': 1.5,
            'scale_max': 2.0,
            'rand_seed0': 1234,
            'N': 50000,
            'v_sd': 4/64*10, #4/64 pixels was the value for active_dsprites, which corresponded to a fraction of the frame size. The range of positions is about 10 m on avg across (x,y,z).
            'a_sd': 4/64*10,            
            'v_ang_sd': 5,
            'a_ang_sd': 5,                        
            'action_frames': (2,),
            'interactive': False,
            'episode_length': 4,
            'bounding_actions': True,
            'gpus': [0,],
            'with_rotation': False,          
            'bgcolor': None, #If this is None, then it gets picked randomly - otherwise it is fixed to the given value            
            'include_bgd_action': True,
            'no_depth_motion': False,
            'with_directional_light': True,
            'rule_goal': 'None',
            'goal_frames': 2,
        }

        for key in ctx: assert key in def_args, "Argument '{}' not in list of accepted arguments".format(key)

        for key in def_args:
            if key not in ctx: ctx[key] = def_args[key]

        if ctx['render_size'] is None: ctx['render_size']=ctx['im_size']*8

        if not (isinstance(ctx['gpus'], list) or isinstance(ctx['gpus'], tuple)):
            ctx['gpus']  = [ctx['gpus'],]

        self.ctx = ctx             
        
        
    def __len__(self):
        return self.ctx['N']
    
    def __getitem__(self, idx):           
        '''
        
        '''
                
        rand_seed = self.ctx['rand_seed0']+idx
        ctx = self.ctx.copy()
        del_keys = ['rand_seed0', 'N', 'action_frames', 'interactive', 'episode_length', 'bounding_actions', 'goal_frames']
        for key in del_keys: del ctx[key]        
        ctx['rand_seed'] = rand_seed        
        env = active_3dsprites_env(ctx)  
  
        if self.ctx['interactive']:        
            return env.obj_data, env.bgcolor, env.ctx
        else:
            if not self.ctx['rule_goal']=='None':
                self.goal = env.get_goal_state()

            ims, masks, action_fields = self.generate_episode(env, self.ctx['episode_length'])        
            if self.ctx['with_rotation']:
                action_fields[:,3:] = action_fields[:,3:]/5*0.625 #The translation actions have a s.d. of 0.625, while the angular actions have a s.d. of 5, so this brings them into the same scale
            return ims, masks, action_fields
        
            
    #@profile
    def generate_episode(self, env, episode_length):
        ims = torch.zeros(episode_length, 3, *(self.ctx['im_size'],)*2)
        masks = torch.zeros(episode_length, 1, *(self.ctx['im_size'],)*2)
        action_fields = torch.zeros(episode_length, 3+3*self.ctx['with_rotation'], *(self.ctx['im_size'],)*2)
        


        for t in range(episode_length):
            ims[t], masks[t] = env.render(keep_render_env=True if t<(episode_length-1) else False)
            
            if t+1 <= (episode_length-1):
                obj_actions = torch.zeros((self.ctx['num_objs']+1,3+3*self.ctx['with_rotation']))
                action_field=torch.zeros(action_fields.shape[1:])
                if self.ctx['rule_goal'] and t+1 >= episode_length-self.ctx['goal_frames']:
                    if t+1 < episode_length-1:
                        frames_to_go = episode_length-(t+1)   #E.g. if the next frame is frame 9 in a 12-frame sequence, then given 0-indexing that is actually the 10th frame, and so there are 2 frames left to go                        
                        goal_pos    = self.goal[:,8:11]
                        curr_pos    = env.obj_data[:,8:11]
                        curr_v      = env.obj_data[:,11:14]                            
                        need_v      = (goal_pos-curr_pos)/(frames_to_go-1) #-1 since the last step is necessary to cancel the remaining velocity; we need objects in their desired positions by the penultimate frame                               
                        obj_actions = need_v-curr_v
                        #Computing this every frame is strictly redundant as the velocity will be constant for all but the last frame
                    elif t+1 == episode_length-1:
                        curr_v      = env.obj_data[:,11:14]                            
                        obj_actions = -curr_v
                    else:
                        raise Exception('This should not be possible')
                    action_field = env.obj_actions_to_action_fields(obj_actions)
                else:
                    if t+1 in self.ctx['action_frames']:
                        obj_actions = env.get_random_action(return_action_field=False)
                        action_field = env.obj_actions_to_action_fields(obj_actions)
                    if self.ctx['bounding_actions']:
                        pred_obj_data = env.step(action_field, do_not_update=True)                    
                        _, _, _, _, pos, _, *_ = torch.split(pred_obj_data, env.split_sizes, -1)
                        depths = pos[:,(-1,)]
                        depth_range = self.ctx['depth_max']-self.ctx['depth_min']                    
                        depths_fraction = (depths-self.ctx['depth_min'])/depth_range if depth_range > 0 else 1.0
                        
                        max_pos = depths_fraction*self.ctx['pos_max_at_max_depth'] + (1-depths_fraction)*self.ctx['pos_max_at_min_depth'] #Maximum (x,y) coordinates as distance from center depend on depth        
                        too_high = torch.cat((pos[:,:-1]>max_pos, depths>self.ctx['depth_max']),-1)
                        too_low = torch.cat((pos[:,:-1]<-max_pos, depths<self.ctx['depth_min']),-1)
                        oob = torch.logical_or(too_high, too_low)
                        if oob.any():
                            _, _, _, _, _, velocity, *_ = torch.split(env.obj_data, env.split_sizes, -1)                                             
                            pos_actions = obj_actions[:,:3]
                            desired_velocity = velocity.clone()
                            desired_velocity[oob] = -velocity[oob]                        
                            req_actions = desired_velocity-velocity                        
                            req_actions=torch.cat((req_actions, torch.zeros(1,3)),0)
                            pos_actions[req_actions!=0] = req_actions[req_actions!=0]
                            obj_actions[:,:3] = pos_actions
                            action_field = env.obj_actions_to_action_fields(obj_actions) 

                action_fields[t+1] = action_field
                                
                env.step(action_field, move_render_objs=True)
            

        return ims, masks, action_fields

        

class active_3dsprites_vecenv():

    def __init__(self, ctx: dict, init_data=None):
        self.N = init_data[0].shape[0]
        self.device=init_data[0].device
        self.envs = []
        for i in range(self.N):
            this_ctx = dict()
            for key in ctx:
                foo = ctx[key]
                if isinstance(foo, list) or torch.is_tensor(foo):
                    if len(foo)==1:
                        foo=foo[0]                
                    foo = foo[i]                        

                if torch.is_tensor(foo):
                    foo=foo.item()
                if key=='gpus': 
                    foo = (foo,)
                this_ctx[key] = foo
            this_env = active_3dsprites_env(ctx=this_ctx, init_data = (init_data[0][i],init_data[1][i]))
            self.envs.append(this_env)

    def destroy_render_envs(self):
        for env in self.envs:
            env.destroy_render_env()

    def step(self, actions, move_render_objs=False):
        for i,env in enumerate(self.envs): 
            env.step(actions[i], move_render_objs=move_render_objs)
        

    def render(self, keep_render_env=False):
        ims = []
        masks = []
        for env in self.envs: 
            im, mask = env.render(keep_render_env)
            ims.append(im)
            masks.append(mask)

        ims = torch.stack(ims).to(self.device)
        masks = torch.stack(masks).to(self.device)        

        return ims, masks
    
    def get_true_states(self):
        obj_data = torch.stack([env.obj_data for env in self.envs])
        bgc = torch.stack([env.bgcolor for env in self.envs])

        return obj_data, bgc
    
    def get_goal_states(self):
                
        goal_states = torch.stack([env.get_goal_state() for env in self.envs])

        return goal_states


def create_glossy_material(color):
    material = Material()            
    amb_color = [c for c in color]
    dif_color = [c for c in color]
    material.set_ambient(VBase4(*amb_color, 1))
    material.set_diffuse(VBase4(*dif_color, 1))
    material.set_specular(VBase4(5,5, 5, 1))    
    material.set_shininess(10)
    material.set_local(True)
    
    return material

def create_mask_material(color):
    material = Material()    
    material.set_ambient(VBase4(color,0,0, 1))
    material.set_diffuse(VBase4(0,0,0,0))
    material.set_specular(VBase4(0,0,0,0))    
    material.set_shininess(0)

    return material

class active_3dsprites_env():
    '''
    Active 3dsprites environment. This is a single environment instance. For batched/vectorized environment,
    we use a separate class: active_3dsprites_vecenv
    '''

    def __init__(self, ctx: dict, init_data=None):
        def_args = {
            'im_size': 64,  
            'num_objs': 3,                       
            'render_size': None,
            'depth_min': 20,
            'depth_max': 30,
            'pos_max_at_min_depth': 4,
            'pos_max_at_max_depth': 6.5,
            'scale_min': 1.0,
            'scale_max': 1.5,
            'rand_seed': 1234,            
            'v_sd': 4/64*10, #4/64 pixels was the value for active_dsprites, which corresponded to a fraction of the frame size. The range of positions is about 10 m on avg across (x,y,z).
            'a_sd': 4/64*10,
            'v_ang_sd': 5,
            'a_ang_sd': 5,
            'gpus': [0,],            
            'with_rotation': False,
            'bgcolor': None, #If this is None, then it gets picked randomly - otherwise it is fixed to the given value        
            'include_bgd_action': True,
            'no_depth_motion': False,
            'with_directional_light': True,
            'rule_goal': 'None',
        }

        for key in ctx: assert key in def_args, "Argument '{}' not in list of accepted arguments".format(key)

        for key in def_args:
            if key not in ctx: ctx[key] = def_args[key]

        if ctx['render_size'] is None: ctx['render_size']=ctx['im_size']*8

        self.ctx = ctx

        self.rand_gen = torch.Generator()
        self.rand_gen.manual_seed(ctx['rand_seed'])

        if init_data is None:
            self.obj_data, self.bgcolor = self.initialize_environment()
        else:
            self.obj_data, self.bgcolor = init_data[:]
        
        self.ToTensor = ToTensor()
        self.Resize_bicubic = Resize((ctx['im_size'],)*2, interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.Resize_nearest = Resize((ctx['im_size'],)*2, interpolation=InterpolationMode.NEAREST)
        self.mask_materials = []
        for i in range(self.ctx['num_objs']):
            mm = create_mask_material((i+1)*2/255)
            self.mask_materials.append(mm)
    
        self.split_sizes = [3,1,1,3,3,3]
        if self.ctx['with_rotation']: self.split_sizes+=[3]

    #@profile
    def create_objects_from_data(self, obj_data):
        # obj_paths = ['/home/rubber/AttentionExperiments/assets/box.egg', '/home/rubber/AttentionExperiments/assets/Cone.egg', '/home/rubber/AttentionExperiments/assets/HalfTorus.egg']
        # obj_paths = ['../assets/box.egg', '../assets/Cone.egg', '../assets/HalfTorus.egg']
        obj_paths = ['box.egg', 'Cone.egg', 'HalfTorus.egg']
        prepath = os.path.expanduser('~') + '/AttentionExperiments/assets/'

        if 'box' in obj_paths[0]:
            scale_corrections = [1.4, 1.07, 6.0]
        elif 'cube' in obj_paths[0]:
            scale_corrections = [1., 1.07, 6.0]
        objects = []

        for i in range(obj_data.shape[0]):
            color, shape, scale, ori, pos, _, *_ = torch.split(obj_data[i], self.split_sizes)
            pos = pos[[0, 2, 1]] #Obj data is in coordinates (horizontal, vertical, depth), and panda3d expects (horizontal, depth, vertical)
            shape = int(shape.item())-1
            scale = scale.item()

            # exists = os.path.exists(obj_paths[shape])
            # print(exists)
            # print('Worker {}, obj path {}, exists: '.format(worker_rank, obj_paths[shape], exists))
            obj = NodePath(self.loader.loadSync(prepath + obj_paths[shape]))
            obj.reparent_to(self.scene)
            obj.set_material(create_glossy_material(color/255))
            obj.setScale((scale/scale_corrections[shape],)*3)
            obj.setHpr(*ori)
            obj.setPos(*pos)

            objects.append(obj)

        return objects

    #@profile
    def get_rendered_image(self):
        # Now we can render the frame manually
        self.graphicsEngine.renderFrame()

        im = torch.frombuffer(self.bgr_tex.getRamImage(), dtype=torch.uint8).view(self.bgr_tex.getYSize(), self.bgr_tex.getXSize(), self.bgr_tex.getNumComponents())
        im = torch.flip(im, (-1,)).permute((2,0,1))/255    
                
        return im


    def step(self, actions=None, do_not_update=False, move_render_objs=False):
        
        new_obj_data=self.obj_data.clone()            
                
        if actions is not None:            
            
            assert actions.ndim==3, 'Action fields have incorrect dimensions'            

            if self.curr_masks==None:
                '''
                Normally we rely on there already being masks available from a previous render step.
                '''
                _, self.curr_masks, _ = self.render()                
                        
            self.curr_masks = self.curr_masks.to(self.obj_data.device)
            actions = torch.einsum('ijk,ljk->il', self.curr_masks*1.0, actions)                                    
            new_obj_data[:,11:] += actions #Increment (angular) velocity by acceleration (this line works with or without rotations)

        new_obj_data[:,8:11] += new_obj_data[:,11:14] #Increment position by velocity        
        if self.ctx['with_rotation']:
            new_obj_data[:, 5:8] += new_obj_data[:, 14:] #Increment angle by angular velocity

        if do_not_update:
            return new_obj_data 
        else:
            self.obj_data = new_obj_data
            self.curr_masks = None #As a precaution, clear the current masks if there were any, since they will no longer be accurate (this prevents that we would call step() again before re-rendering the environment, and thus simulate an action based on outdated masks)            
            if move_render_objs:
                # If we want to reuse the render env, e.g. because we are generating a continuous episode in non-interactive mode, then we need to apply the changes to the render objects
                for [i,obj] in enumerate(self.objs):
                    obj.set_pos(*new_obj_data[i,[8,10,9]])
                    if self.ctx['with_rotation']:
                        obj.set_hpr(*new_obj_data[i,5:8])
        
    #@profile
    def init_render_env(self):
        # Retrieve the worker info
        worker_info = torch.utils.data.get_worker_info()        
        # print(worker_info)
        worker_rank = worker_info.id if worker_info is not None else 0
        gpus = self.ctx['gpus']
        use_gpu = gpus[worker_rank % len(gpus)]
        loadPrcFileData('', 'egl-device-index {}'.format(use_gpu))
        
        loadPrcFileData('', 'notify-output zz_showbase_log.txt') #This just makes it so we don't get a bunch of output in the terminal
               
                
        self.loader = PandaLoader.get_global_ptr()
        selection = GraphicsPipeSelection.get_global_ptr()
        self.pipe = selection.make_default_pipe()
        self.graphicsEngine = GraphicsEngine()
                
        # Create window properties
        win_prop = WindowProperties.size(*(self.ctx['render_size'],)*2)

        # Create frame buffer properties
        fb_prop = FrameBufferProperties()
        fb_prop.setRgbColor(True)
        # Only render RGB with 8 bit for each channel, no alpha channel
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setDepthBits(24)

        # Create window (offscreen)        
        self.window = self.graphicsEngine.makeOutput(self.pipe, "cameraview", 0, fb_prop, win_prop, GraphicsPipe.BFRefuseWindow)                   
        self.window.setClearColor((self.bgcolor.item()/255,)*3 + (1,)) #This allows us to specify a background color 
        
                                
        # Create display region
        # This is the actual region used where the image will be rendered
        disp_region = self.window.makeDisplayRegion()

        self.scene = NodePath('scene')
  
        # Create cameras
        cam = self.scene.attach_new_node(Camera('cam'))                        
        disp_region.set_camera(cam)


        # Set up lighting
        if self.ctx['with_directional_light']:
            light = PointLight("light")
            light.set_color(VBase4(1,1,1,1))
            light.set_attenuation(VBase3(1, 0.2, 0))
            light_node = self.scene.attach_new_node(light)        
            light_node.set_pos(0,0,0)
            self.scene.set_light(light_node)
            ambient_color = (0.5,0.5,0.5,1)
            del light, light_node
        else:
            ambient_color = (1,)*4
        
        alight = AmbientLight('alight')
        alight.setColor(ambient_color)
        alnp = self.scene.attachNewNode(alight)

        
        self.scene.set_light(alnp)

        del alight, alnp, cam

        # Create the texture where the frame will be rendered
        # This is the RGB/RGBA buffer which stores the rendered data
        self.bgr_tex = Texture()
        self.window.addRenderTexture(self.bgr_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)

    #@profile
    def destroy_render_env(self):
        self.graphicsEngine.removeAllWindows()        
        for obj in self.objs:
            obj.removeNode()
        for node in self.scene.get_children():
            node.removeNode()
            del node
        self.scene.removeNode()
        # self.loader.destroy()
        ModelPool.releaseAllModels()
        ModelPool.garbageCollect() 
        TexturePool.releaseAllTextures()
        TexturePool.garbageCollect()
        gc.collect()
        RenderState.clear_munger_cache()

        # states = RenderState.getStates()
        # print(len(states))
                
        del self.bgr_tex, self.objs, self.scene, self.loader, self.graphicsEngine, self.mask_materials, self.pipe, self.window
            
                
    #@profile                
    def render(self, keep_render_env=False):       

        if not hasattr(self, 'graphicsEngine'):
            self.init_render_env()        

        #Instate objects
        if not hasattr(self, 'objs'):
            self.objs = self.create_objects_from_data(self.obj_data)

        im = torch.clamp(self.Resize_bicubic(self.get_rendered_image()), 0., 1.)        
                
        # To get masks, we need to change the materials of our objects so that they become uniformly colored in the render
        materials = [obj.get_material() for obj in self.objs]
        for i, obj in enumerate(self.objs):            
            # obj.set_material(create_mask_material((i+1)*2/255)) #This will use only the red channel since the bgd is uniform grayscale. We multiply by 2 because ambient illumination is only 50%. 
            obj.set_material(self.mask_materials[i])            
            
        masks = self.Resize_nearest(self.get_rendered_image())
        masks = (masks*255).to(torch.uint8)
        mask_values = torch.arange(1, self.ctx['num_objs']+1)        
        masks = (masks[0].unsqueeze(-1) == mask_values).permute((2,0,1))
        self.curr_masks = masks

        masks = (masks*mask_values.view(self.ctx['num_objs'],1,1)).sum(0,True) #These masks are categorical

        if keep_render_env:
            for i, obj in enumerate(self.objs): obj.set_material(materials[i]) #Reset objects to their previous materials
        else:
            self.destroy_render_env()

        for mat in materials:
            del mat

        im = im.to(self.obj_data.device)
        masks = masks.to(self.obj_data.device)

        return im, masks
    
    def initialize_environment(self):
        
        cval = torch.tensor([0, 63, 127, 191, 255])
        if self.ctx['bgcolor'] is None:
            bgcolor = cval[torch.randint(0, len(cval), (1,), generator=self.rand_gen)]
        else:
            bgcolor = torch.tensor(self.ctx['bgcolor'])            

        colors = cval[torch.randint(0, len(cval), (self.ctx['num_objs'], 3), generator=self.rand_gen)]        
        shapes = torch.randint(1, 4, (self.ctx['num_objs'],1), generator=self.rand_gen)        
        scales = torch.rand((self.ctx['num_objs'],1), generator=self.rand_gen)*(self.ctx['scale_max']-self.ctx['scale_min'])+self.ctx['scale_min']        
        orientations = torch.rand((self.ctx['num_objs'],3), generator=self.rand_gen)*360
        
        depth_range = self.ctx['depth_max']-self.ctx['depth_min']
        depths = torch.rand((self.ctx['num_objs'],1), generator=self.rand_gen)*depth_range+self.ctx['depth_min']
        depths_fraction = (depths-self.ctx['depth_min'])/depth_range if depth_range > 0 else 1.0
        
        max_pos = depths_fraction*self.ctx['pos_max_at_max_depth'] + (1-depths_fraction)*self.ctx['pos_max_at_min_depth'] #Maximum (x,y) coordinates as distance from center depend on depth        
        xy = torch.rand((self.ctx['num_objs'], 2), generator=self.rand_gen)*max_pos*2 - max_pos        
        
        positions = torch.cat((xy, depths), -1) #NOTE: This is different from the order in panda3d so we'll have to convert                
        velocities = torch.randn(positions.shape, generator=self.rand_gen)*self.ctx['v_sd']
        if self.ctx['no_depth_motion']:
            velocities[:,-1] = 0

        if self.ctx['with_rotation']:
            ang_velocities = torch.rand(orientations.shape, generator=self.rand_gen)*self.ctx['v_ang_sd']
        else:
            ang_velocities = torch.Tensor([])

        obj_data = torch.cat((colors, shapes, scales, orientations, positions, velocities, ang_velocities), -1)        

        return obj_data, bgcolor

    def get_random_action(self, return_action_field = True):
        '''
        For now we'll just assume we need actions for all objects + background
        '''
        actions = torch.randn((self.ctx['num_objs']+self.ctx['include_bgd_action']*1,3), generator=self.rand_gen)*self.ctx['a_sd']  
        if self.ctx['no_depth_motion']:
            actions[:,-1] = 0
        if self.ctx['with_rotation']:
            ang_actions = torch.randn((self.ctx['num_objs']+self.ctx['include_bgd_action']*1,3), generator=self.rand_gen)*self.ctx['a_ang_sd']  
            actions = torch.cat((actions, ang_actions),-1)
        if return_action_field:
            actions = self.obj_actions_to_action_fields(actions)

        return actions
    
    def get_goal_state(self):
        if self.ctx['rule_goal']=='IfHalfTorus':
            halftorus_idx = self.obj_data[:,3]==3 #Order is 1: box, 2: cone, 3: halftorus
            goal = self.obj_data.clone()
            goal[:,10] = (self.ctx['depth_min']+self.ctx['depth_max'])/2
            goal[:,11:] = 0 #Target velocity is always 0   
            if halftorus_idx.any():
                goal[:,8]=0.15*10-5
            else:
                goal[:,8]=0.85*10-5            
            goal_ypos = [x*10-5 for x in [0.15, 0.5, 0.85]]
            for s_idx in range(1,4):
                this_idx = self.obj_data[:,3]==s_idx
                goal[this_idx,9] = goal_ypos[s_idx-1]
        elif self.ctx['rule_goal']=='None':
            pass
        else:
            raise Exception('Goal rule "{}" not implemented'.format(self.ctx['rule_goal']))

        return goal
    
    def obj_actions_to_action_fields(self, obj_actions):
        '''
        Take a tensor of object actions, K x A, and convert it to an action field,
        A x H x W. If K is 1 larger than self.ct['num_objs'], then the last entry is assumed to encode
        the action on the background. 
        '''

        K = obj_actions.shape[0]

        assert K in (self.ctx['num_objs'], self.ctx['num_objs']+1), 'Size of obj_actions is incompatible with number of sprites'

        include_bgd_action = True if K==self.ctx['num_objs']+1 else False
        if include_bgd_action: bg_mask = torch.logical_not(self.curr_masks.any(0))                

        if hasattr(torch, 'argwhere'):
            aw_fun = torch.argwhere
        else:
            aw_fun = lambda x: np.argwhere(x.cpu()).T

        action_field = torch.zeros(3 + 3*self.ctx['with_rotation'], *(self.ctx['im_size'],)*2)                

        
        for k in range(K):
            if (obj_actions[k]==0).all().item(): continue
            if k < self.ctx['num_objs']:
                this_mask = self.curr_masks[k] 
            else:
                this_mask = bg_mask

            if not this_mask.any().item(): continue

            indices = aw_fun(this_mask)
            pick = torch.randint(0, indices.shape[0], (1,), generator=self.rand_gen).item()
            index = indices[pick]

            action_field[:, index[0], index[1]] = obj_actions[k]

        return action_field
    
   

if __name__=="__main__":
    

    # dataset = active_3dsprites_dataset({'im_size': 64, 'render_size': 64*4, 'num_objs': 3, 'rand_seed0': 4001, 'gpus': [3,], 'episode_length': 12, 'with_rotation': True, 'bgcolor': 127})
    
    dataset = active_3dsprites_dataset({'im_size': 64, 'render_size': 64*4, 'num_objs': 3, 'rand_seed0': 4001, 'gpus': [0,], 'episode_length': 12, 'with_rotation': False, 'bgcolor': 127, 
                                        'action_frames': [2,4,6,8], 'rule_goal': 'IfHalfTorus', 'scale_min': 1.5, 'scale_max': 1.50001, 'goal_frames': 3})
               
    # dataset = active_3dsprites_dataset({
    #         'N': 512,
    #         'interactive': True,            
    #         'action_frames': [],            
    #         'gpus': 9,
    #         'with_rotation': False,
    #         'scale_min': 1.5,
    #         'scale_max': 1.50001,
    #         'rand_seed0': 50000+10000+1234+4343,
    #         'bgcolor': 127,                      
    #     })

    dataloader = DataLoader(dataset, 16, num_workers=0)
    # dataloader = DataLoader(dataset, 16, num_workers=4)
    dataiter = iter(dataloader)
    
    for i in range(2):
        
        foo = dataiter.next()           
                
        Image.fromarray((make_grid(foo[0].view(dataloader.batch_size*dataset.ctx['episode_length'],3,64,64), nrow=dataset.ctx['episode_length']).permute(1,2,0)*255).to(torch.uint8).numpy()).save('foo_rule-goal-IfHalfTorus_{}.png'.format(i))
              
