import numpy as np
import h5py
import main
from models.savn import SAVN
from models.basemodel import BaseModel
from models.gcn import GCN
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from runners.train_util import get_params
from models.model_io import ModelOptions, ModelInput
from utils.net_util import gpuify
from utils.net_util import resnet_input_transform
import torch.nn.functional as F
import time
from runners.train_util import compute_learned_loss, SGD_step

MODEL_PATH_DICT = {'SAVN' : 'pretrained_models/savn_pretrained.dat',
                   'NON_ADAPTIVE_A3C': 'pretrained_models/nonadaptivea3c_pretrained.dat',
                   'GCN':'pretrained_models/gcn_pretrained.dat' }
GLOVE_FILE = './data/thor_glove/glove_map300d.hdf5'
ACTION_LIST = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Done']



class FakeArgs():
    def __init__(self, model='SAVN'):
        self.action_space = 6
        self.glove_dim = 300
        self.hidden_state_sz = 512
        self.dropout_rate = 0.25
        self.num_steps = 6 # initialized in main_eval.py
        self.gpu_id = -1
        self.learned_loss = True if model=='SAVN' else False
        self.inner_lr = 0.0001
        self.model = model
        self.glove_file = GLOVE_FILE
        self.max_gradient_updates = 4
        self.done=False
        
        
class Agent():
    def __init__(self,args,model):
        self.gpu_id = args.gpu_id
        self.model = model
        self.hidden = None #initialized in function call
        self.last_action_probs = None #initialized in function call
        self.resnet18 = None #initialized in function call
        self.hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        self.learned_loss = args.learned_loss
        self.learned_input = None #initialized in function call
        
    def set_target(self,target_glove_embedding):
        self.target_glove_embedding = target_glove_embedding
        
    def eval_at_state(self, model_options,frame):
        model_input = ModelInput()
#         if self.episode.current_frame is None:
#             model_input.state = self.state()
#         else:
#             model_input.state = self.episode.current_frame
        #process_frame to shape [1,3,224,224], for input to resnet18
        processed_frame = self.preprocess_frame(resnet_input_transform(frame, 224).unsqueeze(0))
        resnet18_features = self.resnet18(processed_frame)
        
        model_input.state = resnet18_features
        model_input.hidden = self.hidden
        model_input.target_class_embedding = gpuify(torch.Tensor(self.target_glove_embedding),gpu_id=self.gpu_id)
        model_input.action_probs = self.last_action_probs

        return model_input, self.model.forward(model_input, model_options)
        
    def reset_hidden(self):
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.hidden = (
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                )
        else:
            self.hidden = (
                torch.zeros(1, self.hidden_state_sz),
                torch.zeros(1, self.hidden_state_sz),
            )
        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )
        
    def action(self, model_options, frame,training=False):
        if training:
            self.model.train()    #torch.nn
        else:
            self.model.eval()    

        model_input, out = self.eval_at_state(model_options,frame)  
        self.hidden = out.hidden
        prob = F.softmax(out.logit, dim=1)
        #print(prob)
        action = prob.multinomial(1).data
        #log_prob = F.log_softmax(out.logit, dim=1)
        self.last_action_probs = prob
        
        if self.learned_loss:
            
            res = torch.cat((self.hidden[0], self.last_action_probs), dim=1)
            #if DEBUG: print("agent/action  learned loss", res.size())
            if self.learned_input is None:
                self.learned_input = res
            else:
                self.learned_input = torch.cat((self.learned_input, res), dim=0)
        
        return out.value, prob, action
    
    
    
    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)
    
    def init_resnet18(self):
        
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-2]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False

class Episode:
    def __init__(self, controller, target, model_name, model_path,glove_file_path):
        self.controller = controller
        self.args = FakeArgs(model=model_name)
        self.target = target
        self.event = None
        self.model_name = model_name
        self.action_list=['MoveAhead', 
                          'RotateLeft', 
                          'RotateRight', 
                          'LookUp', 
                          'LookDown', 
                          'Done']
        self.step_count = 0
        self.load_glove_embedding(glove_file_path)
        self.load_model(model_path, self.args)
        self.init_agent()
        self.done = False
        
    def load_glove_embedding(self, glove_file):
        glove_embedding_dict = {}
        with h5py.File(glove_file, "r") as f:
            for key in f.keys():
                glove_embedding_dict[key] = f[key].value
        self.glove_embedding =  glove_embedding_dict
    
    def load_model(self, model_path,args):
        if args.model=='NON_ADAPTIVE_A3C':
            self.model = BaseModel(args)
        elif args.model == 'GCN':
            self.model = GCN(args)
        else:
            self.model = SAVN(args)
        saved_state = torch.load(
                    model_path, map_location=lambda storage, loc: storage
                )
        self.model.load_state_dict(saved_state)

        self.model_options = ModelOptions()
        self.model_options.params = get_params(self.model, args.gpu_id)
        
         
    def init_event(self):
        self.event = self.controller.step(action='Initialize')
    
    def init_agent(self):
        agent = Agent(self.args,self.model)
        agent.reset_hidden()
        agent.init_resnet18()
        agent.set_target(self.glove_embedding[self.target])
        self.agent = agent
        
    def step(self):
        if not self.done:
            if self.event == None:
                self.init_event()
            self.frame = self.event.frame
            _,_, action = self.agent.action(self.model_options, self.frame)
            if action[0,0] == 5: 
                print("Agent Done")
                self.done = True
            print(self.action_list[action[0,0]])
            self.event = self.controller.step(action=self.action_list[action[0,0]])
            self.step_count += 1
            #print(self.args.learned_loss)
            if self.args.learned_loss:
                if self.step_count % self.args.num_steps == 0 \
                    and self.step_count/self.args.num_steps < self.args.max_gradient_updates:
                    
                    learned_loss = compute_learned_loss(self.args, self.agent, self.args.gpu_id, self.model_options)
                    inner_gradient = torch.autograd.grad(
                            learned_loss["learned_loss"],
                            [v for _, v in self.model_options.params.items()],
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    print("gradient update")
                    self.model_options.params = SGD_step(self.model_options.params,inner_gradient, self.args.inner_lr)
    
        else:
            print("agent done")
    def isolated_step(self, image):
        _,_, action = self.agent.action(self.model_options, image)
        return self.action_list[action[0,0]]
        
    def stop(self):
        self.controller.stop()
    
    