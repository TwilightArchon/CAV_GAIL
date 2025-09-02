
import random
import numpy as np

import torch
import torch.nn as nn
from Utils.NeuralNetwork import Discriminator
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")

################################## Discriminator ##################################

class DISCRIMINATOR_FUNCTION:
    def __init__(self, state_dim, action_dim, lr_gail, D_epochs, expert_traj, expert_sample_size):

        self.D_epochs = D_epochs

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # self.expert_traj = expert_traj
        self.expert_sample_size = expert_sample_size

        expert_trajectory = expert_traj[0]
        for i in range(1,len(expert_traj)):
            expert_trajectory = np.concatenate((expert_trajectory, expert_traj[i]), axis=0)

        expert_trajectory = expert_trajectory.reshape(-1,self.state_dim+self.action_dim+self.state_dim)
        expert_trajectory = torch.FloatTensor(expert_trajectory).to(device)
        self.expert_states, self.expert_actions, _ = torch.split(expert_trajectory, (self.state_dim,self.action_dim,self.state_dim), dim=1)
        

        self.discrim_criterion = nn.BCELoss()

        self.discriminator = Discriminator(state_dim,action_dim).to(device)
        self.optimizer_discrim = torch.optim.Adam(params=self.discriminator.parameters(), lr=lr_gail)

    def reward(self, state, action, epsilon = 1e-40):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(np.expand_dims(action, axis=0)).to(device)
            
        with torch.no_grad():
            # epsilon avoids log(0)
            return torch.log(self.discriminator(state, action)+ epsilon).cpu().data.numpy().squeeze()

    def update(self, agent_net, states, actions):

        for _ in range(self.D_epochs):          

            expert_loss = self.discriminator(self.expert_states, self.expert_actions)
            learner_loss = self.discriminator(states, actions)

            # take gradient step
            self.optimizer_discrim.zero_grad()

            expert_loss  = self.discrim_criterion(expert_loss, torch.ones((self.expert_states.size(0), 1)).to(device))
            learner_loss = self.discrim_criterion(learner_loss, torch.zeros((states.shape[0], 1)).to(device))
            discrim_loss =  expert_loss + learner_loss
            discrim_loss.backward()

            self.optimizer_discrim.step()
        
    def save(self, checkpoint_path):
        torch.save(self.discriminator.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.discriminator.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))