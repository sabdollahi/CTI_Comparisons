from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
BERT2DFP: Feedforward Neural Network for amino acid sequences [BERT-based Input Features] and Feedforward NN for 2D drug fingerprints
'''
class BERT2DFP(nn.Module):
    def __init__(self):
        super(BERT2DFP, self).__init__()
        '''[TARGET MODULE]'''
        self.tar_fc1 = nn.Linear(768, 512)
        self.tar_bn1 = nn.BatchNorm1d(num_features=512)
        self.tar_drop1 = torch.nn.Dropout(0.2)
        self.tar_fc2 = nn.Linear(512, 256)
        self.tar_bn2 = nn.BatchNorm1d(num_features=256)
        self.tar_drop2 = torch.nn.Dropout(0.2)
        self.tar_fc3 = nn.Linear(256, 128)
        '''[COMPOUND MODULE]'''
        #3239 -> 2048 -> 512 -> 128 
        self.comp_fc1 = nn.Linear(3239, 2048)
        self.comp_bn1 = nn.BatchNorm1d(num_features=2048)
        self.comp_drop1 = torch.nn.Dropout(0.2)
        self.comp_fc2 = nn.Linear(2048, 512)
        self.comp_bn2 = nn.BatchNorm1d(num_features=512)
        self.comp_drop2 = torch.nn.Dropout(0.2)
        self.comp_fc3 = nn.Linear(512, 128)
        '''[PREDICTION MODULE]'''
        self.pred_fc1 = nn.Linear(256,64)
        self.pred_drop1 = torch.nn.Dropout(0.2)
        self.pred_fc2 = nn.Linear(64,16)
        self.pred_drop2 = torch.nn.Dropout(0.2)
        self.pred_fc3 = nn.Linear(16,4)
        self.pred_drop3 = torch.nn.Dropout(0.2)
        self.pred_fc4 = nn.Linear(4,2)

    def forward(self, compound_input, target_input):
        #Target representative Vector
        target_rep_vec = self.tar_drop1(self.tar_bn1(self.tar_fc1(target_input)))
        target_rep_vec = self.tar_drop2(self.tar_bn2(self.tar_fc2(target_rep_vec)))
        target_rep_vec = self.tar_fc3(target_rep_vec)
        #Compound representative Vector
        compound_rep_vec = self.comp_drop1(self.comp_bn1(self.comp_fc1(compound_input)))
        compound_rep_vec = self.comp_drop2(self.comp_bn2(self.comp_fc2(compound_rep_vec)))
        compound_rep_vec = self.comp_fc3(compound_rep_vec)
        #Concatenate the target representative and the compound representative vectors
        result = torch.cat((compound_rep_vec, target_rep_vec), 1)
        result = self.pred_drop1(F.relu(self.pred_fc1(result)))
        result = self.pred_drop2(F.relu(self.pred_fc2(result)))
        result = self.pred_drop3(F.relu(self.pred_fc3(result)))
        result = self.pred_fc4(result)
        return result