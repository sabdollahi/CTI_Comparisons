from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
DTISeq2DFP: 2D Convolutional Neural Networks for amino acid sequences and Feedforward NN for drug fingerprints
'''
class DTISeq2DFP(nn.Module):
    def __init__(self):
        super(DTISeq2DFP, self).__init__()
        input_channel = 1
        output_channel = 1
        '''[TARGET MODULE]'''
        # Padding = 1; Stride = 1
        # Hout = floor {[Hin + (2*(padding) - kernel[0] )]/stride + 1}
        # Wout = floor {[Win + (2*(padding) - kernel[1] )]/stride + 1}
        # (20, 1, 1400, 20) ->conv-> (20, 1, 800, 15) ->maxpool-> (20, 1, 600, 12)
        self.tar_conv1 = nn.Sequential(
                        nn.Conv2d(input_channel, output_channel, kernel_size=(603,8), padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(201,4), stride=1))
        # (20, 1, 600, 12) ->conv-> (20, 1, 300, 8) ->maxpool-> (20, 1, 200, 5)
        self.tar_conv2 = nn.Sequential(
                        nn.Conv2d(input_channel, output_channel, kernel_size=(303,7), padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(101,4), stride=1))
        self.tar_fc1 = nn.Linear(200 * 5, 512)
        self.tar_drop1 = torch.nn.Dropout(0.2)
        self.tar_fc2 = nn.Linear(512, 256)
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
        target_rep_vec = self.tar_conv1(target_input)
        target_rep_vec = self.tar_conv2(target_rep_vec)
        target_rep_vec = self.tar_drop1(F.relu(self.tar_fc1(target_rep_vec.view(-1, 200 * 5))))
        target_rep_vec = self.tar_drop2(F.relu(self.tar_fc2(target_rep_vec)))
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