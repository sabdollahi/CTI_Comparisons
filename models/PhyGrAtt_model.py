import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv


'''
PhyGrAtt: Accepts Drugs' 2D graphs and Physicochemical properties of the amino acids
'''
class PhyGrAtt(torch.nn.Module):
    def __init__(self, dim_in_cpd, dim_h, dim_out, heads=8):
        super(PhyGrAtt, self).__init__()
        input_channel = 1
        output_channel = 1
        self.dim_out = dim_out
        '''[TARGET MODULE]'''
        # Padding = 1; Stride = 1
        # Hout = floor {[Hin + (2*(padding) - kernel[0] )]/stride + 1}
        # Wout = floor {[Win + (2*(padding) - kernel[1] )]/stride + 1}
        # (20, 1, 1400, 20) ->conv-> (20, 1, 800, 15) ->maxpool-> (20, 1, 600, 12)
        # The first 20 is the size of batch
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
        self.gat1_cpd = GATv2Conv(dim_in_cpd, dim_h, heads=heads, edge_dim=1)
        self.gat2_cpd = GATv2Conv(dim_h*heads, dim_out, heads=1, edge_dim=1)
        self.fc_cpd = Linear(148*dim_out, 128)
        self.bn_cpd = nn.BatchNorm1d(num_features=128)
        '''[CONCATENATION MODULE]'''
        self.fc1 = nn.Linear(256,64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(64,16)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.fc3 = nn.Linear(16,4)
        self.bn3 = nn.BatchNorm1d(num_features=4)
        self.fc4 = nn.Linear(4,2)
        
        
    def forward(self, target_input, ddata, dev):
        batch_size = int(target_input.shape[0])
        ############# PROTEINS REPRESENTATIVE VECTORS #############
        target_rep_vec = self.tar_conv1(target_input)
        target_rep_vec = self.tar_conv2(target_rep_vec)
        target_rep_vec = self.tar_drop1(F.relu(self.tar_fc1(target_rep_vec.view(-1, 200 * 5))))
        target_rep_vec = self.tar_drop2(F.relu(self.tar_fc2(target_rep_vec)))
        target_rep_vec = self.tar_fc3(target_rep_vec)
        #After this line of code, we have target_rep with size of (batch_size, 128) -> Representative vector of the proteins in a batch
        ############# COMPOUNDS REPRESENTATIVE VECTORS #############
        cpd_rep, cpd_edge_index, num_of_atoms = ddata.x, ddata.edge_index, ddata.num_of_atoms
        cpd_rep = F.dropout(cpd_rep, p=0.2, training=self.training)
        cpd_rep = self.gat1_cpd(cpd_rep, cpd_edge_index)
        cpd_rep = F.elu(cpd_rep)
        cpd_rep = F.dropout(cpd_rep, p=0.2, training=self.training)
        cpd_rep = self.gat2_cpd(cpd_rep, cpd_edge_index)
        cpd_rep = cpd_rep.view(batch_size, 148, self.dim_out)
        for idx in range(len(num_of_atoms)):
            mask_cpd = torch.zeros(148).to(torch.device(dev))
            mask_cpd[:num_of_atoms[idx]] = 1
            cpd_rep[idx] = cpd_rep[idx] * mask_cpd.view(-1, 1)
        cpd_rep = cpd_rep.view(batch_size, 148*self.dim_out)
        cpd_rep = F.dropout(self.bn_cpd(self.fc_cpd(cpd_rep)), p=0.4, training=self.training)    
        #After this line of code, we have cpd_rep with size of (batch_size, 128) -> Representative vector of the compounds in a batch    
        ############# CONCATENATING THE REP. VECTORS #############
        res = torch.cat((cpd_rep, target_rep_vec), 1)
        #Now, res has dimension of (batch_size, 256) which we are going to pass it through some dense layers
        res = F.dropout(self.bn1(self.fc1(res)), p=0.3, training=self.training)  
        res = F.dropout(self.bn2(self.fc2(res)), p=0.2, training=self.training)  
        res = self.bn3(self.fc3(res))
        res = self.fc4(res)
        return res