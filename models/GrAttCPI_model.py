import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv


'''
GrAttCPI: Accepts Drugs' 2D graphs and Residue interaction netwoeks (It can be either AlphaFold+RING results or contactMap results)
'''
class GrAttCPI(torch.nn.Module):
    def __init__(self, dim_in_target, dim_in_cpd, dim_h, dim_out, heads=8):
        super().__init__()
        self.dim_out = dim_out
        self.gat1_target = GATv2Conv(dim_in_target, dim_h, heads=heads, edge_dim=1)
        self.gat2_target = GATv2Conv(dim_h*heads, dim_out, heads=1, edge_dim=1)
        self.fc1_target = Linear(1400*self.dim_out, 2048)
        self.bn1_target = nn.BatchNorm1d(num_features=2048)
        self.fc2_target = Linear(2048, 512)
        self.bn2_target = nn.BatchNorm1d(num_features=512)
        self.fc3_target = Linear(512, 128)
        self.bn3_target = nn.BatchNorm1d(num_features=128)
        self.gat1_cpd = GATv2Conv(dim_in_cpd, dim_h, heads=heads, edge_dim=1)
        self.gat2_cpd = GATv2Conv(dim_h*heads, dim_out, heads=1, edge_dim=1)
        self.fc_cpd = Linear(148*self.dim_out, 128)
        self.bn_cpd = nn.BatchNorm1d(num_features=128)
        self.fc1 = nn.Linear(256,64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(64,16)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.fc3 = nn.Linear(16,4)
        self.bn3 = nn.BatchNorm1d(num_features=4)
        self.fc4 = nn.Linear(4,2)
        
        
    def forward(self, tdata, ddata, dev):
        ############# PROTEINS REPRESENTATIVE VECTORS #############
        target_rep, target_edge_index, target_edge_attr, num_of_aas = tdata.x, tdata.edge_index, tdata.edge_attr, tdata.num_of_aa
        batch_size = int(target_rep.shape[0]/1400)
        target_rep = F.dropout(target_rep, p=0.2, training=self.training)
        target_rep = self.gat1_target(target_rep, target_edge_index, target_edge_attr)
        target_rep = F.elu(target_rep)
        target_rep = F.dropout(target_rep, p=0.2, training=self.training)
        target_rep = self.gat2_target(target_rep, target_edge_index, target_edge_attr)
        target_rep = target_rep.view(batch_size, 1400, self.dim_out)
        for idx in range(len(num_of_aas)):
            mask_target = torch.zeros(1400).to(torch.device(dev))
            mask_target[:num_of_aas[idx]] = 1
            target_rep[idx] = target_rep[idx] * mask_target.view(-1, 1)
        target_rep = target_rep.view(batch_size, 1400*self.dim_out)
        target_rep = F.dropout(self.bn1_target(self.fc1_target(target_rep)), p=0.4, training=self.training)
        target_rep = F.dropout(self.bn2_target(self.fc2_target(target_rep)), p=0.3, training=self.training)
        target_rep = F.dropout(self.bn3_target(self.fc3_target(target_rep)), p=0.2, training=self.training)
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
        res = torch.cat((cpd_rep, target_rep), 1)
        #Now, res has dimension of (batch_size, 256) which we are going to pass it through some dense layers
        res = F.dropout(self.bn1(self.fc1(res)), p=0.3, training=self.training)  
        res = F.dropout(self.bn2(self.fc2(res)), p=0.2, training=self.training)  
        res = self.bn3(self.fc3(res))
        res = self.fc4(res)
        return res