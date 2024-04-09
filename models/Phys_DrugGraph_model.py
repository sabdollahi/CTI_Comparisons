import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np



'''
    TargetFeatureCapturer: Captures informative features of the proteins
'''
class TargetFeatureCapturer(nn.Module):
    def __init__(self, hidden_dim, dropout, device): #ADD SOME PARAMETERS DEPENDS ON YOUR NEED!
        super().__init__()    
        self.tar_fc = nn.Linear(20, hidden_dim)
        self.tar_drop = torch.nn.Dropout(dropout)
        
    def forward(self, target_embedding):
        return self.tar_drop(F.relu(self.tar_fc(target_embedding)))


'''
    SelfAttention Class: A model for calculating the attention based on Query-Key-Value (softmax(Q*K/n)*V)
'''
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        assert hidden_dim % n_heads == 0
        #Defining weights for query, key, and value
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        #Fully-connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query, key, value = (batch size * sent len * hidden dim)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        # Q, K, V = (batch size * sent len * hidden dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.hidden_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hidden_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hidden_dim // self.n_heads).permute(0, 2, 1, 3)
        # // -> floor division, .permute(.) -> modify the shape by changing the order
        # K, V = (batch size * num of heads * sent len_K * hidden dim // num of heads)
        # Q = (batch size * n heads * sent len_q * hidden dim // num of heads)
        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # score = (batch size * num of heads * sent len_Q * sent len_K)
        if (mask is not None):
            # Fills elements of 'score' with -1e10 where mask == 0 is True. 
            score = score.masked_fill(mask == 0, -1e10)
        #attention = (batch size * n heads * sent len_Q * sent len_K)
        attention = self.dropout(F.softmax(score, dim=-1))
        #Z will be softmax(Q*K/n)*V
        Z = torch.matmul(attention, V)
        #Z = (batch size * num of heads * sent len_Q * hidden dim // num of heads)
        Z = Z.permute(0, 2, 1, 3).contiguous()
        #Z = (batch size * sent len_Q * num of heads * hid dim // n heads)
        Z = Z.view(batch_size, -1, self.n_heads * (self.hidden_dim // self.n_heads))
        #Z = (batch size * src sent len_Q * hidden dim)
        Z = self.fc(Z)
        #Z = (batch size * sent len_Q * hid dim)
        return Z

    
'''
    PositionBasedConv1d Class: To take care about the position of the elements by utilizing Conv1D as the next step after calculating self-attention 
'''
class PositionBasedConv1d(nn.Module):
    def __init__(self, hidden_dim, middle_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.middle_dim = middle_dim
        #Utilizing Conv1D 
        self.conv1d_1 = nn.Conv1d(hidden_dim, middle_dim, 1)
        self.conv1d_2 = nn.Conv1d(middle_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = (batch size * sent len * hidden dim)
        x = x.permute(0, 2, 1)
        # x = (batch size * hid dim * sent len)
        x = self.dropout(F.relu(self.conv1d_1(x)))
        # x = (batch size * middle dim * sent len)
        x = self.conv1d_2(x)
        # x = (batch size * hid dim * sent len)
        x = x.permute(0, 2, 1)
        # x = (batch size * sent len * hidden dim)
        return x
    
    
'''
    IntertwiningLayer Class: The compound and target embeddings are interwoven with each other based on self-attention scenarios
'''
class IntertwiningLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, middle_dim, self_attention, position_based_conv1d, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention_1 = self_attention(hidden_dim, n_heads, dropout, device)
        self.self_attention_2 = self_attention(hidden_dim, n_heads, dropout, device)
        self.pos_conv = position_based_conv1d(hidden_dim, middle_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cpd_embeddings, target_embeddings, cpd_mask=None, target_mask=None):
        '''
             cpd_embeddings = (batch_size * compound len * atom_dim)
             target_embeddings = (batch_size * protein len * hidden_dim) ===> encode target embeddings based on this dimension
             cpd_mask = (batch size * compound sent len)
             target_mask = (batch size * protein len)
        '''
        #To make attention over the atoms of compounds (Highlight the atoms at specific positions based on the compounds' embeddings)
        #Query: cpd_embeddings, Key: cpd_embeddings, Value: cpd_embeddings
        cpd_embeddings = self.layer_norm(cpd_embeddings + self.dropout(self.self_attention_1(cpd_embeddings, cpd_embeddings, cpd_embeddings, cpd_mask)))
        #To make attention over the amino acids of proteins (Highlights the amino acids at specific positions based on the compounds' highlighted embeddings)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++JUST TRIED TO PUT 'None' INSTEAD OF target_mask++++++++++++++++++++++++++
        result = self.layer_norm(cpd_embeddings + self.dropout(self.self_attention_2(cpd_embeddings, target_embeddings, target_embeddings, target_mask)))
        #Position based Conv1d
        result = self.layer_norm(result + self.dropout(self.pos_conv(result)))
        return result
    
    
    
'''
    IntertwiningTransformer Class: Utilizes multiple IntertwiningLayers (self-attention models)
'''
class IntertwiningTransformer(nn.Module):
    def __init__(self, atom_dim, hidden_dim, n_layers, n_heads, middle_dim, intertwining_layer, self_attention, position_based_conv1d, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.output_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.middle_dim = middle_dim
        self.intertwining_layer = intertwining_layer
        self.self_attention = self_attention
        self.position_based_conv1d = position_based_conv1d
        self.dropout = dropout
        self.device = device
        self.layers = nn.ModuleList([intertwining_layer(hidden_dim, n_heads, middle_dim, self_attention, position_based_conv1d, dropout, device) for _ in range(n_layers)])
        self.linear = nn.Linear(atom_dim, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, 8)
        self.fc_2 = nn.Linear(8, 2)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, cpd_embeddings, target_embeddings, cpd_mask=None, target_mask=None):
        '''
             cpd_embeddings = (batch_size * max_num_atoms * num_of_atom_features)
             target_embeddings = (batch_size * max_target_length * hidden_dim) ===> encode target embeddings based on this dimension
        '''
        cpd_embeddings = self.linear(cpd_embeddings)
        # cpd_embeddings = (batch_size * max_num_atoms * hidden_dim)
        #Usually we pass 3 times through the IntertwiningLayer
        for layer in self.layers:
            cpd_embeddings = layer(cpd_embeddings, target_embeddings, cpd_mask, target_mask)
        # cpd_embeddings = (batch_size * max_num_atoms * hidden_dim)
        # Use norm to determine which atom is significant
        norm = torch.linalg.norm(cpd_embeddings, dim=2)
        # norm = (batch_size * max_num_atoms)
        norm = F.softmax(norm, dim=1)
        sum = torch.zeros((cpd_embeddings.shape[0], self.hidden_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = cpd_embeddings[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        # sum = (batch_size * hidden_dim)
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label    
    
    
'''
    PhyChemDG: The model is based on physicochemical properties for targets and Molecule Graphs (atom interactions) for compounds
'''
class PhyChemDG(nn.Module):
    def __init__(self, target_feature_capturer, intertwining_transformer, device, atom_dim=34):
        super().__init__()
        self.target_feature_capturer = target_feature_capturer
        self.intertwining_transformer = intertwining_transformer
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input = (batch * num_node * atom_dim)
        # adj = (batch * num_node * num_node)
        support = torch.matmul(input, self.weight)
        # support = (batch * num_node * atom_dim)
        output = torch.bmm(adj, support)
        # output = (batch * num_node * atom_dim)
        return output

    def generate_masks(self, all_cpds_num_atoms, all_tar_num_aa, max_num_atoms, max_target_length):
        batch_size = len(all_cpds_num_atoms)
        compound_mask = torch.zeros((batch_size, max_num_atoms))
        protein_mask = torch.zeros((batch_size, max_target_length))
        for i in range(batch_size):
            compound_mask[i, :all_cpds_num_atoms[i]] = 1
            protein_mask[i, :all_tar_num_aa[i]] = 1
        compound_mask = torch.unsqueeze(torch.unsqueeze(compound_mask, 1), 3).to(self.device)
        protein_mask = torch.unsqueeze(torch.unsqueeze(protein_mask, 1), 2).to(self.device) 
        return compound_mask, protein_mask


    #compounds_atom_features, compounds_adjacency_matrices, target_input_features, all_cpds_num_atoms
    def forward(self, compounds_atom_features, compounds_adjacency_matrices, target_input_features, all_cpds_num_atoms, all_tar_num_aa):
         # compounds_adjacency_matrices (batch_size * max_num_atoms * max_num_atoms) in our case, max_num_atoms = 148
        # compounds_atom_features (batch_size * max_num_atoms * num_of_atom_features) in our case, num_of_atom_features = 34 and max_num_atoms = 148
        # target_input_features (batch_size * max_target_length * embedding_size) in our case, max_target_length = 1400, embedding_size = 20
        max_num_atoms = compounds_atom_features.shape[1]
        max_target_length = target_input_features.shape[1]
        compound_mask, protein_mask = self.generate_masks(all_cpds_num_atoms, all_tar_num_aa, max_num_atoms, max_target_length)
        # target_embedding = (batch size * max_target_length * hidden dim)
        target_embedding = self.target_feature_capturer(target_input_features)
        compounds_atom_features = self.gcn(compounds_atom_features, compounds_adjacency_matrices)
        out = self.intertwining_transformer(compounds_atom_features, target_embedding, compound_mask, protein_mask)
        return out

    def __call__(self, data, train=True):
        compounds_atom_features, compounds_adjacency_matrices, target_input_features, batch_Y, all_cpds_num_atoms, all_tar_num_aa = data
        Loss = torch.nn.CrossEntropyLoss()
        if(train): #In the case of training the model
            predicted_interaction = self.forward(compounds_atom_features, compounds_adjacency_matrices, target_input_features, all_cpds_num_atoms, all_tar_num_aa)
            loss = Loss(predicted_interaction, batch_Y)
            return loss
        else: #In the case of testing the model
            predicted_interaction = self.forward(compounds_atom_features, compounds_adjacency_matrices, target_input_features, all_cpds_num_atoms, all_tar_num_aa)
            correct_labels = batch_Y.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores
        
  
'''
    TrainerPhyChemDG class: Takes the model (in this case: CMapDG) and trains it based on the training dataset
'''
class TrainerPhyChemDG(object):
    def __init__(self, model, learning_rate, weight_decay, batch):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.batch = batch

    def train(self, batch_features, device):
        self.model.train()
        self.optimizer.zero_grad()
        # compounds_adjacency_matrices (batch_size * max_num_atoms * max_num_atoms) in our case, max_num_atoms = 148
        # compounds_atom_features (batch_size * max_num_atoms * num_of_atom_features) in our case, num_of_atom_features = 34 and max_num_atoms = 148
        # target_input_features (batch_size * max_target_length * embedding_size) in our case, max_target_length = 1400, embedding_size = 20
        loss = self.model(batch_features)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        

'''
    TesterPhyChemDG: Test the PhyChemDG model
'''
class TesterPhyChemDG(object):
    def __init__(self, model):
        self.model = model

    def test(self, test_features):
        self.model.eval()
        return self.model(test_features, train=False)