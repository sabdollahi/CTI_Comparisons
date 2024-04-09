import os
import csv
import math
import pickle
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, average_precision_score
from sklearn.metrics import confusion_matrix, roc_curve, recall_score, precision_score
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from .DTIConvSeqSMILES_simple import DTIConvSeqSMILES
from .DTISeq2DFP_simple import DTISeq2DFP
from .BERT2DFP_model import BERT2DFP
from .UniRep2DFP_model import UniRep2DFP
from .DTISeqE3FP_model import DTISeqE3FP
from .GrAttCPI_model import GrAttCPI
from .PhyGrAtt_model import PhyGrAtt
from .Phys_DrugGraph_model import TargetFeatureCapturer, SelfAttention, PositionBasedConv1d, IntertwiningLayer, IntertwiningTransformer, PhyChemDG, TrainerPhyChemDG, TesterPhyChemDG
import sys
sys.path.append('..')
from features.FeatureExtraction import FeaturePreparation
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE




class TrainAndSaveModels:
    def __init__(self):
        self.main_path = '../Datasets/BindingDatasets_3DAvailable'
        
        
    def plot_tSNE_DeepDTA_target_LFs(self, use_biophysicochemical_props, use_gpu = False, gpu_device = 0):
        '''
            To plot the learned features of the DeepDTA model utilizing tSNE plot
            NOTE: Before running this method, you MUST train and save the model using 'train_save_DTIConvSeqSMILES' method
            NOTE: The external datasets for extracting the targets' learned features are located on "Datasets/ExternalDatasets/BindingDatasets"
        '''
        feature_preparation = FeaturePreparation()
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        loaded_model = DTIConvSeqSMILES()
        loaded_model.load_state_dict(torch.load('../Datasets/BindingDatasets_3DAvailable/DeepDTA_trained.pth', map_location=dev))
        loaded_model.eval()
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        if(use_biophysicochemical_props):
            aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        smiles_maximum_length = 348
        targets_max_length = 1400
        X = []
        y = []
        i = 0
        isItFirstTime = True
        labels = ['Epigenetic Regulators', 'Ion-Channels', 'Membrane Receptors', 'Transcription Factors', 'Transporters', 'Hydrolases', 'Oxidoreductases', 'Proteases', 'Transferases', 'Other Enzymes']
        ttypes = ['epigenetic-regulators', 'ion-channels', 'membrane-receptors', 'transcription-factors', 'transporters', 'hydrolases', 'oxidoreductases', 'proteases', 'transferases', 'other-enzymes']
        for ttype in ttypes:
            external_df = pd.read_csv(f'../Datasets/ExternalDatasets/BindingDatasets/{ttype}_BD.csv')
            targets_arr = external_df['Sequence'].to_numpy()
            target_input_features = []
            if(use_biophysicochemical_props):
                target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            else:
                target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            target_rep_vec = loaded_model.tar_conv1(target_input_features)
            target_rep_vec = loaded_model.tar_conv2(target_rep_vec)
            target_rep_vec = loaded_model.tar_drop1(F.relu(loaded_model.tar_fc1(target_rep_vec.view(-1, 200 * 5))))
            target_rep_vec = loaded_model.tar_drop2(F.relu(loaded_model.tar_fc2(target_rep_vec)))
            target_rep_vec = loaded_model.tar_fc3(target_rep_vec)
            if(use_gpu):
                target_rep_vec = target_rep_vec.cpu().detach().numpy()
            else:            
                target_rep_vec = target_rep_vec.detach().numpy()
            y_arr = np.full((target_rep_vec.shape[0], ), i)
            if(isItFirstTime):
                isItFirstTime = False
                X = target_rep_vec
                y = y_arr
            else:
                X = np.concatenate((X, target_rep_vec))
                y = np.concatenate((y, y_arr))
            i += 1
        print('Start to plot the tSNE chart!')
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        colors = ['gold', 'darkorange', 'violet', 'deepskyblue', 'lime', 'darkorchid', 'blue', 'green', 'red', 'sienna']
        c = [colors[label] for label in y]
        for i in range(len(labels)):
            go_once = True
            for j in range(len(X)):
                if y[j] == i:
                    if(go_once):
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], label=labels[i], s=7)
                        go_once = False
                    else:
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], s=7)
        plt.legend(fontsize=12, markerscale=2.5, bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig('../Datasets/BindingDatasets_3DAvailable/tSNE_DeepDTA.jpg', format='jpeg', dpi=300, bbox_inches='tight')
    
        
    def train_save_DeepDTA(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0):
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation()    
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")        
        training_df = pd.read_csv(f'{self.main_path}/Drug-Target-Binding-v3-nonmetal.csv')
        smiles_np_array = training_df['Canonical SMILES'].unique()
        smiles_characters, smiles_maximum_length = feature_preparation.extract_all_SMILES_chars_and_max_len(smiles_np_array)
        smiles_maximum_length = 348
        print(f"The maximum size of SMILES is: {smiles_maximum_length}")
        print(f"SMILES characters' embedding length is: {str(len(smiles_characters))}")
        smiles_chars_embedding_length = len(smiles_characters)
        #The maximum number of amino acid is 1400
        targets_max_length = 1400
        smiles_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(smiles_characters)
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        training_df = shuffle(training_df)
        training_index = training_df.index
        if(use_biophysicochemical_props):
            aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        print(f"Amino acids' embedding length is: {str(aa_embedding_length)}")
        model = DTIConvSeqSMILES()
        model.to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("DeepDTA: Training the model has started!")
        #Training Phase
        count_epoch = 1
        for shuffled_epoch in range(num_of_shuffle_epochs):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(num_of_epochs):
                running_loss = 0
                b = 0
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    compound_input_features = feature_preparation.generate_matrices(smiles_onehot_vectors_dict, smiles_arr, smiles_maximum_length)
                    target_input_features = []
                    if(use_biophysicochemical_props):
                        target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    else:
                        target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
                    optimizer.zero_grad()
                    target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
                    compound_input_features = compound_input_features.reshape((compound_input_features.shape[0], 1, smiles_maximum_length, smiles_chars_embedding_length))
                    compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
                    target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
                    Y_hat = model(compound_input_features, target_input_features)
                    loss = criterion(Y_hat, batch_Y)
                    loss.backward()
                    batch_loss = loss.item()
                    running_loss += batch_loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    b += 1
                epoch_loss = running_loss/b
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("DeepDTA: Saving the model has started!")
        torch.save(model.state_dict(), f'{self.main_path}/DeepDTA_trained.pth')
       
    
    def train_save_DeepCAT(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0):
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation()    
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")        
        training_df = pd.read_csv(f'{self.main_path}/Drug-Target-Binding-v3-nonmetal.csv')
        smiles_np_array = training_df['Canonical SMILES'].unique()
        smiles_characters, smiles_maximum_length = feature_preparation.extract_all_SMILES_chars_and_max_len(smiles_np_array)
        smiles_maximum_length = 348
        print(f"The maximum size of SMILES is: {smiles_maximum_length}")
        print(f"SMILES characters' embedding length is: {str(len(smiles_characters))}")
        smiles_chars_embedding_length = len(smiles_characters)
        #The maximum number of amino acid is 1400
        targets_max_length = 1400
        smiles_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(smiles_characters)
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        training_df = shuffle(training_df)
        training_index = training_df.index
        if(use_biophysicochemical_props):
            aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        print(f"Amino acids' embedding length is: {str(aa_embedding_length)}")
        model = DTIConvSeqSMILES()
        model.to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("DeepCAT: Training the model has started!")
        #Training Phase
        count_epoch = 1
        for shuffled_epoch in range(num_of_shuffle_epochs):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(num_of_epochs):
                running_loss = 0
                b = 0
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    compound_input_features = feature_preparation.generate_matrices(smiles_onehot_vectors_dict, smiles_arr, smiles_maximum_length)
                    target_input_features = []
                    if(use_biophysicochemical_props):
                        target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    else:
                        target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
                    optimizer.zero_grad()
                    target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
                    compound_input_features = compound_input_features.reshape((compound_input_features.shape[0], 1, smiles_maximum_length, smiles_chars_embedding_length))
                    compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
                    target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
                    Y_hat = model(compound_input_features, target_input_features)
                    loss = criterion(Y_hat, batch_Y)
                    loss.backward()
                    batch_loss = loss.item()
                    running_loss += batch_loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    b += 1
                epoch_loss = running_loss/b
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("DeepCAT: Saving the model has started!")
        torch.save(model.state_dict(), f'{self.main_path}/DeepCAT_trained.pth')
    
        
        
    def plot_tSNE_AlphaFoldGrAtts_target_LFs(self, use_gpu=False, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):
        dim_out = 4
        dim_hidden = 8
        num_of_aa_features = 20
        num_of_atom_features = 34
        target_graph_adr = '../Datasets/GAT_Prepared_Graphs/RING_based'
        compound_graph_adr = '../Datasets/GAT_Prepared_Graphs/Drugs'
        feature_preparation = FeaturePreparation(target_graph_adr, compound_graph_adr, drug_ids_adr) 
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        loaded_model = GrAttCPI(num_of_aa_features, num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
        loaded_model.load_state_dict(torch.load('../Datasets/BindingDatasets_3DAvailable/AlphaFoldGrAtts_trained.pth', map_location=dev))
        loaded_model.eval()
        X = []
        y = []
        i = 0
        isItFirstTime = True
        labels = ['Epigenetic Regulators', 'Ion-Channels', 'Membrane Receptors', 'Transcription Factors', 'Transporters', 'Hydrolases', 'Oxidoreductases', 'Proteases', 'Transferases', 'Other Enzymes']
        ttypes = ['epigenetic-regulators', 'ion-channels', 'membrane-receptors', 'transcription-factors', 'transporters', 'hydrolases', 'oxidoreductases', 'proteases', 'transferases', 'other-enzymes']
        for ttype in ttypes:
            external_df = pd.read_csv(f'../Datasets/ExternalDatasets/BindingDatasets/{ttype}_BD.csv')
            targets_arr = external_df['3D file'].to_numpy()
            target_pkl_names = []
            for filename in targets_arr:
                pkl_name = filename.split('.pdb')[0] + '.pickle'
                target_pkl_names.append(pkl_name)
            smiles_arr = external_df['Canonical SMILES'].to_numpy()
            target_batch, compound_batch = feature_preparation.get_RING_based_batch_for_GAT(target_pkl_names, smiles_arr, dev)
            target_rep_vec, target_edge_index, target_edge_attr, num_of_aas = target_batch.x, target_batch.edge_index, target_batch.edge_attr, target_batch.num_of_aa
            batch_size = int(target_rep_vec.shape[0]/1400)
            target_rep_vec = F.dropout(target_rep_vec, p=0.2)
            target_rep_vec = loaded_model.gat1_target(target_rep_vec, target_edge_index, target_edge_attr)
            target_rep_vec = F.elu(target_rep_vec)
            target_rep_vec = F.dropout(target_rep_vec, p=0.2)
            target_rep_vec = loaded_model.gat2_target(target_rep_vec, target_edge_index, target_edge_attr)
            target_rep_vec = target_rep_vec.view(batch_size, 1400, loaded_model.dim_out)
            for idx in range(len(num_of_aas)):
                mask_target = torch.zeros(1400).to(torch.device(dev))
                mask_target[:num_of_aas[idx]] = 1
                target_rep_vec[idx] = target_rep_vec[idx] * mask_target.view(-1, 1)
            target_rep_vec = target_rep_vec.view(batch_size, 1400*loaded_model.dim_out)
            target_rep_vec = F.dropout(loaded_model.bn1_target(loaded_model.fc1_target(target_rep_vec)), p=0.4)
            target_rep_vec = F.dropout(loaded_model.bn2_target(loaded_model.fc2_target(target_rep_vec)), p=0.3)
            target_rep_vec = F.dropout(loaded_model.bn3_target(loaded_model.fc3_target(target_rep_vec)), p=0.2)
            if(use_gpu):
                target_rep_vec = target_rep_vec.cpu().detach().numpy()
            else:            
                target_rep_vec = target_rep_vec.detach().numpy()
            y_arr = np.full((target_rep_vec.shape[0], ), i)
            if(isItFirstTime):
                isItFirstTime = False
                X = target_rep_vec
                y = y_arr
            else:
                X = np.concatenate((X, target_rep_vec))
                y = np.concatenate((y, y_arr))
            i += 1
        print('Start to plot the tSNE chart!')
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        colors = ['gold', 'darkorange', 'violet', 'deepskyblue', 'lime', 'darkorchid', 'blue', 'green', 'red', 'sienna']
        c = [colors[label] for label in y]
        for i in range(len(labels)):
            go_once = True
            for j in range(len(X)):
                if y[j] == i:
                    if(go_once):
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], label=labels[i], s=7)
                        go_once = False
                    else:
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], s=7)
        plt.legend(fontsize=12, markerscale=2.5, bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig('../Datasets/BindingDatasets_3DAvailable/tSNE_AlphaFoldGrAtts.jpg', format='jpeg', dpi=300, bbox_inches='tight')
        
        
        
    def train_save_AlphaFoldGrAtts(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):     
        dim_out = 4
        dim_hidden = 8
        num_of_aa_features = 20
        num_of_atom_features = 34
        target_graph_adr = '../Datasets/GAT_Prepared_Graphs/RING_based'
        compound_graph_adr = '../Datasets/GAT_Prepared_Graphs/Drugs'
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(target_graph_adr, compound_graph_adr, drug_ids_adr) 
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training data
        training_df = pd.read_csv(f'{self.main_path}/Drug-Target-Binding-v3-nonmetal.csv')
        training_df = training_df[(training_df['Canonical SMILES'] != '-') & (training_df['3D file'] != 'NoINFO_Cathepsin S_WT_1_D3R.pdb') & (training_df['3D file'] != 'NoINFO_Cathepsin S_C25S_0_D3R.pdb')]
        training_df = training_df[training_df['Canonical SMILES'] != "C=S(=O)(O)c1ccc2c(c1)C(C)(C)C1=[N+]2CCC2OC3CCN4C(=C3C=C12)C(C)(C)c1cc(CC(=O)NCCCC[C@@H](NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCNC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCCCNCCNS(=O)(=O)c2cccc3cnccc23)C(N)=O)ccc14"]
        training_df = training_df[training_df['Canonical SMILES'] != 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)O']
        model = GrAttCPI(num_of_aa_features, num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("AlphaFoldGrAtts: Training the model has started!")
        #Training Phase
        count_epoch = 1
        model.train()
        for shuffled_epoch in range(num_of_shuffle_epochs):
            training_df = shuffle(training_df)
            if(len(training_df)%batch_size == 1):
                training_df = training_df.head(len(training_df)-1)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(num_of_epochs):
                running_loss = 0
                b = 0
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['3D file'].to_numpy()
                    target_pkl_names = []
                    for filename in targets_arr:
                        pkl_name = filename.split('.pdb')[0] + '.pickle'
                        target_pkl_names.append(pkl_name)
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    target_batch, compound_batch = feature_preparation.get_RING_based_batch_for_GAT(target_pkl_names, smiles_arr, dev)
                    optimizer.zero_grad()
                    Y_hat = model(target_batch, compound_batch, dev)
                    loss = criterion(Y_hat, batch_Y)
                    loss.backward()
                    batch_loss = loss.item() 
                    running_loss += batch_loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    b += 1
                epoch_loss = running_loss/b
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("AlphaFoldGrAtts: Saving the model has started!")
        torch.save(model.state_dict(), f'{self.main_path}/AlphaFoldGrAtts_trained.pth')
        
        
        
    def plot_tSNE_PhyGrAtt_target_LFs(self, use_gpu=False, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):
        dim_out = 4
        dim_hidden = 8
        num_of_aa_features = 20
        num_of_atom_features = 34
        targets_max_length = 1400
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        compound_graph_adr = '../Datasets/GAT_Prepared_Graphs/Drugs'
        feature_preparation = FeaturePreparation(compound_graph_adr, drug_ids_adr) 
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        loaded_model = PhyGrAtt(num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
        loaded_model.load_state_dict(torch.load('../Datasets/BindingDatasets_3DAvailable/PhyGrAtt_trained.pth', map_location=dev))
        loaded_model.eval()
        X = []
        y = []
        i = 0
        isItFirstTime = True
        labels = ['Epigenetic Regulators', 'Ion-Channels', 'Membrane Receptors', 'Transcription Factors', 'Transporters', 'Hydrolases', 'Oxidoreductases', 'Proteases', 'Transferases', 'Other Enzymes']
        ttypes = ['epigenetic-regulators', 'ion-channels', 'membrane-receptors', 'transcription-factors', 'transporters', 'hydrolases', 'oxidoreductases', 'proteases', 'transferases', 'other-enzymes']
        for ttype in ttypes:
            external_df = pd.read_csv(f'../Datasets/ExternalDatasets/BindingDatasets/{ttype}_BD.csv')
            targets_arr = external_df['Sequence'].to_numpy()
            target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            target_rep_vec = loaded_model.tar_conv1(target_input_features)
            target_rep_vec = loaded_model.tar_conv2(target_rep_vec)
            target_rep_vec = loaded_model.tar_drop1(F.relu(loaded_model.tar_fc1(target_rep_vec.view(-1, 200 * 5))))
            target_rep_vec = loaded_model.tar_drop2(F.relu(loaded_model.tar_fc2(target_rep_vec)))
            target_rep_vec = loaded_model.tar_fc3(target_rep_vec)
            if(use_gpu):
                target_rep_vec = target_rep_vec.cpu().detach().numpy()
            else:            
                target_rep_vec = target_rep_vec.detach().numpy()
            y_arr = np.full((target_rep_vec.shape[0], ), i)
            if(isItFirstTime):
                isItFirstTime = False
                X = target_rep_vec
                y = y_arr
            else:
                X = np.concatenate((X, target_rep_vec))
                y = np.concatenate((y, y_arr))
            i += 1
        print('Start to plot the tSNE chart!')
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        colors = ['gold', 'darkorange', 'violet', 'deepskyblue', 'lime', 'darkorchid', 'blue', 'green', 'red', 'sienna']
        c = [colors[label] for label in y]
        for i in range(len(labels)):
            go_once = True
            for j in range(len(X)):
                if y[j] == i:
                    if(go_once):
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], label=labels[i], s=7)
                        go_once = False
                    else:
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], s=7)
        plt.legend(fontsize=12, markerscale=2.5, bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig('../Datasets/BindingDatasets_3DAvailable/tSNE_PhyGrAtt.jpg', format='jpeg', dpi=300, bbox_inches='tight')
        
        
        
    def train_save_PhyGrAtt(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):
        dim_out = 4
        dim_hidden = 8
        num_of_aa_features = 20
        num_of_atom_features = 34
        targets_max_length = 1400
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        compound_graph_adr = '../Datasets/GAT_Prepared_Graphs/Drugs'
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(compound_graph_adr, drug_ids_adr) 
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training data
        training_df = pd.read_csv(f'{self.main_path}/Drug-Target-Binding-v3-nonmetal.csv')
        training_df = training_df[training_df['Canonical SMILES'] != "C=S(=O)(O)c1ccc2c(c1)C(C)(C)C1=[N+]2CCC2OC3CCN4C(=C3C=C12)C(C)(C)c1cc(CC(=O)NCCCC[C@@H](NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCNC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCCCNCCNS(=O)(=O)c2cccc3cnccc23)C(N)=O)ccc14"]
        training_df = training_df[training_df['Canonical SMILES'] != 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)O']
        model = PhyGrAtt(num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("PhyGrAtt: Training the model has started!")
        #Training Phase
        count_epoch = 1
        model.train()
        for shuffled_epoch in range(num_of_shuffle_epochs):
            training_df = shuffle(training_df)
            if(len(training_df)%batch_size == 1):
                training_df = training_df.head(len(training_df)-1)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(num_of_epochs):
                running_loss = 0
                b = 0
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    compound_batch = feature_preparation.get_Drug_batch_for_GAT(smiles_arr, dev)
                    optimizer.zero_grad()
                    target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
                    Y_hat = model(target_input_features, compound_batch, dev)
                    loss = criterion(Y_hat, batch_Y)
                    loss.backward()
                    batch_loss = loss.item() 
                    running_loss += batch_loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    b += 1
                epoch_loss = running_loss/b
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("PhyGrAtt: Saving the model has started!")
        torch.save(model.state_dict(), f'{self.main_path}/PhyGrAtt_trained.pth')
        
        
        
    def plot_tSNE_E3FP_target_LFs(self, name_3D_FP="drug_3D_fingerprints_v1.pickle", use_gpu=False, max_num_of_conformers=3):
        feature_preparation = FeaturePreparation(name_3D_FP)   
        e3fp_dict = pickle.load(open('../Datasets/drug_3D_fingerprints_v1.pickle','rb'))
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        targets_max_length = 1400
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        loaded_model = DTISeqE3FP(max_num_of_conformers)
        loaded_model.load_state_dict(torch.load('../Datasets/BindingDatasets_3DAvailable/E3FP_trained.pth', map_location=dev))
        loaded_model.eval()
        X = []
        y = []
        i = 0
        isItFirstTime = True
        labels = ['Epigenetic Regulators', 'Ion-Channels', 'Membrane Receptors', 'Transcription Factors', 'Transporters', 'Hydrolases', 'Oxidoreductases', 'Proteases', 'Transferases', 'Other Enzymes']
        ttypes = ['epigenetic-regulators', 'ion-channels', 'membrane-receptors', 'transcription-factors', 'transporters', 'hydrolases', 'oxidoreductases', 'proteases', 'transferases', 'other-enzymes']
        for ttype in ttypes:
            external_df = pd.read_csv(f'../Datasets/ExternalDatasets/BindingDatasets/{ttype}_BD.csv')
            targets_arr = external_df['Sequence'].to_numpy()
            target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            target_rep_vec = loaded_model.tar_conv1(target_input_features)
            target_rep_vec = loaded_model.tar_conv2(target_rep_vec)
            target_rep_vec = loaded_model.tar_drop1(F.relu(loaded_model.tar_fc1(target_rep_vec.view(-1, 200 * 5))))
            target_rep_vec = loaded_model.tar_drop2(F.relu(loaded_model.tar_fc2(target_rep_vec)))
            target_rep_vec = loaded_model.tar_drop3(F.relu(loaded_model.tar_fc3(target_rep_vec)))
            if(use_gpu):
                target_rep_vec = target_rep_vec.cpu().detach().numpy()
            else:            
                target_rep_vec = target_rep_vec.detach().numpy()
            y_arr = np.full((target_rep_vec.shape[0], ), i)
            if(isItFirstTime):
                isItFirstTime = False
                X = target_rep_vec
                y = y_arr
            else:
                X = np.concatenate((X, target_rep_vec))
                y = np.concatenate((y, y_arr))
            i += 1
        print('Start to plot the tSNE chart!')
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        colors = ['gold', 'darkorange', 'violet', 'deepskyblue', 'lime', 'darkorchid', 'blue', 'green', 'red', 'sienna']
        c = [colors[label] for label in y]
        for i in range(len(labels)):
            go_once = True
            for j in range(len(X)):
                if y[j] == i:
                    if(go_once):
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], label=labels[i], s=7)
                        go_once = False
                    else:
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], s=7)
        plt.legend(fontsize=12, markerscale=2.5, bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig('../Datasets/BindingDatasets_3DAvailable/tSNE_E3FP.jpg', format='jpeg', dpi=300, bbox_inches='tight')
        
        
        
    def train_save_DTISeqE3FP(self,num_of_epochs, num_of_shuffle_epochs, name_3D_FP, max_num_of_conformers, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0):  
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_3D_FP)   
        e3fp_dict = pickle.load(open('../Datasets/drug_3D_fingerprints_v1.pickle','rb'))
        unavailable_smiles = []
        for smiles in e3fp_dict:
            if(e3fp_dict[smiles]=='-'):
                unavailable_smiles.append(smiles)
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training data
        training_df = pd.read_csv(f'{self.main_path}/Drug-Target-Binding-v3-nonmetal.csv')
        #The following drugs have no E3FP!
        training_df = training_df[(training_df['Canonical SMILES'] != '-') & (training_df['Canonical SMILES'] != 'CC[C@]1(O)C[C@@H]2C[N@@](CCc3c([nH]c4ccccc34)[C@@](C(=O)OC)(c3cc4c(cc3OC)N(C)[C@H]3[C@@](O)(C(=O)NNC(=O)OCCSSC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(=O)O)NC(=O)CC[C@H](NC(=O)c5ccc(NCc6c[nH]c7nc(N)nc(=O)c-7n6)cc5)C(=O)O)C(=O)O)[C@H](O)[C@]5(CC)C=CCN6CC[C@]43[C@@H]65)C2)C1') & (training_df['Canonical SMILES'] != 'CC1(C)C(/C=C/C2=C(Oc3ccc(C[C@H](NC(=O)c4ccc(NCc5cnc6nc(N)[nH]c(=O)c6n5)cc4)C(=O)O)cc3)/C(=C/C=C3/N(CCCCS(=O)(=O)O)c4ccc(S(=O)(=O)O)cc4C3(C)C)CCC2)=[N+](CCCCS(=O)(=O)O)c2ccc(S(=O)(=O)[O-])cc21')]
        training_df = training_df[~training_df['Canonical SMILES'].isin(unavailable_smiles)]   
        #The maximum number of amino acid is 1400
        targets_max_length = 1400
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        training_df = shuffle(training_df)
        training_index = training_df.index
        if(use_biophysicochemical_props):
            aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        print(f"Amino acids' embedding length is: {str(aa_embedding_length)}")
        model = DTISeqE3FP(max_num_of_conformers)
        model.to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("E3FP-based: Training the model has started!")
        #Training Phase
        count_epoch = 1
        for shuffled_epoch in range(num_of_shuffle_epochs):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(num_of_epochs):
                running_loss = 0
                b = 0
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    # compound_input_features dimensions: (batch_size, max_num_of_conformers, 2048)
                    compound_input_features = feature_preparation.get_3D_drug_fingerprints(smiles_arr, max_num_of_conformers)
                    # target_input_features dimensions: (batch_size, max_target_length, embedding_size) in our case, max_target_length = 1400, embedding_size = 20
                    target_input_features = []
                    if(use_biophysicochemical_props):
                        target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    else:
                        target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
                    optimizer.zero_grad()
                    target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
                    compound_input_features = compound_input_features.reshape((compound_input_features.shape[0], 1, max_num_of_conformers, 2048))
                    compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
                    target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
                    Y_hat = model(compound_input_features, target_input_features)
                    loss = criterion(Y_hat, batch_Y)
                    loss.backward()
                    batch_loss = loss.item() 
                    running_loss += batch_loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    b += 1
                epoch_loss = running_loss/b
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("E3FP-based: Saving the model has started!")
        torch.save(model.state_dict(), f'{self.main_path}/E3FP_trained.pth')   
        
        
        
    def plot_tSNE_2DFP_target_LFs(self, name_2D_FP="drug_2D_fingerprints.pickle", use_gpu = False):
        feature_preparation = FeaturePreparation(name_2D_FP)   
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        targets_max_length = 1400
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        loaded_model = DTISeq2DFP()
        loaded_model.load_state_dict(torch.load('../Datasets/BindingDatasets_3DAvailable/2DFP_trained.pth', map_location=dev))
        loaded_model.eval()
        X = []
        y = []
        i = 0
        isItFirstTime = True
        labels = ['Epigenetic Regulators', 'Ion-Channels', 'Membrane Receptors', 'Transcription Factors', 'Transporters', 'Hydrolases', 'Oxidoreductases', 'Proteases', 'Transferases', 'Other Enzymes']
        ttypes = ['epigenetic-regulators', 'ion-channels', 'membrane-receptors', 'transcription-factors', 'transporters', 'hydrolases', 'oxidoreductases', 'proteases', 'transferases', 'other-enzymes']
        for ttype in ttypes:
            external_df = pd.read_csv(f'../Datasets/ExternalDatasets/BindingDatasets/{ttype}_BD.csv')
            targets_arr = external_df['Sequence'].to_numpy()
            target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            target_rep_vec = loaded_model.tar_conv1(target_input_features)
            target_rep_vec = loaded_model.tar_conv2(target_rep_vec)
            target_rep_vec = loaded_model.tar_drop1(F.relu(loaded_model.tar_fc1(target_rep_vec.view(-1, 200 * 5))))
            target_rep_vec = loaded_model.tar_drop2(F.relu(loaded_model.tar_fc2(target_rep_vec)))
            target_rep_vec = loaded_model.tar_fc3(target_rep_vec)
            if(use_gpu):
                target_rep_vec = target_rep_vec.cpu().detach().numpy()
            else:            
                target_rep_vec = target_rep_vec.detach().numpy()
            y_arr = np.full((target_rep_vec.shape[0], ), i)
            if(isItFirstTime):
                isItFirstTime = False
                X = target_rep_vec
                y = y_arr
            else:
                X = np.concatenate((X, target_rep_vec))
                y = np.concatenate((y, y_arr))
            i += 1
        print('Start to plot the tSNE chart!')
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        colors = ['gold', 'darkorange', 'violet', 'deepskyblue', 'lime', 'darkorchid', 'blue', 'green', 'red', 'sienna']
        c = [colors[label] for label in y]
        for i in range(len(labels)):
            go_once = True
            for j in range(len(X)):
                if y[j] == i:
                    if(go_once):
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], label=labels[i], s=7)
                        go_once = False
                    else:
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], s=7)
        plt.legend(fontsize=12, markerscale=2.5, bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig('../Datasets/BindingDatasets_3DAvailable/tSNE_2DFP.jpg', format='jpeg', dpi=300, bbox_inches='tight')
        
        
        
    def train_save_DTISeq2DFP(self,num_of_epochs, num_of_shuffle_epochs, name_2D_FP, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0):
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_2D_FP)   
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training data
        training_df = pd.read_csv(f'{self.main_path}/Drug-Target-Binding-v3-nonmetal.csv')
        training_df = training_df[training_df['Canonical SMILES'] != '-']
        #The maximum number of amino acid is 1400
        targets_max_length = 1400
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        training_df = shuffle(training_df)
        training_index = training_df.index
        if(use_biophysicochemical_props):
            aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        print(f"Amino acids' embedding length is: {str(aa_embedding_length)}")
        model = DTISeq2DFP()
        model.to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("2DFP-based: Training the model has started!")
        #Training Phase
        count_epoch = 1
        for shuffled_epoch in range(num_of_shuffle_epochs):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(num_of_epochs):
                running_loss = 0
                b = 0
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    # compound_input_features dimensions: (batch_size, 3239)
                    compound_input_features = feature_preparation.get_2D_drug_fingerprints(smiles_arr)
                    # target_input_features dimensions: (batch_size, max_target_length, embedding_size) in our case, max_target_length = 1400, embedding_size = 20
                    target_input_features = []
                    if(use_biophysicochemical_props):
                        target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    else:
                        target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
                    optimizer.zero_grad()
                    target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
                    compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
                    target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
                    Y_hat = model(compound_input_features, target_input_features)
                    loss = criterion(Y_hat, batch_Y)
                    loss.backward()
                    batch_loss = loss.item()
                    running_loss += batch_loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    b += 1
                epoch_loss = running_loss/b
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("2DFP-based: Saving the model has started!")
        torch.save(model.state_dict(), f'{self.main_path}/2DFP_trained.pth')
        
        
        
    def get_max_num_atoms(self, drug_graphs_and_features_adr):
        '''
            To recompute the maximum number of atoms of the drugs [Returns an integer number]
            drug_graphs_and_features_adr: The complete address of the DIRECTORY that contains the adjacency matrices (2D graphs) and the atoms' features of the drugs
        '''
        max_num_atoms = 0
        for adj_file in os.listdir(drug_graphs_and_features_adr):
            if(adj_file.find("AdjacencyMatrix") != -1):
                atom_num = int(np.load(f'{drug_graphs_and_features_adr}/{adj_file}').shape[0])
                if(atom_num > max_num_atoms):
                    max_num_atoms = atom_num
        return max_num_atoms

    
    
    def plot_tSNE_PhyChemDG_target_LFs(self, use_gpu = False, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv', drug_graphs_and_features_adr = '../Datasets/Drugs_2D_Graphs_AtomFeatures'):
        protein_dim = 100
        atom_dim = 34
        hidden_dim = 64
        n_layers = 3
        n_heads = 8
        middle_dim = 256
        dropout = 0.1
        weight_decay = 1e-4
        decay_interval = 5
        lr_decay = 1.0
        kernel_size = 7
        max_num_atoms = 148
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        device = torch.device(dev)
        print(f"***********Running on '{dev}'***********")
        feature_preparation = FeaturePreparation(drug_graphs_and_features_adr, drug_ids_adr)
        targets_max_length = 1400
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        target_featurizer = TargetFeatureCapturer(hidden_dim, dropout, device)
        intertwining_transformer = IntertwiningTransformer(atom_dim, hidden_dim, n_layers, n_heads, middle_dim, IntertwiningLayer, SelfAttention, PositionBasedConv1d, dropout, device)
        loaded_model = PhyChemDG(target_featurizer, intertwining_transformer, device)
        loaded_model.load_state_dict(torch.load('../Datasets/BindingDatasets_3DAvailable/PhyChemDG_trained.pth', map_location=dev))
        loaded_model.eval()
        X = []
        y = []
        i = 0
        isItFirstTime = True
        labels = ['Epigenetic Regulators', 'Ion-Channels', 'Membrane Receptors', 'Transcription Factors', 'Transporters', 'Hydrolases', 'Oxidoreductases', 'Proteases', 'Transferases', 'Other Enzymes']
        ttypes = ['epigenetic-regulators', 'ion-channels', 'membrane-receptors', 'transcription-factors', 'transporters', 'hydrolases', 'oxidoreductases', 'proteases', 'transferases', 'other-enzymes']
        for ttype in ttypes:
            external_df = pd.read_csv(f'../Datasets/ExternalDatasets/BindingDatasets/{ttype}_BD.csv')
            targets_arr = external_df['Sequence'].to_numpy()
            smiles_arr = external_df['Canonical SMILES'].to_numpy()
            all_tar_num_aa = []
            for target in targets_arr:
                all_tar_num_aa.append(len(target))
            target_input_features = []
            target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            compounds_adjacency_matrices, compounds_atom_features, all_cpds_num_atoms = feature_preparation.get_2Dgraphs_atom_features(smiles_arr, max_num_atoms, device)
            max_target_length = target_input_features.shape[1]
            compound_mask, protein_mask = loaded_model.generate_masks(all_cpds_num_atoms, all_tar_num_aa, max_num_atoms, max_target_length)
            target_rep_vec = loaded_model.target_feature_capturer(target_input_features)
            target_rep_vec = target_rep_vec.reshape((target_rep_vec.shape[0], targets_max_length*hidden_dim))
            target_rep_vec = torch.FloatTensor(target_rep_vec).to(torch.device(dev))
            if(use_gpu):
                target_rep_vec = target_rep_vec.cpu().detach().numpy()
            else:            
                target_rep_vec = target_rep_vec.detach().numpy()
            y_arr = np.full((target_rep_vec.shape[0], ), i)
            if(isItFirstTime):
                isItFirstTime = False
                X = target_rep_vec
                y = y_arr
            else:
                X = np.concatenate((X, target_rep_vec))
                y = np.concatenate((y, y_arr))
            i += 1
        print('Start to plot the tSNE chart!')
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        colors = ['gold', 'darkorange', 'violet', 'deepskyblue', 'lime', 'darkorchid', 'blue', 'green', 'red', 'sienna']
        c = [colors[label] for label in y]
        for i in range(len(labels)):
            go_once = True
            for j in range(len(X)):
                if y[j] == i:
                    if(go_once):
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], label=labels[i], s=7)
                        go_once = False
                    else:
                        plt.scatter(X_tsne[j,0], X_tsne[j,1], c=colors[i], s=7)
        plt.legend(fontsize=12, markerscale=2.5, bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig('../Datasets/BindingDatasets_3DAvailable/tSNE_PhyChemDG.jpg', format='jpeg', dpi=300, bbox_inches='tight')
            
            
        
    def train_save_PhyChemDG(self, num_of_epochs, num_of_shuffle_epochs, batch_size = 64, learning_rate = 0.0001, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0, recompute_max_num_atoms = False, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv', drug_graphs_and_features_adr = '../Datasets/Drugs_2D_Graphs_AtomFeatures'):
        #Parameters of the PhyChemDG model
        protein_dim = 100
        atom_dim = 34
        hidden_dim = 64
        n_layers = 3
        n_heads = 8
        middle_dim = 256
        dropout = 0.1
        weight_decay = 1e-4
        decay_interval = 5
        lr_decay = 1.0
        kernel_size = 7
        #In our dataset, the maximum number of atoms in the compounds is 148 (You can recompute it by setting the input argument 'recompute_max_num_atoms' to True)
        max_num_atoms = 148
        if(recompute_max_num_atoms):
            max_num_atoms = self.get_max_num_atoms(self, drug_graphs_and_features_adr)
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        device = torch.device(dev)
        print(f"***********Running on '{dev}'***********")
        feature_preparation = FeaturePreparation(drug_graphs_and_features_adr, drug_ids_adr)              
        #Load Training data
        training_df = pd.read_csv(f'{self.main_path}/Drug-Target-Binding-v3-nonmetal.csv')
        training_df = training_df[training_df['Canonical SMILES'] != '-']
        training_df = training_df[training_df['Canonical SMILES'] != "C=S(=O)(O)c1ccc2c(c1)C(C)(C)C1=[N+]2CCC2OC3CCN4C(=C3C=C12)C(C)(C)c1cc(CC(=O)NCCCC[C@@H](NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCNC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCCCNCCNS(=O)(=O)c2cccc3cnccc23)C(N)=O)ccc14"]
        training_df = training_df[training_df['Canonical SMILES'] != 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)O']
        #The maximum number of amino acid is 1400
        targets_max_length = 1400
        aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
        aa_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(aminoacids)
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = 20
        training_df = shuffle(training_df)
        training_index = training_df.index
        if(use_biophysicochemical_props):
            aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        print(f"Amino acids' embedding length is: {str(aa_embedding_length)}")
        '''Defining the model instance'''
        target_featurizer = TargetFeatureCapturer(hidden_dim, dropout, device)
        intertwining_transformer = IntertwiningTransformer(atom_dim, hidden_dim, n_layers, n_heads, middle_dim, IntertwiningLayer, SelfAttention, PositionBasedConv1d, dropout, device)
        model = PhyChemDG(target_featurizer, intertwining_transformer, device)
        model.to(device)
        trainer = TrainerPhyChemDG(model, learning_rate, weight_decay, batch_size)
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        count_epoch = 1
        print("PhyChemDG: Training the model has started!")
        for shuffled_epoch in range(num_of_shuffle_epochs):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(num_of_epochs):
                if(epoch % decay_interval == 0):
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay
                running_loss = 0
                b = 0
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    labels_new = torch.zeros(len(batch_Y), dtype=torch.long, device=device)
                    i = 0
                    for label in batch_Y:
                        labels_new[i] = label
                        i += 1
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    all_tar_num_aa = []
                    for target in targets_arr:
                        all_tar_num_aa.append(len(target))
                    compounds_adjacency_matrices, compounds_atom_features, all_cpds_num_atoms = feature_preparation.get_2Dgraphs_atom_features(smiles_arr, max_num_atoms, device)
                    target_input_features = []
                    if(use_biophysicochemical_props):
                        target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    else:
                        target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
                    target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
                    batch_features = (compounds_atom_features, compounds_adjacency_matrices, target_input_features, labels_new, all_cpds_num_atoms, all_tar_num_aa)
                    loss_train = trainer.train(batch_features, device)
                    b += 1
                    running_loss += loss_train
                epoch_loss = running_loss/b
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("PhyChemDG: Saving the model has started!")
        torch.save(model.state_dict(), f'{self.main_path}/PhyChemDG_trained.pth')