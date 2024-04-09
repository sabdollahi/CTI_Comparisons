import numpy as np
import pickle
import torch
import os
import pandas as pd
from sklearn.decomposition import PCA
from torch_geometric.data import Data, Batch

class FeaturePreparation:
    def __init__(self, *args):
        if(len(args) == 0):
            self.drug_fingerprints = ""
            self.drug_graphs_and_features_adr = ""
            self.drug_ids_adr = ""
        elif(len(args) == 1):
            #args[0] is the name of the drug 2D fingerprint pickle file
            #The pickle file MUST be located in the 'Datasets' folder.
            self.drug_fingerprints = pickle.load(open(f'../Datasets/{args[0]}', "rb"))
            self.drug_graphs_and_features_adr = ""
            self.drug_ids_adr = ""
        elif(len(args) == 2):
            self.drug_fingerprints = ""
            self.drug_graphs_and_features_adr = args[0]
            self.drug_ids_adr = args[1]
        elif(len(args) == 3):
            self.target_graph_adr = args[0]
            self.compound_graph_adr = args[1]
            self.drug_ids_adr = args[2]
        elif(len(args) == 5):
            self.drug_graphs_and_features_adr = args[0]
            self.drug_ids_adr = args[1]
            self.drug_fingerprints = pickle.load(open(f'../Datasets/{args[2]}', "rb"))
            self.tar_tensors_dir = args[3]
            self.tar_npy_dir = args[4]
            
            
            
    def extract_all_SMILES_chars_and_max_len(self, smiles_np_array):
        '''
            This function extracts all unique characters of the SMILES notations in 'smiles_np_array' AND it finds the maximum length of the SMILES
            smiles_np_array: The array of all SMILES [FOR ALL DATASETS]
            RETURN: list of the charcters
        '''
        all_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'B', 'I', 'P', 'N', 'O', 'S', 'F', 'H', 'Y', 'W', 'K', 'V', 
                     'e', 's', 'o', 'c', 'n', 'p', '(', ')', '=', '-', '#', '[', ']', '@', '+', '/', '\\', '.', '%', 'Br', 'Si', 
                     'Na', 'Li', 'Zn', 'As', 'Se', 'Sr', 'Lu', 'Gd', 'Bi', 'Ca', 'Fe', 'Al', 'Cu', 'Mg', 'Hg', 'Pt', 'Mo', 'Tc', 
                     'Ga', 'Te', 'Cd', 'La', 'Sm', 'Cr', 'Co', 'Xe', 'Zr', 'He', 'Mn', 'Ag', 'Au', 'Sb', 'Ni', 'Rb', 'In', 'Ne', 
                     'Kr', 'Ru', 'Be', 'Ta', 'Tl', 'Ce', 'Nd', 'Re', 'Ra', 'Ti', 'Bk', 'Ba', 'Cs', 'Ho', 'Ac', 'Pd', 'Nb', 'Cl']
        other_atoms = ["Ar", "Ge", "Rh", "Pr", "Pm", "Hf", "Rf",
                       "Eu", "Tb", "Dy", "Er", "Tm", "Yb", "Rg",  
                       "Os", "Ir", "Pb", "Po", "At", "Rn", "Fr", 
                       "Th", "Pa", "Np", "Pu", "Am", "Cm", "Ds", 
                       "Cf", "Es", "Fm", "Md", "No", "Lr", "Mt", 
                       "Db", "Sg", "Bh", "Hs"]
        characters = []
        maximum_length = 0
        for smiles in smiles_np_array:
            num_of_chars = 0
            i = 0
            while(i < len(smiles)):
                num_of_chars += 1
                if(i < len(smiles) - 1):
                    if(smiles[i].isupper() and smiles[i+1].islower()):
                        if((smiles[i]+smiles[i+1] not in all_chars)):
                            if(smiles[i]+smiles[i+1] in other_atoms):
                                twochar = smiles[i]+smiles[i+1]
                                all_chars.append(twochar)
                                print(f"A new character '{twochar}' is recognized!")
                                i += 1
                            elif(smiles[i] not in all_chars):
                                all_chars.append(smiles[i])
                                print(f"A new character '{smiles[i]}' is recognized!")
                        else:
                            i += 1
                    else:
                        if(smiles[i] not in all_chars):
                            all_chars.append(smiles[i])
                            print(f"A new character '{smiles[i]}' is recognized!")
                else:
                    if(smiles[i] not in all_chars):
                        all_chars.append(smiles[i])
                        print(f"A new character '{smiles[i]}' is recognized!")
                i += 1
            if(maximum_length < num_of_chars):
                maximum_length = num_of_chars
        return all_chars, maximum_length


    def generate_one_hot_vectors(self, characters):
        '''
        Get the dictionary of the one hot vectors of the characters
        '''
        one_hot_vectors_dict = {}
        for i in range(len(characters)):
            one_hot_vector = np.zeros(len(characters))
            one_hot_vector[i] = 1
            one_hot_vectors_dict[characters[i]] = one_hot_vector
        return one_hot_vectors_dict


    def generate_SMILES_one_hot_matrix(self, one_hot_vectors_dict, smiles, smiles_maximum_length):
        '''
            NOTE: We also use this function for obtaining onehot/biophysicochemical_props matrix of proteins
            RETURN: a numpy 2D array with a shape of (Maximum number of SMILES charcters of compounds, Number of unique SMILES characters in the one_hot_vectors_dict)
            For instance, if a compound has 126 SMILES characters and the number of unique characters of SMILES is 98. Then, the function generates a numpy array with size of (126, 98) 
            then, it uses zero vectors for padding. So, if the maximum length is 348, finally, it generates a numpy array with size of (348, 98)
            Note that the number of SMILES characters of a compound is not essentially equal to the length of the SMILES. We might have two-character symbols such as 'Li', 'Fe', etc.
            To generate the matrix of onehot vectors for a compound
            smiles: The SMILES notation of the compound
            one_hot_vectors_dict: The dictionary of the SMILES characters and their corresponding onehot vectors
            smiles_maximum_length: The maximum length of the SMILES in all the datasets [We use it for padding]
        '''
        i = 0
        onehot_matrix = []
        zero_vector = np.zeros(len(one_hot_vectors_dict['C']))
        while(i < len(smiles)):
            if(i < len(smiles) - 1):
                if(smiles[i].isupper() and smiles[i+1].islower()):
                    if(smiles[i]+smiles[i+1] in one_hot_vectors_dict):
                        onehot_matrix.append(one_hot_vectors_dict[smiles[i]+smiles[i+1]])
                        i += 1
                elif(smiles[i] in one_hot_vectors_dict):
                    onehot_matrix.append(one_hot_vectors_dict[smiles[i]])
            else:
                if(smiles[i] not in one_hot_vectors_dict):
                    onehot_matrix.append(one_hot_vectors_dict[smiles[i]])
            i += 1
        i = len(onehot_matrix)
        while(i < smiles_maximum_length):
            onehot_matrix.append(zero_vector)
            i += 1
        return np.array(onehot_matrix)


    def generate_matrices(self, one_hot_vectors_dict, seq_array, maximum_length):
        '''
            seq_array: The array of the SMILES/Sequences belongs to different compounds/proteins
            one_hot_vectors_dict: The dictionary of the SMILES/amino acid characters and their corresponding onehot vectors
            maximum_length: The maximum length of the SMILES/sequences in all the datasets
            RETURN: A numpy array with the shape of (Batch size * Maximum number of SMILES/amino-acid charcters of compounds * Number of unique SMILES/amino-acid characters in the one_hot_vectors_dict)
        '''
        onehot_matrices = []
        for seq in seq_array:
            onehot_matrices.append(self.generate_SMILES_one_hot_matrix(one_hot_vectors_dict, seq, maximum_length))
        return np.array(onehot_matrices)


    def get_2Dgraphs_atom_features(self, smiles_arr, max_num_atoms, device):
        '''
            smiles_arr: An array containing SMILES notations of the drugs in a batch 
            max_num_atoms: Maximum number of atoms of drugs 
            RETURN: Generates two numpy arrays:
                        1. The adjacency matrices of the batch
                        2. The atom faetures of the batch
        '''
        all_cpds_num_atoms = []
        batch_size = len(smiles_arr)
        compounds_new = torch.zeros((batch_size,max_num_atoms,34), device=device)
        adjacency_matrices_new = torch.zeros((batch_size, max_num_atoms, max_num_atoms), device=device)
        i = 0
        all_cpds_num_atoms = []
        for smiles in smiles_arr:
            smiles_drugid_df = pd.read_csv(self.drug_ids_adr)
            arr = smiles_drugid_df[smiles_drugid_df['Drug'] == smiles]['Conformers Set Id'].to_numpy()
            if(len(arr) > 0):
                drug_unique_id = int(arr[0])
                adj_matrix = torch.FloatTensor(np.load(f'{self.drug_graphs_and_features_adr}/{drug_unique_id}_AdjacencyMatrix.npy')).to(device)
                compound = torch.FloatTensor(np.load(f'{self.drug_graphs_and_features_adr}/{drug_unique_id}_AtomFeatures.npy')).to(device)
                num_cpd_atoms = compound.shape[0]
                all_cpds_num_atoms.append(num_cpd_atoms)
                compounds_new[i, :num_cpd_atoms, :] = compound
                adj_matrix = adj_matrix + torch.eye(num_cpd_atoms, device=device)
                adjacency_matrices_new[i, :num_cpd_atoms, :num_cpd_atoms] = adj_matrix
                i += 1
            else:
                print(f'ID cannot be found for the drug with the following SMILES: {smiles}')        
        return adjacency_matrices_new, compounds_new, all_cpds_num_atoms

    
    def get_esm2_representations(self, fasta_ids, prot_len=1400):
        '''
            fasta_ids: The uniprot id of proteins
            RETURN: Returns the ESM-2 representations of the amino acids of the proteins 
        '''
        npys = os.listdir(self.tar_npy_dir)
        embedding_dim = 20
        batch_size = len(fasta_ids)
        pca = PCA(n_components=embedding_dim)
        pca_esm2s = np.zeros((batch_size, prot_len, embedding_dim))
        f = 0
        for fasta_id in fasta_ids:
            if( f'{fasta_id}.npy' in npys):
                pca_esm2s[f, :np.load(f'{self.tar_npy_dir}{fasta_id}.npy').shape[0], :] = np.load(f'{self.tar_npy_dir}{fasta_id}.npy')
            else:
                seq_esm2 = torch.load(f'{self.tar_tensors_dir}{fasta_id}.pt')
                pca_esm2s[f, :seq_esm2['representations'][36].numpy().shape[0], :] = pca.fit_transform(seq_esm2['representations'][36].numpy())
            f += 1
        return np.array(pca_esm2s)    
    
    
    def get_Morgan_drug_fingerprints(self, smiles_arr):
        '''
            smiles_arr: An array containing SMILES notations of the drugs in a batch 
            RETURN: Generates a matrix (batch_size*1024) containing Morgan drug fingerprints of the drugs SMILES in 'smiles_arr' 
        '''
        drug_FPs_matrix = []
        for smiles in smiles_arr:
            # Morgan FP: 1024-D
            drug_FPs_matrix.append(self.drug_fingerprints[smiles]['Morgan vector'])
        return np.array(drug_FPs_matrix)
    
    
    def extract_aa_numbers(self, seq):
        seq_rdic = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "D", "E", "R", "H", "K", "G", "P", "O", "U", "X", "B", "Z"]
        seq_dic = {w: i  for i, w in enumerate(seq_rdic)}
        list_numbers = [seq_dic[aa] for aa in seq]
        lenn = len(list_numbers)
        if(lenn < 1400):
            for i in range(lenn, 1400):
                list_numbers.append(25)
        return list_numbers
    
    
    def get_aa_numbers(self, sequences):
        aa_numbers = []
        for seq in sequences:
            aa_numbers.append(self.extract_aa_numbers(seq))
        return aa_numbers
    
    
    def get_2D_drug_fingerprints(self, smiles_arr):
        '''
            smiles_arr: An array containing SMILES notations of the drugs in a batch 
            RETURN: Generates a matrix (batch_size*3239) containing concatenated 2D drug fingerprints of the drugs SMILES in 'smiles_arr' 
        '''
        drug_FPs_matrix = []
        for smiles in smiles_arr:
            # 1024 (Morgan), 167 (MACCS), 1024 (AtomPairs), 1024 (RDKit-2D) = 3239
            concatenated_2D_fingerprints = np.concatenate((self.drug_fingerprints[smiles]['Morgan vector'], self.drug_fingerprints[smiles]['MACCS vector'], self.drug_fingerprints[smiles]['AtomPair vector'], self.drug_fingerprints[smiles]['RDKit-2D vector']))
            drug_FPs_matrix.append(concatenated_2D_fingerprints)
        return np.array(drug_FPs_matrix)
    
    
    def get_3D_drug_fingerprints(self, smiles_arr, max_num_of_conformers):
        '''
            smiles_arr: An array containing SMILES notations of the drugs in a batch 
            max_num_of_conformers: Maximum number of conformers of the drugs in the whole dataset
            RETURN: Generates a matrix (batch_size * max_num_of_conformers * 2048) containing the 3D drug fingerprints (E3FP) of the drugs SMILES in 'smiles_arr'
            NOTE: If the number of conformers of a drug < max_num_of_conformers, we use padding of zeros
        '''
        complete_matrix = []
        for smiles in smiles_arr:
            #print(type(self.drug_fingerprints[smiles]))
            #print(f"'{self.drug_fingerprints[smiles]}'")
            #print(f"'{smiles}'")
            num_of_conformers = len(self.drug_fingerprints[smiles])
            e3fp_length = len(self.drug_fingerprints[smiles][0])
            if(num_of_conformers == max_num_of_conformers):
                complete_matrix.append(self.drug_fingerprints[smiles])
            else:
                for i in range(num_of_conformers, max_num_of_conformers):
                    self.drug_fingerprints[smiles].append(np.zeros(e3fp_length))
                complete_matrix.append(self.drug_fingerprints[smiles])            
        return np.array(complete_matrix)
    
    
    def get_RING_based_batch_for_GAT(self, target_pkl_names, smiles_arr, dev='cpu'):
        '''
            target_pkl_names: A list of string containing pickle file names located in  'self.target_graph_adr'
            smiles_arr: A list of SMILES notations of the drugs in a batch
            dev: the device type
        '''
        max_num_aa = 1400
        graph_list = []
        for target_pkl_name in target_pkl_names:
            if(os.path.exists(f'{self.target_graph_adr}/Edges/{target_pkl_name}')):
                edges = pickle.load(open(f'{self.target_graph_adr}/Edges/{target_pkl_name}','rb'))
                nodes_features = pickle.load(open(f'{self.target_graph_adr}/Nodes_Features/{target_pkl_name}','rb'))
                num_of_aa_features = nodes_features.shape[1]
                num_of_aa = nodes_features.shape[0]
                #Extend the node features based on the maximum number of amino acids
                padded_node_features = np.zeros((max_num_aa, num_of_aa_features))
                padded_node_features[:num_of_aa, : ] = nodes_features
                #Number of features of each node (aa): 20
                edges = torch.as_tensor(edges, dtype=torch.int64).to(torch.device(dev))
                padded_node_features = torch.as_tensor(padded_node_features, dtype=torch.float32).to(torch.device(dev))
                #Load target edge weights
                target_edge_weight = pickle.load(open(f'{self.target_graph_adr}/Edges_Weight/{target_pkl_name}','rb'))
                target_edge_weight = torch.tensor(target_edge_weight, dtype=torch.float).to(torch.device(dev))
                target_edge_weight = target_edge_weight.reshape(target_edge_weight.shape[0],1)    
                graph_list.append(Data(x=padded_node_features, edge_index=edges, edge_attr=target_edge_weight, num_of_aa=num_of_aa))
            else:
                print(f'WARNING) The following pickle file does not exist: {target_pkl_name}')
        target_batch = Batch.from_data_list(graph_list)
        smiles_drugid_df = pd.read_csv(self.drug_ids_adr)
        max_num_atoms = 148
        graph_list = []
        for smiles in smiles_arr:
            arr = smiles_drugid_df[smiles_drugid_df['Drug'] == smiles]['Conformers Set Id'].to_numpy()
            if(len(arr) > 0):
                drug_unique_id = int(arr[0])
                edges = pickle.load(open(f'{self.compound_graph_adr}/Edges/{drug_unique_id}.pickle','rb'))
                nodes_features = pickle.load(open(f'{self.compound_graph_adr}/Nodes_Features/{drug_unique_id}.pickle','rb'))
                num_of_atom_features = nodes_features.shape[1]
                num_of_atoms = nodes_features.shape[0]
                padded_node_features = np.zeros((max_num_atoms, num_of_atom_features))
                padded_node_features[:num_of_atoms, : ] = nodes_features
                #Number of features of each node (atom): 34
                edges = torch.as_tensor(edges, dtype=torch.int64).to(torch.device(dev))
                padded_node_features = torch.as_tensor(padded_node_features, dtype=torch.float32).to(torch.device(dev))
                graph_list.append(Data(x=padded_node_features, edge_index=edges, num_of_atoms=num_of_atoms))
            else:
                print(f'ID cannot be found for the drug with the following SMILES: {smiles}')  
        compound_batch = Batch.from_data_list(graph_list)        
        return target_batch, compound_batch
    
    
    def get_Drug_batch_for_GAT(self, smiles_arr, dev='cpu'):
        '''
            smiles_arr: A list of SMILES notations of the drugs in a batch
            dev: the device type
        '''
        smiles_drugid_df = pd.read_csv(self.drug_ids_adr)
        max_num_atoms = 148
        graph_list = []
        for smiles in smiles_arr:
            arr = smiles_drugid_df[smiles_drugid_df['Drug'] == smiles]['Conformers Set Id'].to_numpy()
            if(len(arr) > 0):
                drug_unique_id = int(arr[0])
                edges = pickle.load(open(f'{self.drug_graphs_and_features_adr}/Edges/{drug_unique_id}.pickle','rb'))
                nodes_features = pickle.load(open(f'{self.drug_graphs_and_features_adr}/Nodes_Features/{drug_unique_id}.pickle','rb'))
                num_of_atom_features = nodes_features.shape[1]
                num_of_atoms = nodes_features.shape[0]
                padded_node_features = np.zeros((max_num_atoms, num_of_atom_features))
                padded_node_features[:num_of_atoms, : ] = nodes_features
                #Number of features of each node (atom): 34
                edges = torch.as_tensor(edges, dtype=torch.int64).to(torch.device(dev))
                padded_node_features = torch.as_tensor(padded_node_features, dtype=torch.float32).to(torch.device(dev))
                graph_list.append(Data(x=padded_node_features, edge_index=edges, num_of_atoms=num_of_atoms))
            else:
                print(f'ID cannot be found for the drug with the following SMILES: {smiles}')  
        compound_batch = Batch.from_data_list(graph_list)        
        return compound_batch
    
    
    def get_learned_features(self, targets_arr, embeddings_type):
        '''
            targets_arr: A list of protein amino acid sequences
            embeddings_type: 'BERT' for getting the BERT-based input features or 'UniRep' for getting the UniRep-based input features 
        '''
        leaned_featurs = []
        if(embeddings_type == 'BERT'):
            learned_features = pickle.load(open('../Datasets/Learned-Embeddings/BERT_Learned_Features.pickle', "rb"))
        else:
            learned_features = pickle.load(open('../Datasets/Learned-Embeddings/UniRep_Learned_Features.pickle', "rb"))
        target_embeddings = []
        for target in targets_arr:
            if(target not in learned_features):
                print(f'WARNING: The following target sequence does not have a learned features embedding: {target}')
            else:
                target_embeddings.append(learned_features[target])
        return np.array(target_embeddings)