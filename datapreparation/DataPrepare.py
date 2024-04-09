from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
import pickle
import random
import math
import os
import numpy as np


class DataProvider:
    def __init__(self, *args):
        if(len(args) == 0):
            # By default, the working directory is 'DrugBank'
            self.working_dir = "../Datasets/DrugBank"
        if(len(args) == 1):
            # args[0] MUST be a subdirectory of the 'Datasets' folder.
            self.working_dir = f"../Datasets/{args[0]}"

    
    def set_working_dir(self, new_dir):
        # The new working directory MUST be a subdirectory of the 'Datasets' folder
        self.working_dir = f"../Datasets/{new_dir}"
        
        
    def generate_drugs_set_conformers(self, all_smiles_with_ids):    
        '''
            To generate the conformers for a set of drugs 
            all_smiles_with_ids: A CSV file containing two columns with headers of 'Conformers Set Id' and 'Drug', where 'Drug' is the SMILES notation of the drugs and 'Conformers Set Id' is a unique id corresponding to the drugs (The IDs are using for saving the bonds and coordinates files) --> The original file is located in Datasets/Conformers/ConformersSetIds-Drugs.csv
            NOTE: all_smiles_with_ids CSV file MUST be located in the working_dir
        '''
        confids_drug_df = pd.read_csv(f'{self.working_dir}/{all_smiles_with_ids}')
        print('Conformers generation is started!')
        for drug in confids_drug_df.to_numpy():
            drug_unique_id = str(drug[0])
            drug_smiles = drug[1]
            self.generate_conformers(drug_smiles, drug_unique_id)
        print('Conformers generation is finished!')
        
        
        
    def generate_conformers(self, smiles, output, show_details=False):
        '''
            To obtain various conformers of a drug
            For each conformer, this function returns the atoms with their coordinates (x, y, z) and the bonds
            smiles: SMILES notation of a drug
            output: The output file name
            show_details: Set to True if you want to see the details of the compound
        '''
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        #The number of rotatable bonds
        num_of_rotatable_bonds = AllChem.CalcNumRotatableBonds(mol)
        max_num_conformers = 30
        if(num_of_rotatable_bonds == 0):
            max_num_conformers = 1
        elif((num_of_rotatable_bonds >= 1) and (num_of_rotatable_bonds <= 25)):
            max_num_conformers = num_of_rotatable_bonds
        elif((num_of_rotatable_bonds >= 26) and (num_of_rotatable_bonds <= 28)):
            max_num_conformers = 26
        elif((num_of_rotatable_bonds >= 29) and (num_of_rotatable_bonds <= 31)):
            max_num_conformers = 27
        elif((num_of_rotatable_bonds >= 32) and (num_of_rotatable_bonds <= 34)):
            max_num_conformers = 28
        elif((num_of_rotatable_bonds >= 35) and (num_of_rotatable_bonds <= 44)):
            max_num_conformers = 29
        else:
            max_num_conformers = 30
        AllChem.EmbedMultipleConfs(mol, numConfs=max_num_conformers, maxAttempts=100)
        conformers = mol.GetConformers()
        num_of_atoms = mol.GetNumAtoms()
        num_of_bonds = len(mol.GetBonds())
        num_of_conformers = len(conformers)
        bonds_file = open(f'{self.working_dir}/{output}_bonds.txt', 'w')
        if(show_details):
            print(f"Generating conformers for '{smiles}'")
            print(f"Number of atoms: {num_of_atoms}")
            print(f"Number of bonds: {num_of_bonds}")
            print(f"{num_of_conformers} conformers have been generated!")
        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            atoms.append(atom.GetSymbol())
        for conf_idx in range(num_of_conformers):
            coordinates_file = open(f'{self.working_dir}/{output}_conf{conf_idx}_coordinates.txt', 'w')
            for atom_idx in range(num_of_atoms):
                coordinates_file.write(str(atom_idx) + '\t')
                coordinates_file.write(str(atoms[atom_idx]) + '\t')
                coordinates_file.write(str(conformers[conf_idx].GetAtomPosition(atom_idx).x) + '\t')
                coordinates_file.write(str(conformers[conf_idx].GetAtomPosition(atom_idx).y) + '\t')
                coordinates_file.write(str(conformers[conf_idx].GetAtomPosition(atom_idx).z) + '\n')
            coordinates_file.close()
        for bond_idx in range(num_of_bonds):
            bonds_file.write(str(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx()) + '\t')
            bonds_file.write(str(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx()) + '\n')
        bonds_file.close()
        
    
    def assign_random_Ys(self, train_test_sets_dir):
        '''
            What if we wish to assign random values instead of predicted results.
            train_test_sets_dir: The directory of the training and test sets (containing All K folds)
            NOTICE: Each fold MUST have a directory called 'TrainingTestSets-i', where i is a number bettween 0 to K-1
            NOTICE: The aforementioned directory MUST contain two CSV files: Training.csv and Test.csv [with a header of 'bind' status]
        '''
        for traintest_subdir in os.listdir(f'{self.working_dir}/{train_test_sets_dir}'):
            if(traintest_subdir.find('TrainingTestSets') != -1):
                test_set_df = pd.read_csv(f'{self.working_dir}/{train_test_sets_dir}/{traintest_subdir}/Test.csv')
                data = {}
                data['max_prediction'] = []
                data['real'] = []
                data['prediction'] = []
                for bind_value in test_set_df['bind'].to_numpy():
                    data['real'].append(bind_value)
                    random_number = random.uniform(0, 1)
                    data['prediction'].append(random_number)
                    if(random_number > 0.5):
                        data['max_prediction'].append(1)
                    else:
                        data['max_prediction'].append(0)
                df = pd.DataFrame(data)
                df.to_csv(f'{self.working_dir}/{train_test_sets_dir}/{traintest_subdir}/Prediction_Random.csv', index=False)
        print(f"You can find the results (Prediction_Random.csv) on the subdirectories of {self.working_dir}/{train_test_sets_dir}")
    
        
    def generate_negative_samples(self, neg_over_pos_ratio, drug_2D_fingerprints_pickle, drug_smiles_csv, targets_csv, pos_dti_csv, output_csv):
        '''
            Generating Negative samples for DrugBank dataset (Max Distance of {Most similar})
            neg_over_pos_ratio: The ratio of the negative/positive 
            drug_2D_fingerprints_pickle: 2D fingerprints of the drugs (It has 4 various drug fingerprints: 'Morgan FP', 'AtomPair FP', 'RDKit-2D FP', 'MACCS FP')
            drug_smiles_csv: The CSV file of the drugs' information (including 'Canonical SMILES' and 'DrugBank Id')
            targets_csv: The CSV file of the targets' information (including 'Uniprot Id' and '3D file') in which '3D file' shows the availability of the 3D structure of the protein.
            pos_dti_csv: The CSV file of the positive samples (It has two columns: 'Target' and 'Drug')
            output_csv: The name of the output CSV file.
        '''
        print("Generating negative samples...")
        drug_df = pd.read_csv(f"{self.working_dir}/{drug_smiles_csv}")
        drug_df = drug_df[drug_df['Canonical SMILES'] != '-'].rename(columns={"DrugBank Id":"Drug"})
        target_df = pd.read_csv(f"{self.working_dir}/{targets_csv}")
        target_df = target_df[target_df['3D file'] != '-'].rename(columns={"Uniprot Id":"Target"})
        #Load Positive Drug-Target Interactions
        dti_df = pd.read_csv(f"{self.working_dir}/{pos_dti_csv}")
        drug_fingerprints = pickle.load(open(f'{self.working_dir}/{drug_2D_fingerprints_pickle}', 'rb'))
        dti_df = pd.merge(dti_df, target_df, on=['Target'], how='inner')[['Drug', 'Target']]
        dti_df = pd.merge(dti_df, drug_df, on=['Drug'], how='inner')[['Drug', 'Target', 'Canonical SMILES']]
        negative_data = {}
        negative_data['Target'] = []
        negative_data['Drug'] = []
        for target in dti_df['Target'].unique():
            bound_drugs = dti_df[dti_df['Target']==target]['Canonical SMILES'].unique()
            other_drugs = dti_df[dti_df['Target']!=target]['Canonical SMILES'].unique()
            candidate_drugs = {}
            for i in range(neg_over_pos_ratio*len(bound_drugs)):
                candidate_drugs[f"smiles-{str(i)}"] = 0.0
            for bound_smiles in bound_drugs:
                for other_smiles in other_drugs:
                    morgan_distance = 1 - DataStructs.TanimotoSimilarity(drug_fingerprints[bound_smiles]['Morgan FP'], drug_fingerprints[other_smiles]['Morgan FP'])
                    atompair_distance = 1 - DataStructs.TanimotoSimilarity(drug_fingerprints[bound_smiles]['AtomPair FP'], drug_fingerprints[other_smiles]['AtomPair FP'])
                    rdkit_distance = 1 - DataStructs.TanimotoSimilarity(drug_fingerprints[bound_smiles]['RDKit-2D FP'], drug_fingerprints[other_smiles]['RDKit-2D FP'])
                    maccs_distance = 1 - DataStructs.TanimotoSimilarity(drug_fingerprints[bound_smiles]['MACCS FP'], drug_fingerprints[other_smiles]['MACCS FP'])
                    distance_avg = (morgan_distance+atompair_distance+rdkit_distance+maccs_distance)/4
                    min_value = min(list(candidate_drugs.values()))
                    if(distance_avg > min_value):
                        keys = [k for k, v in candidate_drugs.items() if v == min_value]
                        if(other_smiles not in candidate_drugs):
                            del candidate_drugs[keys[0]]
                            candidate_drugs[other_smiles] = distance_avg
                        else:
                            if(min_value < candidate_drugs[other_smiles]):
                                candidate_drugs[other_smiles] = min_value
            for candidate_smiles in candidate_drugs:
                drug_id = dti_df[dti_df['Canonical SMILES'] == candidate_smiles]['Drug'].iloc[0]
                negative_data['Target'].append(target)
                negative_data['Drug'].append(drug_id)
        negative_dti_df = pd.DataFrame(negative_data)
        negative_dti_df.to_csv(f"{self.working_dir}/{output_csv}",index=False)
        print(f"The negative drug-target interactions has been saved in {self.working_dir}/{output_csv}")
       
    
    def generate_training_test_sets(self, folds_directory, k):
        '''
            Generating Training and Test sets from the folds that provided by the k-fold generator functions (such as generate_k_cold_start_folds, etc.)
            folds_directory: The directory containing the k folds
            k: Number of folds
            Generates the training and test sets 
        '''
        print("<<< Generating Training-Test Sets has started... >>>")
        for i in range(k):
            print(f'TrainingTestSets-{str(i)}')
            training_df = pd.DataFrame(columns=['Sequence', 'Canonical SMILES', '3D file', 'bind'])
            test_df = pd.read_csv(f'{self.working_dir}/{folds_directory}/Fold{str(i)}.csv')
            if(not os.path.exists(f'{self.working_dir}/{folds_directory}/TrainingTestSets-{str(i)}')):
                os.makedirs(f'{self.working_dir}/{folds_directory}/TrainingTestSets-{str(i)}')
            for j in range(k):
                if(j != i):
                    training_df = pd.concat([training_df, pd.read_csv(f'{self.working_dir}/{folds_directory}/Fold{str(j)}.csv')])
            training_df.to_csv(f'{self.working_dir}/{folds_directory}/TrainingTestSets-{str(i)}/Training.csv', index=False)
            test_df.to_csv(f'{self.working_dir}/{folds_directory}/TrainingTestSets-{str(i)}/Test.csv', index=False)
        print("<<< Generating Training-Test Sets has finished! >>>")
    
        
    def generate_k_cold_start_folds(self, k, col, csv_file, output_dir):
        '''
            Generating K Cold Start folds datasets ('col' argument decides based on which entity [drug or target])
            k: number of folds (for k-fold cross-validation)
            col: MUST be 'Sequence' or 'Canonical SMILES' [It depends whether you need obtain cold start folds based on target or drug]
            csv_file: The CSV file containing the drug-target pairs and their binding condition
            output_dir: The name of the output directory (CSV files of the folds will be saved on this directory)
        '''
        print(f"<<< Generating K Cold Start Folds (for '{col}') has started... >>>")
        if(not os.path.exists(f'{self.working_dir}/{output_dir}')):
            os.makedirs(f'{self.working_dir}/{output_dir}')
        dti_df = pd.read_csv(f'{self.working_dir}/{csv_file}')
        num_of_rows = dti_df.shape[0]
        folds_max_size = math.ceil(dti_df.shape[0]/k)
        df_folds = []
        for i in range(k):
            df_folds.append(pd.DataFrame(columns=['Sequence', 'Canonical SMILES', '3D file', 'bind']))
        binding_freq = dti_df.groupby([col])[col].count()
        d = binding_freq.to_dict()
        #Sort in descending order
        sorted_entities = sorted(d.items(), key=lambda x: x[1], reverse=True) 
        i = 0
        for entity_info in sorted_entities:
            #entity can be either a target or a compound
            entity = entity_info[0]
            frequency = entity_info[1]
            if(df_folds[i%k].shape[0] + frequency <= folds_max_size):
                df_folds[i%k] = pd.concat([df_folds[i%k], dti_df[dti_df[col]==entity]])
            else:
                j = 0
                while(df_folds[i%k].shape[0] + frequency > folds_max_size):
                    i += 1
                    j += 1
                    if(j > k):
                        print("WARNING: Maximum fold size (Limitation) will be changed!")
                        folds_max_size += 5
                        j = 0
                df_folds[i%k] = pd.concat([df_folds[i%k], dti_df[dti_df[col]==entity]])    
            i = i + 1
        for i in range(k):
            df_folds[i].to_csv(f'{self.working_dir}/{output_dir}/Fold{i}.csv', index=False)
        print("<<< Generating K Cold Start Folds has finished! >>>")
        return df_folds
           
           
    def generate_k_warm_start_folds(self, k, col, csv_file, output_dir):
        '''
            Generating K Warm Start folds datasets ('col' argument decides based on which entity [drug or target])
            k: number of folds (for k-fold cross-validation)
            col: MUST be 'Sequence' or 'Canonical SMILES'
            csv_file: The CSV file containing the drug-target pairs and their binding condition
            output_dir: The name of the output directory (CSV files of the folds will be saved on this directory)
        '''
        print(f"<<< Generating K Warm Start Folds (for '{col}') has started... >>>")
        if(not os.path.exists(f'{self.working_dir}/{output_dir}')):
            os.makedirs(f'{self.working_dir}/{output_dir}')           
        dti_df = pd.read_csv(f'{self.working_dir}/{csv_file}')    
        list_folds = []
        for i in range(k):
            list_folds.append({'Sequence' : [], 'Canonical SMILES' : [], '3D file' : [], 'bind' : []})
        binding_freq = dti_df.groupby([col])[col].count()
        d = binding_freq.to_dict()
        sorted_entities = sorted(d.items(), key=lambda x: x[1], reverse=True) 
        j = 0
        for entity_info in sorted_entities:
            #entity can be either a target or a compound
            entity = entity_info[0]
            for neg_pair in dti_df[(dti_df[col]==entity)&(dti_df['bind']==0)].to_numpy():
                list_folds[j%k]['Sequence'].append(neg_pair[0])
                list_folds[j%k]['Canonical SMILES'].append(neg_pair[1])
                list_folds[j%k]['3D file'].append(neg_pair[2])
                list_folds[j%k]['bind'].append(neg_pair[3])
                j += 1
            for pos_pair in dti_df[(dti_df[col]==entity)&(dti_df['bind']==1)].to_numpy():
                list_folds[j%k]['Sequence'].append(pos_pair[0])
                list_folds[j%k]['Canonical SMILES'].append(pos_pair[1])
                list_folds[j%k]['3D file'].append(pos_pair[2])
                list_folds[j%k]['bind'].append(pos_pair[3])
                j += 1
        df_folds = []
        for i in range(k):
            df_folds.append(pd.DataFrame(list_folds[i], columns=['Sequence', 'Canonical SMILES', '3D file', 'bind']))
            df_folds[i].to_csv(f'{self.working_dir}/{output_dir}/Fold{i}.csv', index=False)
        print("<<< Generating K Warm Start Folds has finished! >>>")
        return df_folds  
              
              
    def extract_2D_graphs(self, all_smiles_with_ids_adr):
        '''
            To extract the 2D graphs of the drugs [Adjacency matrices and the properties of the drugs (Each node's features)]
            all_smiles_with_ids_adr: The address of the drugs and their corresponding IDs (all_smiles_with_ids_adr MUST be a full address starts from a subdirectory of the 'Datasets' directory, for instance, 'Conformers/ConformersSetIds-Drugs.csv')
            OUTPUT: Generates and extracts the adjacency matrix (Number of Atoms * Number of Atoms) and the atoms' features (Number of Atoms * 34), then saves them in format of 'numpy arrays'
            For loading the saved files, use: np.load('0_AtomFeatures.npy')
        '''
        ids_drug_df = pd.read_csv(f'../Datasets/{all_smiles_with_ids_adr}')
        print('Extracting the 2D graphs is started!')
        for drug in ids_drug_df.to_numpy():
            drug_unique_id = str(drug[0])
            drug_smiles = drug[1]
            atom_34_features, adjacency_matrix = self.extract_NodeFeatures_and_2Dgraph(drug_smiles)
            np.save(f'{self.working_dir}/{drug_unique_id}_AdjacencyMatrix', adjacency_matrix)
            np.save(f'{self.working_dir}/{drug_unique_id}_AtomFeatures', atom_34_features)
        print(f'The graphs and the atoms\' features are located in {self.working_dir}')
              
              
    def extract_2D_graphs_for_list_of_smiles(self, all_smiles_with_ids_adr, list_of_smiles):
        '''
            To extract the 2D graphs of the drugs [Adjacency matrices and the properties of the drugs (Each node's features)]
            all_smiles_with_ids_adr: The address of the drugs and their corresponding IDs (all_smiles_with_ids_adr MUST be a full address starts from a subdirectory of the 'Datasets' directory, for instance, 'Conformers/ConformersSetIds-Drugs.csv')
            list_of_smiles: A list containing the new SMILES
            OUTPUT: Generates and extracts the adjacency matrix (Number of Atoms * Number of Atoms) and the atoms' features (Number of Atoms * 34), then saves them in format of 'numpy arrays'
            NOTE: It also appends the new SMILES notations to the end of 'all_smiles_with_ids_adr' dataframe if they do not exist in the dataframe!
            NOTE: In general, the number of compounds in our dataset is 58122 (IDs range from 0 to 58121). But, it can be increased!
            For loading the saved files, use: np.load('0_AtomFeatures.npy')
        '''
        ids_drug_df = pd.read_csv(f'../Datasets/{all_smiles_with_ids_adr}')
        drug_unique_id = len(ids_drug_df)
        print('Extracting the 2D graphs is started!')
        for smiles in list_of_smiles:
            if(smiles not in ids_drug_df['Drug'].unique()):
                atom_34_features, adjacency_matrix = self.extract_NodeFeatures_and_2Dgraph(smiles)
                np.save(f'{self.working_dir}/{drug_unique_id}_AdjacencyMatrix', adjacency_matrix)
                np.save(f'{self.working_dir}/{drug_unique_id}_AtomFeatures', atom_34_features)
                ids_drug_df.loc[drug_unique_id] = {'Conformers Set Id': drug_unique_id, 'Drug':smiles}
                drug_unique_id += 1
            else:
                print(f'The following SMILES notation already exists: {smiles}')
        print(f'The graphs and the atoms\' features are located in {self.working_dir}')              
        ids_drug_df.to_csv('../Datasets/Conformers/ConformersSetIds-Drugs-2.csv', index=False)
        
              
    def extract_NodeFeatures_and_2Dgraph(self, smiles):
        '''
            [[THIS FUNCTION IS COPIED FROM 'TransformerCPI' GitHub AND MODIFIED BASED ON OUR NEEDS]]
            To extract the atoms' features and the adjacency matrix of a drug
            smiles: The SMILES notation of the drug
        '''
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("Parsing error with SMILES")
        number_atom_features = 34
        atom_props = np.zeros((mol.GetNumAtoms(), number_atom_features))
        for atom in mol.GetAtoms():
            atom_props[atom.GetIdx(), :] = self.get_atom_features(atom)
        adj_matrix = self.get_adjacent_matrix(mol)
        return atom_props, adj_matrix
              
              
    def get_adjacent_matrix(self, mol):
        '''
            [[THIS FUNCTION IS COPIED FROM 'TransformerCPI' GitHub AND MODIFIED BASED ON OUR NEEDS]]
            To obtain the 2D graph of a drug (Adjacency Matrix)
        '''
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency) + np.eye(adjacency.shape[0])
              
              
    def get_atom_features(self, atom, explicit_H=False, use_chirality=True):
        '''
            [[THIS FUNCTION IS COPIED FROM 'TransformerCPI' GitHub AND MODIFIED BASED ON OUR NEEDS]]
            To extract the features (A vector with size of 34) of each atom in each drug. The features are as follows:
            1. The atom type (10-dimensional onehot vector)
            2. Degree of the atom (7-dimensional onehot vector)
            3. Formal charge [A value: 0 or 1]
            4. Number of radical electrons [A value: 0 or 1]
            5. Hybridization type (6-dimensional onehot vector)
            6. Aromatic [A value: 0 or 1]
            7. Number of hydrogen atoms attached (5-dimensional onehot vector)
            8. Chirality [A value: 0 or 1]
            9. Configuration (2-dimensional onehot vector)
        '''
        symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  
        degree = [0, 1, 2, 3, 4, 5, 6]  
        hybridizationType = [Chem.rdchem.HybridizationType.SP,
                             Chem.rdchem.HybridizationType.SP2,
                             Chem.rdchem.HybridizationType.SP3,
                             Chem.rdchem.HybridizationType.SP3D,
                             Chem.rdchem.HybridizationType.SP3D2,
                             'other']   
        results = self.one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  self.one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  self.one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if(not explicit_H):
            results = results + self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])   # Num of attached Hydrogens 
        if(use_chirality):
            try:
                results = results + self.one_of_k_encoding_unk(
                        atom.GetProp('_CIPCode'),
                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
        return results
              
              
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return [x == s for s in allowable_set]


    def one_of_k_encoding_unk(self, x, allowable_set):
        '''
            [[THIS FUNCTION IS COPIED FROM 'TransformerCPI' GitHub AND MODIFIED BASED ON OUR NEEDS]]
            Maps inputs not in the allowable set to the last element.
        '''
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]