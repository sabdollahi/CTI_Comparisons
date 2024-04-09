import os
import pandas as pd


class DataConvert:
    def __init__(self, *args):
        if(len(args) == 0):
            # By default, the working directory is 'BindingDatasets_3DAvailable'
            self.working_dir = "../Datasets/BindingDatasets_3DAvailable" 
        elif(len(args) == 1):
            # args[0] MUST be a subdirectory of the 'Datasets' folder.
            self.working_dir = f"../Datasets/{args[0]}"
        elif(len(args) == 2):
            # both args[0] and args[1] MUST be a subdirectory of the 'BindingDatasets_3DAvailable' folder.
            self.working_dir = f"../Datasets/BindingDatasets_3DAvailable"            
            self.source_dir = args[0]
            self.destination_dir = args[1]
        else:
            #If you pass three parameters: args[0] is interpreted as the working directory
            #source_dir MUST be located in the working directory
            #destination_dir MUST be located in a subdirectory of the working directory which is called 'Converted'
            self.working_dir = f"../Datasets/{args[0]}"
            self.source_dir = args[1]
            self.destination_dir = args[2]
       
    
    def set_destination_dir(self, dest_dir):
        print(f'Destination directory has changed to {dest_dir}')
        self.destination_dir = dest_dir
        
        
    def set_source_dir(self, src_dir):
        print(f'Source directory has changed to {src_dir}')
        self.source_dir = src_dir
    
            
    def convert_to_TransformerCPI_format(self):
        '''
            Convert our standard CSV input files (Located in self.source_dir) into the TransformerCPI format (will be located in self.destination_dir)
        '''
        source_adr = f'{self.working_dir}/{self.source_dir}'
        destination_adr = f'{self.working_dir}/Converted/TransformerCPI/{self.destination_dir}'
        for fold_folder in os.listdir(source_adr):
            if(fold_folder.find('TrainingTestSets-') != -1):
                print(f'Data format conversion for: {fold_folder}')
                os.makedirs(f'{destination_adr}/{fold_folder}', exist_ok=True)
                test_df = pd.read_csv(f'{source_adr}/{fold_folder}/Test.csv')[['Canonical SMILES', 'Sequence', 'bind']]
                test_df.to_csv(f'{destination_adr}/{fold_folder}/Test.csv', sep=' ', header=False, index=False)
                training_df = pd.read_csv(f'{source_adr}/{fold_folder}/Training.csv')[['Canonical SMILES', 'Sequence', 'bind']]
                training_df.to_csv(f'{destination_adr}/{fold_folder}/Training.csv', sep=' ', header=False, index=False)
        
    def convert_one_Training_Test_to_TransformerCPI_format(self):
        '''
            Convert our standard CSV input files (Located in self.source_dir) into the TransformerCPI format (will be located in self.destination_dir)
            We can call this method in case of having a test set and a training set in a directory (self.source_dir)
        '''
        source_adr = f'{self.working_dir}/{self.source_dir}'
        destination_adr = f'{self.working_dir}/Converted/TransformerCPI/{self.destination_dir}'
        print(f'Data format conversion for TransformerCPI has just started!')
        os.makedirs(f'{destination_adr}', exist_ok=True)
        test_df = pd.read_csv(f'{source_adr}/Test.csv')[['Canonical SMILES', 'Sequence', 'bind']]
        test_df.to_csv(f'{destination_adr}/Test.csv', sep=' ', header=False, index=False)
        training_df = pd.read_csv(f'{source_adr}/Training.csv')[['Canonical SMILES', 'Sequence', 'bind']]
        training_df.to_csv(f'{destination_adr}/Training.csv', sep=' ', header=False, index=False)