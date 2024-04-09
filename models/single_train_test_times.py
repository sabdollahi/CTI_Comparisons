import os
import csv
import math
import time
import pickle
import numpy as np
import torch
from numpy import interp
import pandas as pd
from sklearn.utils import shuffle
from torch.autograd import Variable
from .GrAttCPI_model import GrAttCPI
from .DTIConvSeqSMILES_simple import DTIConvSeqSMILES
from .DTISeq2DFP_simple import DTISeq2DFP
from .BERT2DFP_model import BERT2DFP
from .UniRep2DFP_model import UniRep2DFP
from .DTISeqE3FP_model import DTISeqE3FP
from .PhyGrAtt_model import PhyGrAtt
from .Phys_DrugGraph_model import TargetFeatureCapturer, SelfAttention, PositionBasedConv1d, IntertwiningLayer, IntertwiningTransformer, PhyChemDG, TrainerPhyChemDG, TesterPhyChemDG
import sys
sys.path.append('..')
from features.FeatureExtraction import FeaturePreparation
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier



class SingleTrainTest:
    def __init__(self, main_path):
        '''
            main_path: The path where a single pair of the training and test sets is located
        '''
        self.main_path = main_path
        
    
    
    def modify_main_path(self, new_main_path):
        '''
            new_main_path: The new path of the folders containing the training and test sets 
        '''
        self.main_path = new_main_path
    
    
    
    def plot_spider_chart(self, title, name_to_be_saved):
        '''
            To plot the Radar (Spider) chart of the performances of various models
            title: The title of the figure to be shown
            name_to_be_saved: the name of the JPG file to be saved
            NOTE: You MUST first run compute_metrics() in your program; then, run this function
        '''
        labels = ['Acc', 'ROC', 'PR', 'F1', 'MCC']
        df = pd.read_csv(f'{self.main_path}/MODELS_EVALUATION.csv', index_col=0)
        values_dict = df.to_dict(orient='index')
        main_colors =   {'RF':'gold',  'PhyChemDG':'limegreen','AlphaFoldGrAtts':'darkorange','PhyGrAtt':'deepskyblue','BERT-based':'darkorchid', 'UniRep-based':'red',    'DeepCAT':'seagreen',   'DeepDTA':'firebrick','E3FP-based':'darkblue', '2DFP-based':'fuchsia', 'TransformerCPI':'blue', 'IIFDTI':'olive', 'DeepConv-DTI':'mediumslateblue'}

        shadow_colors = {'RF':'yellow','PhyChemDG':'palegreen','AlphaFoldGrAtts':'wheat',     'PhyGrAtt':'skyblue',    'BERT-based':'orchid',     'UniRep-based':'tomato', 'DeepCAT':'lime',   'DeepDTA':'salmon',   'E3FP-based':'skyblue',  '2DFP-based':'violet',  'TransformerCPI':'cornflowerblue', 'IIFDTI':'yellowgreen', 'DeepConv-DTI':'lavender'}
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
        # Set the angle and the labels
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        # Set the axis limits and the labels
        ax.set_ylim(-0.2, 1)
        ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels(['-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])   
        # Plot the data for each metric and model
        for model, values in values_dict.items():
            values = list(values.values())
            main_color = main_colors[model]
            shadow_color = shadow_colors[model]
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=main_color, zorder=-1)
            #ax.fill(angles, values, alpha=0.2, color=shadow_color)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(values_dict.keys(), bbox_to_anchor=(0.7, 0.49), loc='upper left')
        fig.savefig(f'{self.main_path}/{name_to_be_saved}.jpg', format='jpeg', dpi=300, bbox_inches='tight')
    
    
    
    def plot_boxplots_comparisons(self, colors, models, y_label, title, comparison_datasets, compare_with_adr, name_to_be_saved):
        '''
            To plot a comparison-based boxplot 
            colors: A list of the colors  corresponding to the models
            models: A list of models' name
            y_label: MUST be chosen from 'Accuracy', 'Auc ROC', 'AUC PR', 'F1 score', or 'MCC'
            title: A string to be assigned as the title of the plot
            comparison_datasets: A list consisting of two names of two datasets to be compared
            compare_with_adr: the address of the second dataset to be compared
            name_to_be_saved: The name of the plot to be saved
        '''
        df1 = pd.read_csv(f'{self.main_path}/MODELS_EVALUATION.csv', index_col=0)
        values_dict1 = df1.to_dict(orient='index')
        df2 = pd.read_csv(f'{compare_with_adr}/MODELS_EVALUATION.csv', index_col=0)
        values_dict2 = df2.to_dict(orient='index')
        data1 = []
        data2 = []
        for model in models:
            data1.append(values_dict1[model][y_label])
            data2.append(values_dict2[model][y_label])
        # Create a list to label the groups
        groups = [comparison_datasets[0]] * len(data1) + [comparison_datasets[1]] * len(data2)
        # Combine the data and groups
        combined_data = [data1, data2]
        # Set up the figure and axes
        fig, ax = plt.subplots()
        # Customize the boxplot properties
        boxprops = dict(linewidth=2, color='black', facecolor='whitesmoke')
        medianprops = dict(linewidth=3, color='gray')
        box = ax.boxplot(combined_data, patch_artist=True, showfliers=False, zorder=1, boxprops=boxprops,medianprops=medianprops, widths=0.5)
        # Set different marker shapes for each data point
        markers = ['X', '^', 'v', 's', 'D', 'P']
        marker_size = 100
        # Plot the individual data points with different marker shapes
        for i, data in enumerate(combined_data):
            x = np.random.normal(i + 1, 0.05, size=len(data))
            for j, (x_val, y_val) in enumerate(zip(x, data)):
                marker = markers[j % len(markers)]
                color = colors[j % len(colors)]
                plt.scatter(x_val, y_val, marker=marker, color=color, s=marker_size, zorder=2,label=f'{models[j]}')
        # Set the x-axis tick labels
        ax.set_xticklabels(comparison_datasets)
        # Set the y-axis label
        ax.set_ylabel(y_label)
        # Set the title
        ax.set_title(title)
        # Create custom legends for each marker shape
        legend_handles = []
        for j in range(len(markers)):
            marker = markers[j]
            color = colors[j % len(colors)]
            legend_handle = mlines.Line2D([], [], color=color, marker=marker, markersize=8, label=f'{models[j]}')
            legend_handles.append(legend_handle)
        # Add the legends to the plot and move it outside
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left')
        # Adjust the plot layout to accommodate the legend
        plt.tight_layout()
        fig.savefig(f'{self.main_path}/{name_to_be_saved}.jpg', format='jpeg', dpi=300, bbox_inches='tight')
    
    
    
    
    
    def plot_multiple_boxplots_comparisons(self, colors, models, y_label, title, comparison_datasets, compare_with_adrs, name_to_be_saved):
        '''
            To plot a comparison-based boxplot 
            colors: A list of the colors  corresponding to the models
            models: A list of models' name
            y_label: MUST be chosen from 'Accuracy', 'Auc ROC', 'AUC PR', 'F1 score', or 'MCC'
            title: A string to be assigned as the title of the plot
            comparison_datasets: A list consisting of two names of two datasets to be compared
            compare_with_adrs: the list of addresses of the second, third, etc. datasets to be compared
            name_to_be_saved: The name of the plot to be saved
        '''
        values_dicts = []
        df = pd.read_csv(f'{self.main_path}/MODELS_EVALUATION.csv', index_col=0)
        values_dicts.append(df.to_dict(orient='index'))
        for compare_with_adr in compare_with_adrs:
            df = pd.read_csv(f'{compare_with_adr}/MODELS_EVALUATION.csv', index_col=0)
            values_dicts.append(df.to_dict(orient='index'))
        data1 = []
        data2 = []
        combined_data = []
        for values_dict in values_dicts:
            data = []
            for model in models:
                data.append(values_dict[model][y_label])
            combined_data.append(data)
        # Set up the figure and axes
        fig, ax = plt.subplots()
        # Customize the boxplot properties
        boxprops = dict(linewidth=2, color='black', facecolor='whitesmoke')
        medianprops = dict(linewidth=3, color='gray')
        box = ax.boxplot(combined_data, patch_artist=True, showfliers=False, zorder=1, boxprops=boxprops,medianprops=medianprops, widths=0.5)
        # Set different marker shapes for each data point
        markers = ['X', '^', 'v', 's', 'D', 'P', '*', 'o']
        marker_size = 100
        # Plot the individual data points with different marker shapes
        for i, data in enumerate(combined_data):
            x = np.random.normal(i + 1, 0.05, size=len(data))
            for j, (x_val, y_val) in enumerate(zip(x, data)):
                marker = markers[j % len(markers)]
                color = colors[j % len(colors)]
                plt.scatter(x_val, y_val, marker=marker, color=color, s=marker_size, zorder=2,label=f'{models[j]}')
        # Set the x-axis tick labels
        ax.set_xticklabels(comparison_datasets)
        # Set the y-axis label
        ax.set_ylabel(y_label)
        # Set the title
        ax.set_title(title)
        # Create custom legends for each marker shape
        legend_handles = []
        for j in range(len(markers)):
            marker = markers[j]
            color = colors[j % len(colors)]
            legend_handle = mlines.Line2D([], [], color=color, marker=marker, markersize=8, label=f'{models[j]}')
            legend_handles.append(legend_handle)
        # Add the legends to the plot and move it outside
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left')
        # Adjust the plot layout to accommodate the legend
        plt.tight_layout()
        fig.savefig(f'{self.main_path}/{name_to_be_saved}.jpg', format='jpeg', dpi=300, bbox_inches='tight')
    
    
        
    def compute_metrics(self):
        '''
            Computes various metrics (Accuracy, F1-score, ROC, PR, and MCC) to evaluate the predictions obtained by different models 
            NOTE: The predictions MUST be saved on CSV files containing at least two columns of 'real' and 'prediction' in which 'prediction' column MUST contain values between 0 and 1
            NOTE: The name of the predictions files (in the self.main_path directory) MUST start with 'Prediction_' and end with '.csv'
            Output: It saves the results in the 'self.main_path' directory as a CSV file
        '''
        metrics_data = {}
        index_names = []
        metrics_data["Accuracy"] = []
        metrics_data["Auc ROC"] = []
        metrics_data["AUC PR"] = []
        metrics_data["F1 score"] = []
        metrics_data["MCC"] = []
        for pred_file in os.listdir(self.main_path):
            if('Prediction_' in pred_file):
                index_names.append(pred_file.split('_')[1].split('.csv')[0])
                # Load the data from the pandas dataframe
                df = pd.read_csv(f'{self.main_path}/{pred_file}')
                 # Extract the real and predicted labels
                y_true = df['real'].values
                y_pred_proba = df['prediction'].values
                y_pred = (y_pred_proba >= 0.5).astype(int)
                #Compute the various metrics
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                auc_roc = roc_auc_score(y_true, y_pred_proba)
                auc_pr = average_precision_score(y_true, y_pred_proba)
                mcc = matthews_corrcoef(y_true, y_pred)
                metrics_data["Accuracy"].append(accuracy)
                metrics_data["Auc ROC"].append(auc_roc)
                metrics_data["AUC PR"].append(auc_pr)
                metrics_data["F1 score"].append(f1)
                metrics_data["MCC"].append(mcc)
        #pd.DataFrame(metrics_data, index=index_names).to_csv(f'{self.main_path}/MODELS_EVALUATION.csv')
    
    
        
    def compute_metrics_folds(self):
        '''
            Computes various metrics (Accuracy, F1-score, ROC, PR, and MCC) to evaluate the predictions obtained by different models for different folds. In addition, it also calculates the average of the results
            NOTE: The predictions MUST be saved on CSV files containing at least two columns of 'real' and 'prediction' in which 'prediction' column MUST contain values between 0 and 1
            NOTE: The name of the predictions files (in the self.main_path directory) MUST start with 'Prediction_' and end with '.csv'
            Output: It saves the results in the 'self.main_path' directory as a CSV file
        '''
        avg_models = {'UniRep-based': {'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'DeepConv-DTI':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0},'IIFDTI':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'TransformerCPI':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'BERT-based':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'AlphaFoldGrAtts':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'PhyGrAtt':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0},'E3FP-based':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, '2DFP-based':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'DeepCAT':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'DeepDTA':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'Random':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'RF':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}, 'PhyChemDG':{'acc':0, 'roc':0, 'pr':0, 'f1':0, 'mcc':0}}
        metrics_data = {}
        metrics_data["Model"] = []
        metrics_data["Accuracy"] = []
        metrics_data["AUC ROC"] = []
        metrics_data["AUC PR"] = []
        metrics_data["F1 score"] = []
        metrics_data["MCC"] = []
        for fold_folder in os.listdir(self.main_path):
            if(fold_folder.find('TrainingTestSets-') != -1): 
                fold_number = fold_folder.split('-')[1]
                print(f'Evaluating Fold-{fold_number}')
                metrics_data["Model"].append(f'Fold-{fold_number}')
                metrics_data["Accuracy"].append('Accuracy')
                metrics_data["AUC ROC"].append('AUC ROC')
                metrics_data["AUC PR"].append('AUC PR')
                metrics_data["F1 score"].append('F1 score')
                metrics_data["MCC"].append('MCC')
                for pred_file in os.listdir(f'{self.main_path}/{fold_folder}'):
                    if(pred_file.find('Prediction_') != -1): 
                        model_name = pred_file.split('_')[1].split('.')[0]
                        metrics_data["Model"].append(model_name)
                        # Load the data from the pandas dataframe
                        df = pd.read_csv(f'{self.main_path}/{fold_folder}/{pred_file}')
                         # Extract the real and predicted labels
                        y_true = df['real'].values
                        y_pred_proba = df['prediction'].values
                        y_pred = (y_pred_proba >= 0.5).astype(int)
                        #Compute the various metrics
                        accuracy = accuracy_score(y_true, y_pred)
                        f1 = f1_score(y_true, y_pred)
                        auc_roc = roc_auc_score(y_true, y_pred_proba)
                        auc_pr = average_precision_score(y_true, y_pred_proba)
                        mcc = matthews_corrcoef(y_true, y_pred)
                        avg_models[model_name]['acc'] += accuracy
                        avg_models[model_name]['roc'] += auc_roc
                        avg_models[model_name]['pr'] += auc_pr
                        avg_models[model_name]['f1'] += f1
                        avg_models[model_name]['mcc'] += mcc
                        metrics_data["Accuracy"].append(accuracy)
                        metrics_data["AUC ROC"].append(auc_roc)
                        metrics_data["AUC PR"].append(auc_pr)
                        metrics_data["F1 score"].append(f1)
                        metrics_data["MCC"].append(mcc)
        metrics_data["Model"].append('AVERAGE')
        metrics_data["Accuracy"].append('Accuracy')
        metrics_data["AUC ROC"].append('AUC ROC')
        metrics_data["AUC PR"].append('AUC PR')
        metrics_data["F1 score"].append('F1 score')
        metrics_data["MCC"].append('MCC')
        for model_id in avg_models:
            metrics_data["Model"].append(model_id)
            metrics_data["Accuracy"].append(avg_models[model_id]['acc']/10)
            metrics_data["AUC ROC"].append(avg_models[model_id]['roc']/10)
            metrics_data["AUC PR"].append(avg_models[model_id]['pr']/10)
            metrics_data["F1 score"].append(avg_models[model_id]['f1']/10)
            metrics_data["MCC"].append(avg_models[model_id]['mcc']/10)
        #pd.DataFrame(metrics_data).to_csv(f'{self.main_path}/MODELS_EVALUATION.csv', index=False)


    def train_test_DTIConvSeqSMILES(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv=''):
        '''
            TRAIN and EVALUATE the DTIConvSeqSMILES model (Utilize SMILES onehot vectors for compounds and physicochemical/onehot vectors for targets)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_biophysicochemical_props: Either to use 'Biophysicochemical properties' or 'onehot vectors' for amino acid representations. (default = False)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]
        '''
        training_elapsed_time = 0
        test_elapsed_time = 0
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        dev = 'cpu'
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")        
        feature_preparation = FeaturePreparation()    
        Evaluation_data = {}
        Evaluation_data['DTIConvSeqSMILES'] = []
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}')
        training_set_SMILES = training_df['Canonical SMILES'].unique()
        test_set_SMILES = test_df['Canonical SMILES'].unique()
        smiles_np_array = np.concatenate((training_set_SMILES, test_set_SMILES))

        smiles_characters, smiles_maximum_length = feature_preparation.extract_all_SMILES_chars_and_max_len(smiles_np_array)
        smiles_maximum_length = 348
        #print(f"The maximum size of SMILES is: {smiles_maximum_length}")
        #print(f"SMILES characters' embedding length is: {str(len(smiles_characters))}")
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
        #print(f"Amino acids' embedding length is: {str(aa_embedding_length)}")
        model = DTIConvSeqSMILES()
        model.to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("DTIConvSeqSMILES: Training the model has started!")
        #Training Phase
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                running_loss = 0
                b = 0
                start_time = time.time()
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    # compound_input_features dimensions: (batch_size, max_smiles_length, embedding_size) in our case, max_smiles_length = 348, embedding_size = 98
                    compound_input_features = feature_preparation.generate_matrices(smiles_onehot_vectors_dict, smiles_arr, smiles_maximum_length)
                    # target_input_features dimensions: (batch_size, max_target_length, embedding_size) in our case, max_target_length = 1400, embedding_size = 20
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
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                #print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        #pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/Loss_DTIConvSeqSMILES.csv', index=False)
        print("DTIConvSeqSMILES: Testing the model has started!")            
        #Test Phase            
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)                
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        b = 0
        concatenated_pred_Y = []
        numpy_pred_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            targets_arr = batch_test_df['Sequence'].to_numpy()
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            compound_input_features = feature_preparation.generate_matrices(smiles_onehot_vectors_dict, smiles_arr, smiles_maximum_length)
            target_input_features = []
            if(use_biophysicochemical_props):
                target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            else:
                target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            compound_input_features = compound_input_features.reshape((compound_input_features.shape[0], 1, smiles_maximum_length, smiles_chars_embedding_length))
            compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            start_time = time.time()
            test_Y_hat = model.forward(compound_input_features, target_input_features)
            end_time = time.time()
            test_elapsed_time = end_time - start_time
            break
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")

        
        
    def train_test_DTISeq2DFP(self,num_of_epochs, num_of_shuffle_epochs, name_2D_FP, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv=''):
        '''
            TRAIN and EVALUATE the DTISeq2DFP model (Utilize the concatenation of 4 drug fingerprints for compounds and physicochemical/onehot vectors for targets)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            name_2D_FP: The name of the 2D drug fingerprints' pickle file (MUST BE LOCATED in 'Datasets' directory)
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_biophysicochemical_props: Either to use 'Biophysicochemical properties' or 'onehot vectors' for amino acid representations. (default = False)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]            
        '''
        training_elapsed_time = 0
        test_elapsed_time = 0
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_2D_FP)   
        Evaluation_data = {}
        Evaluation_data['DTISeq2DFP'] = []
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        training_df = training_df[training_df['Canonical SMILES'] != '-']
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        test_df = test_df[test_df['Canonical SMILES'] != '-']
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}')  
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
        print("DTISeq2DFP: Training the model has started!")
        #Training Phase
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                running_loss = 0
                b = 0
                start_time = time.time()
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
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("DTISeq2DFP: Testing the model has started!")            
        #Test Phase            
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)                
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        b = 0
        concatenated_pred_Y = []
        numpy_pred_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            targets_arr = batch_test_df['Sequence'].to_numpy()
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            compound_input_features = feature_preparation.get_2D_drug_fingerprints(smiles_arr)
            target_input_features = []
            if(use_biophysicochemical_props):
                target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            else:
                target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            start_time = time.time()
            test_Y_hat = model.forward(compound_input_features, target_input_features)
            end_time = time.time()
            test_elapsed_time = end_time - start_time
            break
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")

        
        
    def train_test_BERT2DFP(self,num_of_epochs, num_of_shuffle_epochs, name_2D_FP, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv=''):
        '''
            TRAIN and EVALUATE the BERT2DFP model (Utilize the concatenation of 4 drug fingerprints for compounds and BERT-based features for targets)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            name_2D_FP: The name of the 2D drug fingerprints' pickle file (MUST BE LOCATED in 'Datasets' directory)
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]            
        '''
        training_elapsed_time = 0
        test_elapsed_time = 0
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_2D_FP)   
        Evaluation_data = {}
        Evaluation_data['BERT2DFP'] = []
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        training_df = training_df[training_df['Canonical SMILES'] != '-']
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        test_df = test_df[test_df['Canonical SMILES'] != '-']
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}')  
        training_df = shuffle(training_df)
        training_index = training_df.index
        model = BERT2DFP()
        model.to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("BERT2DFP: Training the model has started!")
        #Training Phase
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                running_loss = 0
                b = 0
                start_time = time.time()
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    # compound_input_features dimensions: (batch_size, 3239)
                    compound_input_features = feature_preparation.get_2D_drug_fingerprints(smiles_arr)
                    # target_input_features dimensions: (batch_size, embedding_size) -> For BERT the embedding size is 768 and for UniRep is 1900
                    target_input_features = feature_preparation.get_learned_features(targets_arr, 'BERT')
                    optimizer.zero_grad()
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
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        #pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/Loss_BERT2DFP.csv', index=False)
        print("BERT2DFP: Testing the model has started!")            
        #Test Phase            
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)                
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        b = 0
        concatenated_pred_Y = []
        numpy_pred_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            targets_arr = batch_test_df['Sequence'].to_numpy()
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            compound_input_features = feature_preparation.get_2D_drug_fingerprints(smiles_arr)
            target_input_features = feature_preparation.get_learned_features(targets_arr, 'BERT')
            compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            start_time = time.time()
            test_Y_hat = model.forward(compound_input_features, target_input_features)
            end_time = time.time()
            test_elapsed_time = end_time - start_time
            break
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")

        
        
    def train_test_UniRep2DFP(self,num_of_epochs, num_of_shuffle_epochs, name_2D_FP, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv=''):
        '''
            TRAIN and EVALUATE the UniRep2DFP model (Utilize the concatenation of 4 drug fingerprints for compounds and UniRep-based features for targets)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            name_2D_FP: The name of the 2D drug fingerprints' pickle file (MUST BE LOCATED in 'Datasets' directory)
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]            
        '''
        training_elapsed_time = 0
        test_elapsed_time = 0
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_2D_FP)   
        Evaluation_data = {}
        Evaluation_data['UniRep2DFP'] = []
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        training_df = training_df[training_df['Canonical SMILES'] != '-']
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        test_df = test_df[test_df['Canonical SMILES'] != '-']
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}')  
        training_df = shuffle(training_df)
        training_index = training_df.index
        model = UniRep2DFP()
        model.to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("UniRep2DFP: Training the model has started!")
        #Training Phase
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                running_loss = 0
                b = 0
                start_time = time.time()
                for index in range(0, len(training_index), batch_size):
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    # compound_input_features dimensions: (batch_size, 3239)
                    compound_input_features = feature_preparation.get_2D_drug_fingerprints(smiles_arr)
                    # target_input_features dimensions: (batch_size, embedding_size) -> For BERT the embedding size is 768 and for UniRep is 1900
                    target_input_features = feature_preparation.get_learned_features(targets_arr, 'UniRep')
                    optimizer.zero_grad()
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
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        #pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/Loss_UniRep2DFP.csv', index=False)
        print("UniRep2DFP: Testing the model has started!")            
        #Test Phase            
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)                
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        b = 0
        concatenated_pred_Y = []
        numpy_pred_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            targets_arr = batch_test_df['Sequence'].to_numpy()
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            compound_input_features = feature_preparation.get_2D_drug_fingerprints(smiles_arr)
            target_input_features = feature_preparation.get_learned_features(targets_arr, 'UniRep')
            compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            start_time = time.time()
            test_Y_hat = model.forward(compound_input_features, target_input_features)
            end_time = time.time()
            test_elapsed_time = end_time - start_time
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")
        
        
        
        
    def train_test_DTISeqE3FP(self,num_of_epochs, num_of_shuffle_epochs, name_3D_FP, max_num_of_conformers, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv=''):  
        '''
            TRAIN and EVALUATE the DTISeqE3FP model (Utilize the 3D drug fingerprints for compounds [in particular E3FP] and physicochemical/onehot vectors for targets)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            name_3D_FP: The name of the 3D drug fingerprints' pickle file (MUST BE LOCATED in 'Datasets' directory)
            max_num_of_conformers: Maximum number of conformers
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_biophysicochemical_props: Either to use 'Biophysicochemical properties' or 'onehot vectors' for amino acid representations. (default = False)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]               
        '''        
        training_elapsed_time = 0
        test_elapsed_time = 0
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_3D_FP)   
        e3fp_dict = pickle.load(open('../Datasets/drug_3D_fingerprints_v1.pickle','rb'))
        unavailable_smiles = []
        for smiles in e3fp_dict:
            if(e3fp_dict[smiles]=='-'):
                unavailable_smiles.append(smiles)
        Evaluation_data = {}
        Evaluation_data['DTISeqE3FP'] = []
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        #The following drugs have no E3FP!
        training_df = training_df[(training_df['Canonical SMILES'] != '-') & (training_df['Canonical SMILES'] != 'CC[C@]1(O)C[C@@H]2C[N@@](CCc3c([nH]c4ccccc34)[C@@](C(=O)OC)(c3cc4c(cc3OC)N(C)[C@H]3[C@@](O)(C(=O)NNC(=O)OCCSSC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(=O)O)NC(=O)CC[C@H](NC(=O)c5ccc(NCc6c[nH]c7nc(N)nc(=O)c-7n6)cc5)C(=O)O)C(=O)O)[C@H](O)[C@]5(CC)C=CCN6CC[C@]43[C@@H]65)C2)C1') & (training_df['Canonical SMILES'] != 'CC1(C)C(/C=C/C2=C(Oc3ccc(C[C@H](NC(=O)c4ccc(NCc5cnc6nc(N)[nH]c(=O)c6n5)cc4)C(=O)O)cc3)/C(=C/C=C3/N(CCCCS(=O)(=O)O)c4ccc(S(=O)(=O)O)cc4C3(C)C)CCC2)=[N+](CCCCS(=O)(=O)O)c2ccc(S(=O)(=O)[O-])cc21')]
        training_df = training_df[~training_df['Canonical SMILES'].isin(unavailable_smiles)]
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        test_df = test_df[test_df['Canonical SMILES'] != '-']
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}')          
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
        print("DTISeqE3FP: Training the model has started!")
        #Training Phase
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                running_loss = 0
                b = 0
                start_time = time.time()
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
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        #pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/Loss_DTISeqE3FP.csv', index=False)
        print("DTISeqE3FP: Testing the model has started!")            
        #Test Phase            
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        b = 0
        concatenated_pred_Y = []
        numpy_pred_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            targets_arr = batch_test_df['Sequence'].to_numpy()
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            compound_input_features = feature_preparation.get_3D_drug_fingerprints(smiles_arr, max_num_of_conformers)
            target_input_features = []
            if(use_biophysicochemical_props):
                target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            else:
                target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            compound_input_features = compound_input_features.reshape((compound_input_features.shape[0], 1, max_num_of_conformers, 2048))
            compound_input_features = torch.FloatTensor(compound_input_features).to(torch.device(dev))
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            start_time = time.time()
            test_Y_hat = model.forward(compound_input_features, target_input_features)
            end_time = time.time()
            test_elapsed_time = end_time - start_time
            break
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")
                        
            
            
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
    
    
    
    def train_test_PhyChemDG(self, num_of_epochs, num_of_shuffle_epochs, batch_size = 64, learning_rate = 0.0001, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv='', recompute_max_num_atoms = False, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv', drug_graphs_and_features_adr = '../Datasets/Drugs_2D_Graphs_AtomFeatures'):
        '''
            TRAIN and EVALUATE the PhyChemDG model (Utilize the 2D graphs for compounds and physicochemical vectors for targets)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            batch_size: The batch size (default = 64)
            learning_rate: The learning rate of the model (default = 0.0001)
            use_biophysicochemical_props: Either to use 'Biophysicochemical properties' or 'onehot vectors' for amino acid representations. (default = False)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            recompute_max_num_atoms: Specify whether you wish to compute the maximum number of atoms of the drugs [In our dataset, the max number of atoms is 148] (default = False)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]             
            drug_ids_adr: The complete address of the CSV FILE containing the drugs and their corresponding unique IDs (default: '../Datasets/Conformers/ConformersSetIds-Drugs.csv')
            drug_graphs_and_features_adr: The complete address of the DIRECTORY that contains the adjacency matrices (2D graphs) and the atoms' features of the drugs (default: '../Datasets/Drugs_2D_Graphs_AtomFeatures')
            *** NOTE *** 
                > The '.npy' files of the adjacency matrix (2D graphs of compounds) MUST be located in whatever specified in 'drug_graphs_and_features_adr' variable
                > The atoms' features of a drug with ID of XXX can be found in XXX_AtomFeatures.npy 
                > The adjacency matrix (graph of interactions between the atoms) can be found in XXX_AdjacencyMatrix.npy
                      (The drugs and their corresponding IDs can be found in whatever specified in 'drug_ids_adr' variable)
                > atom_features.shape => (number of atom in a specific drug * 34) [34 is the number of features of the atoms]
                > adj_matrix.shape => (number of atom in a specific drug * number of atom in a specific drug)
        '''
        test_elapsed_time = 0
        training_elapsed_time = 0
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
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        training_df = training_df[training_df['Canonical SMILES'] != '-']
        training_df = training_df[training_df['Canonical SMILES'] != "C=S(=O)(O)c1ccc2c(c1)C(C)(C)C1=[N+]2CCC2OC3CCN4C(=C3C=C12)C(C)(C)c1cc(CC(=O)NCCCC[C@@H](NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCNC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCCCNCCNS(=O)(=O)c2cccc3cnccc23)C(N)=O)ccc14"]
        training_df = training_df[training_df['Canonical SMILES'] != 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)O']
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        test_df = test_df[test_df['Canonical SMILES'] != '-']
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}')  
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
        tester = TesterPhyChemDG(model)
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        print("PhyChemDG: Training the model has started!")
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                if(epoch % decay_interval == 0):
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay
                running_loss = 0
                b = 0
                start_time = time.time()
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
                    # The input features of the drugs 
                    # compounds_adjacency_matrices dimesnions: (batch_size, max_num_atoms, max_num_atoms) in our case, max_num_atoms = 148
                    # compounds_atom_features dimensions: (batch_size, max_num_atoms, num_of_atom_features) in our case, num_of_atom_features = 34 and max_num_atoms = 148
                    compounds_adjacency_matrices, compounds_atom_features, all_cpds_num_atoms = feature_preparation.get_2Dgraphs_atom_features(smiles_arr, max_num_atoms, device)         
                    # target_input_features dimensions: (batch_size, max_target_length, embedding_size) in our case, max_target_length = 1400, embedding_size = 20
                    target_input_features = []
                    if(use_biophysicochemical_props):
                        target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    else:
                        target_input_features = feature_preparation.generate_matrices(aa_onehot_vectors_dict, targets_arr, targets_max_length)
                    # target_input_features dimensions: (batch_size, 1, max_target_length, embedding_size)
                    # target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
                    target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
                    batch_features = (compounds_atom_features, compounds_adjacency_matrices, target_input_features, labels_new, all_cpds_num_atoms, all_tar_num_aa)
                    loss_train = trainer.train(batch_features, device)
                    b += 1
                    running_loss += loss_train
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                #print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        #pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/Loss_PhyChemDG.csv', index=False)
        print("PhyChemDG: Testing the model has started!")
        #Test Phase            
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        max_prediction_Y = []
        real_Y = []
        prediction_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            targets_arr = batch_test_df['Sequence'].to_numpy()
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            labels_new = torch.zeros(len(batch_test_Y), dtype=torch.long, device=device)
            i = 0
            for label in batch_test_Y:
                labels_new[i] = label
                i += 1
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
            start_time = time.time()
            correct_labels, predicted_labels, predicted_scores = tester.test(batch_features)
            end_time = time.time()
            test_elapsed_time = end_time - start_time
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")
                
                
    def train_test_GrAttCPI(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv='', drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):  
        '''
            TRAIN and EVALUATE the GrAttCPI model (Utilize the 2D drugs' graph for compounds and 2D networks which are obtained from AlphaFold+RING)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]                         
        '''  
        training_elapsed_time = 0
        test_elapsed_time = 0        
        dim_out = 4
        dim_hidden = 8
        num_of_aa_features = 20
        num_of_atom_features = 34
        target_graph_adr = '../Datasets/GAT_Prepared_Graphs/RING_based'
        compound_graph_adr = '../Datasets/GAT_Prepared_Graphs/Drugs'
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        Evaluation_data = {}
        feature_preparation = FeaturePreparation(target_graph_adr, compound_graph_adr, drug_ids_adr) 
        Evaluation_data['GrAttCPI'] = []
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        training_df = training_df[(training_df['Canonical SMILES'] != '-') & (training_df['3D file'] != 'NoINFO_Cathepsin S_WT_1_D3R.pdb') & (training_df['3D file'] != 'NoINFO_Cathepsin S_C25S_0_D3R.pdb')]
        training_df = training_df[training_df['Canonical SMILES'] != "C=S(=O)(O)c1ccc2c(c1)C(C)(C)C1=[N+]2CCC2OC3CCN4C(=C3C=C12)C(C)(C)c1cc(CC(=O)NCCCC[C@@H](NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCNC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCCCNCCNS(=O)(=O)c2cccc3cnccc23)C(N)=O)ccc14"]
        training_df = training_df[training_df['Canonical SMILES'] != 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)O']
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        test_df = test_df[(test_df['Canonical SMILES'] != '-') & (test_df['3D file'] != 'NoINFO_Cathepsin S_WT_1_D3R.pdb') & (test_df['3D file'] != 'NoINFO_Cathepsin S_C25S_0_D3R.pdb')]
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}') 
        model = GrAttCPI(num_of_aa_features, num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("GrAttCPI: Training the model has started!")
        #Training Phase
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        model.train()
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            if(len(training_df)%batch_size == 1):
                training_df = training_df.head(len(training_df)-1)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                running_loss = 0
                b = 0
                start_time = time.time()
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
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("GrAttCPI: Testing the model has started!") 
        #Test Phase            
        model.eval()
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        b = 0
        concatenated_pred_Y = []
        numpy_pred_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            targets_arr = batch_test_df['3D file'].to_numpy()
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            target_pkl_names = []
            for filename in targets_arr:
                pkl_name = filename.split('.pdb')[0] + '.pickle'
                target_pkl_names.append(pkl_name)
            target_batch, compound_batch = feature_preparation.get_RING_based_batch_for_GAT(target_pkl_names, smiles_arr, dev)
            with torch.no_grad():
                start_time = time.time()
                test_Y_hat = model.forward(target_batch, compound_batch, dev)
                end_time = time.time()
                test_elapsed_time = end_time - start_time
                break
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")

        
        
    def train_test_PhyGrAtt(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, is_side_training=False, side_training_csv='', drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):
        '''
            TRAIN and EVALUATE the PhyGrAtt model (Utilize the 2D drugs' graph for compounds and Physicochemical properties of the proteins)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
            is_side_training: In the case that you have an additional training set (Transfer Learning)
            side_training_csv: The name of the CSV file of the Training Set [NOTE: the CSV file MUST be located in the self.main_path]            
        '''  
        training_elapsed_time = 0
        test_elapsed_time = 0          
        dim_out = 4
        dim_hidden = 8
        num_of_aa_features = 20
        num_of_atom_features = 34
        targets_max_length = 1400
        aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
        aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
        compound_graph_adr = '../Datasets/GAT_Prepared_Graphs/Drugs'
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        Evaluation_data = {}
        feature_preparation = FeaturePreparation(compound_graph_adr, drug_ids_adr) 
        Evaluation_data['PhyGrAtt'] = []
        dev = 'cpu'
        if(use_gpu and torch.cuda.is_available()):
            dev = f'cuda:{gpu_device}'
        print(f"***********Running on '{dev}'***********")
        #Load Training and Test data
        training_df = pd.read_csv(f'{self.main_path}/Training.csv')
        training_df = training_df[training_df['Canonical SMILES'] != "C=S(=O)(O)c1ccc2c(c1)C(C)(C)C1=[N+]2CCC2OC3CCN4C(=C3C=C12)C(C)(C)c1cc(CC(=O)NCCCC[C@@H](NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCNC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCCCCNCCNS(=O)(=O)c2cccc3cnccc23)C(N)=O)ccc14"]
        training_df = training_df[training_df['Canonical SMILES'] != 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)O']
        test_df = pd.read_csv(f'{self.main_path}/Test.csv')
        side_training_df = ''
        if(is_side_training):
            side_training_df = pd.read_csv(f'{self.main_path}/{side_training_csv}')         
        model = PhyGrAtt(num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print("PhyGrAtt: Training the model has started!")
        #Training Phase
        Loss_data = {}
        Loss_data['Loss'] = []
        count_epoch = 1
        model.train()
        for shuffled_epoch in range(1):
            training_df = shuffle(training_df)
            if(len(training_df)%batch_size == 1):
                training_df = training_df.head(len(training_df)-1)
            training_index = training_df.index
            Y = training_df[['bind']].values
            Y = Variable(torch.LongTensor(Y.flatten()).to(torch.device(dev)), requires_grad=False)
            for epoch in range(1):
                running_loss = 0
                b = 0
                start_time = time.time()
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
                end_time = time.time()
                training_elapsed_time = end_time - start_time
                epoch_loss = running_loss/b
                Loss_data['Loss'].append(epoch_loss)
                print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                count_epoch += 1
        print("PhyGrAtt: Testing the model has started!") 
        #Test Phase            
        model.eval()
        test_df = shuffle(test_df)
        if(len(test_df)%batch_size == 1):
            test_df = test_df.head(len(test_df)-1)
        test_index = test_df.index     
        Y_test = test_df[['bind']].values
        Y_test = Variable(torch.LongTensor(Y_test.flatten()).to(torch.device(dev)), requires_grad=False)
        b = 0
        concatenated_pred_Y = []
        numpy_pred_Y = []
        for index in range(0, len(test_index), batch_size):
            batch_test_df = test_df.loc[test_index[index : index + batch_size]]
            batch_test_Y = Y_test[index : index + batch_size]
            smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
            targets_arr = batch_test_df['Sequence'].to_numpy()
            target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
            target_input_features = torch.FloatTensor(target_input_features).to(torch.device(dev))
            target_input_features = target_input_features.reshape((target_input_features.shape[0], 1, targets_max_length, aa_embedding_length))
            compound_batch = feature_preparation.get_Drug_batch_for_GAT(smiles_arr, dev)
            with torch.no_grad():
                start_time = time.time()
                test_Y_hat = model.forward(target_input_features, compound_batch, dev)
                end_time = time.time()
                test_elapsed_time = end_time - start_time
                break
        print(f"Training Time (For each epoch): {training_elapsed_time}")
        print(f"Test Time (For each batch): {test_elapsed_time}")                             