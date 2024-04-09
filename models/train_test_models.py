import os
import csv
import math
import pickle
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, recall_score, precision_score, f1_score, roc_auc_score
from sklearn import metrics
import torch
from torch.autograd import Variable
from .DTIConvSeqSMILES_simple import DTIConvSeqSMILES
from .DTISeq2DFP_simple import DTISeq2DFP
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



class ModelsEvaluation:
    def __init__(self, main_path):
        '''
            main_path: The path that the folders of the training and test sets are located
        '''
        self.main_path = main_path
        
    
    
    def modify_main_path(self, new_main_path):
        '''
            new_main_path: The new path of the folders containing the folders of the training and test sets 
        '''
        self.main_path = new_main_path
    
    
    
    def plot_single_loss_curve(self, subdirectory, loss_csv_name, y_min, y_max):
        '''
            Plot the line chart of a specific training loss in various epochs 
            subdirectory: Subdirectory of the main path containing the Loss CSV file
            loss_csv_name: The name of the Loss CSV file 
            y_min: Minimum Error Value
            y_max: Maximum Error Value 
        '''
        df = pd.read_csv(f'{self.main_path}/{subdirectory}/{loss_csv_name}')
        ax = df.plot.line()
        ax.set_ylim(y_min, y_max)
        ax.figure.savefig(f'{self.main_path}/{subdirectory}/Loss_Plot.png')
    
    
    def evaluate_binary_classification(self, prediction_header, groundtruth_header, csv_file, delimiter = ','):
        '''
            Read the CSV file containing the prediction results and the ground truth
            The CSV file must have the headers so that the predicted and ground truth values will be recognizable
            prediction_header: The header of the prediction column in the CSV file
            groundtruth_header: the header of the class label column in the CSV file
            csv_file: The full address and the name of the CSV file
            delimiter: The delimiter which is used in the CSV files ('\t', ';', ',') By default it is ','
        '''
        results = {}
        prediction_df = pd.read_csv(csv_file,sep=delimiter)
        prediction_df["pred_binary"] = np.where(prediction_df[prediction_header] > 0.5, 1, 0)
        #Calculate accuracy, F1, Recall, Precision, AUC of PR and ROC curves
        acc = accuracy_score(y_true=prediction_df[groundtruth_header].to_numpy(), y_pred=prediction_df["pred_binary"].to_numpy())
        results["acc"] = acc
        f1 = f1_score(prediction_df[groundtruth_header].to_numpy(), prediction_df["pred_binary"].to_numpy())
        results["f1"] = f1
        recall = recall_score(prediction_df[groundtruth_header].to_numpy(), prediction_df["pred_binary"].to_numpy(), average='binary')
        results["recall"] = recall
        precision = precision_score(prediction_df[groundtruth_header].to_numpy(), prediction_df["pred_binary"].to_numpy(), average='binary')
        results["precision"] = precision
        auc = roc_auc_score(prediction_df[groundtruth_header].to_numpy(), prediction_df["pred_binary"].to_numpy())
        results["auc"] = auc
        #Plot the ROC curve
        roc_fpr, roc_tpr, _ = roc_curve(prediction_df[groundtruth_header].to_numpy(), prediction_df[prediction_header].to_numpy())
        results["roc_fpr"] = roc_fpr
        results["roc_tpr"] = roc_tpr
        lr_precision, lr_recall, _ = precision_recall_curve(prediction_df[groundtruth_header].to_numpy(), prediction_df[prediction_header].to_numpy())
        pr = metrics.auc(lr_recall, lr_precision)
        results["lr_precision"] = lr_precision
        results["lr_recall"] = lr_recall
        results["pr"] = pr
        results["y_proba"] = prediction_df[prediction_header]
        results["y_real"] = prediction_df[groundtruth_header]
        return results

    
    
    
    def plot_PR_ROC_average_evaluation_folds(self, name, prediction_header, groundtruth_header, model_name, plot_only_average, delimiter = ','):
        '''
            Plots the ROC and PR curves for all the folds and their average and also reports the details
            prediction_header: The header of the prediction column in the CSV files inside the folder
            groundtruth_header: the header of the class label column in the CSV files inside the folder
            model_name: The name of the model for evaluation
            plot_only_average: set 'True' if you only expect to see the average line and its confidence interval; otherwise 'False'
            delimiter: The delimiter which is used in the CSV files ('\t', ';', ',') By default it is ','
            NOTE: self.main_path MUST point to the folder which contains the CSV files of the predictions and the real values
        '''
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        avg_acc=avg_f1=avg_recall=avg_precision=avg_pr=avg_auc  = 0
        Results = [["Name","Accuracy","F1 score","PR","AUC","Recall","Precision"]]
        for f in sorted(os.listdir(self.main_path)):
            if(f.find('.') == -1): 
                if(f'Prediction_{model_name}.csv' in os.listdir(f'{self.main_path}/{f}')):
                    fold_number = int(f.split('-')[1])
                    results = self.evaluate_binary_classification(prediction_header,groundtruth_header, f'{self.main_path}/{f}/Prediction_{model_name}.csv', delimiter)
                    avg_acc += results["acc"]
                    avg_f1 += results["f1"]
                    avg_recall += results["recall"]
                    avg_precision += results["precision"]
                    avg_pr += results["pr"]
                    avg_auc += results["auc"]
                    fpr = results["roc_fpr"]
                    tpr = results["roc_tpr"]
                    row = [f,results["acc"],results["f1"],results["pr"],results["auc"],results["recall"],results["precision"]]
                    Results.append(row)
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    if(not plot_only_average):
                        '''ROC Plot for each individual fold '''
                        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d (AUC = %0.2f)' % (fold_number, roc_auc))
                    i += 1
        avg_acc /= i
        avg_f1 /= i
        avg_precision /= i
        avg_recall /= i
        avg_pr /= i
        avg_auc /= i
        row = ["Average",avg_acc,avg_f1,avg_pr,avg_auc,avg_recall,avg_precision]
        Results.append(row)
        with open(f'{self.main_path}/{name}_results_metric.csv', 'w') as f:
            writer = csv.writer(f)
            for row in Results:
                writer.writerow(row)
        plt.plot([0, 1], [0, 1], linestyle=(0, (3, 1, 1, 1, 1, 1)), lw=2, color='black', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        #A shadow between the lowest and the highest one
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='skyblue', alpha=.2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve")
        plt.legend(loc="lower right", fontsize=7)
        plt.savefig(f"{self.main_path}/ROC-{name}.png", dpi=300)
        plt.close()
        base_x = np.linspace(0, 1, 101)
        ys = []
        y_real = []
        y_proba = []
        i = 0
        for f in sorted(os.listdir(self.main_path)):
            if(f.find('.') == -1): 
                if(f'Prediction_{model_name}.csv' in os.listdir(f'{self.main_path}/{f}')):
                    fold_number = int(f.split('-')[1])
                    results = self.evaluate_binary_classification(prediction_header,groundtruth_header, f'{self.main_path}/{f}/Prediction_{model_name}.csv', delimiter)
                    # Plotting each individual PR Curve
                    precision, recall, _ = precision_recall_curve(results["y_real"], results["y_proba"])
                    if(not plot_only_average):
                        plt.plot(recall, precision, lw=1, alpha=0.3, label='Fold %d (AUC = %0.2f)' % (fold_number, average_precision_score(results["y_real"], results["y_proba"])))
                    y_real.append(results["y_real"])
                    y_proba.append(results["y_proba"])
                    i += 1
                    recall = list(recall)
                    precision = list(precision)
                    recall.reverse()
                    precision.reverse()
                    ys.append(np.interp(base_x, recall, precision))
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        plt.plot(recall, precision, color='b', label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)), lw=2, alpha=.8)
        ys = np.array(ys)
        mean_ys = ys.mean(axis=0)
        std = ys.std(axis=0)
        ys_upper = np.minimum(mean_ys + std, 1)
        ys_lower = np.maximum(mean_ys - std, 0)
        plt.fill_between(base_x, ys_lower, ys_upper, color='skyblue', alpha=0.3)
        plt.plot([0, 1], [1, 0], 'k-.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("PR curve")
        plt.legend(loc="lower left", fontsize=7)
        plt.savefig(f"{self.main_path}/PR-{name}.png", dpi=300)
        plt.close()
        print("Accuracy: " + str(avg_acc))
        print("F1: " + str(avg_f1))
        print("Recall: " + str(avg_recall))
        print("Precision: " + str(avg_precision))
    
    
    
    def compare_pr_auc_models(self, name, prediction_header, groundtruth_header, models_name, delimiter = ','):
        '''
            To compare the models on PR and ROC curves
            prediction_header: The header of the prediction column in the CSV files inside the folder
            groundtruth_header: the header of the class label column in the CSV files inside the folder
            models_name: The name of the models for evaluation and comparison
            delimiter: The delimiter which is used in the CSV files ('\t', ';', ',') By default it is ','
            NOTE: self.main_path MUST point to the folder which contains the CSV files of the predictions and the real values
        '''
        #main_colors =   ['dimgrey',  'gold',  'darkorange', 'seagreen', 'saddlebrown','darkmagenta','crimson','darkblue','limegreen',           'maroon']
        #shadow_colors = ['lightgrey','yellow','wheat',      'palegreen','peru',       'orchid',     'tomato', 'skyblue', 'lime',  'salmon']
        
        
        main_colors =   ['dimgrey',  'gold',  'limegreen','darkorange','deepskyblue','darkorchid', 'red',    'seagreen','firebrick','darkblue', 'fuchsia', 'blue', 'olive', 'mediumslateblue']
        shadow_colors = ['lightgrey','yellow','palegreen','wheat',     'skyblue',    'orchid',     'tomato', 'lime',    'salmon',   'skyblue',  'violet',  'cornflowerblue', 'yellowgreen', 'lavender']
        #################################################################### ROC curves
        l = 0
        for model_name in models_name:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            i = 0
            for f in sorted(os.listdir(self.main_path)):
                if(f.find('.') == -1): 
                    if(f'Prediction_{model_name}.csv' in os.listdir(f'{self.main_path}/{f}')):
                        fold_number = int(f.split('-')[1])
                        results = self.evaluate_binary_classification(prediction_header,groundtruth_header, f'{self.main_path}/{f}/Prediction_{model_name}.csv', delimiter)
                        fpr = results["roc_fpr"]
                        tpr = results["roc_tpr"]
                        tprs.append(interp(mean_fpr, fpr, tpr))
                        tprs[-1][0] = 0.0
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)
                        i += 1
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color=main_colors[l], label=model_name+r' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=1.0, alpha=.8)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            #A shadow between the lowest and the highest one
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=shadow_colors[l], alpha=.2)
            l += 1
        plt.plot([0, 1], [0, 1], linestyle=(0, (3, 1, 1, 1, 1, 1)), lw=2, color='black', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve")
        plt.legend(loc="lower right", fontsize=7)
        plt.savefig(f"{self.main_path}/ROC-{name}.png", dpi=300)
        plt.close()
        #################################################################### Precision-Recall curves
        l = 0
        for model_name in models_name:        
            base_x = np.linspace(0, 1, 101)
            ys = []
            y_real = []
            y_proba = []
            i = 0
            for f in sorted(os.listdir(self.main_path)):
                if(f.find('.') == -1): 
                    if(f'Prediction_{model_name}.csv' in os.listdir(f'{self.main_path}/{f}')):
                        fold_number = int(f.split('-')[1])
                        results = self.evaluate_binary_classification(prediction_header,groundtruth_header, f'{self.main_path}/{f}/Prediction_{model_name}.csv', delimiter)
                        precision, recall, _ = precision_recall_curve(results["y_real"], results["y_proba"])
                        y_real.append(results["y_real"])
                        y_proba.append(results["y_proba"])
                        i += 1
                        recall = list(recall)
                        precision = list(precision)
                        recall.reverse()
                        precision.reverse()
                        ys.append(np.interp(base_x, recall, precision))
            y_real = np.concatenate(y_real)
            y_proba = np.concatenate(y_proba)
            precision, recall, _ = precision_recall_curve(y_real, y_proba)
            plt.plot(recall, precision, color=main_colors[l], label=model_name+r' (PR = %0.2f)' % (average_precision_score(y_real, y_proba)), lw=1.0, alpha=.8)
            ys = np.array(ys)
            mean_ys = ys.mean(axis=0)
            std = ys.std(axis=0)
            ys_upper = np.minimum(mean_ys + std, 1)
            ys_lower = np.maximum(mean_ys - std, 0)
            plt.fill_between(base_x, ys_lower, ys_upper, color=shadow_colors[l], alpha=0.3)
            l += 1
        plt.plot([0, 1], [1, 0], 'k-.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("PR curve")
        plt.legend(loc="lower left", fontsize=7)
        plt.savefig(f"{self.main_path}/PR-{name}.png", dpi=300)
        plt.close()
    
    
        
    def train_test_RF(self):
        '''
            TRAIN and EVALUATE the Random Forest (RF) model -> Basic model
        '''
        feature_preparation = FeaturePreparation()    
        Evaluation_data = {}
        batch_size = 8000
        for training_test_sets_dir in os.listdir(f'{self.main_path}'):
            if(training_test_sets_dir.find('.') == -1):
                print(f'Dataset: {training_test_sets_dir}')
                Evaluation_data[training_test_sets_dir] = []
                training_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Training.csv')
                test_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Test.csv')
                training_set_SMILES = training_df['Canonical SMILES'].unique()
                test_set_SMILES = test_df['Canonical SMILES'].unique()
                smiles_np_array = np.concatenate((training_set_SMILES, test_set_SMILES))
                smiles_characters, smiles_maximum_length = feature_preparation.extract_all_SMILES_chars_and_max_len(smiles_np_array)
                smiles_chars_embedding_length = len(smiles_characters)
                #The maximum number of amino acid is 1400
                targets_max_length = 1400
                smiles_onehot_vectors_dict = feature_preparation.generate_one_hot_vectors(smiles_characters)
                aminoacids = ['M', 'T', 'D', 'L', 'A', 'F', 'Q', 'R', 'H', 'I', 'W', 'P', 'Y', 'S', 'V', 'E', 'G', 'C', 'N', 'K']
                aa_biophysicochemical_props_dict = pickle.load(open('../Datasets/biophysicochemical_PCA.pickle', 'rb'))
                training_df = shuffle(training_df)
                training_index = training_df.index
                aa_embedding_length = len(aa_biophysicochemical_props_dict['A'])
                training_df = shuffle(training_df)
                Y = training_df[['bind']].values
                pca = PCA(n_components=32, random_state=42)
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                has_prev = False
                final_features = []
                final_Y = []
                max_round = math.ceil(len(training_index)/batch_size)
                c = 0
                print("Extracting features phase!")
                for index in range(0, len(training_index), batch_size):
                    c += 1
                    print(f'{c}/{max_round}')
                    batch_df = training_df.loc[training_index[index : index + batch_size]]
                    if(len(batch_df) < 2):
                        break                            
                    batch_Y = Y[index : index + batch_size]
                    first_dim = len(batch_Y)
                    batch_Y = batch_Y.reshape(first_dim)
                    targets_arr = batch_df['Sequence'].to_numpy()
                    smiles_arr = batch_df['Canonical SMILES'].to_numpy()
                    target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    compound_input_features = feature_preparation.generate_matrices(smiles_onehot_vectors_dict, smiles_arr, smiles_maximum_length)
                    target_input_features = target_input_features.reshape(first_dim, 1400*20)
                    compound_input_features = compound_input_features.reshape(first_dim, 342*98)
                    # all_features: (first_dim, 61516)
                    all_features = np.concatenate([target_input_features, compound_input_features], axis=-1)
                    del target_input_features
                    del compound_input_features
                    reduced_features = pca.fit_transform(all_features) 
                    del all_features
                    if(has_prev):
                        final_features = np.concatenate([final_features, reduced_features], axis=0)
                        final_Y = np.concatenate([final_Y, batch_Y])
                    else:
                        final_features = reduced_features
                        final_Y = batch_Y
                    has_prev = True
                print("Training the models has started: RF")
                rf.fit(final_features, final_Y)
                print("Testing phase has just started!")
                if(len(test_df)%batch_size == 1):
                    test_df = test_df.head(len(test_df)-1)                
                test_index = test_df.index     
                Y_test = test_df[['bind']].values
                has_prev = False
                final_features = []
                final_Y = []
                max_round = math.ceil(len(test_index)/batch_size)
                c = 0
                for index in range(0, len(test_index), batch_size):
                    c += 1
                    print(f'{c}/{max_round}')
                    batch_test_df = test_df.loc[test_index[index : index + batch_size]]
                    batch_test_Y = Y_test[index : index + batch_size]
                    first_dim = len(batch_test_Y)
                    batch_test_Y = batch_test_Y.reshape(first_dim)
                    targets_arr = batch_test_df['Sequence'].to_numpy()
                    smiles_arr = batch_test_df['Canonical SMILES'].to_numpy()
                    target_input_features = feature_preparation.generate_matrices(aa_biophysicochemical_props_dict, targets_arr, targets_max_length)
                    compound_input_features = feature_preparation.generate_matrices(smiles_onehot_vectors_dict, smiles_arr, smiles_maximum_length)
                    target_input_features = target_input_features.reshape(first_dim, 1400*20)
                    compound_input_features = compound_input_features.reshape(first_dim, 342*98)
                    # all_features: (first_dim, 61516)
                    all_features = np.concatenate([target_input_features, compound_input_features], axis=-1)
                    del target_input_features
                    del compound_input_features
                    reduced_features = pca.fit_transform(all_features) 
                    del all_features
                    if(has_prev):
                        final_features = np.concatenate([final_features, reduced_features], axis=0)
                        final_Y = np.concatenate([final_Y, batch_test_Y])
                    else:
                        final_features = reduced_features
                        final_Y = batch_test_Y
                    has_prev = True
                rf_y_pred = rf.predict(final_features)
                rf_y_proba = rf.predict_proba(final_features)
                rf_results_data = {}
                rf_results_data["max_prediction"] = rf_y_pred
                rf_results_data['prediction'] = rf_y_proba[:, 1]
                rf_results_data["real"] = final_Y
                pd.DataFrame(rf_results_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Prediction_RF.csv', index=False)
        
        
        
    def train_test_DTIConvSeqSMILES(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0):
        '''
            TRAIN and EVALUATE the DTIConvSeqSMILES model (Utilize SMILES onehot vectors for compounds and physicochemical/onehot vectors for targets)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_biophysicochemical_props: Either to use 'Biophysicochemical properties' or 'onehot vectors' for amino acid representations. (default = False)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
        '''
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation()    
        Evaluation_data = {}
        for training_test_sets_dir in os.listdir(f'{self.main_path}'):
            if(training_test_sets_dir.find('.') == -1):
                Evaluation_data[training_test_sets_dir] = []
                dev = 'cpu'
                if(use_gpu and torch.cuda.is_available()):
                    dev = f'cuda:{gpu_device}'
                print(f"***********Running on '{dev}'***********")
                #Load Training and Test data
                training_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Training.csv')
                test_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Test.csv')

                training_set_SMILES = training_df['Canonical SMILES'].unique()
                test_set_SMILES = test_df['Canonical SMILES'].unique()
                smiles_np_array = np.concatenate((training_set_SMILES, test_set_SMILES))

                smiles_characters, smiles_maximum_length = feature_preparation.extract_all_SMILES_chars_and_max_len(smiles_np_array)
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
                print(f"{training_test_sets_dir}: Training the model has started!")
                #Training Phase
                Loss_data = {}
                Loss_data['Loss'] = []
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
                            batch_loss = loss.item() #* batch_size
                            running_loss += batch_loss
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                            optimizer.step()
                            b += 1
                        epoch_loss = running_loss/b
                        Loss_data['Loss'].append(epoch_loss)
                        print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                        count_epoch += 1
                pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Loss_DTIConvSeqSMILES.csv', index=False)
                print(f"{training_test_sets_dir}: Testing the model has started!")            
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
                    test_Y_hat = model.forward(compound_input_features, target_input_features)
                    dummy, preds_test = torch.max(test_Y_hat, dim = 1)
                    Y_prediction = torch.softmax(test_Y_hat, dim=1)
                    Y_prediction = np.array(Y_prediction.tolist())
                    if(b == 0):
                        concatenated_pred_Y = preds_test
                        numpy_pred_Y = Y_prediction
                    else:
                        concatenated_pred_Y = torch.cat((concatenated_pred_Y,preds_test))
                        numpy_pred_Y = np.concatenate((numpy_pred_Y, Y_prediction), axis=0)
                    b += 1

                num_of_ones = 0
                for el in concatenated_pred_Y:
                    if(el == 1):
                        num_of_ones += 1
                print(f'The number of <drug,target> pairs that predicted as positive: {str(num_of_ones)}')
                prediction_read_data = {}
                if(use_gpu):
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.cpu().numpy()
                    prediction_read_data['real'] = Y_test.cpu().numpy()
                else:
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.numpy()
                    prediction_read_data['real'] = Y_test.numpy()
                prediction_read_data['prediction'] = numpy_pred_Y[:, 1]
                
                pd.DataFrame(prediction_read_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Prediction_DTIConvSeqSMILES.csv', index=False)
                accuracy_test = (concatenated_pred_Y == Y_test).long().sum().float() /  concatenated_pred_Y.size()[0]    
                Y_real = np.array([[1,0] if y == 0 else [0,1] for y in Y_test])
                fpr = dict()
                tpr = dict()
                precision = dict()
                recall = dict()
                roc_auc = dict()
                PR_auc = dict()
                
                for i in range(2):
                    fpr[i], tpr[i], _ = roc_curve(Y_real[:, i], numpy_pred_Y[:, i])                    
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(Y_real[:, i], numpy_pred_Y[:, i])
                    PR_auc[i] = auc(recall[i], precision[i])
                if(use_gpu):
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.cpu().numpy()))
                else:            
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.numpy()))
                Evaluation_data[training_test_sets_dir].append(roc_auc[1])
                Evaluation_data[training_test_sets_dir].append(PR_auc[1])
                Evaluation_data[training_test_sets_dir].append(float(num_of_ones))
                if(use_gpu):
                    print(f"Accuracy: {str(float(accuracy_test.cpu().numpy()))}")
                else:
                    print(f"Accuracy: {str(float(accuracy_test.numpy()))}")
                print(f"ROC AUC: {str(roc_auc[1])}")
                print(f"PR AUC: {str(PR_auc[1])}")
        pd.DataFrame(Evaluation_data, index=['Accuracy','ROC AUC','PR AUC', 'Num of Positive Predicted']).to_csv(f'{self.main_path}/Evaluation_DTIConvSeqSMILES.csv')
        
        
    def train_test_DTISeq2DFP(self,num_of_epochs, num_of_shuffle_epochs, name_2D_FP, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0):
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
        '''
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_2D_FP)   
        Evaluation_data = {}
        for training_test_sets_dir in os.listdir(f'{self.main_path}'):
            if(training_test_sets_dir.find('.') == -1):
                Evaluation_data[training_test_sets_dir] = []
                dev = 'cpu'
                if(use_gpu and torch.cuda.is_available()):
                    dev = f'cuda:{gpu_device}'
                print(f"***********Running on '{dev}'***********")
                #Load Training and Test data
                training_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Training.csv')
                training_df = training_df[training_df['Canonical SMILES'] != '-']
                test_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Test.csv')
                test_df = test_df[test_df['Canonical SMILES'] != '-']
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
                print(f"{training_test_sets_dir}: Training the model has started!")
                #Training Phase
                Loss_data = {}
                Loss_data['Loss'] = []
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
                        Loss_data['Loss'].append(epoch_loss)
                        print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                        count_epoch += 1
                pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Loss_DTISeq2DFP.csv', index=False)
                print(f"{training_test_sets_dir}: Testing the model has started!")            
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
                    test_Y_hat = model.forward(compound_input_features, target_input_features)
                    dummy, preds_test = torch.max(test_Y_hat, dim = 1)
                    Y_prediction = torch.softmax(test_Y_hat, dim=1)
                    Y_prediction = np.array(Y_prediction.tolist())
                    if(b == 0):
                        concatenated_pred_Y = preds_test
                        numpy_pred_Y = Y_prediction
                    else:
                        concatenated_pred_Y = torch.cat((concatenated_pred_Y,preds_test))
                        numpy_pred_Y = np.concatenate((numpy_pred_Y, Y_prediction), axis=0)
                    b += 1
                num_of_ones = 0
                for el in concatenated_pred_Y:
                    if(el == 1):
                        num_of_ones += 1
                print(f'The number of <drug,target> pairs that predicted as positive: {str(num_of_ones)}')
                prediction_read_data = {}
                if(use_gpu):
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.cpu().numpy()
                    prediction_read_data['real'] = Y_test.cpu().numpy()
                else:
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.numpy()
                    prediction_read_data['real'] = Y_test.numpy()
                prediction_read_data['prediction'] = numpy_pred_Y[:, 1]

                pd.DataFrame(prediction_read_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Prediction_DTISeq2DFP.csv', index=False)
                accuracy_test = (concatenated_pred_Y == Y_test).long().sum().float() /  concatenated_pred_Y.size()[0]    
                Y_real = np.array([[1,0] if y == 0 else [0,1] for y in Y_test])
                fpr = dict()
                tpr = dict()
                precision = dict()
                recall = dict()
                roc_auc = dict()
                PR_auc = dict()

                for i in range(2):
                    fpr[i], tpr[i], _ = roc_curve(Y_real[:, i], numpy_pred_Y[:, i])                    
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(Y_real[:, i], numpy_pred_Y[:, i])
                    PR_auc[i] = auc(recall[i], precision[i])
                if(use_gpu):
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.cpu().numpy()))
                else:            
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.numpy()))
                Evaluation_data[training_test_sets_dir].append(roc_auc[1])
                Evaluation_data[training_test_sets_dir].append(PR_auc[1])
                Evaluation_data[training_test_sets_dir].append(float(num_of_ones))
                if(use_gpu):
                    print(f"Accuracy: {str(float(accuracy_test.cpu().numpy()))}")
                else:
                    print(f"Accuracy: {str(float(accuracy_test.numpy()))}")
                print(f"ROC AUC: {str(roc_auc[1])}")
                print(f"PR AUC: {str(PR_auc[1])}")
        pd.DataFrame(Evaluation_data, index=['Accuracy','ROC AUC','PR AUC', 'Num of Positive Predicted']).to_csv(f'{self.main_path}/Evaluation_DTISeq2DFP.csv')      
        
        
        
    def train_test_DTISeqE3FP(self,num_of_epochs, num_of_shuffle_epochs, name_3D_FP, max_num_of_conformers, batch_size, learning_rate = 0.1, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0):  
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
        '''        
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        feature_preparation = FeaturePreparation(name_3D_FP)   
        Evaluation_data = {}
        for training_test_sets_dir in os.listdir(f'{self.main_path}'):
            if(training_test_sets_dir.find('.') == -1):
                Evaluation_data[training_test_sets_dir] = []
                dev = 'cpu'
                if(use_gpu and torch.cuda.is_available()):
                    dev = f'cuda:{gpu_device}'
                print(f"***********Running on '{dev}'***********")
                #Load Training and Test data
                training_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Training.csv')
                training_df = training_df[training_df['Canonical SMILES'] != '-']
                test_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Test.csv')
                test_df = test_df[test_df['Canonical SMILES'] != '-']
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
                print(f"{training_test_sets_dir}: Training the model has started!")
                #Training Phase
                Loss_data = {}
                Loss_data['Loss'] = []
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
                        Loss_data['Loss'].append(epoch_loss)
                        print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                        count_epoch += 1
                pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Loss_DTISeqE3FP.csv', index=False)
                print(f"{training_test_sets_dir}: Testing the model has started!")            
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
                    test_Y_hat = model.forward(compound_input_features, target_input_features)
                    dummy, preds_test = torch.max(test_Y_hat, dim = 1)
                    Y_prediction = torch.softmax(test_Y_hat, dim=1)
                    Y_prediction = np.array(Y_prediction.tolist())
                    if(b == 0):
                        concatenated_pred_Y = preds_test
                        numpy_pred_Y = Y_prediction
                    else:
                        concatenated_pred_Y = torch.cat((concatenated_pred_Y,preds_test))
                        numpy_pred_Y = np.concatenate((numpy_pred_Y, Y_prediction), axis=0)
                    b += 1
                num_of_ones = 0
                for el in concatenated_pred_Y:
                    if(el == 1):
                        num_of_ones += 1
                print(f'The number of <drug,target> pairs that predicted as positive: {str(num_of_ones)}')
                prediction_read_data = {}
                if(use_gpu):
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.cpu().numpy()
                    prediction_read_data['real'] = Y_test.cpu().numpy()
                else:
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.numpy()
                    prediction_read_data['real'] = Y_test.numpy()
                prediction_read_data['prediction'] = numpy_pred_Y[:, 1]

                pd.DataFrame(prediction_read_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Prediction_DTISeqE3FP.csv', index=False)
                accuracy_test = (concatenated_pred_Y == Y_test).long().sum().float() /  concatenated_pred_Y.size()[0]    
                Y_real = np.array([[1,0] if y == 0 else [0,1] for y in Y_test])
                fpr = dict()
                tpr = dict()
                precision = dict()
                recall = dict()
                roc_auc = dict()
                PR_auc = dict()

                for i in range(2):
                    fpr[i], tpr[i], _ = roc_curve(Y_real[:, i], numpy_pred_Y[:, i])                    
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(Y_real[:, i], numpy_pred_Y[:, i])
                    PR_auc[i] = auc(recall[i], precision[i])
                if(use_gpu):
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.cpu().numpy()))
                else:            
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.numpy()))
                Evaluation_data[training_test_sets_dir].append(roc_auc[1])
                Evaluation_data[training_test_sets_dir].append(PR_auc[1])
                Evaluation_data[training_test_sets_dir].append(float(num_of_ones))
                if(use_gpu):
                    print(f"Accuracy: {str(float(accuracy_test.cpu().numpy()))}")
                else:
                    print(f"Accuracy: {str(float(accuracy_test.numpy()))}")
                print(f"ROC AUC: {str(roc_auc[1])}")
                print(f"PR AUC: {str(PR_auc[1])}")
        pd.DataFrame(Evaluation_data, index=['Accuracy','ROC AUC','PR AUC', 'Num of Positive Predicted']).to_csv(f'{self.main_path}/Evaluation_DTISeqE3FP.csv')      
                        
            
            
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
            
        
            
    def train_test_PhyChemDG(self, num_of_epochs, num_of_shuffle_epochs, batch_size = 64, learning_rate = 0.0001, use_biophysicochemical_props = False, use_gpu = False, gpu_device = 0, recompute_max_num_atoms = False, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv', drug_graphs_and_features_adr = '../Datasets/Drugs_2D_Graphs_AtomFeatures'):
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
        for training_test_sets_dir in os.listdir(f'{self.main_path}'):
            if(training_test_sets_dir.find('.') == -1):               
                #Load Training and Test data
                training_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Training.csv')
                training_df = training_df[training_df['Canonical SMILES'] != '-']
                test_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Test.csv')
                test_df = test_df[test_df['Canonical SMILES'] != '-']
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
                print(f"{training_test_sets_dir}: Training the model has started!")
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
                        epoch_loss = running_loss/b
                        Loss_data['Loss'].append(epoch_loss)
                        print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                        count_epoch += 1
                pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Loss_PhyChemDG.csv', index=False)
                print(f"{training_test_sets_dir}: Testing the model has started!")
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
                    correct_labels, predicted_labels, predicted_scores = tester.test(batch_features)
                    max_prediction_Y.extend(predicted_labels)
                    real_Y.extend(correct_labels)
                    prediction_Y.extend(predicted_scores)
                prediction_real_data = {}
                prediction_real_data['max_prediction'] = max_prediction_Y
                prediction_real_data['real'] = real_Y
                prediction_real_data['prediction'] = prediction_Y
                pd.DataFrame(prediction_real_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Prediction_PhyChemDG.csv', index=False)
                
                
                
    def train_test_GrAttCPI(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):  
        '''
            TRAIN and EVALUATE the GrAttCPI model (Utilize the 2D drugs' graph for compounds and 2D networks which are obtained from AlphaFold+RING)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
        '''     
        dim_out = 4
        dim_hidden = 8
        num_of_aa_features = 20
        num_of_atom_features = 34
        target_graph_adr = '../Datasets/GAT_Prepared_Graphs/RING_based'
        compound_graph_adr = '../Datasets/GAT_Prepared_Graphs/Drugs'
        total_epochs = num_of_epochs*num_of_shuffle_epochs
        Evaluation_data = {}
        feature_preparation = FeaturePreparation(target_graph_adr, compound_graph_adr, drug_ids_adr) 
        for training_test_sets_dir in os.listdir(f'{self.main_path}'):
            if(training_test_sets_dir.find('.') == -1):
                Evaluation_data[training_test_sets_dir] = []
                dev = 'cpu'
                if(use_gpu and torch.cuda.is_available()):
                    dev = f'cuda:{gpu_device}'
                print(f"***********Running on '{dev}'***********")
                #Load Training and Test data
                training_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Training.csv')
                training_df = training_df[(training_df['Canonical SMILES'] != '-') & (training_df['3D file'] != 'NoINFO_Cathepsin S_WT_1_D3R.pdb') & (training_df['3D file'] != 'NoINFO_Cathepsin S_C25S_0_D3R.pdb')]
                test_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Test.csv')
                test_df = test_df[(test_df['Canonical SMILES'] != '-') & (test_df['3D file'] != 'NoINFO_Cathepsin S_WT_1_D3R.pdb') & (test_df['3D file'] != 'NoINFO_Cathepsin S_C25S_0_D3R.pdb')]
                model = GrAttCPI(num_of_aa_features, num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                print(f"{training_test_sets_dir}: Training the model has started!")
                #Training Phase
                Loss_data = {}
                Loss_data['Loss'] = []
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
                        Loss_data['Loss'].append(epoch_loss)
                        print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                        count_epoch += 1
                pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Loss_GrAttCPI.csv', index=False)
                print(f"{training_test_sets_dir}: Testing the model has started!") 
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
                        test_Y_hat = model.forward(target_batch, compound_batch, dev)
                        dummy, preds_test = torch.max(test_Y_hat, dim = 1)
                        Y_prediction = torch.softmax(test_Y_hat, dim=1)
                        Y_prediction = np.array(Y_prediction.tolist())
                        if(b == 0):
                            concatenated_pred_Y = preds_test
                            numpy_pred_Y = Y_prediction
                        else:
                            concatenated_pred_Y = torch.cat((concatenated_pred_Y,preds_test))
                            numpy_pred_Y = np.concatenate((numpy_pred_Y, Y_prediction), axis=0)
                        b += 1
                num_of_ones = 0
                for el in concatenated_pred_Y:
                    if(el == 1):
                        num_of_ones += 1
                print(f'The number of <drug,target> pairs that predicted as positive: {str(num_of_ones)}')
                prediction_read_data = {}
                if(use_gpu):
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.cpu().numpy()
                    prediction_read_data['real'] = Y_test.cpu().numpy()
                else:
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.numpy()
                    prediction_read_data['real'] = Y_test.numpy()
                prediction_read_data['prediction'] = numpy_pred_Y[:, 1]
                pd.DataFrame(prediction_read_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Prediction_GrAttCPI.csv', index=False)    
                accuracy_test = (concatenated_pred_Y == Y_test).long().sum().float() /  concatenated_pred_Y.size()[0]    
                Y_real = np.array([[1,0] if y == 0 else [0,1] for y in Y_test])
                fpr = dict()
                tpr = dict()
                precision = dict()
                recall = dict()
                roc_auc = dict()
                PR_auc = dict()

                for i in range(2):
                    fpr[i], tpr[i], _ = roc_curve(Y_real[:, i], numpy_pred_Y[:, i])                    
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(Y_real[:, i], numpy_pred_Y[:, i])
                    PR_auc[i] = auc(recall[i], precision[i])
                if(use_gpu):
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.cpu().numpy()))
                else:            
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.numpy()))
                Evaluation_data[training_test_sets_dir].append(roc_auc[1])
                Evaluation_data[training_test_sets_dir].append(PR_auc[1])
                Evaluation_data[training_test_sets_dir].append(float(num_of_ones))
                if(use_gpu):
                    print(f"Accuracy: {str(float(accuracy_test.cpu().numpy()))}")
                else:
                    print(f"Accuracy: {str(float(accuracy_test.numpy()))}")
                print(f"ROC AUC: {str(roc_auc[1])}")
                print(f"PR AUC: {str(PR_auc[1])}")
        pd.DataFrame(Evaluation_data, index=['Accuracy','ROC AUC','PR AUC', 'Num of Positive Predicted']).to_csv(f'{self.main_path}/Evaluation_GrAttCPI.csv')  

        
        
    def train_test_PhyGrAtt(self,num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate = 0.1, use_gpu = False, gpu_device = 0, drug_ids_adr = '../Datasets/Conformers/ConformersSetIds-Drugs.csv'):
        '''
            TRAIN and EVALUATE the PhyGrAtt model (Utilize the 2D drugs' graph for compounds and Physicochemical properties of the proteins)
            num_of_epochs: Number of epochs for a constant order of a training set
            num_of_shuffle_epochs: Number of epochs in which for each epoch we shuffle the training set
            batch_size: The batch size
            learning_rate: The learning rate of the model (default = 0.1)
            use_gpu: True -> The model uses gpu, and False -> The model uses cpu. (default = False)
            gpu_device: In the case that you are using gpu, declare the number of the gpu device. (default = 0)
        '''  
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
        for training_test_sets_dir in os.listdir(f'{self.main_path}'):
            if(training_test_sets_dir.find('.') == -1):
                Evaluation_data[training_test_sets_dir] = []
                dev = 'cpu'
                if(use_gpu and torch.cuda.is_available()):
                    dev = f'cuda:{gpu_device}'
                print(f"***********Running on '{dev}'***********")
                #Load Training and Test data
                training_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Training.csv')
                test_df = pd.read_csv(f'{self.main_path}/{training_test_sets_dir}/Test.csv')
                model = PhyGrAtt(num_of_atom_features, dim_hidden, dim_out).to(torch.device(dev))
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                print(f"{training_test_sets_dir}: Training the model has started!")
                #Training Phase
                Loss_data = {}
                Loss_data['Loss'] = []
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
                        # 100 to len(training_index)
                        for index in range(0, 100, batch_size):
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
                        Loss_data['Loss'].append(epoch_loss)
                        print(f">>>>>>>>>>{str(count_epoch)}/{str(total_epochs)}>>>>>>>>>>Epoch loss: {str(running_loss/b)}")
                        count_epoch += 1
                pd.DataFrame(Loss_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Loss_PhyGrAtt.csv', index=False)
                print(f"{training_test_sets_dir}: Testing the model has started!") 
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
                        test_Y_hat = model.forward(target_input_features, compound_batch, dev)
                        dummy, preds_test = torch.max(test_Y_hat, dim = 1)
                        Y_prediction = torch.softmax(test_Y_hat, dim=1)
                        Y_prediction = np.array(Y_prediction.tolist())
                        if(b == 0):
                            concatenated_pred_Y = preds_test
                            numpy_pred_Y = Y_prediction
                        else:
                            concatenated_pred_Y = torch.cat((concatenated_pred_Y,preds_test))
                            numpy_pred_Y = np.concatenate((numpy_pred_Y, Y_prediction), axis=0)
                        b += 1
                num_of_ones = 0
                for el in concatenated_pred_Y:
                    if(el == 1):
                        num_of_ones += 1
                print(f'The number of <drug,target> pairs that predicted as positive: {str(num_of_ones)}')
                prediction_read_data = {}
                if(use_gpu):
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.cpu().numpy()
                    prediction_read_data['real'] = Y_test.cpu().numpy()
                else:
                    prediction_read_data['max_prediction'] = concatenated_pred_Y.numpy()
                    prediction_read_data['real'] = Y_test.numpy()
                prediction_read_data['prediction'] = numpy_pred_Y[:, 1]
                pd.DataFrame(prediction_read_data).to_csv(f'{self.main_path}/{training_test_sets_dir}/Prediction_PhyGrAtt.csv', index=False)    
                accuracy_test = (concatenated_pred_Y == Y_test).long().sum().float() /  concatenated_pred_Y.size()[0]    
                Y_real = np.array([[1,0] if y == 0 else [0,1] for y in Y_test])
                fpr = dict()
                tpr = dict()
                precision = dict()
                recall = dict()
                roc_auc = dict()
                PR_auc = dict()
                for i in range(2):
                    fpr[i], tpr[i], _ = roc_curve(Y_real[:, i], numpy_pred_Y[:, i])                    
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    precision[i], recall[i], _ = precision_recall_curve(Y_real[:, i], numpy_pred_Y[:, i])
                    PR_auc[i] = auc(recall[i], precision[i])
                if(use_gpu):
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.cpu().numpy()))
                else:            
                    Evaluation_data[training_test_sets_dir].append(float(accuracy_test.numpy()))
                Evaluation_data[training_test_sets_dir].append(roc_auc[1])
                Evaluation_data[training_test_sets_dir].append(PR_auc[1])
                Evaluation_data[training_test_sets_dir].append(float(num_of_ones))
                if(use_gpu):
                    print(f"Accuracy: {str(float(accuracy_test.cpu().numpy()))}")
                else:
                    print(f"Accuracy: {str(float(accuracy_test.numpy()))}")
                print(f"ROC AUC: {str(roc_auc[1])}")
                print(f"PR AUC: {str(PR_auc[1])}")
                
                
                
        pd.DataFrame(Evaluation_data, index=['Accuracy','ROC AUC','PR AUC', 'Num of Positive Predicted']).to_csv(f'{self.main_path}/Evaluation_PhyGrAtt.csv')         