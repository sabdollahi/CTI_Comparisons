from models.train_test_models import ModelsEvaluation
from datapreparation.DataPrepare import DataProvider
from datapreparation.DataConversion import DataConvert
from models.single_train_test import SingleTrainTest
from models.train_save_models import TrainAndSaveModels


# In general, the number of compounds in our dataset is 58122 (IDs range from 0 to 58121). But, it can be increased!
'''To generate 2D graphs of a set of drugs and the nodes' (atoms') features'''
#data_provider = DataProvider('Drugs_2D_Graphs_AtomFeatures')
#data_provider.extract_2D_graphs('Conformers/ConformersSetIds-Drugs.csv')
'''To generate 2D graphs of a NEW set of drugs (Given a list of SMILES notations)'''
#data_provider = DataProvider('Drugs_2D_Graphs_AtomFeatures')
#list_of_smiles = ["COc1cc(OC)c(NC(=O)Nc2cnc(C#N)cn2)cc1Cl","COc1cc(Nc2nccc(Nc3c(Cl)ccc4c3OCO4)n2)cc(OC)c1OC","CC(C)c1nnc2ccc(-c3ocnc3-c3ccc(F)cc3)cn12","CN1CCN(c2cnc3cc(C(F)(F)F)cc(-c4cccc(O)c4)c3n2)CC1"]
#data_provider.extract_2D_graphs_for_list_of_smiles('Conformers/ConformersSetIds-Drugs.csv', list_of_smiles)




'''To generate negative samples'''
#neg_over_pos_ratio = 20
#data_provider = DataProvider()
#data_provider.generate_negative_samples(neg_over_pos_ratio, 'DrugBank_drug_fingerprints.pickle', 'Drugs_Canonical_SMILES_3DProtsAvailable.csv', 'Targets_leq_1500_3D_Availability.csv', 'DTI.csv', 'negative_DTI_2times_bigger.csv')
#data_provider.generate_negative_samples(neg_over_pos_ratio, 'DrugBank_drug_fingerprints.pickle', 'Drugs_Canonical_SMILES_3DProtsAvailable.csv', 'Targets_leq_1500_3D_Availability.csv', 'DTI_nonmetal_drugs.csv', 'negative_DTI_20times_nonmetals.csv')




'''To generate K Cold Start Folds and then construct the training-test sets'''
#k = 10       
#data_provider = DataProvider("BindingDatasets_3DAvailable")   
#col_smiles = 'Canonical SMILES'
#col_seq = 'Sequence'
#csv_file = 'Drug-Target-Binding-v3-nonmetal.csv'
#Warm start for drugs
#data_provider.generate_k_warm_start_folds(k, col_smiles, csv_file, 'WarmStart-Drugs-v3')
#data_provider.generate_training_test_sets('WarmStart-Drugs-v3', k)
#Cold start for drugs
#data_provider.generate_k_cold_start_folds(k, col_smiles, csv_file, 'ColdStart-Drugs-v3')
#data_provider.generate_training_test_sets('ColdStart-Drugs-v3', k)
#Warm start for targets
#data_provider.generate_k_warm_start_folds(k, col_seq, csv_file, 'WarmStart-Targets-v3')
#data_provider.generate_training_test_sets('WarmStart-Targets-v3', k)
#Cold start for targets
#data_provider.generate_k_cold_start_folds(k, col_seq, csv_file, 'ColdStart-Targets-v3')       
#data_provider.generate_training_test_sets('ColdStart-Targets-v3', k)




'''To assign random values instead of Y_hat (predicted binding status)'''
#data_provider = DataProvider("BindingDatasets_3DAvailable")
#train_test_sets_dir = "WarmStart-Drugs-v1"    
#data_provider.assign_random_Ys(train_test_sets_dir)




''' In the case that you have a single pair of training and test sets in a specific directory'''
#name_3D_FP = "drug_3D_fingerprints_v1.pickle"
#name_2D_FP = "drug_2D_fingerprints.pickle"
#num_of_epochs = 50
#num_of_shuffle_epochs = 4
#batch_size = 64
#learning_rate = 0.0001
#use_biophysicochemical_props = True
#gpu_device = 0
#use_gpu = False
#max_num_of_conformers = 3
#main_path = '../Datasets/BindingDatasets_3DAvailable/TrainWT-TestMut'
#single_train_test = SingleTrainTest(main_path)
#single_train_test.train_test_DTIConvSeqSMILES(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#single_train_test.modify_main_path('../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/Davis')
#single_train_test.train_test_DTIConvSeqSMILES(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#single_train_test.modify_main_path('../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/DrugBank')
#single_train_test.train_test_DTIConvSeqSMILES(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#single_train_test.modify_main_path('../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/KiBA')
#single_train_test.train_test_DTIConvSeqSMILES(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#single_train_test.modify_main_path('../Datasets/BindingDatasets_3DAvailable/Train4Datasets-TestSoren')
#is_side_training = True
#side_training_csv = 'Side-Training.csv'
#single_train_test.train_test_DTIConvSeqSMILES(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device, is_side_training, side_training_csv)




''' To Train the models on whole dataset and save the model '''
name_3D_FP = "drug_3D_fingerprints_v1.pickle"
name_2D_FP = "drug_2D_fingerprints.pickle"
num_of_epochs = 1
num_of_shuffle_epochs = 1
batch_size = 64
learning_rate = 0.0001
use_biophysicochemical_props = True
gpu_device = 0
use_gpu = False
max_num_of_conformers = 3
train_save = TrainAndSaveModels()
#train_save.train_save_DeepDTA(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#use_biophysicochemical_props = True
#train_save.train_save_DeepCAT(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#train_save.plot_tSNE_DeepDTA_target_LFs(use_biophysicochemical_props, use_gpu, gpu_device)
#train_save.train_save_AlphaFoldGrAtts(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_gpu, gpu_device)
#train_save.plot_tSNE_AlphaFoldGrAtts_target_LFs()
#train_save.train_save_PhyGrAtt(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_gpu, gpu_device)
#train_save.plot_tSNE_PhyGrAtt_target_LFs()
#train_save.train_save_DTISeqE3FP(num_of_epochs, num_of_shuffle_epochs, name_3D_FP, max_num_of_conformers, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#train_save.plot_tSNE_E3FP_target_LFs()
#train_save.train_save_DTISeq2DFP(num_of_epochs, num_of_shuffle_epochs, name_2D_FP, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#train_save.plot_tSNE_2DFP_target_LFs()
#train_save.train_save_PhyChemDG(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#train_save.plot_tSNE_PhyChemDG_target_LFs()




''' To evaluate the models for 10 folds '''
main_path = '../Datasets/BindingDatasets_3DAvailable/WarmStart-Drugs-v3'
#single_train_test = SingleTrainTest(main_path)
#single_train_test.compute_metrics_folds()

''' To evaluate the models using MCC, Accuracy, F1 score, AUC PR, and AUC ROC metrics '''
main_path = '../Datasets/BindingDatasets_3DAvailable/TrainWT-TestMut'
#single_train_test = SingleTrainTest(main_path)
#single_train_test.compute_metrics()

''' To plot Spider (Radar) chart of the results '''
#NOTE: You first NEED to run: single_train_test.compute_metrics() to get the MODELS_EVALUATION.csv file
main_path = '../Datasets/BindingDatasets_3DAvailable/TrainWT-TestMut'
title = 'Train on WT/Test on Mut'
name_to_be_saved = 'WTMut-Spider-Plot'
#single_train_test = SingleTrainTest(main_path)
#single_train_test.plot_spider_chart(title, name_to_be_saved)

''' To compare the models based on a crietrion: 'Accuracy', 'Auc ROC', 'AUC PR', 'F1 score', or 'MCC' using boxplots '''
#NOTE: You first NEED to run: single_train_test.compute_metrics() to get the MODELS_EVALUATION.csv file
#colors = ['blue',           'darkorchid', 'darkorange',      'deepskyblue', 'darkblue',   'fuchsia']
#models = ['TransformerCPI', 'BERT-based', 'AlphaFoldGrAtts', 'PhyGrAtt',    'E3FP-based', '2DFP-based']
#y_label = 'F1 score'
#title = 'KiBA Performance Comparison based on RB'
#name_to_be_saved = 'KiBA_RB_F1_Boxplots_Comparison'
#comparison_datasets = ['W/O Limitation', 'LRB (#RB < 10)']
#main_path = '../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/KiBA'
#compare_with_adr = '../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/KiBA_Limited_Rotatable_Bonds'
#single_train_test = SingleTrainTest(main_path)
#single_train_test.plot_boxplots_comparisons(colors, models, y_label, title, comparison_datasets, compare_with_adr, name_to_be_saved)

''' To compare more than two boxlots '''
#NOTE: You first NEED to run: single_train_test.compute_metrics() to get the MODELS_EVALUATION.csv file
colors = ['mediumslateblue', 'olive',  'blue',           'darkorchid', 'darkorange',      'deepskyblue', 'darkblue',   'fuchsia']
models = ['DeepConv-DTI',    'IIFDTI', 'TransformerCPI', 'BERT-based', 'AlphaFoldGrAtts', 'PhyGrAtt',    'E3FP-based', '2DFP-based']
y_label = 'MCC'
dataset_name = 'DrugBank'
title = f'{dataset_name} Performance Comparison based on RB'
name_to_be_saved = f'{dataset_name}_RB_MCC_Boxplots_Comparison_including_Ratio'
comparison_datasets = ['W/O Limitation', 'LRB (#RB < 10)', 'RRB (Ratio < 0.184)']
main_path = f'../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/{dataset_name}'
compare_with_adrs = [f'../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/{dataset_name}_Limited_Rotatable_Bonds', f'../Datasets/BindingDatasets_3DAvailable/Drug-Dissimilar/{dataset_name}_Ratio_LRB']
#single_train_test = SingleTrainTest(main_path)
#single_train_test.plot_multiple_boxplots_comparisons(colors, models, y_label, title, comparison_datasets, compare_with_adrs, name_to_be_saved)




'''Train and Test a model -> K-fold (K Training and Test sets)'''
#name_3D_FP = "drug_3D_fingerprints_v1.pickle"
#name_2D_FP = "drug_2D_fingerprints.pickle"
#num_of_epochs = 50
#num_of_shuffle_epochs = 4
#batch_size = 64
#learning_rate = 0.0001
#use_biophysicochemical_props = True
#gpu_device = 0
#use_gpu = False
#max_num_of_conformers = 3
# Dataset 'v1' is using for the DTIConvSeqSMILES and DTISeq2DFP models and dataset 'v2' is using for the DTISeqE3FP model
#main_path = '../Datasets/BindingDatasets_3DAvailable/ColdStart-Drugs-v3'
#model_evaluation = ModelsEvaluation(main_path)
#model_evaluation.train_test_DTIConvSeqSMILES(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#model_evaluation.train_test_DTISeq2DFP(num_of_epochs, num_of_shuffle_epochs, name_2D_FP, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#model_evaluation.train_test_DTISeqE3FP(num_of_epochs, num_of_shuffle_epochs, name_3D_FP, max_num_of_conformers, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#model_evaluation.train_test_PhyChemDG(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_biophysicochemical_props, use_gpu, gpu_device)
#model_evaluation.train_test_GrAttCPI(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_gpu, gpu_device)
#model_evaluation.train_test_RF()
#model_evaluation.train_test_PhyGrAtt(num_of_epochs, num_of_shuffle_epochs, batch_size, learning_rate, use_gpu, gpu_device)




'''To plot PR and ROC curves and save the details in a CSV file'''
#main_path = '../Datasets/BindingDatasets_3DAvailable/WarmStart-Targets-v3'
#prediction_header_in_csv = 'prediction'
#real_header_in_csv = 'real'
#model_name = 'TransformerCPI'
#name_to_be_saved = 'TransformerCPI-WarmStart-Targets-v3'
#plot_only_average = False
#model_evaluation = ModelsEvaluation(main_path)
#model_evaluation.plot_PR_ROC_average_evaluation_folds(name_to_be_saved, prediction_header_in_csv, real_header_in_csv, model_name, plot_only_average) 




'''To compare the models based on PR and ROC curves'''
#models = ['Random', 'RF', 'PhyChemDG', 'AlphaFoldGrAtts', 'PhyGrAtt', 'BERT-based', 'UniRep-based', 'DeepCAT', 'DeepDTA', 'E3FP-based', '2DFP-based', 'TransformerCPI', 'IIFDTI', 'DeepConv-DTI']  
#name_to_be_saved = 'CD-Evaluation-2'
#main_path = '../Datasets/BindingDatasets_3DAvailable/ColdStart-Drugs-v3'
#prediction_header_in_csv = 'prediction'
#real_header_in_csv = 'real'
#model_evaluation = ModelsEvaluation(main_path)
#model_evaluation.compare_pr_auc_models(name_to_be_saved, prediction_header_in_csv, real_header_in_csv, models, delimiter = ',') 




'''To plot the loss curve'''
#y_min = 0.35
#y_max = 0.6
#model_evaluation.plot_single_loss_curve('TrainingTestSets-3', 'Loss_DTIConvSeqSMILES.csv', y_min, y_max)




'''To convert the original input format to the other formats [for the other state-of-the-art models]'''
#data_conversion = DataConvert('ColdStart-Drugs-v3', 'ColdStart-Drugs-v3')
#data_conversion.convert_to_TransformerCPI_format()
#data_conversion.set_destination_dir('ColdStart-Targets-v3')
#data_conversion.set_source_dir('ColdStart-Targets-v3')
#data_conversion.convert_to_TransformerCPI_format()
#data_conversion.set_destination_dir('WarmStart-Targets-v3')
#data_conversion.set_source_dir('WarmStart-Targets-v3')
#data_conversion.convert_to_TransformerCPI_format()
#data_conversion.set_destination_dir('WarmStart-Drugs-v3')
#data_conversion.set_source_dir('WarmStart-Drugs-v3')
#data_conversion.convert_to_TransformerCPI_format()
#data_conversion.set_destination_dir('Train4Datasets-TestSoren')
#data_conversion.set_source_dir('Train4Datasets-TestSoren')
#data_conversion.convert_one_Training_Test_to_TransformerCPI_format()
#data_conversion.set_destination_dir('TrainWT-TestMut')
#data_conversion.set_source_dir('TrainWT-TestMut')
#data_conversion.convert_one_Training_Test_to_TransformerCPI_format()
#data_conversion.set_destination_dir('Drug-Dissimilar/Davis')
#data_conversion.set_source_dir('Drug-Dissimilar/Davis')
#data_conversion.convert_one_Training_Test_to_TransformerCPI_format()
#data_conversion.set_destination_dir('Drug-Dissimilar/DrugBank')
#data_conversion.set_source_dir('Drug-Dissimilar/DrugBank')
#data_conversion.convert_one_Training_Test_to_TransformerCPI_format()
#data_conversion.set_destination_dir('Drug-Dissimilar/KiBA')
#data_conversion.set_source_dir('Drug-Dissimilar/KiBA')
#data_conversion.convert_one_Training_Test_to_TransformerCPI_format()
