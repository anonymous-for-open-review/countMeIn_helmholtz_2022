# required packages
import os
import argparse
import pdb
 
try:
    from osgeo import gdal
    from osgeo import osr, ogr
except:
    import gdal
    import osr

from utils import feature_engineering, validation_reg, get_perf

from rf_regression import rf_regressor
from adaboost_regression import adaboost_regressor
from gradientboosting_regression import gradientboosting_regressor
from voting_regression import voting_regressor
from mlp_regression import mlp_regressor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path_So2Sat_pop_part1",
        type=str, help="Enter the path to So2Sat POP Part1 folder", required=True)
    
    parser.add_argument(
        "--data_path_So2Sat_pop_part2", required=True,
        type=str, help="Enter the path to So2Sat POP Part2 folder")
    
    parser.add_argument(
        "--learning_method", required=True,
        type=str, help="Enter the learning algorithm to use within: ['random_forest', 'adaboost', 'gradient_boosting', 'voting', 'mlp']")
    
    parser.add_argument(
        "--tuning_method", required=True,
        type=str, help="Enter the HPO tuning algorithm to use within: ['grid', 'halving']")
    
    parser.add_argument(
        "--seed", required=True,
        type=int, help="Enter the seed")
 
    parser.add_argument(
        "--training_no_engineering", required=True, type=int, 
        help="Enter if performing the training (no feature engineering done) or just the feature engineering step [1 or 0]")
    
    parser.add_argument(
        "--data_path_feature_folder", required=False,
        type=str, help="Enter the path to the feature folder")
 
    args = parser.parse_args()
    
    learning_method = args.learning_method
    tuning_method = args.tuning_method
    seed = args.seed
    training_no_engineering = args.training_no_engineering
    data_path_feature_folder = args.data_path_feature_folder
    
    all_patches_mixed_part1 = args.data_path_So2Sat_pop_part1
    all_patches_mixed_part2 = args.data_path_So2Sat_pop_part2
    
    all_patches_mixed_train_part1 = os.path.join(all_patches_mixed_part1, 'train')   # path to train folder
    all_patches_mixed_test_part1 = os.path.join(all_patches_mixed_part1, 'test')  # path to test folder
    
    all_patches_mixed_train_part2 = os.path.join(all_patches_mixed_part2, 'train')   # path to train folder
    all_patches_mixed_test_part2 = os.path.join(all_patches_mixed_part2, 'test')   # path to test folder

    print('\nPath to So2Sat POP Part1: ', all_patches_mixed_part1)
    print('Path to So2Sat POP Part2: ', all_patches_mixed_part2)
    
    assert learning_method in ['random_forest', 'adaboost', 'gradient_boosting', 'voting', 'mlp']
    assert tuning_method in ['grid', 'halving']
    assert training_no_engineering in [1, 0]
    if training_no_engineering == 1:
        assert len(data_path_feature_folder) > 0
    #pdb.set_trace()
    
    if training_no_engineering == 0:
        # create features for training and testing data from So2Sat POP Part1 and So2Sat POP Part2
        feature_folder = feature_engineering(all_patches_mixed_part1)
        feature_folder = feature_engineering(all_patches_mixed_part2)
        print("feature_folder: ", feature_folder)
    
    elif training_no_engineering == 1:
        #elif training_no_engineering:
        feature_folder = data_path_feature_folder
        
        # Perform regression, ground truth is population count (POP)
        if learning_method == 'random_forest':
            prediction_csv = rf_regressor(feature_folder, 
                                          hp_strategy=tuning_method, 
                                          seed=seed)
            
        elif learning_method == 'adaboost':
            prediction_csv = adaboost_regressor(feature_folder, 
                                          hp_strategy=tuning_method, 
                                          seed=seed)
            
        elif learning_method == 'gradient_boosting':
            prediction_csv = gradientboosting_regressor(feature_folder, 
                                          hp_strategy=tuning_method, 
                                          seed=seed)
            
        elif learning_method == 'voting':
            prediction_csv = voting_regressor(feature_folder, 
                                          hp_strategy=tuning_method, 
                                          seed=seed)
            
        elif learning_method == 'mlp':
            prediction_csv = mlp_regressor(feature_folder, 
                                          hp_strategy=tuning_method, 
                                          seed=seed)
            
        else:
            print('No learning')
            pass
        
        
        validation_csv_path = prediction_csv.replace('prediction', 'validation')
        validation_reg(prediction_csv, validation_csv_path, all_patches_mixed_test_part1)
    
        mae, rmse, rsq = get_perf(prediction_csv, validation_csv_path, 
                                  all_patches_mixed_test_part1)
    
    
    
    