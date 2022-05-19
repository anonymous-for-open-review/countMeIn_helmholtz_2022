# Random forest regression implementation, creates rf_model directory in the current directory to save all the logs
import glob
import os
import time
import _pickle

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

from utils import plot_feature_importance
from constants import min_fimportance, kfold, n_jobs, params_grid_voting, covariate_list, current_dir_path, file_name_voting, ground_truth_col_reg
import numpy as np

def voting_regressor(feature_folder, hp_strategy=None, seed=0):
    """
    :param feature_folder: path to feature folder
    :return: prediction csv path
    """
    print("Starting regression")
    np.random.seed(seed)
    
    # get all training cities
    feature_folder_train = os.path.join(feature_folder, 'train')
    all_train_cities = glob.glob(os.path.join(feature_folder_train, '*'))
    # prepare the training dataframe
    training_df = pd.DataFrame()
    for each_city in all_train_cities:
        #city_csv = glob.glob(os.path.join(each_city, '*_features.csv'))[0]  # get the feature csv
        #city_df = pd.read_csv(city_csv)
        city_csv = glob.glob(os.path.join(each_city, '*_features.pkl'))[0]  # get the feature csv        
        city_df = pd.read_pickle(city_csv)        
        training_df = training_df.append(city_df, ignore_index=True)  # append data from all the training cities

    training_df.fillna(0, inplace=True)
    # Get the dependent variables
    y = training_df[ground_truth_col_reg]
    # Get the independent variables
    x = training_df[covariate_list]

    print("Starting training...\n")
    
    # Initialize the model
    
    gb_model = GradientBoostingRegressor(n_estimators=500, random_state=0)  # random_state is fixed to allow exact replication
    adaboost_model = AdaBoostRegressor(n_estimators=500, random_state=0)  # random_state is fixed to allow exact replication
    rfmodel = RandomForestRegressor(n_estimators=500, oob_score=True, n_jobs=-1,
                                    random_state=0)  # random_state is fixed to allow exact replication
    
    #voting_model = VotingRegressor(estimators=[('ada', adaboost_model), ('rf', rfmodel), ('gb', gb_model)], n_jobs=-1)
    
    
    sel = SelectFromModel(rfmodel, threshold=min_fimportance, importance_getter='auto')
    fited = sel.fit(x, y)
    feature_idx = fited.get_support()  # Get list of T/F for covariates for which OOB score is upper the threshold
    list_covar = list(x.columns[feature_idx])  # Get list of covariates with the selected features
    x = fited.transform(x) # Update the dataframe with the selected features only

    # Instantiate the grid search model
    print("Starting Grid search with cross validation...\n")
    search = None
    
    assert hp_strategy is not None
    if hp_strategy == 'grid':
        search = GridSearchCV(estimator=VotingRegressor(estimators=[('ada', adaboost_model), ('rf', rfmodel), ('gb', gb_model)], n_jobs=-1), 
                              param_grid=params_grid_voting, 
                              cv=kfold, 
                              n_jobs=n_jobs, 
                              verbose=0)
    elif hp_strategy == 'halving':
        search = HalvingGridSearchCV(estimator=VotingRegressor(estimators=[('ada', adaboost_model), ('rf', rfmodel), ('gb', gb_model)], n_jobs=-1), 
                                     param_grid=params_grid_voting, 
                                     cv=kfold, 
                                     n_jobs=n_jobs, 
                                     verbose=0, 
                                     factor=2, 
                                     max_resources=75)
    
    search.fit(x, y)  # Fit the grid search to the data
    regressor = search.best_estimator_  # Save the best regressor
    regressor.fit(x, y)  # Fit the best regressor with the data
    # mean cross-validated score (OOB) and stddev of the best_estimator
    best_score = search.cv_results_['mean_test_score'][search.best_index_]
    best_std = search.cv_results_['std_test_score'][search.best_index_]

    rf_model_folder = os.path.join(current_dir_path, "rf_logs")  # path to the folder "rf_model"
    if not os.path.exists(rf_model_folder):
        os.mkdir(rf_model_folder)  # creates rf_logs folder inside the project folder

    model_folder = os.path.join(rf_model_folder, time.strftime("%Y%m%d-%H%M%S_") + file_name_voting)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)  # creates folder inside the rf_logs folder, named as per time stamp and file_name

    model_name = time.strftime("%Y%m%d-%H%M%S_") + file_name_voting  # model name
    rf_model_path = os.path.join(model_folder, model_name)  # path to saved model

    # save the best regressor
    with open(rf_model_path, 'wb') as f:
        _pickle.dump(regressor, f)
        f.close()

    # Save the log
    log = ""
    message = 'Parameter grid for Random Forest tuning :\n'
    for key in params_grid_voting.keys():
        message += '    ' + key + ' : ' + ', '.join([str(i) for i in list(params_grid_voting[key])]) + '\n'
    message += '    ' + 'min_fimportance' + ' : ' + str(min_fimportance) + '\n'
    log += message + '\n'

    message = 'Optimized parameters for Random Forest after grid search %s-fold cross-validation tuning :\n' % kfold
    for key in search.best_params_.keys():
        message += '    %s : %s' % (key, search.best_params_[key]) + '\n'
    log += message + '\n'

    message = "Mean cross-validated score (OOB) and stddev of the best_estimator : %0.3f (+/-%0.3f)" % (
    best_score, best_std) + '\n'
    log += message + '\n'

    # Print mean OOB and stddev for each set of parameters
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']
    message = "Mean cross-validated score (OOB) and stddev for every tested set of parameter :\n"
    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        message += "%0.3f (+/-%0.03f) for %r" % (mean, std, params) + '\n'
    log += message + '\n'

    # Save the log
    fout = open(os.path.join(model_folder, '%s_training_log.txt' % model_name), 'w')
    fout.write(log)
    fout.close()
    #################################################################################################

    # Start the predictions on completely unseen test data set
    print("Starting testing...\n")
    feature_folder_test = os.path.join(feature_folder, 'test')
    all_test_cities = glob.glob(os.path.join(feature_folder_test, '*'))# get all test cities
    test_df = pd.DataFrame()
    for each_test_city in all_test_cities:
         #test_city_csv = glob.glob(os.path.join(each_test_city, '*features.csv'))[0]  # get the feature csv
        #test_city_df = pd.read_csv(test_city_csv)
        
        test_city_csv = glob.glob(os.path.join(each_test_city, '*features.pkl'))[0]  # get the feature csv
        test_city_df = pd.read_pickle(test_city_csv)
        test_df = test_df.append(test_city_df, ignore_index=True)  # append all test cities together

    # Get features
    x_test = test_df[list_covar]

    # load the trained model
    with open(rf_model_path, 'rb') as f:
        regressor = _pickle.load(f)

    # Predict on test data set
    prediction = regressor.predict(x_test)

    # Save the prediction
    df_pred = pd.DataFrame()
    df_pred["CITY"] = test_df['CITY']
    df_pred["GRD_ID"] = test_df['GRD_ID']
    df_pred['Predictions'] = prediction

    pred_csv_path = os.path.join(model_folder, '%s_predictions.csv' % model_name)
    df_pred.to_csv(pred_csv_path, index=False)

    ## Feature importances
    #print("Creation of feature importance plot...\n")
    #importances = regressor.feature_importances_  # Save feature importances from the model
    #path_plot = os.path.join(model_folder, "%s_RF_feature_importance" % model_name)  # path to saved plot
    #plot_feature_importance(importances, x_test, path_plot)
    
    return pred_csv_path