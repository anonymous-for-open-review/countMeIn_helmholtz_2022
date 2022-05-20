import os

# constants
img_rows = 100
img_cols = 100
osm_features = 56

# paths to the current folder
current_dir_path = os.getcwd()

# suffix for the file names
file_name_reg = 'rf_reg'
ground_truth_col_reg = 'POP'

# covariates used to train the model
covariate_list = ['DEM_MEAN', 'DEM_MAX', 'LCZ_CL', 'LU_1_A', 'LU_2_A', 'LU_3_A', 'LU_4_A', 'VIIRS_MEAN', 'VIIRS_MAX',
                  'SEN2_AUT_MEAN_R', 'SEN2_AUT_MEAN_G', 'SEN2_AUT_MEAN_B', 'SEN2_AUT_MED_R', 'SEN2_AUT_MED_G',
                  'SEN2_AUT_MED_B', 'SEN2_AUT_STD_R', 'SEN2_AUT_STD_G', 'SEN2_AUT_STD_B', 'SEN2_AUT_MAX_R',
                  'SEN2_AUT_MAX_G', 'SEN2_AUT_MAX_B', 'SEN2_AUT_MIN_R', 'SEN2_AUT_MIN_G', 'SEN2_AUT_MIN_B',
                  'SEN2_SPR_MEAN_R', 'SEN2_SPR_MEAN_G', 'SEN2_SPR_MEAN_B', 'SEN2_SPR_MED_R', 'SEN2_SPR_MED_G',
                  'SEN2_SPR_MED_B', 'SEN2_SPR_STD_R', 'SEN2_SPR_STD_G', 'SEN2_SPR_STD_B', 'SEN2_SPR_MAX_R',
                  'SEN2_SPR_MAX_G', 'SEN2_SPR_MAX_B', 'SEN2_SPR_MIN_R', 'SEN2_SPR_MIN_G', 'SEN2_SPR_MIN_B',
                  'SEN2_SUM_MEAN_R', 'SEN2_SUM_MEAN_G', 'SEN2_SUM_MEAN_B', 'SEN2_SUM_MED_R', 'SEN2_SUM_MED_G',
                  'SEN2_SUM_MED_B', 'SEN2_SUM_STD_R', 'SEN2_SUM_STD_G', 'SEN2_SUM_STD_B', 'SEN2_SUM_MAX_R',
                  'SEN2_SUM_MAX_G', 'SEN2_SUM_MAX_B', 'SEN2_SUM_MIN_R', 'SEN2_SUM_MIN_G', 'SEN2_SUM_MIN_B',
                  'SEN2_WIN_MEAN_R', 'SEN2_WIN_MEAN_G', 'SEN2_WIN_MEAN_B', 'SEN2_WIN_MED_R', 'SEN2_WIN_MED_G',
                  'SEN2_WIN_MED_B', 'SEN2_WIN_STD_R', 'SEN2_WIN_STD_G', 'SEN2_WIN_STD_B', 'SEN2_WIN_MAX_R',
                  'SEN2_WIN_MAX_G', 'SEN2_WIN_MAX_B', 'SEN2_WIN_MIN_R', 'SEN2_WIN_MIN_G', 'SEN2_WIN_MIN_B', 'aerialway',
                  'aeroway', 'amenity', 'barrier', 'boundary', 'building', 'craft', 'emergency', 'geological',
                  'healthcare', 'highway', 'historic', 'landuse', 'leisure', 'man_made', 'military', 'natural',
                  'office', 'place', 'power', 'public Transport', 'railway', 'route', 'shop', 'sport', 'telecom',
                  'tourism', 'water', 'waterway', 'addr:housenumber', 'restrictions', 'other', 'n', 'm', 'k_avg',
                  'intersection_count', 'streets_per_node_avg', 'streets_per_node_counts_argmin',
                  'streets_per_node_counts_min', 'streets_per_node_counts_argmax', 'streets_per_node_counts_max',
                  'streets_per_node_proportion_argmin', 'streets_per_node_proportion_min',
                  'streets_per_node_proportion_argmax', 'streets_per_node_proportion_max', 'edge_length_total',
                  'edge_length_avg', 'street_length_total', 'street_length_avg', 'street_segments_count',
                  'node_density_km', 'intersection_density_km', 'edge_density_km', 'street_density_km', 'circuity_avg',
                  'self_loop_proportion']


# Parameters for baseline experiment
min_fimportance = 0.002

# Parameter for the Grid Search for hyperparameter optimization
param_grid = {'oob_score': [True], 'bootstrap': [True],
              'max_features': ['sqrt', 0.05, 0.1, 0.2, 0.3, 0.4],
              'n_estimators': [250, 350, 500, 750, 1000]
              }

#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.tree import DecisionTreeRegressor

file_name_ada = 'ada_reg'
param_grid_adaboost = { #'base_estimator':[DecisionTreeRegressor, GaussianProcessRegressor],
              'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],  #, 2.0, 3.0],
              'n_estimators': [250, 350, 500, 750, 1000],
              'loss': ['linear', 'square', 'exponential']
}

file_name_gb = 'gb_reg'
param_grid_gradientboosting = {
              'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'n_estimators': [250, 350, 500, 750, 1000],
              'criterion':['friedman_mse', 'squared_error'],
              'max_features': ['sqrt', 0.05, 0.1, 0.2, 0.3, 0.4],

              }
file_name_voting = 'voting_reg'
params_grid_voting = {'rf__n_estimators': [250, 350, 500, 750, 1000], 
                      'ada__n_estimators': [250, 350, 500, 750, 1000], 
                      'gb__n_estimators': [250, 350, 500, 750, 1000],
                      #'rf__n_estimators': [250, 350, 500], 
                      #'ada__n_estimators': [250, 350, 500],
                      #'gb__n_estimators': [250, 350, 500], 
                      'gb__loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                      'gb__max_depth': [3, 4, 5, 6, 7, 8, 9],
                      'gb__criterion':['friedman_mse', 'squared_error'],
                      'gb__max_features': ['sqrt', 0.05, 0.1, 0.2, 0.3, 0.4],          
                      'ada__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],  #, 2.0, 3.0],
                      'ada__loss': ['linear', 'square', 'exponential'],
                      'rf__oob_score': [True], 
                      'rf__bootstrap': [True],
                      'rf__max_features': ['sqrt', 0.05, 0.1, 0.2, 0.3, 0.4],                                    
                      }

file_name_mlp = 'mlp_reg'
params_grid_mlp = {'solver': ['lbfgs', 'sgd', 'adam'], 
                      #'alpha':[1e-5],
                      'activation': ['logistic', 'tanh', 'relu'],
                      #'early_stopping': [True, False],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],
                      #'hidden_layer_sizes':[(5,), (50,), (100,), ],
}                       


# Kfold parameter
kfold = 3
n_jobs = -1
