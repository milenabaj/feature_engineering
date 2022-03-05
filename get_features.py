# coding=utf-8
"""
@author: Milena Bajic (DTU Compute)
"""
import sys, os, json, time
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import argparse, json
import tsfel
import pickle
from utils.data_transforms import *
from utils.plotting import *
from utils.data_loaders import *
from utils.analysis import *
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import FormatStrFormatter
from math import ceil, floor
import gc, os, sys, glob             

#=================================#
# SETTINGS
#=================================#
# Script arguments
parser = argparse.ArgumentParser(description='Please provide command line arguments.')

parser.add_argument('--route', nargs='+', help='Process all trips on those routes, which are found in json file.')
parser.add_argument('--trip', type = int, help='Process this trip only.')

# Vehicle type to align with GM: you can pass multiple
parser.add_argument('--p79', action='store_true', help = 'If you want to load the aligned dataset with p79 data, pass true.')
parser.add_argument('--aran', action='store_true', help = 'If you want to load the aligned dataset with aran data, pass true.')
parser.add_argument('--viafrik', action='store_true', help = 'If you want to load the aligned dataset with Viafrik friction data, pass true.')

parser.add_argument('--target', help = 'Target for machine learning. Select between: IRI_mean, DI, KPI.')
parser.add_argument('--load_add_sensors', action='store_true', help = 'Load input dataset containing additional sensors.') 
parser.add_argument('--use_3dacc', action='store_true', help = 'Use 3D acceleration sensors.') 
parser.add_argument('--use_add_sensors', action='store_true', help = 'Use additional sensors in fe and fs.') 
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--step', type=int, default=10)

parser.add_argument('--json_route', default= "json/routes.json", help='Json file with route information.')
parser.add_argument('--in_dir', default= "data", help='Input directory base.')
parser.add_argument('--recreate_fe', action="store_true", help = 'Recreate fe files, even if present. If False and the files are present, the data will be loaded from them.')
parser.add_argument('--recreate_fs', action="store_true", help = 'Recreate fs files, even if present. If False and the files are present, the data will be loaded from them.')

parser.add_argument('--mode',  default = 'trainvalidkfold_test', help = 'If you want to use the the loaded data for feature extraction and selection, use Choose between: trainvalid and trainvalidkfold. If you also want to prepare a part of it as a test dataset, use: trainvalid_test or trainvalidkfold_test. ')    
parser.add_argument('--dev_mode', action="store_true", help = 'Run on a subset of lines only. Use for debugging purposes.')
parser.add_argument('--predict_mode',  action="store_true", help = 'Prediction mode - use if you want to treat all loaded data as test data or in production mode. The code will load a list of selected features and extract only them.')    
parser.add_argument('--no_filter_speed', action="store_true", help = 'Do not filter speed.')
    
# Parse arguments
args = parser.parse_args()
if args.route:
    routes = args.route
else:
    routes = ['M3_VH','M3_HH','M13_VH','M13_HH']
trip  = args.trip

# Vehicle
p79 = args.p79
aran = args.aran
viafrik = args.viafrik

# Target
target_name = args.target

filter_speed = not args.no_filter_speed
use_3dacc = args.use_3dacc
load_add_sensors = args.load_add_sensors
use_add_sensors = args.use_add_sensors
use_3dacc = args.use_3dacc
window_size = args.window_size
step = args.step

json_route = args.json_route
in_dir_base = args.in_dir
recreate_fe = args.recreate_fe
recreate_fs = args.recreate_fs

mode = args.mode
predict_mode = args.predict_mode
dev_mode = args.dev_mode
dev_nrows = 2


make_plots = False
only_test = False
resample = False
#=================================#  
# Check mode
if mode and mode not in ['trainvalid','trainvalidkfold','trainvalid_test','trainvalidkfold_test']:
    print('Wrong mode passed - set mode to one of trainvalid, trainvalidkfold, trainvalid_test, trainvalidkfold_test')
    sys.exit(0)
  
# None passed
if not predict_mode and mode not in ['trainvalid','trainvalidkfold','trainvalid_test','trainvalidkfold_test']:
    print('Pass either --predict_mode or set mode to one of trainvalid, trainvalidkfold, trainvalid_test, trainvalidkfold_test')
    sys.exit(0) 
    
suff = ''        
if filter_speed:
    suff = 'filter_speed'


# Basic sensors: acceleration and speed
if use_3dacc:
   input_feats = ['GM.obd.spd_veh.value','GM.acc.xyz.x', 'GM.acc.xyz.y', 'GM.acc.xyz.z']
   suff = suff + '_3daccspeed'
else:
    input_feats = ['GM.obd.spd_veh.value', 'GM.acc.xyz.z']  
    suff = suff + '_accspeed'
  
# Add additional sensors
steering_sensors = ['GM.obd.strg_pos.value', 'GM.obd.strg_acc.value','GM.obd.strg_ang.value'] 
wheel_pressure_sensors =  ['GM.obd.whl_prs_rr.value', 'GM.obd.whl_prs_rl.value','GM.obd.whl_prs_fr.value','GM.obd.whl_prs_fl.value'] 
other_sensors = ['GM.obd.acc_yaw.value','GM.obd.trac_cons.value']
add_sensors = steering_sensors + wheel_pressure_sensors + other_sensors 
if use_add_sensors:
   input_feats = input_feats + add_sensors
   suff = suff + '_add_sensors'
       
# Route and trip
if trip and not len(routes)==1 and not predict_mode:
    print('If you pass trip, pass its route.')
    sys.exit(0)
   
with open(json_route, "r") as f:
    route_data = json.load(f)

# Trips
if trip:
   trips = [trip]
else:
   trips = []
   for route in routes:
       trips  =  trips + route_data[route]['GM_trips']

# DRD type  
if not aran and not p79 and not viafrik:
    print('Set p79 or aran or both to True')
    sys.exit(0)

# Input directories
drd_veh = ''     
if p79: 
    drd_veh = drd_veh + '_P79'
if aran:
    drd_veh = drd_veh + '_ARAN'
if not p79 and aran and not viafrik:
    drd_veh = drd_veh + '_VIAFRIK'
if drd_veh.startswith('_'):
   drd_veh =  drd_veh[1:]  
   
in_dirs = []
for route in routes:
    in_dir = '{0}/aligned_GM_{1}_data_window-{2}-step-{3}/{4}'.format(in_dir_base, drd_veh, window_size, step, route) 
    
    if load_add_sensors:
        in_dir = in_dir.replace(route, route+'_add_sensors')
        
    in_dirs.append(in_dir)

# Out directory
routes_string = '_'.join(routes)
out_dir_base = '/'.join(in_dir.split('/')[0:-1]).replace('aligned_','aligned_fe_fs_')
out_dir = '{0}/{1}_{2}'.format(out_dir_base, routes_string, suff)
if load_add_sensors and not use_add_sensors:
    out_dir = out_dir.replace('_add_sensors','')
    
# Create dir
out_dir_base = out_dir
if not os.path.exists(out_dir_base):
    os.makedirs(out_dir_base)
    
# FE output
out_dir_plots_fe = '{0}/plots_fe'.format(out_dir_base)
if not os.path.exists(out_dir_plots_fe):
    os.makedirs(out_dir_plots_fe)
           
print('p79 data? ', p79)
print('Aran data? ', aran)
print('Viafrik data? ',viafrik)
print('Additional sensors? ', load_add_sensors)
print('Trip: ', trips)
print('Dev mode?: ',dev_mode)
print('Input directories: ', in_dirs)
print('Output directory: ', out_dir)
time.sleep(3)  

# =====================================================================   #
# Process aligned files
# =====================================================================   #     
filenames = []
for trip in trips:
    print('Checking trip: ',trip)
    for in_dir in in_dirs:
        print('Checking directory: ',in_dir)
          
        # Find filenames
        filenames = filenames + glob.glob('{0}/*{1}*.pickle'.format(in_dir, trip))
filenames=list(set(filenames)) 

# Load
dfs = []
for filename in filenames:
    print('Loading :',filename)
    df = pd.read_pickle(filename)
    
    # Drop other sensors
    if not use_add_sensors:
        df.drop(add_sensors, axis = 1, inplace = True)
        
    # Speed filter
    if filter_speed:
        df['b']= df['GM.obd.spd_veh.value'].apply(lambda row: (row<20).sum())
        df = df[ df['b']==0]
        df.reset_index(drop=True, inplace=True)
        df.drop(['b'],axis=1, inplace=True)

    if dev_mode:
        df = df.head(dev_nrows)
        
    dfs.append(df)
    
print('Loaded files: ',filenames) 
         
# Data            
df = pd.concat(dfs)  #without and with speed filter = 3300, 33.1 km

# Prepare data
non_GM_cols = [col for col in  df.columns if not col.startswith('GM.')]
clean_nans(df, exclude_cols=non_GM_cols) # 1609
df.reset_index(inplace=True, drop = True)

# Plot 
vars_to_plot = ['IRI_mean', 'GM.obd.spd_veh.value']
for var in vars_to_plot:
    x = df[var]
    get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fe, suff = '_all')
    
# Predict mode
if predict_mode:
    trainvalid_df = None
    test_df = df
    vars_to_plot = ['GM.obd.spd_veh.value']
  
# Split
elif 'test'in mode:
    
    # Trainvalid
    trainvalid_n = int(0.8*df.shape[0])
    trainvalid_df = df[0:trainvalid_n]
    trainvalid_df.reset_index(inplace=True, drop=True)
    
    # Test df
    test_df = df[trainvalid_n:]
    test_df.reset_index(inplace=True, drop=True)
else:
    trainvalid_df = df #1287
    test_df = None # 322

   
# =====================================================================   #
# Feature extraction for trainvalid
# =====================================================================   # 
if (trainvalid_df is not None) and (not only_test):
        to_lengths_dict = {}
        for feat in input_feats:
            a =  trainvalid_df[feat].apply(lambda seq: seq.shape[0])
            l = int(a.quantile(0.90))
            to_lengths_dict[feat] = l
            #print(to_lengths_dict)
            #to_lengths_dict = {'GM.acc.xyz.z': 369, 'GM.obd.spd_veh.value':309} # this was used for motorway
            
        # Plot trainvalid
        for var in vars_to_plot:
            x = trainvalid_df[var]
            get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fe, suff = '_trainvalid')
        
        
        # Resample
        if resample:
            trainvalid_df, feats_resampled = resample_df(trainvalid_df, feats_to_resample = input_feats, to_lengths_dict = to_lengths_dict, window_size = window_size)
        
        # Do feature extraction 
        keep_cols = trainvalid_df.columns.to_list()
        print(trainvalid_df.shape)
        sys.exit(0)
        trainvalid_df, fe_filename = feature_extraction(trainvalid_df, keep_cols = keep_cols, feats = input_feats, 
                                                        out_dir = out_dir_base, 
                                                   file_suff = routes_string + suff +'_trainvalid', 
                                                   write_out_file = True, recreate = recreate_fe, predict_mode = predict_mode)
     
    
        cols = trainvalid_df.columns.to_list()
        fe_cols = list(set(cols).difference(keep_cols))
        
        fe = {}
        for input_sensor in input_feats:
            print('Exploring extracted features for: {0}'.format(input_sensor))
            fe_this_sensor = [col for col in fe_cols if input_sensor in col]
            n_fe_this_sensor = len(fe_this_sensor)
            print('=== Sensor: {0}, extracted: {1} features'.format(input_sensor,  n_fe_this_sensor))
            fe[input_sensor] =  fe_this_sensor
            
        #=== Sensor: GM.obd.spd_veh.value, extracted: 35 features
        #=== Sensor: GM.acc.xyz.z, extracted: 35 features
        # ['GM.acc.xyz.z-0_Neighbourhood peaks', 'GM.acc.xyz.z-0_Entropy', 'GM.acc.xyz.z-0_Mean absolute diff', 'GM.acc.xyz.z-0_Area under the curve', 'GM.acc.xyz.z-0_ECDF Percentile 0.8', 'GM.acc.xyz.z-0_ECDF Percentile 0.2', 'GM.acc.xyz.z-0_Interquartile range', 'GM.acc.xyz.z-0_Median absolute deviation', 'GM.acc.xyz.z-0_Mean diff', 'GM.acc.xyz.z-0_Zero crossing rate', 'GM.acc.xyz.z-0_Variance', 'GM.acc.xyz.z-0_Root mean square', 'GM.acc.xyz.z-0_Skewness', 'GM.acc.xyz.z-0_Centroid', 'GM.acc.xyz.z-0_Signal distance', 'GM.acc.xyz.z-0_Negative turning points', 'GM.acc.xyz.z-0_Max', 'GM.acc.xyz.z-0_Absolute energy', 'GM.acc.xyz.z-0_Min', 'GM.acc.xyz.z-0_Sum absolute diff', 'GM.acc.xyz.z-0_ECDF Percentile 0.05', 'GM.acc.xyz.z-0_Mean absolute deviation', 'GM.acc.xyz.z-0_Autocorrelation', 'GM.acc.xyz.z-0_Peak to peak distance', 'GM.acc.xyz.z-0_Maxmin diff', 'GM.acc.xyz.z-0_Median', 'GM.acc.xyz.z-0_Positive turning points', 'GM.acc.xyz.z-0_Kurtosis', 'GM.acc.xyz.z-0_ECDF Percentile 0.1', 'GM.acc.xyz.z-0_Slope', 'GM.acc.xyz.z-0_Median absolute diff', 'GM.acc.xyz.z-0_Median diff', 'GM.acc.xyz.z-0_Total energy', 'GM.acc.xyz.z-0_Mean', 'GM.acc.xyz.z-0_Standard deviation']
        
        # Plot histogram of each extracted feature and correlation with the target
        for var in fe_cols:
            x = trainvalid_df[var]
            if make_plots:
                get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fe, suff = 'trainvalid_'+suff, norm = False)


        # Remove some columns if needed
        to_rem = []
        avail_cols = list(trainvalid_df.columns)
        for col in avail_cols:
            if any(x in col for x in ['ECDF Percentile Count', 'ECDF_']):
                 to_rem.append(col)  
    
        trainvalid_df.drop(to_rem,axis=1,inplace=True)
        trainvalid_df.reset_index(drop=True, inplace = True)
        
        # Compute target if DI or KPI
        if target_name=='DI' or 'KPI':
            compute_di_aran(trainvalid_df)
            compute_kpi_aran(trainvalid_df)
            
        # Select X and target 
        X_trainvalid_fe = trainvalid_df[fe_cols] 
        y_trainvalid =  trainvalid_df[target_name]
        
        # Get valid indices
        if 'kfold' in mode:
            print('SFS will be done with kfold validation')
            X_valid_indices = None # the trainvalid fs will be done in kfold manner
        else:
            print('SFS will be done with train-valid split validation')
            valid_nrows = int(0.25*trainvalid_df.shape[0]) # valid = 0.2 of the whole dataset, 0.25 of trainvalid
            X_valid_indices = trainvalid_df.iloc[-valid_nrows:].index.tolist()


# =====================================================================   #
# Feature extraction for test
# =====================================================================   #  
if test_df is not None:
    to_lengths_dict = {}
    for feat in input_feats:
        a =  test_df[feat].apply(lambda seq: seq.shape[0])
        l = int(a.quantile(0.90))
        to_lengths_dict[feat] = l
        
    # Plot test
    for var in vars_to_plot:
        x = test_df[var]
        get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fe, suff = '_test')
    # Write some info
    
    # Resample
    if resample:
        test_df, feats_resampled = resample_df(test_df, feats_to_resample = input_feats, to_lengths_dict = to_lengths_dict, window_size = window_size)
    
    # Do feature extraction 
    keep_cols = test_df.columns.to_list()
    test_df, fe_filename = feature_extraction(test_df, keep_cols = keep_cols, feats = input_feats, 
                                               out_dir = out_dir_base, 
                                               file_suff = routes_string + suff +'_test', 
                                               write_out_file = True, recreate = recreate_fe, predict_mode = predict_mode)
 

    cols = test_df.columns.to_list()
    fe_cols = list(set(cols).difference(keep_cols))
    
    # Plot extracted
    fe = {}
    for input_sensor in input_feats:
        print('Exploring extracted features for: {0}'.format(input_sensor))
        fe_this_sensor = [col for col in fe_cols if input_sensor in col]
        n_fe_this_sensor = len(fe_this_sensor)
        print('=== Sensor: {0}, extracted: {1} features'.format(input_sensor,  n_fe_this_sensor))
        fe[input_sensor] =  fe_this_sensor
         
    # Plot histogram of each extracted feature and correlation with the target
    for var in fe_cols:
        x = test_df[var]
        if make_plots:
            get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fe, suff = 'test_'+suff, norm = False)

    # Compute target if DI or KPI
    if target_name=='DI' or 'KPI':
        compute_di_aran(test_df)
        compute_kpi_aran(test_df)
        
    # Select target 
    X_test_fe = test_df[fe_cols]
    y_test =  test_df[target_name]  


# =====================================================================   #
# Feature selection for trainvalid and test
# =====================================================================   # 
# Out dir for FS
out_dir = '{0}/feature_selection_{1}'.format(out_dir_base, target_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
model_names = ['random_forest']
for model_name in model_names:
    
    # Create dir for this model
    out_dir = '{0}/{1}'.format(out_dir, model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_dir_plots_fs = '{0}/plots_fs'.format(out_dir)
    if not os.path.exists(out_dir_plots_fs):
        os.makedirs(out_dir_plots_fs)

            
    # Do FS
    trainvalid_fs, sel_feature_names, feature_selector = find_optimal_subset(X_trainvalid_fe, y_trainvalid, 
                                                                               valid_indices = X_valid_indices, 
                                                                               reg_model = True, 
                                                                               target_name = target_name,
                                                                               out_dir =  out_dir, 
                                                                               outfile_suff = 'trainvalid_' + suff + '_'+target_name, 
                                                                               recreate = recreate_fs)
    #sys.exit(0)
    # Write json file
    json_feats_file_1 = 'json/selected_features_{0}_route-{0}_GM_trip-{1}_sensors-{2}_model-{3}.json'.format(drd_veh, routes_string, suff, model_name)  
    json_feats_file_2 = json_feats_file_1.replace('json/', out_dir+'/')
    for f_name in [json_feats_file_1, json_feats_file_2]:
        f =  open(f_name, "w") 
        sel_feat_data = {"features":sel_feature_names}
        json.dump(sel_feat_data, f)    
        f.close()
    
    sel_feat_data = pd.DataFrame(sel_feat_data)
        
    # Write feature info into a tex file
    tex_feats_file = json_feats_file_2.replace('.json','.tex')
    sel_feat_data.to_latex(tex_feats_file)
    
    csv_feats_file = json_feats_file_2.replace('.json','.csv')
    sel_feat_data.to_csv(csv_feats_file)
    #sel_feat_data.to_latex(tex_feats_file, columns = feats.columns, index = False, 
    #                float_format = lambda x: '%.2e' % x, label = 'table:selected_features',  
    #                header=[format_col(col) for col in feats.columns], escape=False)
    
    # Print selected features
    n_sel_features = len(sel_feature_names) 
    print('Selected features are: {0}'.format(sel_feature_names))
    print('Number of selected features is:{0}'.format(n_sel_features))
    
    # Plot histogram of each selected feature and correlation with the target
    for var in fe_cols:
        x = trainvalid_df[var]
        if make_plots:
            get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fs, suff = 'trainvalid_'+suff+target_name, norm = False)
            # add correlation plots
            #plot_correlation(trainvalid_df, method = 'pearson', out_dir = out_dir_plots_fs, suff = 'trainvalid_'+suff+target_name)
   
    X_trainvalid_fs = trainvalid_fs[sel_feature_names]
    y_trainvalid = trainvalid_fs[target_name]
    
    # Fit on full trainvalid
    if feature_selector:
        model = feature_selector.estimator
        model.fit(X_trainvalid_fs, y_trainvalid)
        s_trainvalid = model.score(X_trainvalid_fs, y_trainvalid)
        print('Score (trainvalid): ',s_trainvalid)
        
    # Load or save the best model, its parameters and predictions
    model_path = '{0}/best_model_{1}.pickle'.format(out_dir, model_name)
    if not recreate_fs and os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        params = model.get_params()
    else:
        with open(model_path, 'wb') as handle:
            pickle.dump(model, handle, protocol=4)
            print('Wrote best model to: {0}'.format(model_path))
            
        params = model.get_params()
        params = pd.DataFrame([params]).T
        
        params_path = model_path.replace('best_model','best_model_parameters').replace('.pickle','.tex')
        params.to_latex(params_path)  
        
        params_path = model_path.replace('best_model','best_model_parameters').replace('.pickle','.csv')
        params.to_csv(params_path)  
        
   # =====================================================================   #
   # Feature selection for test
   # =====================================================================   # 
    # Load selected features
    with open(json_feats_file_1) as json_file:
        data = json.load(json_file)
    sel_feature_names = data["features"]
    n_sel_features = len(sel_feature_names)
        
    print('Selected features are: {0}'.format(sel_feature_names))
    print('Number of selected features is:{0}'.format(n_sel_features))
    
    
    # Do FS (only selection will be done and output created)
    test_fs, sel_feature_names, _ = find_optimal_subset(X_test_fe, y_test, reg_model = True, target_name = target_name, sel_features_names =  sel_feature_names,
                                                                 out_dir = out_dir, outfile_suff = 'test_' + suff + '_'+target_name, recreate = recreate_fs)

    print('Number of selected features is:{0}'.format(n_sel_features))
    
    # Plot histogram of each selected feature and correlation with the target
    for var in fe_cols:
        x = test_df[var]
        if make_plots:
            get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fs, suff = 'test_'+suff, norm = False)
   
    X_test_fs = test_fs[sel_feature_names]
    y_test = test_fs[target_name]
    
    # Obtain prediction with SFS model 
    try:
        s_test = model.score(X_test_fs, y_test)
        print('Score (trainvalid): ',s_trainvalid)
        print('Score (test): ',s_test)
    except:
        pass

    
   # =====================================================================   #
   # Check Performance
   # =====================================================================   # 
    y_trainvalid_pred = model.predict(X_trainvalid_fs)
    y_test_pred = model.predict(X_test_fs)
   
    # Plot train regression
    model_name='random_forest'
    plot_regression_true_vs_pred(y_trainvalid, y_trainvalid_pred, title='Train: {0}'.format(model_name),
                                 out_dir = out_dir_plots_fs, var_label = target_name, filename = '{0}_train'.format(model_name))
    
    for var in sel_feature_names:
        scatter_plots(trainvalid_fs, var = var, targets = [target_name], out_dir = out_dir_plots_fs,  plot_suff='_'+var+'_train')
  
    # Plot test regression
    plot_regression_true_vs_pred(y_test, y_test_pred, title= 'Test: {0}'.format(model_name),
                                     out_dir = out_dir_plots_fs, var_label = target_name, filename = '{0}_test'.format(model_name))
     
    for var in sel_feature_names:
        scatter_plots(test_fs, var = var, targets = [target_name], out_dir = out_dir_plots_fs,  plot_suff='_'+var+'_train')
        
    s = pd.DataFrame(X_trainvalid_fe.columns)
    s.index +=1
    
    f = '{0}/trainvalid_fe.tex'.format(out_dir)
    sourceFile = open(f, 'w')
    print(s.to_latex(), file = sourceFile)
    sourceFile.close()
    #print(s.to_latex())
    
    