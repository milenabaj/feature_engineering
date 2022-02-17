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

parser.add_argument('--route', '--list', nargs='+', help='Process all trips on those routes, which are found in json file.')
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
routes = args.route
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


# TEMP
routes = ['M3_VH','M3_HH']
p79 = True
aran = True
load_add_sensors = True
recreate_fs = True
#dev_mode = True
#=================================#  
# Check mode
if mode and mode not in ['trainvalid','trainvalidkfold','trainvalid_test','trainvalidkfold_test']:
    print('Wrong mode passed - set mode to one of trainvalid, trainvalidkfold, trainvalid_test, trainvalidkfold_test')
    sys.exit(0)
  
# None passed
if not predict_mode and mode not in ['trainvalid','trainvalidkfold','trainvalid_test','trainvalidkfold_test']:
    print('Pass either --predict_mode or set mode to one of trainvalid, trainvalidkfold, trainvalid_test, trainvalidkfold_test')
    sys.exit(0)
 
#  Both true
if predict_mode and mode:
    print('Do not pass --predict_mode and --mode at the same time. Pass either --predict_mode or set mode to one of trainvalid, trainvalidkfold, trainvalid_test, trainvalidkfold_test.')
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

      
# Predict mode
if predict_mode:
    input_feats = [var.split('GM.')[1] for var in input_feats] 
       
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


# Load json file with sel features
sel_features = None
if predict_mode:
    with open(json_feats_file, "r") as f:
        sel_features = json.load(f)['features']
  
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
    
    if predict_mode:
        in_dir = in_dir.replace('aligned','predict_mode')
            
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
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
out_dir_plots_fe = '{0}/plots_fe'.format(out_dir)
if not os.path.exists(out_dir_plots_fe):
    os.makedirs(out_dir_plots_fe)

out_dir_plots_fs = '{0}/plots_fs'.format(out_dir)
if not os.path.exists(out_dir_plots_fs):
    os.makedirs(out_dir_plots_fs)


json_feats_file = 'json/selected_features_{0}_route-{0}_GM_trip-{1}_sensors-{2}.json'.format(drd_veh, routes_string, suff)
 
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
    get_normalized_hist_fe(x, var_name = var, out_dir = out_dir_plots_fs, suff = '_all')
    
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

                                    
# Resample -  FE - FS on trainvalid
if trainvalid_df is not None:
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
        get_normalized_hist(x, var_name = var, out_dir = out_dir_plots, suff = '_trainvalid')
    
    # Write some info
    
    # Resample
    trainvalid_df, feats_resampled = resample_df(trainvalid_df, feats_to_resample = input_feats, to_lengths_dict = to_lengths_dict, window_size = window_size)
    
    # Do feature extraction 
    keep_cols = trainvalid_df.columns.to_list()
    trainvalid_df, fe_filename = feature_extraction(trainvalid_df, keep_cols = keep_cols, feats = input_feats, out_dir = out_dir, 
                                               file_suff = routes_string + suff +'_trainvalid', 
                                               write_out_file = True, recreate = recreate_fe, sel_features = sel_features, 
                                               predict_mode = predict_mode)
 

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
        get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fe, suff = 'trainvalid_'+suff, norm = False)
       
    # Write info about trainvalid
    # todo
    
    # Remove some columns if needed
    to_rem = []
    avail_cols = list(trainvalid_df.columns)
    for col in avail_cols:
        if any(x in col for x in ['ECDF Percentile Count', 'ECDF_']):
             to_rem.append(col)  

    trainvalid_df.drop(to_rem,axis=1,inplace=True)
    trainvalid_df.reset_index(drop=True, inplace = True)
    
    # Select X and target 
    X_trainvalid_fe = X_trainvalid_df[fe_cols] 
    y_trainvalid =  train_df[target_name]
    
    # Get valid indices
    if 'kfold' in mode:
        print('SFS will be done with kfold validation')
        X_valid_indices = None # the trainvalid fs will be done in kfold manner
    else:
        print('SFS will be done with train-valid split validation')
        valid_nrows = int(0.2*trainvalid_df.shape[0])
        X_valid_indices = trainvalid_df.iloc[-valid_nrows:].index.tolist()
        
    # Do FS
    if dev_mode:
        fmax = 1
    else:
        fmax = X_trainvalid_fe.shape[0]-1
    X_trainvalid_fs, sel_feature_names = find_optimal_subset(X_trainvalid_fe, y_trainvalid, valid_indices = X_valid_indices, reg_model = True, target_name = target_name,
                                                                 out_dir = out_dir, fmax = fmax, outfile_suff = 'trainvalid_' + suff, recreate = recreate_fs)
    
    # Write json file
    f =  open(json_feats_file, "w") 
    sel_feat_data = {"features":sel_feature_names}
    json.dump(sel_feat_data, f)    
    f.close()
    
    n_sel_features = len(sel_feature_names) 
    print('Selected features are: {0}'.format(sel_feature_names))
    print('Number of selected features is:{0}'.format(n_sel_features))
    
    # Plot histogram of each selected feature and correlation with the target
    for var in fe_cols:
        x = trainvalid_df[var]
        get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_ff, suff = 'trainvalid_'+suff, norm = False)
       
    sys.exit(0)
    
# Plot test
for var in vars_to_plot:
    x = test_df[var]
    get_normalized_hist(x, var_name = var, out_dir = out_dir_plots_fs, suff = '_test')
       
# Resample - FS on test        
if test_df is not None:
    to_lengths_dict = {}
    for feat in input_feats:
        a =  test_df[feat].apply(lambda seq: seq.shape[0])
        l = int(a.quantile(0.90))
        to_lengths_dict[feat] = l
        #print(to_lengths_dict)
        #to_lengths_dict = {'GM.acc.xyz.z': 369, 'GM.obd.spd_veh.value':309} # this was used for motorway
    test_df, _ = resample_df(test_df, feats_to_resample = input_feats, to_lengths_dict = to_lengths_dict, window_size = window_size)
    
    # Do FE for only selected feature
       

sys.exit(0)       
            
# =====================================================================   #
# Prepare and split
# =====================================================================   # 
print('Starting splitting')
trips_string = [ str(trip) for trip in trips]
trips_string = '_'.join(trips_string)

# Get filenames
filenames = []
for trip in trips: 
    filenames = filenames + glob.glob('{0}/*{1}*.pickle'.format(chunks_dir, trip))
 

# Predict stage
if predict_mode and sel_features:
    for filename in filenames:
        print('Loading: ',filename)
        key = filename.split('/')[-1]
        df = pd.read_pickle(filename)
        
        # Extract and clean
        extract_inner_df(df, feats = input_feats, do_clean_nans=True)    
        
        sel_features = [feat.split('GM.')[1] for feat in sel_features]
        df = df[sel_features]
        
        filename = '{0}/route-{1}_trips-{2}.pickle'.format(out_dir, route, trips_string)
        df.to_pickle(filename)
        print('Saved to: ',filename)
        
 
elif not predict_mode:
      
    # Split
    train_dfs = []
    valid_dfs = []
    test_dfs = []
    
    for filename in filenames:
        print('Loading: ',filename)
        key = filename.split('/')[-1]
        df = pd.read_pickle(filename)
        
        # Extract and clean
        extract_inner_df(df, feats = input_feats, do_clean_nans=True)    
            
        # Train
        train_n = int(0.6*df.shape[0])
        valid_n = int(0.2*df.shape[0])
        train_df = df[0:train_n]
        train_df.reset_index(inplace=True, drop=True)
        train_dfs.append(train_df)  
        
        # Valid
        valid_df = df[train_n:train_n+valid_n]
        valid_df.reset_index(inplace=True, drop=True)
        valid_dfs.append(valid_df) 
        
        # Test
        test_df = df[train_n+valid_n:]
        test_df.reset_index(inplace=True, drop=True)
        test_dfs.append(test_df)
    
    
    train_merged = pd.concat(train_dfs,ignore_index=True)
    train_merged.reset_index(inplace=True, drop=True)
    train_filename = '{0}/train_route-{1}_trips-{2}.pickle'.format(out_dir, route, trips_string)
    train_merged.to_pickle(train_filename)
    print('Saved to: ',train_filename)
     
    
    valid_merged = pd.concat(valid_dfs,ignore_index=True)
    valid_merged.reset_index(inplace=True, drop=True)
    valid_filename = train_filename.replace('train','valid')
    valid_merged.to_pickle(valid_filename)
    print('Saved to: ',valid_filename)
     
    
    test_merged = pd.concat(test_dfs,ignore_index=True)   
    test_merged.reset_index(inplace=True, drop=True)
    test_filename = train_filename.replace('train','test')
    test_merged.to_pickle(test_filename)
    print('Saved to: ',test_filename)
    
        