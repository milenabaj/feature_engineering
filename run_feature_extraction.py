ad
# coding=utf-8
"""
@author: Milena Bajic (DTU Compute)
"""
import sys, os, json, time
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from utils.data_transforms import *
from utils.plotting import *
from utils.data_loaders import *
import argparse
from utils.analysis import *
import tsfel
import pickle
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

parser.add_argument('--route', help='Process all trips on this route, given in json file. This needs tspyo be set to find the DRD file.')
parser.add_argument('--trip', type=int, help='Process this trip only.')
parser.add_argument('--is_p79', action='store_true', help = 'If this is p79 data, pass true.')
parser.add_argument('--is_aran', action='store_true', help = 'If this is aran data, pass true.')
parser.add_argument('--json', default= "json/routes.json", help='Json file with route information.')
parser.add_argument('--in_dir_base', default= "data", help='Input directory base.')
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--step', type=int, default=10)
parser.add_argument('--aran_target',  default='DI')
parser.add_argument('--recreate', action="store_true", help = 'Recreate files, even if present. If False and the files are present, the data will be loaded from them.')
parser.add_argument('--only_load_results', action='store_true', help='Only load files with results.')
parser.add_argument('--only_prepare_df', action='store_true', help='Only extract chosen features from tsfel objects')
parser.add_argument('--load_add_sensors', action='store_true', help = 'Load additional sensors.') 
parser.add_argument('--predict_mode', action="store_true", help = 'Run in prediction mode.')
parser.add_argument('--dev_mode',  action="store_true", help = 'Development mode. A small portion of data will be loaded and processes.')    
   
# Parse arguments
args = parser.parse_args()
route = args.route
trip  = args.trip
is_p79 = args.is_p79
is_aran = args.is_aran
json_file = args.json
in_dir_base = args.in_dir_base
window_size = args.window_size
aran_target = args.aran_target
step = args.step
recreate = args.recreate
only_load_results = args.only_load_results
predict_mode = args.predict_mode
dev_mode = args.dev_mode
dev_nrows = 10
json_feats_file = 'json/selected_features.json'
load_add_sensors = args.load_add_sensors


# TEMP
step = 10
is_p79 = True
is_aran = True
recreate = True
#predict_mode = True
#load_add_sensors = True
#only_load_results = True
args.trips = [7792]      
#is_p79 = True
route = 'CPH1_VH'
load_add_sensors = True
only_prepare_df = True
#=================================#  

# Additional sensors to load
input_feats = ['GM.obd.spd_veh.value','GM.acc.xyz.x', 'GM.acc.xyz.y', 'GM.acc.xyz.z']
if load_add_sensors:
    steering_sensors = ['GM.obd.strg_pos.value', 'GM.obd.strg_acc.value','GM.obd.strg_ang.value'] 
    wheel_pressure_sensors =  ['GM.obd.whl_prs_rr.value', 'GM.obd.whl_prs_rl.value','GM.obd.whl_prs_fr.value','GM.obd.whl_prs_fl.value'] 
    other_sensors = ['GM.obd.acc_yaw.value','GM.obd.trac_cons.value']
    add_sensors = steering_sensors + wheel_pressure_sensors + other_sensors 
    input_feats = input_feats + add_sensors
       
if predict_mode:
    input_feats = [var.split('GM.')[1] for var in input_feats] 
    
# Load json file with route info
with open(json_file, "r") as f:
    route_data = json.load(f)
   #print(route_data)


# Use the user passed trip, the ones in json file for the route or the default one
if trip:
    GM_trips = [trip]
elif (not trip and route):             
    # Load route data
    GM_trips =  route_data[route]['GM_trips']
# Default trip
else:
    GM_trips = [7792] 
    
    
# Try to find a route if only a trip is given
if not route:
    # Try to find it from json file and the given trip
    for route_cand in route_data.keys():
        if GM_trips[0] in route_data[route_cand]['GM_trips']:
            route = route_cand
            break
    # If not found in the json file, set to default name if predict mode
    if not route and predict_mode:
        route = 'Copenhagen'
        print('Route not found. Setting to default name: ', route)    
    elif not route:
        print('Please pass the route or add it into the json file. The route is required unless in predict_mode.')
        sys.exit(0)
 
# Load json file with sel features
sel_features = None
if predict_mode:
    with open(json_feats_file, "r") as f:
        sel_features = json.load(f)['features']
  
# DRD type           
if is_p79 and not is_aran:
    if predict_mode:
        in_dir = '{0}/predict_mode_GM_data_window-{1}-step-{2}/{3}'.format(in_dir_base, window_size, step, route)
        DRD_trips =  None
        target_name = None
    else:
        in_dir = '{0}/aligned_GM_p79_data_window-{1}-step-{2}/{3}'.format(in_dir_base, window_size, step, route) 
        DRD_trips =  route_data[route]['P79_trips']
        target_name = 'IRI_mean'
elif is_aran and not is_p79:
    if predict_mode:
        in_dir = '{0}/predict_mode_GM_aran_data_window-{1}-step-{2}/{3}'.format(in_dir_base, window_size, step, route)
        DRD_trips =  None
        target_name = None
    else:
        in_dir = '{0}/aligned_GM_ARAN_data_window-{1}-step-{2}/{3}'.format(in_dir_base, window_size, step, route) 
        DRD_trips =  route_data[route]['ARAN_trips'] 

        target_name = aran_target
elif is_aran and is_p79:
    in_dir = '{0}/aligned_GM_ARAN_p79_data_window-{1}-step-{2}/{3}'.format(in_dir_base, window_size, step, route) 
    DRD_trips =  None
    target_name = 'iri_widths_sum'
        
elif not is_aran and not is_p79:
    print('Set p79 or aran or both to True')
    sys.exit(0)
 
if load_add_sensors:
    in_dir = in_dir.replace(route, route+'_add_sensors')

# Out dir name
out_dir = in_dir.replace('step-{0}'.format(step),'step-{0}-feature-extraction'.format(step))  

# Create output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
    
# Output directory for chunks
chunks_dir = '{0}/chunks'.format(out_dir)
if not os.path.exists(chunks_dir):
    os.makedirs(chunks_dir)

print('p79 data? ', is_p79)
print('Aran data? ', is_aran)
print('Additional sensors? ', load_add_sensors)
print('Trip: ', trip)
print('Dev mode?: ',dev_mode)
print('Input directory (GM): ', in_dir)
print('Output directory: ', out_dir)
time.sleep(3)  

# Only load fe results and exit
if only_load_results:
    # Get filenames
    filenames = []
    for GM_trip in GM_trips: 
        filenames.extend(glob.glob('{0}/aligned_extracted_features_GM*.pickle'.format(chunks_dir)))
       
    # Load
    GM_data = {}
    for filename in filenames:
        print('Loading: ',filename)
        key = filename.split('/')[-1]
        GM_data[key]= pd.read_pickle(filename)
        
    sys.exit(0) 

# =====================================================================   #
# Process aligned files
# =====================================================================   # 

# For each trip
if not only_prepare_df:
    for GM_trip in GM_trips:
        print('Starting extraction for trip: ',GM_trip)
            
        # Find filenames
        filenames = glob.glob('{0}/*{1}*.pickle'.format(in_dir, GM_trip))
        print('Filenames: ',filenames)
        
        # For each pass
        for filename in filenames:
            print('Loading :',filename)
            df = pd.read_pickle(filename)
            if dev_mode:
                df = df.head(dev_nrows)
              
            if predict_mode:
                keep_cols = []
            else: 
                '''                
                use_cols = ['TS_or_Distance_start','street_name_start', 'street_name_end'] + input_feats + [target_name]
                try:
                    df = df[use_cols]
                except:
                    pass #do not consider this trip if something is missing in it
                df[target_name] = df[target_name].astype(np.float16)
                df['TS_or_Distance_start'] = df['TS_or_Distance_start'].astype(np.int)
                keep_cols = [col for col in df.columns if not col.startswith('GM')]
                '''
                keep_cols = df.columns.to_list()

            df.reset_index(inplace=True, drop=True)  
            df_len = df.shape[0]
            print(df.memory_usage())
            #sys.exit(0)
            # Create and process chunks (only chunks can be processed as they fit into RAM)
            #fe_dfs = []
            #n_chunks = ceil(df.shape[0]/chunk_size)
            #n_chunks = 1
            to_lengths_dict = {}
  
            # Make chunk df
            df_chunk = df
            #df_chunk = df.head(5)
            #file_suff='GM_id-{0}_chunk-{1}_start-{2}_end-{3}'.format(GM_trip_id, chunk, split_idx_start, split_idx_end)
            df_chunk.reset_index(drop=True, inplace=True)
 
            # Output file suffix
            file_suff =  filename.split('/')[-1].replace('aligned_','aligned_extracted_features_')
            #file_suff = file_suff.replace('.pickle','_chunk_{0}.pickle'.format(chunk))
                                                         
            # Resample length
            for feat in input_feats:
                a = df_chunk[feat].apply(lambda seq: seq.shape[0])
                l = int(a.quantile(0.90))
                to_lengths_dict[feat] = l
                #print(to_lengths_dict)
                #to_lengths_dict = {'GM.acc.xyz.z': 369, 'GM.obd.spd_veh.value':309} # this was used for motorway
                    
            # Resample chunk df
            df_chunk, feats_resampled = resample_df(df_chunk, feats_to_resample = input_feats, to_lengths_dict = to_lengths_dict, window_size = window_size)
              
            # Do feature extraction on the  chunk df
            df_chunk, fe_filename = feature_extraction(df_chunk, keep_cols = keep_cols, feats = input_feats, out_dir = chunks_dir, 
                                                       file_suff = file_suff, 
                                                       write_out_file = True, recreate = recreate, sel_features = sel_features, 
                                                       predict_mode = predict_mode)
            
# =====================================================================   #
# Prepare and split
# =====================================================================   # 
print('Starting splitting')
GM_trips_string = [ str(trip) for trip in GM_trips]
GM_trips_string = '_'.join(GM_trips_string)

# Get filenames
filenames = []
for GM_trip in GM_trips: 
    filenames = filenames + glob.glob('{0}/*{1}*.pickle'.format(chunks_dir, GM_trip))
 

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
        
        filename = '{0}/route-{1}_trips-{2}.pickle'.format(out_dir, route, GM_trips_string)
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
    train_filename = '{0}/train_route-{1}_trips-{2}.pickle'.format(out_dir, route, GM_trips_string)
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
    
        