
"""
@author: Milena Bajic (DTU Compute)
"""
import sys,os, glob, pickle, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsfel
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import classification_report, plot_confusion_matrix
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import scipy.stats
from scipy import stats, interpolate
from numpy.random import choice
from sklearn.model_selection import  TimeSeriesSplit
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sklearn.model_selection import  TimeSeriesSplit, GridSearchCV
import seaborn as sns
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import *
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import linear_model
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.signal import find_peaks, argrelmin, argrelextrema, find_peaks_cwt
from mlxtend.evaluate import PredefinedHoldoutSplit

def sort2(f):
    try:
        num = int(f.split('/')[-1].split('_')[4] )
    except:
        if '_fulltrip.png' in f:
            num = -10
        elif '_clusters.png' in f:
            num = -9
        elif '_removed_outliers.png' in f:
            num = -8
        elif '_fulltrip_minima.png' in f:
            num = -7
        else:
            num = -1
    return num

def sort3(f):
    if 'mapmatched_map_printout.png' in f:
        num = -10
    elif'interpolated_300th_map_printout.png' in f:
        num = -9
    else:
        num = -1
    return num

def polynomial_model(degree=10):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)  #normalize=True normalize data
    pipeline = Pipeline([("polynomial_features", polynomial_features),#Add polynomial features
                         ("linear_regression", linear_regression)])
    return pipeline
    
    
def plot_fs(nf, res, var_label = 'MSE',title='', size=2,
                               out_dir = '.', save_plot=True, filename='plot-fs'):        
      if size==2:
          plt.rcParams.update({'font.size': 6})
          figsize=[2.5,2]
          dpi= 1000
          ms = 3  
          ls = 7
      if size==3:
          plt.rcParams.update({'font.size': 7})
          figsize=[4,3]
          dpi= 1000
          ms = 6
          ls = 9

      #var_min = true.min() - 0.3*true.min()
      #var_max = true.max() + 0.35*true.max()

      #var_min = 0.3
      #var_max = 3.5
      
      plt.figure(figsize=figsize, dpi=dpi)
      #plt.plot(true, m*true + b, c='blue', label='Best fit') 
      plt.scatter(nf, res, marker='o',s=ms, facecolors='b', edgecolors='b', label='MSE')
      plt.plot(nf, res, linewidth=1)
      plt.ylabel('{0}'.format(var_label), fontsize=ls)
      plt.xlabel('Number of features', fontsize=ls)
      #plt.xlim([var_min, var_max])
      #plt.ylim([var_min, var_max])
      #plt.title(title)
      ax = plt.gca()
      ax.yaxis.set_major_formatter('{x:.2e}')
      ax.xaxis.set_major_formatter('{x:.0f}')
      # For the minor ticks, use no labels; default NullFormatter.
      ax.xaxis.set_minor_locator(AutoMinorLocator())
      #ax.yaxis.set_minor_locator(AutoMinorLocator())
      plt.tight_layout()
      
      if save_plot:
          out_file_path = filename
          plt.savefig(out_file_path, dpi=dpi, bbox_inches = "tight")
          plt.savefig(out_file_path.replace('.png','.eps'),format='eps',dpi=dpi, bbox_inches = "tight")
          plt.savefig(out_file_path.replace('.png','.pdf'),dpi=dpi, bbox_inches = "tight")
          print('file saved as: ',out_file_path)
          
      return
def format_col(x):
    if x=='R2':
        return r'$\textbf{R^2}$'
    else:
        return r'\textbf{' + x + '}'
    
def get_flattened(row, vars):
    row_data = []
    for var in vars:
        row_data.append(pd.Series(row[var]))
    df = pd.concat(row_data, axis = 1) 
    return df
    
    
def set_class(y_cont, bins = [0,2,5,50]):
    labels = list(range(len(bins)-1))
    y_cat = pd.cut(y_cont, bins, labels=labels).astype(np.int8)
    print(y_cat.value_counts())
    return y_cat


def get_nan_cols(df, nan_percent=0.01, exclude_cols = ['IRI_mean_end']):
    # return cols to remove
    threshold = len(df.index) * nan_percent  
    res = [c for c in df.drop(exclude_cols,axis=1).columns if sum(df[c].isnull()) >= threshold]        
    return res

def clean_nans(df, col_nan_percent = 0.01, exclude_cols = ['IRI_mean_end']):
    cols_to_remove = get_nan_cols(df, nan_percent=col_nan_percent, exclude_cols = exclude_cols) 
    
    # Replace infinities with nans 
    for col in df.columns:
        try:
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        except:
            pass
        
    # Drop
    df.drop(columns=cols_to_remove,axis=1, inplace=True) # remove features with more than 1% nans
    df.dropna(axis=0, inplace=True)   # drop rows with at least 1 nan
    
    # Reset index
    df.reset_index(inplace=True, drop = True)
    return 
    
def compute_features_per_series(seq, cfg):
    try:
        t = tsfel.time_series_features_extractor(cfg, seq, window_size=len(seq),overlap=0, fs=None)
    except:
        t = None
    return t


def extract_col(tsfel_obj, col):
    if col in list(tsfel_obj.columns):
        t = tsfel_obj[col] 
    else:
        t = pd.Series() # not None!!
    return t

       
def feature_extraction(df, out_dir, keep_cols = [], feats = ['GM.obd.spd_veh.value','GM.acc.xyz.x', 'GM.acc.xyz.y', 'GM.acc.xyz.z'], 
                       file_suff='', recreate=False, write_out_file = True, feats_list_to_extract = None,  predict_mode = False, sel_features = None):

    # Output filename
    out_filename = '{0}/{1}'.format(out_dir, file_suff)

    # Load if it exists
    if not recreate and os.path.exists(out_filename):
        with open(out_filename, 'rb') as handle:
            df = pickle.load(handle)
            print('File succesfully loaded.')
                  
    # Compute features 
    else:
  
        # Remove those
        stat_to_rem = ['Histogram', 'ECDF', 'ECDF Percentile Count','ECDF Percentile']
        
        # Set cfg
        cfg = tsfel.get_features_by_domain() 
        
        # Delete spectral
        del cfg['spectral']
        
        # Set stat
        for key in stat_to_rem:
            del cfg['statistical'][key]
                
        # Features to save in the output dataframe
        for var in feats:
            if var not in keep_cols:
                keep_cols.append(var)
                
         # Compute features   
        for var in feats:
            print('===== Computing features for: {0} ======'.format(var))
            
            # Additional features: maxmin
            df[var+'-0_Maxmin diff'] = df[var].apply(lambda seq: seq.max()-seq.min())
            keep_cols.append(var+'-0_Maxmin diff')
             
            # ECDF percentils
            for i, p in enumerate([0.05, 0.10, 0.20, 0.80]):
                perc_col_name = var+'-0_ECDF Percentile '+str(p)
                df[perc_col_name] = df[var].apply(lambda seq:  tsfel.ecdf_percentile(seq, percentile=[p]) if seq.shape[0]>20 else None)
                keep_cols.append(perc_col_name)
                
            # Compute default tsfel features
            colname_tsfel  = var+'_fe'
            df[colname_tsfel] = df[var].apply(lambda seq: compute_features_per_series(seq, cfg) )
            keep_cols.append(colname_tsfel)
          
        # Drop rows where tsfel is not computed  
        df = df[keep_cols] #here
        #df.dropna(axis=0, inplace=True)
        df.reset_index(inplace=True, drop = True)

        # Save  
        if write_out_file:
            df.to_pickle(out_filename) 
            print('Wrote to ',out_filename)
          
    return df, out_filename



def extract_inner_df(df, feats = ['GM.obd.spd_veh.value','GM.acc.xyz.z'], remove_vars = ['GM.acc.xyz.x', 'GM.acc.xyz.y'], do_clean_nans = False):
    
    # Extract to nice structure
    for var in feats: 
    
       var_tsfel = var + '_fe'
       # Check if present
       if   var_tsfel  not in df.columns:
           print('Feature {0} not present in df'.format(var))
           continue
        
       # Get feature names for this var
       #try:
       col_names = df[var_tsfel].iloc[0].columns
       print('Will extract: {0}\n'.format(col_names.to_list()))
       time.sleep(2)
       #except:
       #    print('Extraction failed')
       #    time.sleep(3)
       #    return None, None

       # Extract
       for col in col_names:
           print('Extracting: ',col)
           df[var+'-'+col] = df[var_tsfel].apply(lambda tsfel: extract_col(tsfel, col))
           df[var+'-'+col].astype(np.float16)
     
    # Remove tsfel and additional variables
    cols_to_rem = [col for col in df.columns if col.endswith('_fe')] # all tsfel
    for var in remove_vars:
        add_cols_rem = [col for col in df.columns if var in col and var not in cols_to_rem]
        cols_to_rem.extend(add_cols_rem)
    df.drop(cols_to_rem,axis=1,inplace=True)

    # Clean the dataframe
    if do_clean_nans:
        exclude = [col for col in df.columns if not col.startswith('GM')]
        clean_nans(df, exclude_cols = exclude)
    
    # Rename 
    #for col in df.columns:
    #    if '_resampled' in col:
    #        new_col = col.replace('_resampled','')
    #        df.rename(columns={col:new_col}, inplace=True)
    
    return
    
def find_optimal_subset(X, y, valid_indices = None, n_trees=50, fmax = None, reg_model = True, bins = None, target_name = 'target', sel_features_names = None,
                        out_dir = '.', outfile_suff = 'feature_selection', recreate = False,  save_output = True):
        
  
    # Iinput filenames
    if reg_model:
        x_filename = '{0}/{1}_regression.pickle'.format(out_dir, outfile_suff)
    else:
        x_filename = '{0}/{1}_bins-{2}_classification.pickle'.format(out_dir, GM_trip_id, '-'.join([str(b) for b in bins]))
        
    # Load files if they exists
    if not recreate and os.path.exists(x_filename):
        with open(x_filename, 'rb') as handle:
            Xy_filt = pickle.load(handle)
        
        X_filt = Xy_filt.drop([target_name],axis=1) 
        y = Xy_filt[target_name]
        
        sel_features_names = list(X_filt.columns)
        
        print('Files loaded.')
    
    # Create files if they do not exist
    elif not sel_features_names:
        
        print('Starting SFS')
        
        # Remove features with zero variance
        features = list(X.columns)
        for col in features:
            if X[col].var()==0:
                print('=== Removing: {0} (0 variance)'.format(col)) # Zero crossing rate
                X.drop(col,axis=1,inplace=True)
                #test.drop(col,axis=1,inplace=True)
    
        # Feature search
        tscv = TimeSeriesSplit(n_splits=5)
        if not fmax:
            fmax = X.shape[1]-1
            
        if reg_model:
            f=(1,fmax) 
            if valid_indices is not None:
                valid_subset = PredefinedHoldoutSplit(valid_indices)
                feature_selector = SequentialFeatureSelector(RandomForestRegressor(n_trees, bootstrap = True, min_impurity_decrease=1e-2), 
                                                                               n_jobs=-1,
                                                                               k_features=f,
                                                                               forward=True,
                                                                               verbose=2,
                                                                               scoring='neg_mean_squared_error',
                                                                               cv = tscv)
                                                                               #cv=valid_subset)
            else:
                feature_selector = SequentialFeatureSelector(RandomForestRegressor(n_trees, bootstrap = True, min_impurity_decrease=1e-2), 
                                                                               n_jobs=-1,
                                                                               k_features=f,
                                                                               forward=True,
                                                                               verbose=2,
                                                                               scoring='neg_mean_squared_error',
                                                                               cv=tscv)
             
        else:
            f=(1,fmax)
            if valid_indices is not None:
                valid_subset = PredefinedHoldoutSplit(valid_indices)
                feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_trees,class_weight = 'balanced_subsample', max_depth=5, min_impurity_decrease=1e-6),
                   n_jobs=-1,
                   k_features=f,
                   forward=True,
                   verbose=2,
                   scoring=make_scorer(f1_score, average='macro'),
                   cv=valid_subset)
            else:
                feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_trees,class_weight = 'balanced_subsample', max_depth=5, min_impurity_decrease=1e-6),
                   n_jobs=-1,
                   k_features=f,
                   forward=True,
                   verbose=2,
                   scoring=make_scorer(f1_score, average='macro'),
                   cv=tscv)
        
            
        features = feature_selector.fit(X,y)
        sel_features_names = list(feature_selector.k_feature_names_)
        print('Selected features ', sel_features_names)
        
        # Plot
        fig1 = plot_sfs(feature_selector.get_metric_dict(), kind='std_dev', figsize=(25,20))
        plot_name = x_filename.replace('.pickle','.png')
        if save_output:
            plt.savefig(plot_name)
        #print(feature_selector_backward.asubsets_)
       
        # Get metrics per feature        
        feats = pd.DataFrame.from_dict(feature_selector.get_metric_dict()).T
        feats['Added Feature'] = None
        for i in range(1,feats.shape[0]+1):
            if i==1:
                prev  = set()
            else:
                prev  = set(feats.at[i-1,'feature_names'])
            curr = set(feats.at[i,'feature_names'])
            diff = curr.difference(prev)
            diff =  list(diff)[0]
            feats.at[i, 'Added Feature'] = diff
            print(i,diff)
        feats['MSE (subset)'] = feats['avg_score'].apply(lambda row:abs(row))                  
        feats['Added Feature'] = feats['Added Feature'].apply(lambda row: get_var_name(row))
        feats['Added Feature'] = feats['Added Feature'].apply(lambda row: row.replace('GM.obd.spd_veh.value-0_','Vehicle speed '))
       
        if save_output:
            feats.to_pickle(x_filename.replace('.pickle','_feats_info.pickle'))
            print('Saved: ',x_filename.replace('.pickle','_feats_info.pickle'))
        
        # Save latex
        #feats = feats[['Added Feature','MSE (subset)']]
        feats = feats[['Added Feature', 'MSE (subset)']]
        feats.to_latex('reg_fs_table.tex', columns = feats.columns, index = True, 
                            float_format = lambda
                            x: '%.2e' % x, label = 'table:reg_fs',  
                            header=[ format_col(col) for col in feats.columns] ,escape=False)

        latex_file = x_filename.replace('.pickle','_table.tex')
        print('Wrote latex to: ',latex_file)
          
        # Plot
        plot_filename =  x_filename.replace('.pickle','_sfs.pdf')
        plot_fs(feats.index, res = feats['MSE (subset)'], var_label='MSE',filename=plot_filename)
        print('Saved: ',plot_filename)
        
        # Select best features
        X_filt = X[sel_features_names]
            
        # Merge with the target
        X_filt[target_name] = y
        
        # Dump them
        if save_output:
            X_filt.to_pickle(x_filename)
            print('Wrote to ',x_filename)
          
  
    # If test, only select features and save files 
    else:
        print('Selecting given features.')
        
        X_filt = X[sel_features_names]
        
        # Merge with the target
        X_filt[target_name] = y
        
        # Dump them
        if save_output:
            X_filt.to_pickle(x_filename)
            print('Wrote to ',x_filename)
            
    
    return X_filt, sel_features_names


def compute_di_aran(data):
    print('Computing DI')
    
    # DI
    data['DI'] = (data["AlligCracksSmall"]*3+data["AlligCracksMed"]*4+data["AlligCracksLarge"]*5)**0.3 + (data["CracksLongitudinalSmall"]**2+data["CracksLongitudinalMed"]**3+data["CracksLongitudinalLarge"]**4+data["CracksLongitudinalSealed"]**2+data["CracksTransverseSmall"]*3+data["CracksTransverseMed"]*4+data["CracksTransverseLarge"]*5+data["CracksTransverseSealed"]*2)**0.1 + (data["PotholeAreaAffectedLow"]*5+data["PotholeAreaAffectedMed"]*7+data["PotholeAreaAffectedHigh"]*10+data["PotholeAreaAffectedDelam"]*5)**0.1

    # DI reduced
    data['DI_red'] = (data["AlligCracksMed"]*4+data["AlligCracksLarge"]*5)**0.3 + (data["CracksTransverseMed"]*4+data["CracksTransverseLarge"]*5)**0.1 + (data["PotholeAreaAffectedMed"]*7+data["PotholeAreaAffectedHigh"]*10)**0.1
   
    return 

def compute_kpi_aran(data):
    #kpi = (data['DI'] + ((data['p79.RutDepthLeft']+data['p79.RutDepthRight'])/2)**0.5)*((data['p79.IRI5']+data['p79.IRI21'])/2)**0.2
    data['KPI'] = data['DI'] + data['IRI_mean']
   
    return
 
#custom function for ecdf﻿
def empirical_cdf(data):
  percentiles = []
  n = len(data)
  sort_data = np.sort(data)
  
  for i in np.arange(1,n+1):
    p = i/n
    percentiles.append(p)
  return sort_data,percentiles


def ent(data):
    """Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts()           # counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy

def resample(seq, to_length, window_size):
    '''
    Resample a sequence/

    Parameters
    ----------
    seq : np.array
        Sequence to be resampled.
    to_length : int
        Resample to this number of points.

    Returns
    -------
    d_resampled : np.array
        resampled distance (0,10)
    y_resampled : np.array
        resampled input sequence.
    '''    
    # Downsample if needed
    seq_len = seq.shape[0] 
    if seq_len>to_length:
        seq = choice(seq, to_length)
        seq_len = seq.shape[0] #
          
    # Current
    d = np.linspace(0, window_size, seq_len)
    f = interpolate.interp1d(d, seq)
    
    # Generate new points 
    d_new = np.random.uniform(low=0, high=d[-1], size=(to_length - seq_len))
    
    # Append new to the initial
    d_resampled = sorted(np.concatenate((d, d_new)))
    
    # Estimate y at points
    y_resampled = f(d_resampled) 
    
    return d_resampled, y_resampled


def resample_df(df, feats_to_resample, to_lengths_dict = {}, window_size = None):
    input_feats_resampled = []
    
    # Filter rows with less than 2 points (can't resample those)
    for feat in feats_to_resample:
        df[feat+'_len'] =  df[feat].apply(lambda seq: 1 if isinstance(seq, float) else seq.shape[0])  
        df.mask(df[feat+'_len']<2, inplace = True)
        
    # Drop nans (rows with NaN/len<2) and reset index
    df.dropna(subset = feats_to_resample, inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    # Resample to the maximum
    for feat in feats_to_resample:
        print('Resampling feature: ',feat)
        #max_len = max(df[feat].apply(lambda seq: seq.shape[0]))
        to_length = to_lengths_dict[feat]
        new_feats_resampled = ['{0}_d_resampled'.format(feat), '{0}_resampled'.format(feat)]
        df[new_feats_resampled ] = df.apply(lambda seq: resample(seq[feat], to_length = to_length, window_size = window_size), 
                                        axis=1, result_type="expand")
        input_feats_resampled.append('{0}_resampled'.format(feat))
    
    return df,  input_feats_resampled 



def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true))

def get_var_name(row, short = False):
    new_row = row.replace(' diff', ' difference')
    if short:
        if 'GM.acc.xyz.z-0' in new_row: 
            return '{0} (Acc-z)'.format(new_row.split('-0_')[1])
        elif 'spd' in new_row:
            return '{0} (Speed)'.format(new_row.split('-0_')[1])   
        else:
            return new_row
    
    else:
        if 'GM.acc.xyz.z-0' in new_row: 
            return '{0} (Acceleration-z)'.format(new_row.split('-0_')[1])
        elif 'spd' in new_row:
            return '{0} (Vehicle Speed)'.format(new_row.split('-0_')[1])
        else:
            return new_row
    
def get_regression_model(model, f_maxsel, random_state = None, use_default = False, is_pca = False, fs = 10):
    model_title = model.replace('_',' ').title()
    nt = 200
    # Define models
    
    if model=='dummy':
        rf =  DummyRegressor(strategy="mean")
        parameters = {}
    if model=='linear':
        model_title='Multiple Linear'
        rf = linear_model.LinearRegression()
        parameters = {}
    elif model=='lasso':
        rf = linear_model.Lasso(random_state=random_state)
        #lasso_alpha = np.logspace(-5,5,11)
        lasso_alpha = np.array([0.001,0.001,0.1,0.5]) #M3
        #lasso_alpha = np.array([0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.0005, 0.001,0.002, 0.005, 0.01,1]) #CPH1
        #lasso_alpha = np.linspace(0,1,21) #M3
        parameters={'alpha':lasso_alpha}
    elif model=='kNN':
        rf = KNeighborsRegressor()
        k = np.arange(5,41,step=5) 
        parameters={'n_neighbors':k}
    elif model=='ridge':
        rf = linear_model.Ridge(random_state=random_state)
        ridge_alpha = np.linspace(0,1500,510)
        parameters={'alpha': ridge_alpha}
    elif model=='elastic_net':
        rf = linear_model.ElasticNet(random_state=random_state)
        alpha =  np.array([0.001, 0.01,10, 20, 50, 100, 500, 700, 1000])
        parameters={'alpha':alpha, 'l1_ratio':np.linspace(0,1,21)}    
    elif model=='random_forest':
        rf = RandomForestRegressor(random_state=random_state)
        depths = np.arange(2,8,2)
        n_estimators = [200]
        #np.arange(300,500,100)
        #fs =  np.arange(6,16,2) motorway
        max_f = np.arange(2,fs-1,2)
        min_samples_split = (2,4,6)
        parameters = {'n_estimators': n_estimators, 'min_samples_split': min_samples_split, 'max_depth':depths,'min_impurity_decrease':[0,1e-2,],'max_features':max_f}
    elif model=='SVR_poly':
        rf = SVR(kernel='poly')
        C = [0.01,0.8,1,1.5,2,3,5,10,15,20,50]
        epsilon = [0.01,0.01,0.05,0.1,0.2,0.3,0.4,0.5]
        gamma = [0.001,0.01,0.02, 0.05,0.1]
        parameters = {'C':C,
                      'epsilon': epsilon,
                       'gamma':gamma}
    elif model=='SVR_rbf':
        model_title = 'SVR'
        rf = SVR(kernel='rbf', epsilon = 0.1)
        #C = np.linspace(0,20,5)
        #C = np.append(C,[1])
        #C.sort()
        #gamma = np.logspace(-4,-2,3) 
        C = np.array([0.1,10,100])
        gamma = np.array([0.01, 0.1, 1])
        n = C.shape[0]*gamma.shape[0]
        print(n)
        parameters = {'C':C, 'gamma':gamma}
    elif model=='ANN':
        model_title = 'ANN'
        
        # L2 regularization parameter
        ann_alpha = np.array([0.1,10,50])
        
        # Learning rate init
        #learning_rate_init = np.logspace(-5,0,6)
        learning_rate_init = np.array([0.01, 0.1, 1])
          
        # Architecture
        hs1 = [(2),(4),(8)]
        hs2 = [(16, 8),(12,6),(12,4),(12,2),(10,4)]
        hs3 = [(2, 4, 6), (2,6,8), (2,6,10),(4,8,10), (4,8,12), (4,8,16), (12,8,6), (10,8,6),(10,8,4),(8, 6, 4)]
        hs4 = [(2,4,8,12),(12,8,6,2)]
        hs5 = [(2,4,6,8,12),(12,8,6,4,2)]
        #hs = hs2+hs3+hs4
        hs = [(4,6,8)]
        
        #if use_default and is_pca:
        #    parameters = {'hidden_layer_sizes':[(8,6,4)], 'alpha':[1], 'learning_rate_init':[0.1], 'random_state':rs}
        #elif use_default and not is_pca:
        #    parameters = {'hidden_layer_sizes':[(2,4,6)], 'alpha':[1], 'learning_rate_init':[0.01], 'random_state':rs}
        #else:
        #    parameters = {'hidden_layer_sizes':hs, 'alpha':ann_alpha, 'learning_rate_init':learning_rate_init, 'random_state':rs}
        
        parameters = {'hidden_layer_sizes':hs, 'alpha':ann_alpha, 'learning_rate_init':learning_rate_init}
        
        # Model
        rf = MLPRegressor(learning_rate='adaptive', max_iter=1000)
   
 

    return rf, parameters, model_title
    
        
def grid_search(rf, parameters, X, y, score, n_splits = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    clf = GridSearchCV(rf, parameters, cv=tscv, scoring=score, verbose=1)
    clf.fit(X,y)
    best_parameters = clf.best_params_
    print(best_parameters)   
    rf = clf.best_estimator_
    return clf, rf



def get_classification_predictions(X_trainvalid, y_trainvalid, X_test, y_test, rf, 
                                   test_results = None, model_title='', row = 0, labels = None,
                                   save_plots = True, out_dir = '.', is_pca = False):
     
    labels_train = [l for l in labels if int(l) in y_trainvalid.unique()]    
    labels_test = [l for l in labels if int(l) in y_test.unique()]
    labels_plot = ['Low', 'Medium', 'High']
    
    # Train results     
    y_trainvalid_pred = rf.predict(X_trainvalid)         
    train_report = classification_report(y_trainvalid,  y_trainvalid_pred, labels=labels_train)
    train_cm = confusion_matrix(y_trainvalid,y_trainvalid_pred, labels=labels_train)
    #plot_confusion_matrix(rf, X_trainvalid, y_true = y_trainvalid, labels=labels)
    #plt.title('Train')

    # Test results
    y_test_pred = rf.predict(X_test)
    test_report = classification_report(y_test,  y_test_pred, labels=labels_test)
    test_cm = confusion_matrix(y_test,y_test_pred, labels=labels_test)
    
    plt.rcParams.update({'font.size': 19})
    plot_confusion_matrix(rf, X_test, y_true = y_test,labels=labels_test,display_labels=labels_plot, colorbar = False) 
    #ax=plt.gca()
    #ax.set_xticklabels(labels)
    #ax.set_yticklabels(labels)
    
    #if is_pca:
    #    plt.title(model_title + ' (PCA)')
    #else:
    #    plt.title(model_title 
    
    # Compute average metrics over all classes
    test_report_dict = classification_report(y_test,  y_test_pred, labels=labels_test, output_dict=True)
    test_report_dict = { str(label): test_report_dict[str(label)] for label in labels_test}
    test_report_df = pd.DataFrame(test_report_dict)

    # Compute average metrics over all classes and update results with all models
    test_results.at[row, 'Model'] = model_title
    test_results.at[row, 'Precision'] = test_report_df.loc['precision',:].mean()
    test_results.at[row, 'Recall'] = test_report_df.loc['recall',:].mean()
    test_results.at[row, 'F1-Score'] =  test_report_df.loc['f1-score',:].mean()
    
    # Save
    if save_plots:
          dpi=1000
          out_file_path = '{0}/{1}_test.png'.format(out_dir, model_title.replace(' ','_'))
          if is_pca:
              out_file_path = out_file_path.replace('_test.png','_pca_test.png')
                  
          plt.savefig(out_file_path, dpi=dpi, bbox_inches = "tight")
          plt.savefig(out_file_path.replace('.png','.eps'),format='eps',dpi=dpi, bbox_inches = "tight")
          plt.savefig(out_file_path.replace('.png','.pdf'),dpi=dpi, bbox_inches = "tight")
          print('file saved as: ',out_file_path)
      
    # Print
    print('=== Train ====')
    print(train_report)
    print('=== Test ====')
    print(test_report)
    
    return train_report, test_report
    
def get_regression_predictions(X_trainvalid, y_trainvalid, X_test, y_test, rf, train_results = None, test_results = None, model_title='', row = 0, labels = None):

    # Predict
    y_trainvalid_pred = rf.predict(X_trainvalid)
    y_test_pred = rf.predict(X_test)
    
    # MSE: train
    rmse_train = np.sqrt(mean_squared_error(y_true =  y_trainvalid, y_pred = y_trainvalid_pred))
    mae_train = mean_absolute_error(y_true =  y_trainvalid, y_pred = y_trainvalid_pred)
    mape_train = mean_absolute_percentage_error(y_true =  y_trainvalid, y_pred = y_trainvalid_pred)
    r2_train = r2_score(y_true =  y_trainvalid, y_pred = y_trainvalid_pred)
    print('\nMODEL: \n',model_title)
    print('==== Train error: ==== ')
    print('MRSE: ', rmse_train)
    print('MAE: ', mae_train)
    print('R2: ', r2_train)
    print('MRE: ',mape_train)
    print('====================== \n')

    # MSE: test
    rmse_test = np.sqrt(mean_squared_error(y_true =  y_test, y_pred = y_test_pred))
    mae_test = mean_absolute_error(y_true =  y_test, y_pred = y_test_pred)
    r2_test = r2_score(y_true =  y_test, y_pred = y_test_pred)
    mape_test = mean_absolute_percentage_error(y_true =  y_test, y_pred = y_test_pred)
    print('==== Test error: ==== ')
    print('RMSE: ', rmse_test)
    print('MAE: ', mae_test)
    print('R2: ', r2_test)
    print('MRE: ',mape_test)
    print('====================== \n')   

    # Update results
    train_results.at[row, 'Model'] = model_title
    train_results.at[row, 'R2'] = r2_train
    train_results.at[row, 'MAE'] = mae_train
    train_results.at[row, 'RMSE'] = rmse_train
    train_results.at[row, 'MRE'] = mape_train
    
    # Update results
    test_results.at[row, 'Model'] = model_title
    test_results.at[row, 'R2'] = r2_test
    test_results.at[row, 'MAE'] = mae_test
    test_results.at[row, 'RMSE'] = rmse_test
    test_results.at[row, 'MRE'] = mape_test
    
    
    
    return y_trainvalid_pred,  y_test_pred
    
def get_classification_model(model, f_maxsel, random_state = None, is_pca= False):
    
    model_title = model.replace('_',' ').title()
    nt = 500
    
    # Define models
    if model=='dummy':
        rf =  DummyClassifier(strategy='most_frequent')
        parameters = {}
    if model=='logistic_regresion':
        #rf = linear_model.LogisticRegression(class_weight = 'balanced')
        rf = linear_model.LogisticRegression(random_state=random_state)
        C = np.linspace(0,20,5)
        #C = np.arange(0,10,2)
        parameters = {'C':C}
    if model=='naive_bayes':
        rf = GaussianNB()
        parameters = {}  
    if model=='kNN':
        rf = KNeighborsClassifier()
        k = np.arange(1,41,step=1) 
        #n = np.arange(0,10,2)
        parameters = {'n_neighbors':k}
    if model=='random_forest':
        #rf = RandomForestClassifier(nt, class_weight = 'balanced_subsample')
        rf = RandomForestClassifier(nt, random_state=random_state)
        depths = np.arange(5,10,1)
        n_estimators = np.arange(300,500,100)
        fs =  np.arange(6,16,2)
        parameters = {'n_estimators': n_estimators, 'max_depth':depths, 'max_features':fs}
    elif model=='SVC_rbf':
        model_title = 'SVC'
        rf = SVC(class_weight='balanced', kernel ='rbf', random_state=random_state)
        #C = np.linspace(0,20,5)
        #C = np.append(C,[1])
        #C.sort()
        #gamma = np.logspace(-4,-2,3)
        C = np.array([1,5])
        gamma = np.array([0.001])
        n = C.shape[0]*gamma.shape[0]
        print(n)
        parameters = {'C':C, 'gamma':gamma}
    elif model=='ANN':
        model_title = 'ANN'
        
        ann_alpha = np.array([1,5,10,12])
        
        rs = [0]
        
        hs1 = [(4),(8),(12),(16)]
        hs2 = [(8,4),(12,4),(4,8)]
        hs3 = [(4,6,8),(4,8,16)]
        hs4 = [(2,4,8,12),(12,8,6,2)]
        hs5 = [(2,4,6,8,12),(12,8,6,4,2)]
        #hs = hs1+hs2+hs3+hs4+hs5
        hs = hs1
        #hs = [(4,6,8)]
        #learning_rate_init = np.logspace(-5,0,6)
        learning_rate_init = np.array([0.01,0.1])
        acct = ['identity', 'relu']
        rf = MLPClassifier(max_iter=1000)
        parameters = {'hidden_layer_sizes':hs, 'alpha':ann_alpha, 'learning_rate_init':learning_rate_init, 'random_state':rs, 'activation':acct}
        #parameters = {'hidden_layer_sizes':hs} 
        #print(parameters)
        
    return rf, parameters, model_title