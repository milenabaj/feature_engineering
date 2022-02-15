"""
@author: Milena Bajic (DTU Compute)
"""
import sys, os, pickle, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplleaflet
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.analysis import mean_absolute_percentage_error
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator,LogFormatter)
from sklearn.model_selection import  TimeSeriesSplit, GridSearchCV

def plot_geolocation(longitudes, latitudes, name = 'map', out_dir = '.', plot_firstlast=0, plot_html_map = True, title=None, full_filename = None, preload = False):
    """
    
    A function which plots and saves GPS coordinates on Open Street Map in a web-browser.
    
    Parameters
    ----------
    longitudes : array-like object
        Trip longitudes.
    latitudes :  array-like object
        Trip latitudes.
    name : STRING, optional
        The plot prefix. The default is 'map'.
    out_dir : STRING, optional
        Output directory. The default is '.'.
    plot_firstlast : INT, optional
        The number of the initial and ending point to plot. The default is 0.
    plot_html_map : BOOL, optional
        Plot GPS coordinates in web browser. The default is True.
    title : STRING, optional
        Plot title. The default is None.
    full_filename : STRING, optional
        Full filename when saving the plot. The default is None.
    preload : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """        
    # Name
    if full_filename:
        name = full_filename.replace('.pickle','.png')
    else:
        name = '{0}/{1}_map.png'.format(out_dir, name)
    
    
    # Matplotlib figure
    if os.path.exists(name) and preload:
        pickle_name = name.replace('.png','.pickle')
        print('Loading ',pickle_name)
        with open(pickle_name,'rb') as f:
            fig = pickle.load(f)
        #plt.show()
        
    else:
        print('Plotting')
        
        # Figure, axis
        fig,ax = plt.subplots()
        
        # Title
        if title:
            ax.set_title(title)
           
        # Plot first/last
        if plot_firstlast!=0:
            ax.scatter(longitudes[0:plot_firstlast], latitudes[0:plot_firstlast], s = 50, c='red',marker='o',alpha=0.3, label = 'Start')
            ax.scatter(longitudes[0:2], latitudes[0:2], s = 90, c='red',marker='x',alpha=0.5, label = 'Start')
            
            ax.scatter(longitudes[-plot_firstlast:], latitudes[-plot_firstlast:], s = 50, c='black',marker='o',alpha=0.3, label = 'End') 
            ax.scatter(longitudes[-2:], latitudes[-2:], s = 90, c='black',marker='x',alpha=0.5, label = 'End') 
         
        # Plot else
        ax.scatter(longitudes, latitudes, s = 8, c='dodgerblue',marker='o',alpha=0.3)
        ax.legend()
        
        # Save as png
        fig.savefig(name)
        print('Figure saved: {0}'.format(name)) 
              
        # Save also as pickle
        fig_name = name.replace('.png','.pickle')
        with open(fig_name,'wb') as f:
            pickle.dump(fig, f)  
        
    # Html map
    if plot_html_map:
        try:
            from selenium import webdriver
            from selenium_firefox import Firefox
            
            print('Will try to open web browser')
            html_name = name.replace('.png','.html')
            
            # Html can't plot with legend
            if fig.axes[0].get_legend():
                fig.axes[0].get_legend().remove()
                
            # If webbrowser available, plot it
            mplleaflet.display(fig, tiles='cartodb_positron')
            mplleaflet.show(fig, html_name)
            print('File saved: {0}'.format(html_name))
             
            # Save pdf printout of the webpage
            html_link = 'file://{0}/{1}'.format(os.getcwd(),html_name)
            printout_name = name.replace('.png','_printout.png')
            #print(html_link)
            
            browser = Firefox()
            browser.get('https://www.google.com')
            #browser = webdriver.Firefox()
            browser.get(html_link)
            
            #Give the map tiles some time to load
            time.sleep(10)
            try:
                browser.save_screenshot(printout_name)
            except:
                pass
            browser.quit()
        except:
            print('Not able to open a webbrowser. Skipping plotting on OSM.')
            pass
            
    return fig



def get_np_array(df, var_name):
    var_data = df[var_name]
    var_data = var_data.to_numpy()
    var_data = np.concatenate(var_data)
    return var_data 
    
def plot_sequences(use_sliding_segments = False):
    #if not use_sliding_segments:
        # filter
    return

def plot_correlation(df, method = 'pearson'):
    print('Plotting method: ', method)
    dfCorr = df.corr(method = method)
    #filteredDf = dfCorr[((dfCorr >= .6) | (dfCorr <= -.6)) & (dfCorr !=1.000)]
    fig, ax = plt.subplots(figsize=(8.5,8.5))
    #sns.set(font_scale = 16)
    font_size = 6 # 12 too much
    #res = sns.heatmap(dfCorr, annot=True, cmap="Reds", fmt=".2f",annot_kws={"fontsize":font_size})
    res = sns.heatmap(dfCorr, annot=True, cmap="Reds", fmt=".2f",annot_kws={"fontsize":font_size})
    res.set_xticklabels(res.get_xmajorticklabels(), rotation = 90)
    #res.set_xticklabels(res.get_xmajorticklabels(), rotation = 45, fontsize = font_size)
    #res.set_yticklabels(res.get_ymajorticklabels(), rotation = 0, fontsize = font_size)
    plt.title(method)
    plt.tight_layout()
    return fig
    
def plot_type(df, var, lower_limit = -np.inf, upper_limit = np.inf, 
              target_name = 'KPI', is_bad = False, n_plots = np.inf):
    
    bad_df = df[(df[target_name]>=lower_limit) & (df[target_name]<=upper_limit)]
    bad_df.reset_index(inplace=True)

    plt.figure()
    for row in range(0,bad_df.shape[0]):
        bad_data = bad_df[var].iloc[row]
        
        if row==n_plots:
            break
        
        if row==0:
            if is_bad:
                plt.plot(bad_data,label='{0} >= {1:.1f}'.format(target_name, lower_limit), alpha=0.5)
            else:
                plt.plot(bad_data,label='{0} <= {1:.1f}'.format(target_name, upper_limit), alpha=0.5)
        else:
            plt.plot(bad_data, alpha=0.5) 
            plt.plot(bad_data, alpha=0.5) 
       
    plt.legend()
    return

    

def plot_geolocation_2(gps_points_1, gps_points_2, name = None, out_dir='.', plot_firstlast=1):
    (lon1,lat1) = gps_points_1
    (lon2,lat2) = gps_points_2
    
    fig,ax=plt.subplots()
    
    ax.scatter(lon1, lat1, s = 15, c='dodgerblue',marker='o',alpha=0.6)
    if plot_firstlast!=0:
        ax.scatter(lon1[0:plot_firstlast], lat1[0:plot_firstlast], s = 50, c='yellow',marker='o',alpha=1)
    
    ax.scatter(lon2, lat2, s = 8, c='yellow',marker='o',alpha=0.3) 
    if plot_firstlast!=0:
        ax.scatter(lon2[-plot_firstlast:], lat2[-plot_firstlast:], s = 50, c='black',marker='o',alpha=0.4) 
     
    # Name
    if name:
        mplleaflet.show(path='{0}/map_{1}.html'.format(out_dir, name))
    else:
        mplleaflet.show() 
    return
 
    
def plot_DRD_vars(data, string=''):

    
    
    plt.figure()
    plt.title('{0} DRD'.format(string))
    plt.scatter(data.DRD_TS_or_Distance, 10*data.DRD_Acceleration, c='r',s=1, alpha=0.5, label= 'p79 Acceleration * 10')
    plt.scatter(data.DRD_TS_or_Distance, data.DRD_Laser5,s=1,  alpha=0.5,label='Laser5')
    plt.scatter(data.DRD_TS_or_Distance, data.DRD_Laser21,s=1, alpha=0.5, label='Laser21')
    plt.legend()
    plt.show()
    return

def plot_DRD_singlevars(data,string=''):
    fig,ax=plt.subplots()
    plt.title('{0} DRD data'.format(string))
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Velocity, s = 2, c='b',marker='.', label = 'DRD Velocity') 
    plt.legend()
    plt.show()
    
    fig,ax=plt.subplots()
    plt.title('{0} DRD data'.format(string))
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Acceleration, s = 2, c='b',marker='.', label = 'DRD Acceleration') 
    plt.legend()
    plt.show()
    
    scatter_plot(data,'DRD_Raw_Flytning', '', plot_title='DRD Flytning')
    
    fig,ax=plt.subplots()
    plt.title('{0} DRD data'.format(string))
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Raw_Flytning, s = 2, c='g',marker='.', label = 'Flytning') 
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Raw_Laser5, s = 2, c='b',marker='.', label = 'Raw Laser 5') 
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Prof_Laser5, s = 2, c='r',marker='.', label = 'Prof Laser 5') 
    plt.legend()
    plt.show()
    
    fig,ax=plt.subplots()
    plt.title('{0} DRD data'.format(string))
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Prof_Laser5, s = 2, c='b',marker='.', label = 'Prof Laser 5') 
    plt.legend()
    plt.show()
    
    
    fig,ax=plt.subplots()
    plt.title('{0} DRD data'.format(string))
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Raw_Laser21, s = 2, c='b',marker='.', label = 'Raw Laser 21') 
    plt.legend()
    plt.show()
    
    fig,ax=plt.subplots()
    plt.title('{0} DRD data'.format(string))
    ax.scatter(data.DRD_TS_or_Distance, data.DRD_Prof_Laser21, s = 2, c='b',marker='.', label = 'Prof Laser 21') 
    plt.legend()
    plt.show()
    return


def plot_DRD_oneplot_singlevars(data,string=''):
    fig,ax=plt.subplots()
    plt.title('{0} DRD data'.format(string))
    #ax.scatter(data.DRD_TS_or_Distance, data.GM_Acceleration_z*300, s=2, label = 'GM acceleration_z') 
    ax.plot(data.DRD_TS_or_Distance, data.DRD_Raw_Flytning, label = 'Flytning') 
    ax.plot(data.DRD_TS_or_Distance, data.DRD_Raw_Rotation*100, label = 'Rotation*100')
   
    #ax.scatter(data.DRD_TS_or_Distance, data.DRD_Raw_Flytning, s = 2, c='g',marker='.', label = 'Flytning') 
    #ax.scatter(data.DRD_TS_or_Distance, data.DRD_Raw_Rotation*100, s=2, marker='.', label = 'Rotation')
    #ax.scatter(data.DRD_TS_or_Distance, data.DRD_Raw_Laser5, s = 2, c='b',marker='.', label = 'Raw Laser 5') 
    #ax.scatter(data.DRD_TS_or_Distance, data.DRD_Prof_Laser5, s = 2, c='r',marker='.', label = 'Prof Laser 5') 
    #ax.scatter(data.DRD_TS_or_Distance, data.GM_Acceleration_z, s = 2, c='b',marker='.', label = 'GM acceleration_z') 
    #ax.scatter(data.DRD_TS_or_Distance, data.DRD_Raw_Laser5 + data.DRD_Raw_Flytning-10, s = 2, marker='*', label = 'd') 
    plt.legend()
    plt.show()
    return

def plot_GM_acc(data,string=''):
    fig,ax=plt.subplots()
    plt.title('GM')
    ax.plot(data.DRD_TS_or_Distance, data.GM_Acceleration_x, label='GM Acc_x') 
    ax.plot(data.DRD_TS_or_Distance, data.GM_Acceleration_y, label='GM Acc_y') 
    ax.plot(data.DRD_TS_or_Distance, data.GM_Acceleration_z, label='GM Acc_z')      
    #ax.scatter(data.DRD_TS_or_Distance, data.GM_Acceleration_z, s = 2, c='b', marker='.', label='GM Acc_z') 
    plt.legend()
    plt.show()
    return

def plot_DRD_lasers(data, laser_type='Prof'):
    match_string = 'DRD_{0}_Laser'.format(laser_type)
    laser_vars = [var for var in matched_data.columns if match_string in var]
    
    # Filter
    laser_ints = [5,15,21]
    filt_laser_vars=[]
    for var in laser_vars:
        for num in laser_ints:
            if not var.endswith('Laser'+str(num)):
                continue
            filt_laser_vars.append(var)
    laser_vars =  filt_laser_vars  
    
    # Plot
    fig,ax=plt.subplots(figsize=(20,20))
    plt.title('DRD data: {0} Lasers'.format(laser_type))
    for var in laser_vars:
        #ax.scatter(data.DRD_TS_or_Distance, data[var], s = 12, marker='.', label=var.replace('DRD_','') )
        ax.plot(data.DRD_TS_or_Distance, data[var],  label=var.replace('DRD_','') )
    plt.legend()
    plt.show()
    return

def scatter_plots(data, var = 'GM_Acceleration_z_segment', targets = ['DRD_IRI5', 'DRD_IRI21','DRD_IRI_mean'],path='plot.png'):
    features = [c for c in data.columns if var+'_' in c]
    for target in targets:
        for feature in features:
            fig,ax=plt.subplots()
            # plt.title('{0} DRD data: {1}'.format(data_string,plot_title))
            sns.regplot(data[feature], data[target])
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            plt.legend()
            plt.savefig(path)
            #plt.show()
    return

def pair_plot(data, input_vars = ['GM_Acceleration_z_segment','GM_Speed_segment'], targets = ['DRD_IRI5', 'DRD_IRI21','DRD_IRI_mean'],data_string='', plot_title=''):
    vars = []
    for var in input_vars:
        vars = vars + [c for c in data.columns if var+'_' in c] 
    vars_targets = vars + targets
    sns.pairplot(data, kind = 'reg')
    plt.show()
    return

def plot_iri_segments(data):
    from scipy.stats import pearsonr
    var_list = ['GM_Acceleration_z']
    n_segments= data.shape[0]
    seg_lengths = data.Time.apply(lambda row:row.shape[0])

    # Segments data
    means = []
    stds = []
    irimeans=[]
    iri5s = []
    iri21s = []
    
    for s in range(0, n_segments):
        seg = data.iloc[s]
        n_seg_points = seg.shape[0] #points in the segment 
        print(n_seg_points)
        if n_seg_points<2:
            continue  
        
        for var in var_list:
            t0 = seg.Time[:-1]
            t1 = seg.Time[1:]
            dt = t1-t0
            iri5 = round(seg.DRD_IRI5,2)
            iri21 = round(seg.DRD_IRI21,2) 
            irim = round((iri5+iri21)/2,2)
            
            stds.append(seg[var].std()/n_seg_points)
            means.append(seg[var].mean())
            
            irimeans.append(irim)
            iri5s.append(iri5)
            iri21s.append(iri21)
             
            #fig,ax=plt.subplots()
            #plt.title('Segment: {0}, {1} vs distance, IRI5 = {2}, IRI21={3}'.format(s,var,iri5, iri21))
            #ax.plot(seg.DRD_Distance, seg[var], marker='o', label=var)
            #plt.legend()
            #plt.show()
            #print(round(seg[var].mean(),3), iri5, iri21, irim)
        
            #fig,ax=plt.subplots()
            #plt.title('Seg: {0}, IRI5 = {1}, IRI21={2}, IRIm={3}'.format(s,iri5, iri21,irim))
            #ax.plot(seg.GM_Time, seg[var], marker='o', label=var)
            #ax.plot(seg.GM_Time[:-1], dt, marker='o', label='delta_time')
            #plt.legend()
            #plt.show()
            
                
    pm = round(pearsonr(means,irimeans)[0],3)
    print('irim:', pm)
    p5 = round(pearsonr(means,iri5s)[0],3)
    print('iri5:',p5)
    p21 = round(pearsonr(means,iri21s)[0],3)

    print('iri:',p5,p21)
    
    data['mean(acc_z series)'] = means
    data['std(acc_z series)'] = stds   
    data['DRD_IRI_mean']=irimeans
    
    fig,ax=plt.subplots(figsize=(50,30))
    data.boxplot('mean(acc_z series)','DRD IRI (mean)')
    ax = data.boxplot('std(acc_z series)','DRD IRI (mean)')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


    data.boxplot(by = 'mean_iri', column = ['std_acc_z'],grid=False, rot=45, fontsize=7)
    
    fig,ax=plt.subplots()
    ax.scatter(irimeans,stds, s=3)

    # bins
    x = pd.cut(data['mean_iri'],bins=5)
    ax = x.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(10,6))
    
    # boxplot for binned
    cut = pd.cut(data['mean_iri'],bins=5)
    boxdf = data.groupby(cut).apply(lambda df: df.std_acc_z.reset_index(drop=True)).unstack(0)
    sns.boxplot(data=boxdf)



def plot_segments(data, n_segments_to_plot = 4, color = 'blue'):

    var_list = ['GM_Acceleration_z_segment']
    n_segments= data.shape[0]
    
    stats={}
    # Loop over segments
    for s in range(0, n_segments_to_plot):
        seg = data.iloc[s] 

        # Plot features
        for var in var_list:
            stat_desc = pd.Series(seg[var]).describe().round(4)
            #print('===== Sequence: {0}'.format(s))
            #print(stat_desc)
            stats[str(s)] = stat_desc
            l = seg[var].shape[0]
            fig,ax=plt.subplots()
            plt.title('IRI5 = {0:.2f}, IRI21={1:.2f}, IRIm={1:.2f}'.format(seg['DRD_IRI5'], seg['DRD_IRI21'], seg['DRD_IRI_mean']))
            ax.plot(range(l), seg[var], marker='o', label=var, c = color)
            plt.text(0.7, 0.1,stat_desc.to_string(), size = 8,  transform=ax.transAxes, bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
            plt.legend()
            plt.show()
    return stats
      
def make_all_plots(data):       
    plot_segments(data)
    scatter_plots(data)
    pair_plot(data)
    return

def make_plots(df, var_names = ['acc.xyz.z','obd.spd_veh'], target = 'IRI_mean_end', start = 0, n_plots=3):
    acc_z = df[var_names[0]]
    spd = df[var_names[1]]
    target = round(df[target],2)
    
    fig, axs = plt.subplots(nrows = n_plots, ncols = 2, figsize=(170, 70))
    
    for i in range(n_plots):
        acc_z_i =  acc_z.iloc[start+i] 
        target_i = target.iloc[start+i]       
        axs[i,0].plot(acc_z_i, linestyle="None", marker='o',markersize = 20, label = target_i)
        axs[i,0].legend(fontsize= 80)
        axs[i,0].tick_params(axis='both', which='major', labelsize=75)
        axs[i,0].tick_params(axis='both', which='minor', labelsize=8)
        
        spd_i = spd.iloc[start+i]  
        axs[i,1].plot(spd_i, linestyle="None", marker='o',markersize = 20, label = target_i)
        axs[i,1].legend(fontsize= 80)
        axs[i,1].tick_params(axis='both', which='major', labelsize=75)
        axs[i,1].tick_params(axis='both', which='minor', labelsize=8)
        
    plt.tight_layout()
    plt.show()
    
    return

def make_plots_allvars(df, target = 'IRI_mean_end', start = 0, n_plots=10):
    acc_x = df['acc.xyz.x']
    acc_y = df['acc.xyz.y']
    acc_z = df['acc.xyz.z']
    spd = df['obd.spd_veh']
    target = round(df[target],2)
    
    for i in range(n_plots):
        acc_x_i =  acc_x.iloc[start+i] 
        acc_y_i =  acc_y.iloc[start+i] 
        acc_z_i =  acc_z.iloc[start+i] 
        spd_i = spd.iloc[start+i]  
        target_i = target.iloc[start+i]   
        
        fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize=(170, 70))
        
        axs[0,0].plot(acc_x_i, linestyle="None", marker='o',markersize = 20, label = target_i)
        axs[0,0].legend(fontsize= 80)
        axs[0,0].set_ylim([0, 0.6])
        axs[0,0].set_ylabel('acc x', fontsize= 80) 
        axs[0,0].tick_params(axis='both', which='major', labelsize=75)
        axs[0,0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[0,1].plot(acc_y_i, linestyle="None", marker='o',markersize = 20, label = target_i)
        axs[0,1].legend(fontsize= 80)
        axs[0,1].set_ylim([0, 0.6])
        axs[0,1].set_ylabel('acc y', fontsize= 80) 
        axs[0,1].tick_params(axis='both', which='major', labelsize=75)
        axs[0,1].tick_params(axis='both', which='minor', labelsize=8)
      
        axs[1,0].plot(acc_z_i, linestyle="None", marker='o',markersize = 20, label = target_i)
        axs[1,0].legend(fontsize= 80)
        axs[1,0].set_ylim([0.6, 1.4])
        axs[1,0].set_ylabel('acc z', fontsize= 80) 
        axs[1,0].tick_params(axis='both', which='major', labelsize=75)
        axs[1,0].tick_params(axis='both', which='minor', labelsize=8)
        
        axs[1,1].plot(spd_i, linestyle="None", marker='o',markersize = 20, label = target_i)
        axs[1,1].set_ylim([0, 50])
        axs[1,1].set_ylabel('speed', fontsize= 80) 
        axs[1,1].legend(fontsize= 80)
        axs[1,1].tick_params(axis='both', which='major', labelsize=75)
        axs[1,1].tick_params(axis='both', which='minor', labelsize=8)
        
    plt.tight_layout()
    plt.show()
    
    return

def to_numpy(df):
    for var in  ['acc.xyz.x', 'acc.xyz.y', 'acc.xyz.z','obd.spd_veh']:
        x = df[var].apply(lambda row: row.replace('\n','').replace('[','').replace(']',''))
        x = x.apply(lambda row: np.array([float(x) for x in row.split(' ') if (x!='' and x[0].isdigit())]))
        df[var+'.value'] = x
        df.drop([var],axis=1, inplace=True)
        df.rename({var+'.value':var},axis='columns',inplace=True)
    return df


def plot_raw_vs_resampled(df, feats, n_rows_to_plot =10):
    for feat in feats:
        var = df[feat]
        d_res_name = '{}_d_resampled'.format(feat)
        var_res_name = '{}_resampled'.format(feat)
        d_res = df[d_res_name]
        var_res = df[var_res_name]
        
        # Plot
        for i in range(0, n_rows_to_plot):
            d_res_i = d_res.iloc[i]
            var_i = var.iloc[i]
            var_res_i = var_res.iloc[i]
            var_len = var_i.shape[0]
            d_i = np.arange(0,10,10./var_len)
                      
            plt.figure()
            plt.plot(d_i, var_i, marker='o',color = 'blue', alpha=0.5, label='Raw')
            plt.plot(d_res_i, var_res_i, marker='.',color = 'red', alpha=0.5, label='Resampled')
            plt.xlabel('Distance')
            plt.ylabel(feat)
            plt.legend()
            plt.show()
        
    return
    
        
def plot_regression_true_vs_pred(true, pred, var_label = 'IRI (m/km)',title='', size=2,
                                 out_dir = '.', save_plot=True, filename='plot'):        
        try:
            m, b = np.polyfit(true, pred, 1)
        except:
            m = 0
            b = 0
        if size==2:
            plt.rcParams.update({'font.size': 6})
            figsize=[2.2,2]
            dpi= 1000
            ms = 2  
            ls = 9
        if size==3:
            plt.rcParams.update({'font.size': 7})
            figsize=[3.3,3]
            dpi= 1000
            ms = 6
            ls = 11

        rmse = np.sqrt(mean_squared_error(y_true =  true, y_pred = pred))
        mae = mean_absolute_error(y_true =  true, y_pred = pred)
        r2 = r2_score(y_true = true, y_pred = pred)
        mape = mean_absolute_percentage_error(y_true =  true, y_pred = pred)
        
        #var_min = true.min() - 0.3*true.min()
        var_max = true.max() + 0.35*true.max()

        var_min = 0.3
        #var_max = 3.5
        
        plt.figure(figsize=figsize, dpi=dpi)

        #plt.plot(true, m*true + b, c='blue', label='Best fit') 
        plt.scatter(true, pred, marker='o',s=ms, facecolors='none', edgecolors='r', label='Prediction')
        plt.xlabel('Actual {0}'.format(var_label), fontsize=ls+1)
        plt.ylabel('Predicted {0}'.format(var_label), fontsize=ls+1)
        plt.xlim([var_min, var_max])
        plt.ylim([var_min, var_max])
        plt.title(title)
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=ls+1)
        
        # Text
        eq='{0:.2f}*x+{1:.2f}'.format(m,b)
        plt.text(0.63,0.3,'{0}\n$R^2$  = {1:.2f}\nRMSE = {2:.2f}\nMAE = {3:.2f} '.format(eq,r2,rmse,mae),
                 style='italic', horizontalalignment='left',verticalalignment='top', transform = ax.transAxes)    
        
        #plt.text(0.75,0.3,'{0}\n$R^2$ = {1:.2f}\nRMSE = {2:.2f}\nMAE = {3:.2f}\nMAPE = {3:.2f} '.format(eq,r2,rmse,mae,mape),
        #         style='italic', horizontalalignment='left',verticalalignment='top', transform = ax.transAxes)    
                
        # Diagonal
        plt.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='dashed',color='grey', linewidth=1)
        plt.legend(prop={'size': ls-1})
        plt.tight_layout()
        
        if save_plot:
            out_file_path = '{0}/{1}.png'.format(out_dir, filename)
            plt.savefig(out_file_path, dpi=dpi, bbox_inches = "tight")
            plt.savefig(out_file_path.replace('.png','.eps'),format='eps',dpi=dpi, bbox_inches = "tight")
            plt.savefig(out_file_path.replace('.png','.pdf'),dpi=dpi, bbox_inches = "tight")
            print('file saved as: ',out_file_path)
            
        return
    
    
def plot_fs(nf, res, var_label = 'MSE',title='', size=2,
                               out_dir = '.', save_plot=True, filename='plot-fs'):        
      if size==2:
          plt.rcParams.update({'font.size': 6})
          figsize=[2.5,2]
          dpi= 1000
          ms = 3  
          ls = 8
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
      plt.ylabel('{0}'.format(var_label), fontsize=ls+1)
      plt.xlabel('Number of features', fontsize=ls+1)
      #plt.xlim([var_min, var_max])
      #plt.ylim([var_min, var_max])
      #plt.title(title)
      ax = plt.gca()
      ax.yaxis.set_major_formatter('{x:.2e}')
      ax.xaxis.set_major_formatter('{x:.0f}')
      # For the minor ticks, use no labels; default NullFormatter.
      ax.xaxis.set_minor_locator(AutoMinorLocator())
      ax.tick_params(axis='both', labelsize=ls+1)     
    
      #ax.yaxis.set_minor_locator(AutoMinorLocator())
      plt.tight_layout()
      
      if save_plot:
          out_file_path = '{0}/{1}'.format(out_dir, filename)
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
    
    


def plot_feature_importance(importance, names, save_plot = True, out_dir = '.', filename='fs_ranking.png',plot_type=''):

    figsize=[7,5]
    dpi=1000
    #sns.set(rc={"figure.dpi":dpi, 'savefig.dpi':dpi,'font.size': 2, 'ytick.labelsize':3})
        
    if plot_type =='mse':
        xlabel = 'MSE'
        sns.set(rc={'xtick.labelsize':9,'ytick.labelsize':9})
    else:
        xlabel = 'MSE'
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=figsize)
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], palette = "Spectral")
    ax = plt.gca()
    if plot_type =='mse':
        ax.set_xlim([0.04,0.045])
        #ax.set(xscale="log")
        #ax.xaxis.set_major_locator(MultipleLocator(20))
        #ax.set_xticks([0.045, 0.044, 0.043, 0.042, 0.041])
        #ax.xaxis.set_major_formatter('{x:.0f}')
        #ax.xaxis.set_major_formatter(LogFormatter(10,  labelOnlyBase=True))
        
    #Add chart labels
    plt.xlabel(xlabel)
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Turns off grid 
    ax = plt.gca()
    #ax.grid(False)
    ax.yaxis.label.set_size(24)
    
    if save_plot:
          out_file_path = '{0}/{1}'.format(out_dir, filename)
          plt.savefig(out_file_path, dpi=dpi, bbox_inches = "tight")
          plt.savefig(out_file_path.replace('.png','.eps'),format='eps',dpi=dpi, bbox_inches = "tight")
          plt.savefig(out_file_path.replace('.png','.pdf'),dpi=dpi, bbox_inches = "tight")
          print('file saved as: ',out_file_path)

    return

def save_hist(aligned, var, route = '', label = '', plot_suff = '', out_dir_plots = None):
    fig, axis = plt.subplots()
    aligned[var].hist(ax = axis, density=1)
    axis.set_title(label)
    axis.set_xlabel(label + ' [m]')
    axis.set_ylabel('Counts')
    axis.text(0.8, 0.9, 'Route: '+ route,
              horizontalalignment="center",
              verticalalignment="center",
              transform = axis.transAxes)
    if out_dir_plots:
        plot_name = '{0}/{1}_{2}.png'.format(out_dir_plots, plot_suff, var)
        fig.savefig(plot_name)
        print('Wrote {0}'.format(plot_name))
    return fig



def get_normalized_hist(x, var_name = '', out_dir='.', suff = ''):
    
    if x.shape[0]==0:
        return
    
    x = x.to_numpy()
            
    if 'object' in str(x.dtype):
        x = np.concatenate(x)
    
    fig, ax = plt.subplots()
    xmin = x.min()
    if xmin-0.1*xmin>0:
        xmin = xmin-0.1*xmin
    xmax = x.max()
    xmax = xmax + 0.1*xmax
    binrange = [xmin, xmax]  
    
    dpi  = 300
    
    if 'IRI' in var_name:
        xlabel = 'IRI [m/km]'
        bins = 20
    else:
        xlabel=''
        bins = 20
        
    sns.histplot(x, binrange=binrange, bins=bins, stat = 'probability', common_norm = True, ax = ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Percent [%]')
    
    # Set yticks to show percent
    yticks = ax.get_yticks()*100    
    yticks = ax.get_yticks()
    yticks = yticks*100
    yticks = yticks.astype(int) 
    #ax.yaxis.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_yticklabels(yticks)
    
    # Xticks
    ml = MultipleLocator(5)
    ax.xaxis.set_minor_locator(ml)
    ml2 = MultipleLocator(10)
    ax.xaxis.set_minor_locator(ml2)
    
    # Figure size
    xs = 4
    fig.set_size_inches([xs,0.66*xs])
    out_filename = '{0}/normalized_{1}{2}.png'.format(out_dir, var_name, suff)
    fig.savefig(out_filename,bbox_inches='tight', dpi = dpi) 
    fig.savefig(out_filename.replace('.png','.eps'),bbox_inches='tight', dpi = dpi)     
    print('Saved: ',out_filename)    

    return 
