"""
@author: Milena Bajic (DTU Compute)
"""
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from utils.data_transforms import *
from utils.plotting import *
import osrm
import requests
import json
import os, sys
from haversine import haversine, Unit

def haversine_distance(lat1, lat2, lon1, lon2, in_meters = True):
     
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371.0088
      
    # calculate the result
    res = c*r
    if in_meters :
        return 1000*res
    else:
        return res



def latlon_to_cartesian(lats,lons):
    R = 6371008.8 # radius of the Earth in meters
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    x = R * np.cos(lats) * np.cos(lons)
    y = R * np.cos(lats) * np.sin(lons)
    z = R *np.sin(lats)
    return x,y,z

def get_segment(i_start, i_end, df, var):
    x=  df[int(i_start): int(i_end)+1][var].values
    x= x[~np.isnan(x)] 
    return x

def clean(df):
    for var in  ['acc.xyz.x', 'acc.xyz.y', 'acc.xyz.z','obd.spd_veh']:
        mask = (df[var].apply(lambda row: len(row))>0)
        df = df[mask]
        df.reset_index(inplace=True, drop=True)  
    return df
 

def do_kNN_matching_irisegments_only(GM_data, iri, GM_task_id, DRD_id, max_distance = 1, gps_points_to_plot = -1,  out_dir='results'):
    
    out_filename = '{0}/matched_trip{1}_{2}_filter10_maxdistance{3}.pkl'.format(out_dir, GM_task_id, 'GM', max_distance)
    try:
        matched_data = pd.read_pickle(laser_out_filename)
        dist_idx = None
    except:
        # Convert latlon to eucleadian
        DRD_x, DRD_y, DRD_z = latlon_to_cartesian(iri.lat.values,  iri.lon.values)
        GM_x, GM_y, GM_z  = latlon_to_cartesian(GM_data.lat.values,  GM_data.lon.values)
        
        # Match drd iri to the closest gm point
        print('Matching DRD to GM')
        latlon = np.vstack([GM_x, GM_y, GM_z]).T
        kd_tree = KDTree(latlon)
        findNearestRegion = kd_tree.query
        
        # Find the closest neigbour (the one with the minimum eucledian distance)
        dist_idx = [findNearestRegion(coords) for coords in zip(DRD_x, DRD_y, DRD_z)] # dist_idx: (distance, DRD_index)
        
        # Set a dataframe to store the matches
        matched_data = pd.DataFrame()
        for col in GM_data.columns:
            new_col=col if col.startswith('GM') else 'GM_'+col
            matched_data[new_col] = pd.Series(dtype = GM_data.dtypes[col])
        for col in iri.columns:
            matched_data[col] = pd.Series(dtype = iri.dtypes[col])
        matched_data['Distance'] = pd.Series(dtype = np.float32)
        
        print('Filtering matches')
        index=0
        # Filter on maximum distance and save the matching results
        for DRD_index, (distance, GM_index) in enumerate(dist_idx): 
            #print(DRD_index, GM_index, distance)
            
            # Save only matched with distance smaller than this
            if distance > max_distance:
                continue
        
            # GM match
            GM_match = GM_data.iloc[GM_index]
            
            # DRD match
            DRD_match = iri.iloc[DRD_index]
            
            # Append the matched data
            for col_name in GM_data.columns:
                new_col_name=col_name if col_name.startswith('GM') else 'GM_'+col_name
                matched_data.at[index,new_col_name] = GM_match[col_name]   
                
            for col_name in iri.columns:
                matched_data.at[index,col_name] = DRD_match[col_name]
                
            matched_data.at[index,'Distance'] = distance
            
            index=index+1
        #matched_data.drop(['DRD_DRDMeasurementId','DRD_message', 'Distance','DRD_T'],
        #                  axis=1,inplace=True)
        #matched_data = matched_data.reindex(columns=['GM_TS_or_Distance','GM_Date', 'GM_Time','GM_lat', 'GM_lon','GM_Acceleration_x', 'GM_Acceleration_y', 'GM_Acceleration_z','GM_Acceleration_full','DRD_TS_or_Distance', 'DRD_lat', 'DRD_lon','DRD_IRI5', 'DRD_IRI21'])
    
    # Min and max distance in matched data
    segments = [] #all segments
    min_d =  int(matched_data.TS_or_Distance.min())
    max_d = int(matched_data.TS_or_Distance.max())
    for d in range(min_d,max_d,10): 
        print('Segment: {0}-{1}'.format(d,d+10))
        seg0 =  matched_data[matched_data.TS_or_Distance==d]
        seg1 =  matched_data[matched_data.TS_or_Distance==d+10]
        if seg0.empty or seg1.empty:
            continue
        
        #seg0.drop(['DRD_IRI5', 'DRD_IRI21','GM_Acceleration_x','GM_Acceleration_y','GM_Acceleration_z','GM_Acceleration_full'],axis=1,inplace=True)
        seg0.reset_index(inplace=True)
        #seg1.drop(['GM_Acceleration_x','GM_Acceleration_y','GM_Acceleration_z','GM_Acceleration_full'],axis=1,inplace=True)
        seg1.reset_index(inplace=True)
     
        # Column names
        for col_name in seg0.columns:
            seg0.rename(columns = {col_name:col_name+'_start'}, inplace = True) 
        for col_name in seg1.columns:
            if 'IRI' in col_name:
                continue
            seg1.rename(columns = {col_name:col_name+'_end'}, inplace = True) 
            
        for col_name in seg1.columns:
            seg0[col_name] = seg1[col_name] 
        seg0.drop(['index_start', 'index_end'],axis=1,inplace=True)
        
        segments.append(seg0)
     
    #Prepare
    matched_iri_segments = pd.concat(segments)
    matched_iri_segments.reset_index(inplace=True, drop=True)
    
    # Compute
    matched_iri_segments['GM_TS_or_Distance_start'] = matched_iri_segments.apply(lambda row: datetime.datetime.combine(row.GM_Date_start, row.GM_Time_start), axis=1)
    matched_iri_segments['GM_TS_or_Distance_end'] = matched_iri_segments.apply(lambda row: datetime.datetime.combine(row.GM_Date_end, row.GM_Time_end), axis=1)
   
    # Save
    #matched_iri_segments.to_pickle('{0}/GMtrip-{1}_DRD-{2}_maxdistance-{3}_irisegments_startend.pkl'.format(out_dir, GM_task_id, DRD_id, max_distance))
    #matched_iri_segments.to_excel('{0}/GMtrip-{1}_DRD-{2}__maxdistance-{3}_irisegments_startend.xlsx'.format(out_dir, GM_task_id, DRD_id, max_distance))
    return matched_iri_segments
  

          
def do_kNN_matching(GM_data, DRD_data, iri, GM_task_id, var_name,  max_distance = 5, gps_points_to_plot = -1,  out_dir='results'):
    
    laser_out_filename = '{0}/matched_trip{1}_{2}_filter10_maxdistance{3}_lasers.pkl'.format(out_dir, GM_task_id, var_name, max_distance)
    
    try:
        matched_data = pd.read_pickle(laser_out_filename)
        dist_idx = None
    except:
        if 'Date' not in GM_data.columns:
            GM_data['Date'] = GM_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).date())
            GM_data['Time'] = GM_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).time())
        
        # Convert latlon to eucleadian
        DRD_x, DRD_y, DRD_z = latlon_to_cartesian(DRD_data.DRD_lat.values,  DRD_data.DRD_lon.values)
        GM_x, GM_y, GM_z  = latlon_to_cartesian(GM_data.lat.values,  GM_data.lon.values)
        
        # Do matching
        print('Matching GM to DRD')
        latlon = np.vstack([DRD_x, DRD_y, DRD_z ]).T
        kd_tree = KDTree(latlon)
        findNearestRegion = kd_tree.query
        
        # Find the closest neigbour (the one with the minimum eucledian distance)
        dist_idx = [findNearestRegion(coords) for coords in zip(GM_x, GM_y, GM_z)] # dist_idx: (distance, DRD_index)
              
        # Set a dataframe to store the matches
        matched_data = pd.DataFrame()
        for col in GM_data.columns:
            matched_data[col] = pd.Series(dtype = GM_data.dtypes[col])
        for col in DRD_data.columns:
            matched_data[col] = pd.Series(dtype = DRD_data.dtypes[col])
        matched_data['Distance'] = pd.Series(dtype = np.float32)
        index = 0
        
        print('Filtering matches')
        # Filter on maximum distance and save the matching results
        for GM_index, (distance, DRD_index) in enumerate(dist_idx): 
            #print(DRD_index, GM_index, distance)
            
            # Save only matched with distance smaller than this
            if distance > max_distance:
                continue
        
            # GM match
            GM_match = GM_data.iloc[GM_index]
            
            # DRD match
            DRD_match = DRD_data.iloc[DRD_index]
            
            # Append the matched data
            for col_name in GM_data.columns:
                matched_data.at[index,col_name] = GM_match[col_name]   
                
            for col_name in DRD_data.columns:
                matched_data.at[index,col_name] = DRD_match[col_name]
                
            matched_data.at[index,'Distance'] = distance
            
            index=index+1
    
       
        # Drop what is not needed
        matched_data.dropna(inplace=True)
        matched_data.drop(['GPS_TS_or_Distance_start', 'GPS_lat_start','GPS_lon_start','GPS_dt', 'GPS_dlat', 'GPS_dlon', 'dtx'],axis=1,inplace=True)
        matched_data.sort_values(by ='DRD_TS_or_Distance', inplace=True)
        matched_data.reset_index(inplace=True, drop=True)
        
        # Save match to lasers
        matched_data.to_pickle(laser_out_filename)


    # Filter iri to make it faster
    mind = matched_data.DRD_TS_or_Distance.min()
    maxd = matched_data.DRD_TS_or_Distance.max()
    iri = iri[(iri.DRD_TS_or_Distance>mind) & (iri.DRD_TS_or_Distance<maxd+10)]
    matched_data.reset_index(inplace=True, drop=True)

    # Find iri
    iri_matched = df_to_iri_seq(matched_data, iri, var_name, out_dir)
    iri_matched.reset_index(inplace=True, drop=True)
    iri_matched['DRD_IRI_mean'] = iri_matched.apply(lambda row: (row.DRD_IRI5+row.DRD_IRI21)/2, axis=1)
    iri_matched.to_pickle('{0}/matched_trip{1}_{2}_filter10_maxdistance{3}_iri.pkl'.format(out_dir, GM_task_id, var_name, max_distance))
    
    # Plot
    #plot_geolocation_2((matched_data.lon.values[0:gps_points_to_plot:50], matched_data.lat.values[0:gps_points_to_plot:50]),
    #                   (matched_data.DRD_lon.values[0:gps_points_to_plot:50], matched_data.DRD_lat.values[0:gps_points_to_plot:50]), name='_matched_trip{0}_{1}'.format(GM_task_id, var_name))
    

    return matched_data, dist_idx, iri_matched


def do_kNN_matching_iri(GM_data, iri, GM_task_id, DRD_id, gps_points_to_plot = -1,  out_dir='results'):
    
    out_filename = '{0}/matched_trip{1}_{2}.pkl'.format(out_dir, GM_task_id, DRD_id)
    
    # Convert latlon to eucleadian: DRD
    iri_suffix = '_map_start'
    DRD_start_x, DRD_start_y, DRD_start_z = latlon_to_cartesian( iri['lat'+iri_suffix].values,  iri['lat'+iri_suffix].values )
    iri_suffix = '_map_end'
    DRD_end_x, DRD_end_y, DRD_end_z = latlon_to_cartesian( iri['lat'+iri_suffix].values,  iri['lat'+iri_suffix].values )
 
    # Convert latlon to eucleadian: GM
    GM_x, GM_y, GM_z  = latlon_to_cartesian( GM_data['lat'].values,  GM_data['lat'].values )
    
    # Match drd iri to the closest gm point
    print('Matching DRD to GM')
    latlon = np.vstack([GM_x, GM_y, GM_z]).T
    findNearestRegion = KDTree(latlon).query
    
    # Find the closest neigbour (the one with the minimum eucledian distance)
    dist_idx_start = [findNearestRegion(coords) for coords in zip(DRD_start_x, DRD_start_y, DRD_start_z)] # dist_idx: (distance, DRD_index)
    dist_idx_end = [findNearestRegion(coords) for coords in zip(DRD_end_x, DRD_end_y, DRD_end_z)] # dist_idx: (distance, DRD_index)
    
    # Remove unmatched 
    dist_idx_start  = [(x,y) for (x,y) in dist_idx_start if (x!=np.inf and y!=np.inf)]
    dist_idx_end  = [(x,y) for (x,y) in dist_idx_end if (x!=np.inf and y!=np.inf)]
      
    # Set a dataframe to store the matches
    iri['GM_lat_map_start'] = pd.Series(dtype = np.float32) #match to the GPS start map
    iri['GM_lon_map_start'] = pd.Series(dtype = np.float32) #match to the GPS start map
    iri['DRD_index_start'] = pd.Series(dtype = np.int64) #match to the GPS start map
    iri['GM_index_start'] = pd.Series(dtype = np.int64) #match to the GPS start map
    iri['Distance_start'] = pd.Series(dtype = np.float32)
    
    iri['GM_lat_map_end'] = pd.Series(dtype = np.float32)  # match to the GPS end 
    iri['GM_lon_map_end'] = pd.Series(dtype = np.float32) #match to the GPS end
    iri['DRD_index_end'] = pd.Series(dtype = np.int64) #match to the GPS start map
    iri['GM_index_end'] = pd.Series(dtype = np.int64) #match to the GPS start map
    iri['Distance_end'] = pd.Series(dtype = np.float32)
    

    for DRD_index, (distance, GM_index) in enumerate(dist_idx_start): 
    
        # GM match
        GM_match = GM_data.iloc[GM_index]
        
        # DRD match
        DRD_match = iri.iloc[DRD_index]
        
        iri.at[DRD_index, 'GM_lat_map_start']  = GM_match.lat
        iri.at[DRD_index, 'GM_lon_map_start']  = GM_match.lon
        iri.at[DRD_index, 'DRD_index_start'] = DRD_index
        iri.at[DRD_index, 'GM_index_start'] = GM_index
        iri.at[DRD_index, 'Distance_start'] = distance
        
        
    for DRD_index, (distance, GM_index) in enumerate(dist_idx_end): 

        
        # GM match
        GM_match = GM_data.iloc[GM_index]
        
        # DRD match
        DRD_match = iri.iloc[DRD_index]
        
        iri.at[DRD_index, 'GM_lat_map_end']  = GM_match.lat
        iri.at[DRD_index, 'GM_lon_map_end']  = GM_match.lon
        iri.at[DRD_index, 'DRD_index_end'] = DRD_index
        iri.at[DRD_index, 'GM_index_end'] = GM_index
        iri.at[DRD_index, 'Distance_end'] = distance
      
    return iri    


def filter(iri, max_distance = 5, gps_points_to_plot = -1,  out_dir='results'):
    #matched_data.drop(['DRD_DRDMeasurementId','DRD_message', 'Distance','DRD_T'],
    #                  axis=1,inplace=True)
    #matched_data = matched_data.reindex(columns=['GM_TS_or_Distance','GM_Date', 'GM_Time','GM_lat', 'GM_lon','GM_Acceleration_x', 'GM_Acceleration_y', 'GM_Acceleration_z','GM_Acceleration_full','DRD_TS_or_Distance', 'DRD_lat', 'DRD_lon','DRD_IRI5', 'DRD_IRI21'])

    # Min and max distance in matched data
    segments = [] #all segments
    min_d =  int(matched_data.TS_or_Distance.min())
    max_d = int(matched_data.TS_or_Distance.max())
    for d in range(min_d,max_d,10): 
        print('Segment: {0}-{1}'.format(d,d+10))
        seg0 =  matched_data[matched_data.TS_or_Distance==d]
        seg1 =  matched_data[matched_data.TS_or_Distance==d+10]
        if seg0.empty or seg1.empty:
            continue
        
        #seg0.drop(['DRD_IRI5', 'DRD_IRI21','GM_Acceleration_x','GM_Acceleration_y','GM_Acceleration_z','GM_Acceleration_full'],axis=1,inplace=True)
        seg0.reset_index(inplace=True)
        #seg1.drop(['GM_Acceleration_x','GM_Acceleration_y','GM_Acceleration_z','GM_Acceleration_full'],axis=1,inplace=True)
        seg1.reset_index(inplace=True)
     
        # Column names
        for col_name in seg0.columns:
            seg0.rename(columns = {col_name:col_name+'_start'}, inplace = True) 
        for col_name in seg1.columns:
            if 'IRI' in col_name:
                continue
            seg1.rename(columns = {col_name:col_name+'_end'}, inplace = True) 
            
        for col_name in seg1.columns:
            seg0[col_name] = seg1[col_name] 
        seg0.drop(['index_start', 'index_end'],axis=1,inplace=True)
        
        segments.append(seg0)
     
    #Prepare
    matched_iri_segments = pd.concat(segments)
    matched_iri_segments.reset_index(inplace=True, drop=True)
    
    # Compute
    matched_iri_segments['GM_TS_or_Distance_start'] = matched_iri_segments.apply(lambda row: datetime.datetime.combine(row.GM_Date_start, row.GM_Time_start), axis=1)
    matched_iri_segments['GM_TS_or_Distance_end'] = matched_iri_segments.apply(lambda row: datetime.datetime.combine(row.GM_Date_end, row.GM_Time_end), axis=1)
   
    # Save
    #matched_iri_segments.to_pickle('{0}/GMtrip-{1}_DRD-{2}_maxdistance-{3}_irisegments_startend.pkl'.format(out_dir, GM_task_id, DRD_id, max_distance))
    #matched_iri_segments.to_excel('{0}/GMtrip-{1}_DRD-{2}__maxdistance-{3}_irisegments_startend.xlsx'.format(out_dir, GM_task_id, DRD_id, max_distance))
    return matched_data, matched_iri_segments 

def map_match(gps, host, lat_name = 'lat', lon_name = 'lon', r=50):
    # Map match a chunk of gps data
    lat = gps[lat_name]
    lon = gps[lon_name]
    c = list(zip(lon,lat))
    radiuses = [r]*len(lat) 
    print(lat)
    print(lon)
    
    timeout = 100
    
    try:
        client = osrm.Client(host=host, profile='car', timeout=timeout)
        match_response = client.match(coordinates=c, radiuses=radiuses)
        
        t = match_response['tracepoints']
        wp = [i['location'] if i!=None else (np.NaN,np.NaN) for i in match_response['tracepoints'] ]
        gps['lon_map'] = list(zip(*wp))[0]
        gps['lat_map'] = list(zip(*wp))[1]
        
        gps['way_id'] = [i['waypoint_index'] if i!=None else (np.NaN,np.NaN) for i in match_response['tracepoints'] ]
        gps['street_name'] = [i['name'] if i!=None else (np.NaN,np.NaN) for i in match_response['tracepoints'] ]
        #gps['hint'] = [i['hint'] if i!=None else (np.NaN,np.NaN) for i in match_response['tracepoints'] ]
        return gps
    except:
        return pd.DataFrame()
    


def map_match_gps_data(gps_data, is_GM, out_dir ='.', out_file_suff = '', recreate = False):
    # Pass a pd dataframe with gps only and returned matched gps dataframe
    
    out_filename = '{0}/map_matched_data{1}.pickle'.format(out_dir, out_file_suff)
    if os.path.exists(out_filename) and not recreate:
        print('Loading map matched data from file')
        gps_result = pd.read_pickle(out_filename)
        return gps_result  
    
    else:
        # Create GPS chunks
        n = ceil(gps_data.shape[0]/100)
        gps_chunks = np.array_split(gps_data, n)
                  
        # Map Match GPS chunks
        for i, gps_chunk in enumerate(gps_chunks):
            if i%100==0:
                print('Chunk: {0}/{1}'.format(i, n))
            map_match(gps_chunk) #it appends the result into the original dataframe
            
        gps_result = pd.concat(gps_chunks)
      
        # Clean non-matched points
        rem_ind =  gps_result[gps_result['lat_map'].isna()].index 
        gps_result.drop(index = rem_ind, inplace=True) 
        rem_ind =  gps_result[gps_result['lon_map'].isna()].index
        gps_result.drop(index = rem_ind, inplace=True) 
        
        out_df = gps_result
          
        # Extract Message column if GM
        if is_GM:
            # Set Message to nan if from GPS 
            out_df['Message'].mask( out_df['T']=='track.pos',inplace=True)
            
            # Extract Message for all sensors
            print('- Extracting message')
            out_df = pd.concat([out_df.drop(['Message'], axis=1), out_df['Message'].apply(pd.Series)], axis=1)
            #out_df.drop([0,'@vid'], axis=1,inplace=True)
            
        # Save results
        out_df.to_pickle(out_filename)
        #out_df.to_csv(out_filename.replace('.pickle', '.csv'))
        
        print('Saved map matching output to: {0}'.format(out_filename))
        
        return gps_result




def route_between_2GPS_points(gps):
    # Return nodes between gps points. Input gps is a dataframe.
    
    # Map matched GPS coordinates
    lat = gps.lat_map 
    lon = gps.lon_map
    c = list(zip(lon,lat))
    
    # Call the Route service
    timeout = 20
    client = osrm.Client(host='http://liradbdev.compute.dtu.dk:5000', profile='car', timeout=timeout)
    match_response = client.route(coordinates=c, overview=osrm.overview.full, annotations = True)
    
    # Distance between the gps points
    distance_between_2GPS = match_response['routes'][0]['distance']
    
    # Compute time and speed
    dt = (gps.TS_or_Distance.iloc[1]-gps.TS_or_Distance.iloc[0]).total_seconds()
    speed = (distance_between_2GPS/dt)*3.6
    
    gps['GPS_distance'] = [0, distance_between_2GPS]
    gps['GPS_dt'] = [0, dt]
    gps['GPS_speed'] = (0, speed)
    
    # Nodes
    geom =  match_response['routes'][0]['geometry']    
    nodes = pd.DataFrame(geom['coordinates'],columns=['lon','lat'])
    d = match_response['routes'][0]['legs'][0]['annotation']['distance']
    d.insert(0,0)   # The distance from the first node to itself is 0
    nodes['distance'] = d

    return nodes, distance_between_2GPS, speed
   
    
def route_distance(lon1, lat1, lon2, lat2):
    # Routing distance is not very accurate
    
    c = [(lon1, lat1),(lon2, lat2)]
    
    # Call the Route service
    timeout = 20
    client = osrm.Client(host='http://liradbdev.compute.dtu.dk:5000', profile='car', timeout=timeout)
    response = client.route(coordinates=c, overview=osrm.overview.full, annotations = True)
    
    # Distance between the gps points
    d = response['routes'][0]['distance']
    #gps['distance'] = [0, distance_between_2GPS]

    return d


def interpolate_trip(all_sensor_data, out_dir='.', add_sensors = [], file_suff = 'GM', recreate = False):
    # Interpolate
    # lx = l1+(dl*dtx/dt)
     #gps_data['lat_start'] + (gps_data['dlat']*/gps_data['dt'])

    print('Interpolating')
    out_filename = '{0}/interpolated_{1}.pickle'.format(out_dir, file_suff)
     
    if os.path.exists(out_filename) and not recreate:
        print('Loading interpolated data from file: {0}'.format(out_filename))
        all_sensor_data = pd.read_pickle(out_filename)
        gps_out_filename = out_filename.replace('.pickle','_GPSonly.pickle')
        gps =  pd.read_pickle(gps_out_filename)
        return all_sensor_data, gps

    else:
        # Create segment dataframe
        gps = all_sensor_data[all_sensor_data['T']=='track.pos']
        gps = gps[['TS_or_Distance', 'T',  'lon', 'lat','lon_map','lat_map', 'way_id','street_name']]
        t1 = gps[0:-1]
        t1.reset_index(inplace=True) #the last el. not included
        t2 = gps[1:]
        t2.reset_index(inplace=True)
        gps_data_segments = pd.merge(t1.drop(['lat','lon'],axis=1),t2.drop(['lat','lon'],axis=1), left_index=True, right_index=True, suffixes=('_start','_end'))
         
        # Filter
        #gps_data_segments = gps_data_segments[gps_data_segments['dt']<2] #consider only these segmengps_data_segments for interpolation
        #gps_data_segments.reset_index(inplace=True)
            
        # Get time diff. between 2 meas.
        gps_data_segments['dt'] = (gps_data_segments['TS_or_Distance_end']  - gps_data_segments['TS_or_Distance_start'])
        gps_data_segments['dt'] = gps_data_segments['dt'].apply(lambda row: row.total_seconds())
            
        # Get lat and long dif in gps_data_segments 
        gps_data_segments['dlat'] = gps_data_segments['lat_map_end']-gps_data_segments['lat_map_start']
        gps_data_segments['dlon'] = gps_data_segments['lon_map_end']-gps_data_segments['lon_map_start']
         
        # Find a matching segment
        all_sensor_data['gps_segment'] = all_sensor_data.TS_or_Distance.apply(lambda t: gps_data_segments[(t>gps_data_segments.TS_or_Distance_start) & (t<gps_data_segments.TS_or_Distance_end) ])
        all_sensor_data['gps_segment'] = all_sensor_data['gps_segment'].apply(lambda df: np.NaN if df.empty else df)
        all_sensor_data.dropna(subset = ['gps_segment'], inplace=True)
        all_sensor_data.reset_index(inplace=True, drop=True)  
        
        # Extract
        all_sensor_data['GPS_TS_or_Distance_start'] =  all_sensor_data['gps_segment'].apply(lambda df: df['TS_or_Distance_start'].iloc[0])
        all_sensor_data['GPS_lat_map_start'] =  all_sensor_data['gps_segment'].apply(lambda df: df['lat_map_start'].iloc[0])
        all_sensor_data['GPS_lon_map_start'] =  all_sensor_data['gps_segment'].apply(lambda df: df['lon_map_start'].iloc[0])
        all_sensor_data['GPS_dt'] =  all_sensor_data['gps_segment'].apply(lambda df: df['dt'].iloc[0])
        all_sensor_data['GPS_dlat'] =  all_sensor_data['gps_segment'].apply(lambda df: df['dlat'].iloc[0])
        all_sensor_data['GPS_dlon'] =  all_sensor_data['gps_segment'].apply(lambda df: df['dlon'].iloc[0]) 
        all_sensor_data['way_id'] =  all_sensor_data['gps_segment'].apply(lambda df: df['way_id_start'].iloc[0]) 
        all_sensor_data['street_name_start'] =  all_sensor_data['gps_segment'].apply(lambda df: df['street_name_start'].iloc[0]) 
        all_sensor_data['street_name_end'] =  all_sensor_data['gps_segment'].apply(lambda df: df['street_name_end'].iloc[0]) 
        #all_sensor_data['hint'] =  all_sensor_data['gps_segment'].apply(lambda df: df['hint_start'].iloc[0]) 
        all_sensor_data.drop(columns=['gps_segment'],inplace=True,axis=1)
        
        # Filter?
        all_sensor_data = all_sensor_data[all_sensor_data['GPS_dt']<5]
        
        # Compute dtx 
        all_sensor_data['dtx'] = (all_sensor_data['TS_or_Distance'] - all_sensor_data['GPS_TS_or_Distance_start'])
        all_sensor_data['dtx'] = all_sensor_data['dtx'].apply(lambda row: row.total_seconds()) 
       
        # Compute lat and lon
        all_sensor_data['lat_int'] = all_sensor_data['GPS_lat_map_start'] + (all_sensor_data['GPS_dlat']* all_sensor_data['dtx'])/all_sensor_data['GPS_dt'] 
        all_sensor_data['lon_int'] = all_sensor_data['GPS_lon_map_start'] + (all_sensor_data['GPS_dlon']* all_sensor_data['dtx'])/all_sensor_data['GPS_dt'] 
        
        #var_data.drop(['GPS_TS_or_Distance_start', 'GPS_lat_start','GPS_lon_start', 'GPS_dt', 'GPS_dlat', 'GPS_dlon', 'dtx'],axis=1,inplace=True)
    
        # Select columns
        all_sensor_data.drop(['lat', 'lon','way_id','GPS_dlat', 'GPS_dlon', 'dtx','street_name','GPS_TS_or_Distance_start','GPS_lat_map_start','GPS_lon_map_start', 'lat_map','lon_map'], axis=1, inplace=True)
        all_sensor_data.reset_index(inplace=True, drop=True)  
       
        # Extract Message
        #print('- Extracting message')
        all_sensor_data = pd.concat([all_sensor_data.drop(['Message'], axis=1), all_sensor_data['Message'].apply(pd.Series)], axis=1)
        #GM_map_matched_data.drop([0,'@vid'], axis=1,inplace=True)
        
        # Save results
        #all_sensor_data.to_csv(out_filename)
        out_filename = out_filename.replace('.csv','.pickle')
        all_sensor_data.to_pickle(out_filename)
        print('Saved interpolation output to: {0}'.format(out_filename))
        
        gps_out_filename = out_filename.replace('.pickle','_GPSonly.pickle')
        gps.to_pickle(gps_out_filename)
        print('Saved GPS recorded data to: {0}'.format(gps_out_filename))
        
        return all_sensor_data, gps
      
    
    
def make_sliding_windows_df(iri, window_size = 50, step = 10):
    
    # Set window dataframe
    iri_window = pd.DataFrame(columns = iri.columns)
        
    # Get IRI windows
    n_samples  = iri.shape[0]
    d_max =  int(iri['TS_or_Distance'].iloc[-1])
    row_i = 0
    for d_start in range(0, d_max, step):
        d_end = d_start + window_size
        #print(d_start, d_end)
        iri_this = iri[iri['TS_or_Distance'].between(d_start,d_end)]
        for col in iri.columns:
            #print(col)
            if col in ['lat_map', 'lon_map','TS_or_Distance','street_name']:
                iri_window.at[row_i, col+'_start'] = iri_this[col].to_numpy()[0]
                iri_window.at[row_i, col+'_end'] = iri_this[col].to_numpy()[-1]
            elif col in ['IRI_mean']:
                iri_window.at[row_i, col] = iri_this[col].mean()
        row_i = row_i + 1
      
    # Clean and reset the window dataframe
    iri_window.dropna(axis=1, inplace=True)
    iri_window.reset_index(drop=True, inplace=True)
    iri_data_segments = iri_window
    return iri_data_segments
    
    
def align_GM_DRD(GM, iri_data_segments, GM_trip_id = None, DRD_trip_id = None, window_size = 50, step = 10, out_dir = '.', file_suff = '', 
                 lat_prefix = 'lat_map', lon_prefix='lon_map'):
    # used
     
    # Add GM column names
    for gm_col in GM.columns:
        iri_data_segments['GM.'+gm_col] = None
    iri_data_segments['dt_segment_mins'] = None
            
    # Match segments with GM
    n_samples =  iri_data_segments.shape[0]
    for row_i in range(0, n_samples):
        if row_i%500==0:
            print(row_i)
            
        # Window GPS borders
        lat_start = iri_data_segments.iloc[row_i][lat_prefix + '_start_x'] #rem _x
        lon_start = iri_data_segments.iloc[row_i][lon_prefix + '_start_x'] #rem _x
        lat_end = iri_data_segments.iloc[row_i][lat_prefix + '_end_x']
        lon_end = iri_data_segments.iloc[row_i][lon_prefix + '_end_x']
        
        c1 = (GM.lat_int.between(min(lat_start, lat_end), max(lat_start, lat_end)))
        c2 = (GM.lon_int.between(min(lon_start, lon_end), max(lon_start, lon_end))) 
        gm_this = GM[c1 & c2]
        
        try:
            dt_seg = gm_this['TS_or_Distance'].iloc[-1]-gm_this['TS_or_Distance'].iloc[0]
            dt_mins = dt_seg.total_seconds()/60
        except:
            dt_mins = None

        iri_data_segments.at[row_i,'dt_segment_mins'] = dt_mins
        
        for gm_col in gm_this.columns:
            gm_this_np = gm_this[gm_col].dropna().to_numpy()
            #n = gm_this_np.shape[0]
            #print(gm_col,n)
            iri_data_segments.at[row_i, 'GM.'+gm_col] = gm_this_np

     
    iri_data_segments.dropna(subset=['dt_segment_mins'], inplace=True)
    iri_data_segments = iri_data_segments[iri_data_segments['dt_segment_mins']<30]
    
    # N aligned segments
    n_segments_init = iri_data_segments.shape[0]
    
    # Clean (require more than 2 points per seq)
    var_feats = ['GM.acc.xyz.z','GM.obd.spd_veh.value']
    for var in var_feats:
        try:
            iri_data_segments = iri_data_segments[ iri_data_segments[var].apply(lambda row: len(row)>1) ]
        except:
            pass
    iri_data_segments.reset_index(inplace=True, drop=True)
    
    # N of segments after cleaning
    n_segments_clean = iri_data_segments.shape[0]
    if n_segments_clean==0:
        return iri_data_segments, n_segments_init, n_segments_clean
    
    # Clean
    iri_data_segments.dropna(inplace=True)
    
    # Save
    out_filename = '{0}/aligned_GM-{1}_DRD-{2}{3}.pickle'.format(out_dir, GM_trip_id, DRD_trip_id, file_suff)
    
    iri_data_segments['GM.DRD_d_start_h'] =  iri_data_segments.apply(lambda row: haversine((row['lat_map_start_x'], row['lon_map_start_x']), 
                                                                      (row['GM.lat_int'][0], row['GM.lon_int'][0]), unit = Unit.METERS), axis=1)
   
    iri_data_segments['GM.DRD_d_end_h'] =  iri_data_segments.apply(lambda row: haversine((row['lat_map_end_x'], row['lon_map_end_x']), 
                                                                      (row['GM.lat_int'][-1], row['GM.lon_int'][-1]), unit = Unit.METERS), axis=1)

    #Select only close matches
    #c1 = iri_data_segments['GM.DRD_d_start_h']<5
    #c2 = iri_data_segments['GM.DRD_d_end_h']<5
    #iri_data_segments =  iri_data_segments[c1&c2] # contains matched ARAN/p79 windows
    
    iri_data_segments.to_pickle(out_filename)
    print('Saved aligned output to: {0}'.format(out_filename))
    
    print('Initial number of segments: {0}'.format(n_segments_init))
    print('Cleaned number of segments: {0}'.format(n_segments_clean))
    
    return iri_data_segments, n_segments_init, n_segments_clean
    


    
def align_GM_ARAN(GM, drd, GM_trip_id, DRD_trip_id = '', col_suffix = 'GM.', is_p79 = False, out_dir = '.', file_suff = '', load_from_file = False, clean = True, save = True):
    # drd is any df with segments, and GM is any df which is to be aligned
    
    out_filename = '{0}/aligned_GM-{1}_DRD-{2}{3}.csv'.format(out_dir, GM_trip_id, DRD_trip_id, file_suff)
    
    # Load from file
    if load_from_file and os.path.exists(out_filename):
        iri_data_segments = pd.read_pickle(out_filename.replace('.csv','.pickle'))
        return iri_data_segments, None, None
    elif load_from_file:
        print('Asked to load from file but no file found.')
        sys.exit(0)
        
    drd.reset_index(drop=True,inplace=True)
    iri_data_segments = drd
       
    # Add GM column names
    if is_p79:
        cols = ['RutDepthLeft','RutDepthRight','IRI5','IRI21']
    else:
        cols = GM.columns
        
    for gm_col in cols:
        iri_data_segments[col_suffix+gm_col] = None
            
    # Find matching segments using GPS
    n_samples =  iri_data_segments.shape[0]
    for row_i in range(0, n_samples):
        if row_i%500==0:
            print(row_i)
            
        # Window GPS borders
        lat_start = iri_data_segments.iloc[row_i]['lat_map_start_start']
        lon_start = iri_data_segments.iloc[row_i]['lon_map_start_start']
        lat_end = iri_data_segments.iloc[row_i]['lat_map_end_end']
        lon_end = iri_data_segments.iloc[row_i]['lon_map_end_end']
        
        # you need to check the dicretion before with other methods or in the plot
        if is_p79:
            c1 = (GM.lat.between(min(lat_start, lat_end), max(lat_start, lat_end)))
            c2 = (GM.lon.between(min(lon_start, lon_end), max(lon_start, lon_end)))     
        else:
            c1 = (GM.lat_int.between(min(lat_start, lat_end), max(lat_start, lat_end)))
            c2 = (GM.lon_int.between(min(lon_start, lon_end), max(lon_start, lon_end))) 
        gm_this = GM[c1 & c2]
        
        for gm_col in gm_this.columns:
            print(gm_col)
            if is_p79:
                iri_data_segments.at[row_i, col_suffix +gm_col] = gm_this[gm_col].dropna().to_numpy().mean()
            else:
                iri_data_segments.at[row_i, col_suffix +gm_col] = gm_this[gm_col].dropna().to_numpy()

     
    n_segments_init = iri_data_segments.shape[0]
    
    if clean:
        # Clean (require more than 2 points per seq to do resampling and interpolation)
        var_feats = [col_suffix + 'acc.xyz.z',col_suffix+'obd.spd_veh.value']
        for var in var_feats:
            iri_data_segments = iri_data_segments[ iri_data_segments[var].apply(lambda row: len(row)>1) ]
        iri_data_segments.reset_index(inplace=True, drop=True)
        
        n_segments_clean = iri_data_segments.shape[0]
        
        print('Initial number of segments: {0}'.format(n_segments_init))
        print('Cleaned number of segments: {0}'.format(n_segments_clean))
    else:
        n_segments_clean = None
    
    if is_p79:
        iri_data_segments['KPI'] = compute_kpi_aran(iri_data_segments)
        
    if save:
        # Save results
        #iri_data_segments.to_csv(out_filename)
        iri_data_segments.to_pickle(out_filename.replace('.csv','.pickle'))
        print('Saved aligned output to: {0}'.format(out_filename))
     
    return iri_data_segments, n_segments_init, n_segments_clean


 
def align_ARAN_p79(ARAN_windows, p79_windows, out_dir = '.', out_filename= 'ARAN_p79.pickle'):
    #USED
   # out_filename = '{0}/aligned_GM-{1}_DRD-{2}{3}.csv'.format(out_dir, GM_trip_id, DRD_trip_id, file_suff)
    

    ARAN_windows.reset_index(drop=True,inplace=True)
    p79_windows.reset_index(drop=True,inplace=True)

    ARAN_start_x, ARAN_start_y, ARAN_start_z = latlon_to_cartesian(ARAN_windows.lat_map_start.values.astype(float),  ARAN_windows.lon_map_start.values.astype(float))
    p79_data_x, p79_data_y, p79_data_z  = latlon_to_cartesian(p79_windows.lat_map_start.values.astype(float),  p79_windows.lon_map_start.values.astype(float))
        
    # Match drd iri to the closest gm point
    print('Matching p79 to ARAN')
    latlon = np.vstack([p79_data_x, p79_data_y, p79_data_z]).T
    kd_tree = KDTree(latlon)
    findNearestRegion = kd_tree.query
    
    # Find the closest neigbour (the one with the minimum eucledian distance)
    dist_idx = [findNearestRegion(coords) for coords in zip(ARAN_start_x, ARAN_start_y, ARAN_start_z)] # dist_idx: (distance, p79_index)
    ARAN_windows['match_p79_index'] = [i[1] for i in dist_idx]
    ARAN_windows['match_p79_d'] = [i[0] for i in dist_idx]
    
    #10m selection
    ARAN_windows = ARAN_windows[ARAN_windows['match_p79_d']<10]
    
    ARAN_windows =  ARAN_windows.merge(p79_windows, left_on='match_p79_index', right_index=True)
    ARAN_windows.to_pickle(out_dir+out_filename)
   
    return ARAN_windows

def align_DRD_GM(DRD_windows, GM_data, out_dir = '.', out_filename= 'DRD_GM.pickle', GM_prefix = 'GM.'):
    
   # out_filename = '{0}/aligned_GM-{1}_DRD-{2}{3}.csv'.format(out_dir, GM_trip_id, DRD_trip_id, file_suff)
    

    DRD_windows.reset_index(drop=True,inplace=True)
    GM_data.reset_index(drop=True,inplace=True)

    DRD_start_x, DRD_start_y, DRD_start_z = latlon_to_cartesian(DRD_windows.lat_map_start_x.values.astype(float), DRD_windows.lon_map_start_x.values.astype(float))
    DRD_end_x, DRD_end_y, DRD_end_z = latlon_to_cartesian(DRD_windows.lat_map_end_x.values.astype(float), DRD_windows.lon_map_end_x.values.astype(float))
    GM_data_x, GM_data_y, GM_data_z  = latlon_to_cartesian(GM_data.lat_int.values.astype(float),  GM_data.lon_int.values.astype(float))
        
    # Match drd iri to the closest gm point
    print('Matching DRD start to GM')
    latlon = np.vstack([GM_data_x, GM_data_y, GM_data_z]).T
    kd_tree = KDTree(latlon)
    findNearestRegion = kd_tree.query

    # Find the closest neigbour (the one with the minimum eucledian distance)
    dist_idx_start = [findNearestRegion(coords) for coords in zip(DRD_start_x, DRD_start_y, DRD_start_z)] # dist_idx is (distance, GM_index)
    DRD_windows['match_GM_start_index'] = [i[1] for i in dist_idx_start]
    DRD_windows['match_GM_start_d'] = [i[0] for i in dist_idx_start]
    
    print('Matching DRD end to GM')
    # Find the closest neigbour (the one with the minimum eucledian distance)
    dist_idx_end = [findNearestRegion(coords) for coords in zip(DRD_end_x, DRD_end_y, DRD_end_z)] # dist_idx is (distance, GM_index)
    DRD_windows['match_GM_end_index'] = [i[1] for i in dist_idx_end]
    DRD_windows['match_GM_end_d'] = [i[0] for i in dist_idx_end]
    #ARAN_windows_sel = ARAN_windows[ARAN_windows['match_p79_d']<10]
    
    # Rough filter
    c1 = DRD_windows['match_GM_start_d']<5
    c2 = DRD_windows['match_GM_end_d']<5
    DRD_windows = DRD_windows[c1 & c2]
    DRD_windows.reset_index(drop=True,inplace=True)
    
    for col in GM_data.columns:
        DRD_windows[GM_prefix+col]=None
        
    nrows =  DRD_windows.shape[0]
    for row in range(nrows):
        if row%100==0:
            print('Filling row: {0}/{1}'.format(row,nrows))
        gm_i_start = DRD_windows.at[row,'match_GM_start_index']
        gm_i_end = DRD_windows.at[row,'match_GM_end_index']
        gm_this = GM_data.iloc[gm_i_start: gm_i_end]
        for gm_col in gm_this.columns:
            #print(gm_col)
            DRD_windows.at[row, GM_prefix + gm_col] = gm_this[gm_col].dropna().to_numpy() 
            
    DRD_windows[GM_prefix + 'ARAN_d_start_h'] =   DRD_windows.apply(lambda row: haversine((row['lat_map_start_x'], row['lon_map_start_x']), 
                                                                      (row['GM.lat_int'][0], row['GM.lon_int'][0]), unit = Unit.METERS), axis=1)
   
    DRD_windows[GM_prefix + 'ARAN_d_end_h'] =   DRD_windows.apply(lambda row: haversine((row['lat_map_end_x'], row['lon_map_end_x']), 
                                                                      (row['GM.lat_int'][-1], row['GM.lon_int'][-1]), unit = Unit.METERS), axis=1)

    
    DRD_windows[GM_prefix + 'p79_d_start_h'] =   DRD_windows.apply(lambda row: haversine((row['lat_map_start_y'], row['lon_map_start_y']), 
                                                                      (row['GM.lat_int'][0], row['GM.lon_int'][0]), unit = Unit.METERS), axis=1)
   
    DRD_windows[GM_prefix + 'p79_d_end_h'] =   DRD_windows.apply(lambda row: haversine((row['lat_map_end_y'], row['lon_map_end_y']), 
                                                                      (row['GM.lat_int'][-1], row['GM.lon_int'][-1]), unit = Unit.METERS), axis=1)

    out_filename =  '{0}/{1}'.format(out_dir, out_filename)
    DRD_windows.to_pickle(out_filename)
    
   
    return  DRD_windows


def map_match_gps_data(data, host, is_GM = False, lat_name = 'lat', lon_name = 'lon', out_dir ='.', out_file_suff = '', preload = False):
    
    out_filename = '{0}/map_matched_data{1}.pickle'.format(out_dir, out_file_suff)
    if os.path.exists(out_filename) and preload:
        print('Loading map matched data from file')
        gps_result = pd.read_pickle(out_filename)
        return gps_result  
    
    else:
        gps_data = data[[lat_name, lon_name]]
        
        # Create GPS chunks
        n = ceil(gps_data.shape[0]/100)
        gps_chunks = np.array_split(gps_data, n)
                  
        # Map Match GPS chunks
        for i, gps_chunk in enumerate(gps_chunks):
            if i%100==0:
                print(' === Chunk: {0}/{1} ===='.format(i, n))
            map_match(gps_chunk, host, lat_name, lon_name) #it appends the result into the original dataframe
        gps_result = pd.concat(gps_chunks)
      
        # Clean non-matched points
        rem_ind =  gps_result[gps_result['lat_map'].isna()].index 
        gps_result.drop(index = rem_ind, inplace=True) 
        rem_ind =  gps_result[gps_result['lon_map'].isna()].index
        gps_result.drop(index = rem_ind, inplace=True) 
    
          
        # Extract Message column if GM
        if is_GM:
                    
            # Set Message to nan if from GPS 
            data['Message'].mask( out_df['T']=='track.pos',inplace=True)
            
            # Extract Message for all sensors
            print('- Extracting message')
            data = pd.concat([data.drop(['Message'], axis=1),  data['Message'].apply(pd.Series)], axis=1)

        
        # Merge with initial dataset
        data = pd.concat([data, gps_result[['lat_map', 'lon_map']]],axis=1)
                  
        # Save results
        data.to_pickle(out_filename)
        #out_df.to_csv(out_filename.replace('.pickle', '.csv'))                          
        print('Saved map matching output to: {0}'.format(out_filename))
        
        return data





def get_windows_GM_predict_mode(df, window_size = 100, step = 10):
    df.reset_index(inplace=True, drop=True)
    df_window = pd.DataFrame(columns = df.columns)
    
    step = 500
    # Get df windows
    n_slices  = int(df.shape[0]/step)-1
    n_samples = n_slices * step
    
    row_i = 0
    for row_0 in range(0, n_samples, step):
        row_end  = row_0 + step
        #print(row_0, row_end)
        
        df_this = df[row_0:row_end]
        
        for col in df.columns:
            #print(col)
            if col in ['lat_int', 'lon_int','TS_or_Distance','street_name_start','street_name_end']:
                df_window.at[row_i, col+'_start'] = df_this[col].to_numpy()[0]
                df_window.at[row_i, col+'_end'] = df_this[col].to_numpy()[-1]
            elif col in ['@vid','T']:
                continue
            else:
                x = df_this[col].dropna().to_numpy()
                df_window.at[row_i, col] = x 
        
        row_i = row_i + 1
    
    # Clean and reset the window dataframe
    df_window.dropna(axis=1, inplace=True)
    df_window.reset_index(drop=True, inplace=True)
    df_data_segments = df_window
    
    return df_data_segments


def match_start_end_points(df1, df2, df1_latlon = ('lat','lon'),df2_latlon = ('lat_int','lon_int')):

    df1.reset_index(drop=True,inplace=True)
    from haversine import haversine, Unit
    
    # Starting and ending point
    df1_start = latlon_to_cartesian( df1.iloc[0][df1_lonlat[0]], df1.iloc[0][df1_lonlat[1]] ) #lat, lon to x,y,z
    df1_end = latlon_to_cartesian( df1.iloc[-1][df1_lonlat[0]], df1.iloc[-1][df1_lonlat[1]] ) #lot, lon
    
    df1_s = (df1.iloc[0][df1_lonlat[0]], df1.iloc[0][df1_lonlat[1]])
    haversine(df1_s,gm,unit=Unit.METERS)
    
    # Convert latlon to eucleadian
    GM_x, GM_y, GM_z  = latlon_to_cartesian(df2[df2_latlon[0]].values,  df2[df2_latlon[1]].values)
    
    from dtwalign import dtw
    
    # Match df1 to the closest df2 point
    latlon = np.vstack([GM_x, GM_y, GM_z]).T
    kd_tree = KDTree(latlon)
    findNearestRegion = kd_tree.query
        
   # Find the closest neigbour (the one with the minimum eucledian distance)
    dist_idx = [findNearestRegion(coords) for coords in [df1_start]] # dist_idx: (distance, DRD_index)
    
    
       
