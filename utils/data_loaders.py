"""
@author: Milena Bajic (DTU Compute)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2 # pip install psycopg2==2.7.7 or pip install psycopg2==2.7.7
from json import loads
import sys,os, glob
from datetime import datetime
from utils.analysis import *
import pickle
from json import loads

def filter_DRDtrips_by_year(DRD_trips, sel_2021 = False, sel_2020 = False):
    DRD_trips['Datetime']=pd.to_datetime(DRD_trips['Created_Date'])
    DRD_trips['Year'] = DRD_trips['Datetime'].apply(lambda row: row.year)
    if sel_2020: 
        DRD_trips = DRD_trips[ DRD_trips['Year']==2020]
    elif sel_2021:
        DRD_trips = DRD_trips[ DRD_trips['Year']==2021]
    DRD_trips.drop(['Datetime'], axis=1, inplace=True)
    return DRD_trips
    

def drop_duplicates(DRD_data, iri):
    # Drop duplicate columns (due to ocassical errors in database)
    DRD_data = DRD_data.T.drop_duplicates().T #
    if iri is not None:
        iri = iri.T.drop_duplicates().T
    return DRD_data, iri
    
def load_GM_data(GM_TaskId, out_dir, all_sensors = False, add_sensors = [], load_nrows = -1):
    
    conn = psycopg2.connect(database="postgres", user="..", password="..", host="liradbdev.compute.dtu.dk", port=5432)
   
    # quory = 'SELECT "lat", "lon" FROM "Measurements" 
    # quory = 'SELECT  "lat", "lon" FROM "Measurements" WHERE "Trips"."TaskId"=\'{0}\' ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(GM_TaskId, load_nrows)
    # quory = 'SELECT  "lat", "lon" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE "Trips"."TaskId"=\'{0}\' ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(GM_TaskId, load_nrows)
      
    
    if all_sensors:
        if load_nrows!=-1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(GM_TaskId, load_nrows)
        else:   
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC'.format(GM_TaskId)
    else:
        sensors = ['track.pos','acc.xyz','obd.spd_veh']
        if add_sensors:
            sensors = sensors + add_sensors
        sensors = str(tuple(sensors))
        print('Loading: ',sensors)
        if load_nrows!=-1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC LIMIT {2}'.format(GM_TaskId, sensors, load_nrows)  
        else:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC'.format(GM_TaskId, sensors)     
  

    cursor = conn.cursor()
    meas_data = pd.read_sql(quory, conn, coerce_float = True)
    meas_data.reset_index(inplace=True, drop=True)   
    
    # Extract message
    #=================# 
    meas_data['Message'] = meas_data.message.apply(lambda msg: filter_keys(loads(msg)))
    meas_data.drop(columns=['message'],inplace=True,axis=1)
    meas_data.reset_index(inplace=True, drop=True)
    meas_data = meas_data[['TS_or_Distance','T', 'lat', 'lon','Message']]
    
    # Extract day and time
    #=================#
    meas_data['Date'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).date())
    meas_data['Time'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).time())
    meas_data.sort_values(by='Time',inplace=True)
    
  
    # Get GM trips info #
    #=================#
    print('Loading GM trip information')
    quory = 'SELECT * FROM "Trips"'
    cursor = conn.cursor()
    trips = pd.read_sql(quory, conn) 
    trips.reset_index(inplace=True, drop=True)   
    
    
    # Close connection
    #==============#
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
    
    # Save files
    #==============#
    if all_sensors:
        filename = '{0}/GM_db_meas_data_{1}_allsensors.csv'.format(out_dir, GM_TaskId)
    else:
        filename = '{0}/GM_db_meas_data_{1}.csv'.format(out_dir, GM_TaskId)
        
    #meas_data.to_csv(filename)
    meas_data.to_pickle(filename.replace('.csv','.pickle'))
    
    #meas_data.to_csv('{0}/GM_db_trips_info.csv'.format(out_dir, GM_TaskId))
    meas_data.to_pickle('{0}/GM_db_trips_info.pickle'.format(out_dir, GM_TaskId))
    
    return meas_data, trips
 
    
def get_trips_info(task_ids = None, only_GM = False):
    
    #conn = psycopg2.connect(database="postgres", user="mibaj", password="mibajLira123", host="liradb.compute.dtu.dk", port=5435) # regular, prod
    conn = psycopg2.connect(database="postgres", user="mibaj", password="Vm9jzgBH", host="liradbdev.compute.dtu.dk", port=5432) # dev
    # Quory
    #quory = 'SELECT * FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC LIMIT 1000'.format(GM_TaskId)
    if task_ids:
        task_ids=str(tuple(task_ids))
        quory = 'SELECT * FROM public."Trips" WHERE ("Trips"."Fully_Imported"=\'True\' AND "Trips"."TaskId" IN {0}) ORDER BY "TaskId" ASC'.format(task_ids)
    else:
        quory = 'SELECT * FROM public."Trips" WHERE "Trips"."Fully_Imported"=\'True\' ORDER BY "TaskId" ASC'
    
    # Set cursor
    cursor = conn.cursor()
    
    d = pd.read_sql(quory, conn, coerce_float = True) 
    d['Datetime']=pd.to_datetime(d['Created_Date'])
    
    if only_GM:
        d = d[d['TaskId']!=0]
    
    # Close the connection
    cursor.close()
    conn.close()  
    
    return d


def get_matching_DRD_info(GM_trip_id):
    
    DRD_info = {}
    if GM_trip_id==4955:
        DRD_trip_id = 'a34887d6-46df-496c-bb82-0c6b205fb199'
    elif GM_trip_id==4957:
        DRD_trip_id = 'a34887d6-46df-496c-bb82-0c6b205fb199'
    elif GM_trip_id==4959:
        DRD_trip_id = '468f5e4c-2977-4785-bea5-d1aca6923435'
    elif GM_trip_id==5683:
        DRD_info['NS'] = 'e9efaba7-a322-4793-a019-9592fb3ee73f' #NS
        DRD_info['SN'] =  'ef5eb740-198e-45c6-a66a-fcbdfb0016fe' #SN
    else:
        print('No DRD trip id set for this GM trip')
        sys.exit(0)

    return DRD_info

def get_matching_ARAN_info(GM_trip_id):
    ARAN_info = {}
    if GM_trip_id==5683:
        ARAN_info['V'] = '974a5c25-ee35-43c6-a2d5-2a486ec6ab0e' #from Brondby up the M3
        ARAN_info['H'] =  '538bd787-06f7-417b-a412-24b0d3caa594' 
    else:
        print('No ARAN trip id set for this GM trip')
        sys.exit(0)
    return ARAN_info

        
def get_GM_passes(GM_trip_id):
    print('Getting GM passes')
    passes = {}
    if GM_trip_id==5683:  
        # up to 136000: not useful
        passes['NS'] = [(136000, 270000)]
        passes['SN'] = [(285000, 457000)]
    else:
        print('No passes info found for this GM trip')
        sys.exit(0)
    return passes

def filter_keys(msg):
    remove_list= ['id', 'start_time_utc', 'end_time_utc','start_position_display',
                  'end_position_display','device','duration','distanceKm','tag', 
                  'personal', '@ts','@uid', '@t','obd.whl_trq_est', '@rec']
    msg = {k : v for k,v in msg.items() if k not in remove_list}
    return msg
 

def extract_string_column(sql_data, col_name = 'message'):
    # if json
    try: 
        sql_data[col_name] = sql_data[col_name].apply(lambda message: loads(message))
    except:
        pass
    keys = sql_data[col_name].iloc[0].keys()
    n_keys =  len(keys)
    for i, key in enumerate(keys):
        print('Key {0}/{1}'.format(i, n_keys))
        sql_data[key] = sql_data[col_name].apply(lambda col_data: col_data[key])
        
    sql_data.drop(columns=[col_name],inplace=True,axis=1)
    return sql_data
    
def check_nans(sql_data, is_aran = False, exclude_cols = []):   
    n_rows = sql_data.shape[0]
    for col in  sql_data.columns:
        if col in exclude_cols:
            continue
        n_nans = sql_data[col].isna().sum()
        n_left = n_rows - n_nans
        print('Number of nans in {0}: {1}/{2}, left: {3}/{2}'.format(col, n_nans, n_rows, n_left ))
    return

def load_GM_data(GM_TaskId, out_dir, all_sensors = False, add_sensors = [], load_nrows = -1):
    
    # Set up connection
     #==============#
    print("\nConnecting to PostgreSQL database to load GM")
    conn = psycopg2.connect(database="postgres", user="mibaj", password="Vm9jzgBH", host="liradbdev.compute.dtu.dk", port=5432)
   
    
    # Get measurements #
    #========================#
    print('Loading GM measurements from the db')
    #quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T"=\'track.pos\' OR "Measurements"."T"=\'acc.xyz\' OR "Measurements"."T"=\'obd.spd_veh\')) LIMIT 500'.format(GM_TaskId)
    if all_sensors:
        if load_nrows!=-1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC LIMIT {1}'.format(GM_TaskId, load_nrows)
        else:   
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\') ORDER BY "TS_or_Distance" ASC'.format(GM_TaskId)
    else:
        sensors = ['track.pos','acc.xyz','obd.spd_veh']
        if add_sensors:
            sensors = sensors + add_sensors
        sensors = str(tuple(sensors))
        print('Loading: ',sensors)
        #sensors = '(\'track.pos\', \'acc.xyz\')' # works
        #sensors = "('track.pos', 'acc.xyz')" #works
        if load_nrows!=-1:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC LIMIT {2}'.format(GM_TaskId, sensors, load_nrows)  
        else:
            quory = 'SELECT "TS_or_Distance", "T", "lat", "lon", "message" FROM ("Measurements" INNER JOIN "Trips" ON "Measurements"."FK_Trip"="Trips"."TripId") WHERE ("Trips"."TaskId"=\'{0}\' AND "Trips"."Fully_Imported"=\'True\' AND ("Measurements"."T" IN {1})) ORDER BY "TS_or_Distance" ASC'.format(GM_TaskId, sensors)     
  

     
    cursor = conn.cursor()
    meas_data = pd.read_sql(quory, conn, coerce_float = True)
    meas_data.reset_index(inplace=True, drop=True)   
    meas_data['Message'] = meas_data.message.apply(lambda msg: filter_keys(loads(msg)))
    meas_data.drop(columns=['message'],inplace=True,axis=1)
    meas_data.reset_index(inplace=True, drop=True)
    meas_data = meas_data[['TS_or_Distance','T', 'lat', 'lon','Message']]
    
    # Extract day and time
    #=================#
    meas_data['Date'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).date())
    meas_data['Time'] = meas_data.TS_or_Distance.apply(lambda a: pd.to_datetime(a).time())
    meas_data.sort_values(by='Time',inplace=True)
    
  
    # Get GM trips info #
    #=================#
    print('Loading GM trip information')
    quory = 'SELECT * FROM "Trips"'
    cursor = conn.cursor()
    trips = pd.read_sql(quory, conn) 
    trips.reset_index(inplace=True, drop=True)   
    
    
    # Close connection
    #==============#
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
    
    # Save files
    #==============#
    if all_sensors:
        filename = '{0}/GM_db_meas_data_{1}_allsensors.csv'.format(out_dir, GM_TaskId)
    else:
        filename = '{0}/GM_db_meas_data_{1}.csv'.format(out_dir, GM_TaskId)
        
    #meas_data.to_csv(filename)
    meas_data.to_pickle(filename.replace('.csv','.pickle'))
    
    #meas_data.to_csv('{0}/GM_db_trips_info.csv'.format(out_dir, GM_TaskId))
    meas_data.to_pickle('{0}/GM_db_trips_info.pickle'.format(out_dir, GM_TaskId))
    
    return meas_data, trips
 
        
def load_DRD_data(DRD_trip, conn_data, prod_db = True, p79 = False, aran = False, viafrik = False, dev_mode = False, load_n_rows = 500):
    '''
    
    Use this function to load and examin DRD data.
    
    Parameters
    ----------
    DRD_trip : string
        DRD trip id.
    conn_data: dictionary
        Database connection information.
    prod_db: BOOL
        Use production database. If False, will use development database. The default is True.
    p79 : BOOL, optional
        DESCRIPTION. The default is False.
    aran : BOOL, optional
        DESCRIPTION. The default is False.
    dev_mode: BOOL, optional
        The code will load load_n_rows lines. The default is False.
    load_n_rows: INT, optinal
        DESCRIPTION. The default is 500.
    Returns
    -------
    sql_data : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    trips : TYPE
        DESCRIPTION.

    '''
       
    # Set up connection
    print("\nConnecting to PostgreSQL database to load the DRD data")
    
    if prod_db:
        print("\nConnecting to production database")
        db_data = conn_data['prod']
    else:
        print("\nConnecting to development database")
        db_data = conn_data['dev']   
        
    db = db_data['database']
    username = db_data['user']
    password = db_data['password']
    host = db_data['host']
    port = db_data['port']

    # Connection    
    conn = psycopg2.connect(database=db, user=username, password=password, host=host, port=port)
    
    # Execute quory: get sensor data
    print('Selecting data for trip: {0}'.format(DRD_trip))
    if aran:
        quory = 'SELECT * FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    elif (p79 or viafrik):
        quory = 'SELECT "DRDMeasurementId","TS_or_Distance","T","lat","lon","message" FROM "DRDMeasurements" WHERE "FK_Trip"=\'{0}\' ORDER BY "TS_or_Distance" ASC'.format(DRD_trip)
    else:
        print('Set either p79 or aran or viafrik to True.')
        sys.exit(0)
    
    # Set the number of rows to load
    if dev_mode:
        quory = '{0} LIMIT {1}'.format(quory, load_n_rows)
        
    # Load and sort data
    cursor = conn.cursor()
    sql_data = pd.read_sql(quory, conn, coerce_float = True)

    # Sort also in pandas after conversion to float
    sql_data.TS_or_Distance = sql_data.TS_or_Distance.map(lambda raw: float(raw.replace(',','.')))
    sql_data['TS_or_Distance'] = sql_data['TS_or_Distance'].astype(float)
    sql_data.sort_values(by ='TS_or_Distance', inplace=True)
    sql_data.reset_index(drop = True, inplace=True)
    
    # Preparation depending on data type
    if aran:
        drop_cols = ['DRDMeasurementId', 'T', 'isComputed', 'FK_Trip', 'FK_MeasurementType', 'Created_Date',
       'Updated_Date','BeginChainage','EndChainage']
        extract_string_column(sql_data)
        sql_data.drop(drop_cols, axis=1, inplace = True)
        sql_data.replace(np.NaN, 0, inplace = True)
    elif viafrik:
        sql_data['Message'] = sql_data.message.apply(lambda msg: filter_keys(loads(msg), remove_gm=False))
        sql_data.drop(columns=['message'],inplace=True,axis=1)
        sql_data.reset_index(inplace=True, drop=True)
        sql_data = sql_data[['TS_or_Distance','T', 'lat', 'lon','Message']]
        sql_data = pd.concat([sql_data, extract_string_column(sql_data,'Message')],axis=1)
    elif p79:
        iri =  sql_data[sql_data['T']=='IRI']
        extract_string_column(iri)
        iri.dropna(subset=['IRI5','IRI21'],inplace=True)
        iri['IRI_mean'] = iri.apply(lambda row: (row['IRI5']+row['IRI21'])/2,axis=1)
        iri.drop(columns=['DRDMeasurementId', 'T','IRI5','IRI21'],inplace=True,axis=1)
        iri = iri[(iri.lat>0) & (iri.lon>0)]
        iri.reset_index(drop=True, inplace=True)
    
    # Get information about the trip
    print('Getting DRD trips information')
    quory = 'SELECT * FROM "Trips" WHERE "TaskId"=\'0\''
    cursor = conn.cursor()
    trips = pd.read_sql(quory, conn)         
        
    # Close connection
    if(conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")
        
    if p79:   
        return sql_data, iri, trips
    else:
        return sql_data, None, trips

        

def filter_latlon(data, col_string, lat_min, lat_max, lon_min, lon_max):
    data = data[data['lat'].between(lat_min,lat_max)]
    data = data[data['lon'].between(lon_min,lon_max)]
    data.reset_index(inplace=True, drop=True)
    return data


def select_platoon(filename):
    id = filename.split('/')[1].split('GMtrip-')[1].split('_DRD')[0]
    platoon_ids = ['4955','4957','4959']
    if id in platoon_ids:
        return True
    else:
        return False
    
def get_filenames(input_dir, mode, filetype='pkl'):
    # mode: accspeed_all
    
    all_filenames = glob.glob('{0}/*.{1}'.format(input_dir, filetype))
    print(all_filenames)
    
    if mode =='acc': #all files
        return all_filenames
    elif (mode=='speed' or mode=='accspeed'):
        return [filename for filename in all_filenames if 'accspeed' in filename]
    elif mode=='platoon_all':
        return list(filter(select_platoon, all_filenames))
    


def prepare_data(data, target_name = 'DRD_IRI_mean', bins = [0,2,5,50]):
    data['DRD_IRI_mean'] = data.apply(lambda x: (x['DRD_IRI5']+x['DRD_IRI21'])/2, axis=1)
    data['len']=data.GM_Time_segment.apply(lambda row:row.shape[0])
    data['GM_time_start'] = data.GM_Time_segment.apply(lambda t:t[0]) 
    data['GM_time_end'] = data.GM_Time_segment.apply(lambda t:t[-1]) 
    
    # Filter and classify
    data = data[data['len']<5000]
    data = set_class(data, target_name = target_name, bins = bins)
    data.reset_index(drop=True, inplace=True)
    
    return data

    
def get_segments_p79(df, window_size, step, is_p79 = True):
    # Columns
    cols_to_take_ends = ['lat_map', 'lon_map','TS_or_Distance']
    cols_to_average = ['IRI_mean']
    #other_cols = list( set(df.columns).difference( set(cols_to_take_ends+cols_to_average)) )
    other_cols = ['IRI_mean']
    
    # Prepare df window
    final_cols = [col + '_start' for col in cols_to_take_ends] + [col + '_end' for col in cols_to_take_ends] + cols_to_average + ['IRI_sequence'] #remove
    df_window = pd.DataFrame(columns =  final_cols)
        
    # Get df windows
    df.reset_index(inplace=True, drop=True)
    n_samples  = df.shape[0]
    d_max =  int(df['TS_or_Distance'].iloc[-1])
    
    row_i = 0
    for d_start in range(0, d_max, step):
        d_end = d_start + window_size
        #print(d_start, d_end)
        df_this = df[df['TS_or_Distance'].between(d_start,d_end)]
        
        #print(df_this.shape)
        
        if df_this.shape[0]<1:
            continue
        
        for col in cols_to_take_ends:
            df_window.at[row_i, col+'_start'] = df_this[col].to_numpy()[0]
            df_window.at[row_i, col+'_end'] = df_this[col].to_numpy()[-1]
            
        # Average
        for col in cols_to_average:
            df_window.at[row_i, col] = df_this[col][:-1].mean() 
                
        # Take whole array and also sum
        for col in other_cols:
            #df_this[col] = df_this[col].fillna(0)
            if col=='IRI_mean':
                col_sequence = 'IRI_sequence'
            else:
                 col_sequence = col
            a = df_this[col][:-1].to_numpy() 
            df_window.at[row_i, col_sequence] = a 
            #s.at[row,col] = data_this_seg[col].sum()

        row_i = row_i + 1
       
    # Clean and reset the window dataframe
    df_window.dropna(inplace=True)
    df_window.reset_index(drop=True, inplace=True)
    
    return df_window


def get_segments_aran(data, window_size = 10, step = 10, is_aran = False, is_aran_sel = False, is_combined = False):
    
    # Take start and end or mean for those columns
    if is_aran:
        cols_to_take_ends = ['TS_or_Distance', 'lat', 'lon', 'lat_map', 'lon_map']
        ignore_cols = ['BeginDistanceStamp','EndDistanceStamp']
        cols_to_average = []
        other_cols = list( set(data.columns).difference( set(cols_to_take_ends+ignore_cols)) )
        ts_column = 'TS_or_Distance'
    elif is_aran_sel:
        cols_to_take_ends = ['TS_or_Distance_start', 'TS_or_Distance_end', 'lat_start', 'lon_start',
       'lat_end', 'lon_end', 'lat_map_start', 'lon_map_start', 'lat_map_end',
       'lon_map_end', 'street_name_start', 'street_name_end']
        ignore_cols = []
        other_cols = []
        cols_to_average = list( set(data.columns).difference( set(cols_to_take_ends+ignore_cols+ ignore_cols+other_cols)) )
        ts_column = 'TS_or_Distance_start' 
    elif is_combined:
        data['Chainage (km)'] = data['Chainage (km)'].apply(lambda row: (row+0.005)*1000)
        cols_to_take_ends = ['Chainage (km)', 'Latitude From (rad)','Longitude From (rad)', 'Latitude To (rad)','Longitude To (rad)']
        ignore_cols = []
        cols_to_average = ['RUT','IRI','CRACK INDEX','Potholes']
        other_cols = ['RUT','IRI','CRACK INDEX','Potholes']
        ts_column = 'Chainage (km)'
    else:
        cols_to_take_ends = []
        ignore_cols = []
        cols_to_average = []
        cols_to_average = []
        other_cols = list( set(data.columns).difference( set(cols_to_take_ends+ignore_cols+cols_to_average)) )
    
    # Final df
    final_cols = [col + '_start' for col in cols_to_take_ends] + [col + '_end' for col in cols_to_take_ends] + [col + '_mean' for col in cols_to_average]+ other_cols +[col+'_seg' for col in other_cols]
    s = pd.DataFrame(columns=final_cols)
    
    # First and last TS_or_Distance divisable by window_size
    i=0
    first = data[ts_column].iloc[0]
    while first%step!=0:
        first = data[ ts_column].iloc[i]
        i = i+1
    
    i=-1
    last = data[ts_column].iloc[-1]
    while last%window_size!=0:
        last = data[ ts_column].iloc[i]
        i = i-1
        
        
    # Make segments
    segments = []
    row=0
    start_ts = first
    while start_ts < last:
        end_ts = start_ts + window_size 
            
        # Print
        #print(start_ts, end_ts)
        
        if start_ts%500==0:
            print(start_ts, end_ts)
        if is_aran_sel==True:
            c1 = data[ts_column] >= start_ts
            c2 = data[ts_column.replace('_start','_end')] <= end_ts
            data_this_seg = data[c1 & c2]
        else:
            data_this_seg = data[data[ts_column].between(start_ts, end_ts)]
        
        # Start, end
        for col in cols_to_take_ends:  
            s.at[row, col+'_start'] = data_this_seg[col].iloc[0]
            s.at[row, col+'_end'] = data_this_seg[col].iloc[-1]
        
        # Means
        for col in cols_to_average:
            #print('mean for: ',col,data_this_seg[col].mean())
            s.at[row,col+'_mean'] = data_this_seg[col].mean()
            
            
        # Sums
        #for col in other_cols:
        #    s.at[row,col] = data_this_seg[col].sum()
            
        # Take whole array and also sum
        for col in other_cols:
            data_this_seg[col] = data_this_seg[col].fillna(0)
            s.at[row,col+'_seg'] = data_this_seg[col][:-1].to_numpy()
            s.at[row,col] = data_this_seg[col][:-1].sum()
        
        row = row+1
        start_ts = start_ts + step

    return s

  
