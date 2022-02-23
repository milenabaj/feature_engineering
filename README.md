This package performs feature extraction and selection together.

* Feature extraction:  
In the feature extraction, a set of features is computed per each input sensor. By default, used are vertical acceleration and speed sensors. If you want to use 3D acceleration and speed, pass --use_3dacc. If you want to use all additional sensors on top, pass --use_3dacc and --use_add_sensors. The feature extraction process is time consuming and does not depend on the target - hence do not recreate it for different targets. 

* Feature selection:
Feature selection will load extracted features and find optimal subset to model the chosen target. Hence, the output will be in new subdirectories containing the target name. For different targets, pass the target name --target <name>. As a target, choose: IRI_mean, KPI or DI.

* Run as:  
python -i get_features.py --in_dir /dtu-compute/lira/ml_data/data --target IRI_mean --load_add_sensors --p79 --aran  

* To recreate the feature selection step only, run:  
python -i get_features.py --in_dir /dtu-compute/lira/ml_data/data --target IRI_mean --load_add_sensors --p79 --aran --recreate_fs

* The output files will be in:
  
  * Feature extraction output files:  
    /dtu-compute/lira/ml_data/data/aligned_fe_fs_GM_P79_ARAN_data_window-100-step-10  (then look into the -route and sensor- subdirectory)
    
    For example, the output with the default [M3_VH, M3_HH] routes and default input [acceleration, speed] sensors will be in:      
    /dtu-compute/lira/ml_data/data/aligned_fe_fs_GM_P79_ARAN_data_window-100-step-10/M3_VH_M3_HH_filter_speed_accspeed/  

  * Feature selection output files:  
    This will be created in a new subdirectory. In the previous example:  
     /dtu-compute/lira/ml_data/data/ aligned_fe_fs_GM_P79_ARAN_data_window-100-step-10/M3_VH_M3_HH_filter_speed_accspeed/feature_selection_<target_name>
    
