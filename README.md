This package does feature extraction and selection. 
* Run as:  
python -i get_features.py --in_dir /dtu-compute/lira/ml_data/data --target IRI_mean --load_add_sensors --p79 --aran  
As a target, choose: IRI_mean, KPI or DI.

* To recreate feature selection, run:  
python -i get_features.py --in_dir /dtu-compute/lira/ml_data/data --target IRI_mean --load_add_sensors --p79 --aran --recreate_fs

* The output files will be in:  
/dtu-compute/lira/ml_data/data/aligned_fe_fs_GM_P79_ARAN_data_window-100-step-10
