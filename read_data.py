import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False) # widen num cols displayed
import os
import pickle


def dump_pickle(dictionary):
    with open('trimester.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

go_pickle = False
centre1_folder = "/home/xelburn/Dropbox/Citi and SGUL_shared folder/Centre1"
centre2_folder = "/home/xelburn/Dropbox/Citi and SGUL_shared folder/Centre2"
centre_1_files = ["PREST1_v2.xlsx", "PREST2_v2.xlsx", "PREST3_v2.xlsx"]
merge_key = 'Patients ID'

# TO DO: make all column names small_like_this

try:
    with open('trimester.pickle', 'rb') as handle:
        centre1_trims = pickle.load(handle)
except:

    # read in all trimesters and rename columns
    centre1_trims = dict()
    overlapping_patient_ids = set()
    for i in range(1,4):

        centre1_trims[i] = pd.read_excel(os.path.join(centre1_folder,
                                                    centre_1_files[i-1]) )
        trim_cols= [c+'_t'+str(i) for c in centre1_trims[i].columns if c !=merge_key]
        trim_cols.insert(0, merge_key)

        centre1_trims[i].columns = trim_cols

# Patient ID's for whom we have all 3 trim readings
all_trim_patients =set(centre1_trims[1].loc[:,merge_key].values).intersection(
                       centre1_trims[2].loc[:,merge_key].values).intersection(
                       centre1_trims[3].loc[:,merge_key].values)

print ( 'Trimester file # rows:' + str({k:len(v) for k,v in centre1_trims.items() } ) )
print("We have all 3 trim readings for %i patients"%(len(all_trim_patients)))

# TO DO: df.groupby('Patients ID').sort('date of exam_t1')

# join all trimesters on "Patients ID"
centre1_trims12 = pd.merge(left=centre1_trims[1], right=centre1_trims[2],
                           on="Patients ID", how = 'inner')
centre1_all = pd.merge(left=centre1_trims12, right=centre1_trims[3],
                         on="Patients ID", how='outer' )

#centre1_all.to_csv(os.path.join(centre1_folder,'all_trims.csv') )

# Percentage of missingness by feature name for Centre 1 Trimester 1
missingness_by_col_name = centre1_trims[1].isnull().sum() / len(centre1_trims[1])
missingness_by_col_name.to_csv(os.path.join(centre1_folder,'all_trims_nan_percentages.csv'))
