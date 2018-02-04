import pandas as pd
import bench_utils as utils
import logging
import classifier_benchmark as classify
import numpy as np
import logging
import datetime

dt = datetime.datetime.now().strftime("%Y%m%d_%H_%M")

# Display progress logs on stdout
logging.basicConfig(filename='exp_results_v3_%s.log'%(dt),
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logging.debug('Experiment Date & Time: %s' % (dt))

filepath = "/home/xelburn/Dropbox/Citi and SGUL_shared folder/data_staging/all_by_baby_enriched_v3.csv"

# Data-related Parameters
label = "outcome"
temporary = False
train_ratio = 0.8
num_strategy = 'median'
cat_strategy = 'explicit'
# util_feats = ["idx", "baby_id", "debug_comments", "debug_filename_xs1",
#               "debug_filename_xs2", "dem_edd", "dem_dob", "dem_edd_lmp", "dem_edd_us",
#               "dem_episode_lmp", "dem_height_cm", "dem_weight_kg", "filename",
#               "outcome_date", "outcome_died_on", "patients_id", "t1_date_of_exam",
#               "t1_msb_date", "t2_date_of_exam", "t3_date_of_exam"]
util_feats = [ "idx", "debug_comments", "debug_ecd", "debug_patients_id", "debug_t1_mat_age_at_exam",
               "debug_t2_mat_age_at_exam", "debug_t3_mat_age_at_exam", "dem_edd", "dem_edd_lmp",
               "dem_edd_us", "dem_episode_lmp", "dem_height_cm", "dem_weight_kg", 'dem_dob',
               "outcome_date", "outcome_died_on", "t1_date_of_exam", "t1_msb_date", "t2_date_of_exam",
               "t3_date_of_exam"]
ignore_feats =["dem_conception", "dem_ethnic_group", "t1_l_uterine_a_ri",
               "t1_msb_plgf_pgml", "t1_pappa", "t1_r_uterine_a_ri", "t2_l_uterine_a_ri",
               "t2_r_uterine_a_ri", "t3_fh4", "t3_l_uterine_a_ri", "t3_r_uterine_a_ri",
               "t3_fetus_mid_cerebral_a_pi", "t3_fetus_umbilical_a_pi" ]
NLP = False
X_VALID = None;
PLOT = False
USE_HASHING = False


if __name__ == '__main__':

    # Read in train and test data separately
    if temporary:
        ## Temporary, just load the iris datase
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer()
        # Fake categorical columns
        cat_dat = pd.DataFrame(data = np.stack([np.random.choice(['Black', 'White'], size=len(bc.data)),
                                                np.random.choice(['A', 'B', 'AB', '0'], size=len(bc.data))], axis=1),
                               columns= ['cat_col_1', 'cat_col_2'])
        data = pd.concat([pd.DataFrame(bc.data), cat_dat,pd.DataFrame(bc.target) ],
                        axis=1, ignore_index=True)
        data.columns = bc.feature_names.tolist() + ['cat_col_1', 'cat_col_2']+ ['label']

        ## End temporary data load, to be replaced with a preprocessing function
    else:
        # Load and massage the data
        data = pd.read_csv(filepath)
        data = utils.tidy_data(data)
        data = utils.drop_irrelevant_cols(data, set(util_feats+ignore_feats))
        data = utils.process_outcome(data, label) # drop rows with no outcome
    shuffled_data = data.sample(frac=1)  # shuffle rows

    #with open(TRAIN_PATH, encoding="latin-1") as trainfile:
    train_data = shuffled_data.head(int(len(data)*train_ratio))
    #with open(TEST_PATH, encoding="latin-1") as testfile:
    test_data = shuffled_data.tail(int(len(data)*(1-train_ratio)))

    # Impute data
    train_data, test_data = utils.fill_missingness(train_data, test_data, label, num_strategy, cat_strategy)

    data_dict = utils.load_normal_data(train_data, test_data, label, X_VALID)

    perf_list = []
    for i, fold_dict in data_dict.items():

        # extract training data and other objects for fold i
        X_train = fold_dict['X_train']
        X_test = fold_dict['X_test']
        y_train = fold_dict['y_train']
        y_test = fold_dict['y_test']
        label_priors = fold_dict['label_priors']
        x_scaler = fold_dict['x_scaler']
        x_encoders = fold_dict['x_encoders']
        y_encoder = fold_dict['y_encoder']

        # Get feature names
        if NLP:
            # mapping from integer feature name to original token string
            vectorizer = fold_dict['vectorizer']
            if USE_HASHING:
                feature_names = None
            else:
                feature_names = vectorizer.get_feature_names()

            if feature_names:
                feature_names = np.asarray(feature_names)
        else:
            feature_names = fold_dict['feature_names']

        # Benchmark classifiers
        results = []
        for clf, name in classify.CLASSIFIER_LIST_FULL:
            logging.debug('=' * 80)
            logging.debug(name)
            results.append(classify.benchmark(clf, name, label_priors, X_train, y_train,
                                     X_test, y_test, feature_names, y_encoder.classes_))

        # Benchmark ensemble
        ensembler = classify.Ensembler([r['classifier'] for r in results])
        results.append(classify.benchmark(ensembler, 'Ensembler', label_priors, X_train, y_train,
                                 X_test, y_test, feature_names, y_encoder.classes_))

        # Evaluate Results
        perf_df = pd.concat([r['performance'] for r in results], axis=1).transpose()
        perf_df['fold'] = i
        perf_list.append(perf_df)

        # Plot results
        if PLOT:
            classify.plot_results(results)

    # Take Descriptive stats of X-validation results (if applicable)
    if X_VALID is not None:
        all_perf = pd.concat(perf_list)
        all_perf['recall'] = all_perf['recall'].apply(lambda x: x[1])  # hard-coded for the binary case
        all_perf['precision'] = all_perf['precision'].apply(lambda x: x[1])  # hard-coded for the binary case
        all_perf['prior'] = all_perf['prior'].apply(lambda x: x[1])  # hard-coded for the binary case
        all_perf[classify.PERFORMANCE_COLS] = all_perf[classify.PERFORMANCE_COLS].apply(pd.to_numeric, axis=1)

        mean_perf = all_perf.groupby('classifier_name').mean()
        std_perf = all_perf.groupby('classifier_name').std()

        print('%i fold x-validation, performance results:' %(X_VALID))
        print(mean_perf)
        print(std_perf)
    else:
        all_perf = perf_df
        print('No x-validation, performance results:')
        print(all_perf)