from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from keras.utils import np_utils
import pandas as pd
import numpy as np
import logging

class explicit_imputer():

    def __init__(self):
        pass

    def transform(self, data):

        if not isinstance(data, pd.DataFrame):
            raise Exception("Input to explicit imputer has to be a pandas df")
        data_out = data.fillna('None')

        return data_out

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y) # convert label vector to dummy binaries matrix
    return y, encoder

def drop_irrelevant_cols(data, cols_to_drop):
    """
    Drop the cols_to_drop from the input data
    :param data: pd.DataFrame
    :param cols_to_drop: list
    :return: pd.DataFrame, reduced dataframe
    """
    reduced_data = data.drop(cols_to_drop, axis=1)

    return reduced_data

def tidy_data(X):
    """
    Calculate the additinoal fields, based on the raw fields in the dataset.

    :param X: pd.DataFrame, input dataset
    :return: X
    """
    if not isinstance(X, pd.DataFrame):
        raise Exception("Input to derive_data() has to be a pandas df")

    # Calculate Age
    if 'dem_mat_age' not in X.columns:
        if ('dem_dob' in X.columns) and ('t1_date_of_exam' in X.columns):
            X['t1_date_of_exam'] = pd.to_datetime(X['t1_date_of_exam'], format='%Y-%m-%d')
            X['dem_dob'] = pd.to_datetime(X['dem_dob'], format='%Y-%m-%d')
            X['dem_age'] = (X['t1_date_of_exam'] - X['dem_dob'])/ np.timedelta64(1, 'Y')
            X['dem_age'] = X['dem_age'].astype('int')

    # Number of vessels in the cord
    if 't2_cord' in X.columns:
        valid_cord_vals = ['3 vessels', '2 vessels', 'abnormal']
        X['t2_cord']= X['t2_cord'].apply(lambda x: x if x in valid_cord_vals else 'None')

    # Treat alcohol field
    if 'dem_alcohol' in X.columns:
        X['dem_alcohol'] = X['dem_alcohol'].apply(lambda x: 'None' if x=='n/k' else x)

    # Treat prior pregnancies field
    if 'dem_para' in X.columns:
        X['dem_para'] = X['dem_para'].apply(lambda x: x if x<=9 else np.nan)

    # Treat msb manufacturer field
    if 't1_msb_manufacturer' in X.columns:
        X['t1_msb_manufacturer'] = X['t1_msb_manufacturer'].apply(lambda x: 'Kryptor compact (Brahms)' if x == 'Kryptor (Brahms)' else x)

    # Treat cord field
    if 't2_cord' in X.columns:
        X['t2_cord'] = X['t2_cord'].apply(lambda x: 'None' if x == 'not examined' else x)

    # Map dem field to 4 categories
    if 'dem_ethnic_group2' in X.columns:
        # Fill all NA as "None"
        X['dem_ethnic_group2'].fillna('None', inplace=True)
        dem_mapping = { "White": "Caucasian",
                        "Black": "Afro-Caribbean",
                        "South Asian": "Asian",
                        "White-Black": "Afro-Caribbean",
                        "East Asian (Oriental)": "Asian",
                        "East Asian": "Asian",
                        "White-South Asian": "Asian",
                        "White-East Asian": "Asian",
                        "South Asian-East Asian": "Asian",
                        "Black-South Asian": "Asian",
                        "Other": "Other",
                        "Black-East Asian": "Asian",
                        "None": "None",
                        "Mixed": "Other"}
        X['dem_ethnic_group2'] = X['dem_ethnic_group2'].apply(lambda x: dem_mapping[x])

    return X


def process_outcome(data, label):
    """
    Organise the outcome field to binary by dropping missing vals and
    mapping the free text into 0 or 1
    :param data: pd.DataFrame the input data with the label column
    :param label: str, the name of the outcome field
    :return:
    """

    # drop rows with missing outcome
    data.dropna(subset=[label], axis=0, inplace=True)

    # Map the categories to binary (0/1) outcome
    outcome_mapping = { 'Live Birth':1,
                        'Live birth':1,
                        'Termination':0,
                        'IUD':0,
                        'Stillbirth':0,
                        'NND >1 Week postpartal':0,
                        'NND <1 Week postpartal': 0,
                        'Neonatal death':0,
                        'Miscarriage':0,
                        'Lost to follow-up':1,
                        'Stillbirth or LB?':1,
                        'Live Birth??':1,
                        'No followup':1,
                        'Ongoing':1}
    data[label] = data[label].apply(lambda x: outcome_mapping[x])

    return data


def calc_missingness_ratio(X):
    """
    Simply calculate the missingness percentages by column name
    :param X: pd.DataFrame input data
    :return missingness_by_col_name: pd.Series, missingness by column name
    """
    missingness_by_col_name = X.isnull().sum() / len(X)

    return missingness_by_col_name

def impute_data(data, num_imputer, num_strategy, cat_imputer, cat_strategy, label):
    """
    Imptue the input data (train or test) creating new imputers or using the supplied
    :param data: pd.DataFrame, inputy train or test data
    :param num_imputer: imputer class for numeric variables
    :param num_strategy: str, method to impute for numeric features
    :param cat_imputer: imputer class for categorical variables
    :param cat_strategy: str, method to impute for categorical features
    :return: X_out, pd.DataFrame imputed data
    """
    X = data.drop(label, axis=1)  # exclude the label from imputing
    X_label = data[[label]]

    # Get the numeric and categorical column names to treat them differently
    numeric_cols = X.columns[[dt != 'object' for dt in X.dtypes]]
    cat_cols = X.columns[[dt == 'object' for dt in X.dtypes]]

    if num_imputer is None:
        if num_strategy == 'median':  # 1- Median-impute Numeric features
            num_imputer = Imputer(strategy='median')
        else:
            raise Exception("No other method for num impute implemented yet - 29 Dec 17")
        num_imputer.fit(X[numeric_cols])
    X_num = pd.DataFrame(num_imputer.transform(X[numeric_cols]), columns=numeric_cols,
                         index = data.index)

    # 2- Explicit impute Categorical features
    if cat_imputer is None:
        if cat_strategy == "explicit":
            cat_imputer = explicit_imputer()
        else:
            raise Exception("No other method for cat impute implemented yet - 29 Dec 17")
    X_cat = pd.DataFrame(cat_imputer.transform(X[cat_cols]), columns=cat_cols,
                         index=data.index)

    # 3- Concatenate Numeric and one-hot-encoded categorical cols
    X_out = pd.concat([X_num, X_cat, X_label], axis=1,)

    assert(X_out.isnull().sum().sum() == 0)

    return X_out, num_imputer, cat_imputer


def fill_missingness(train_data, test_data, label, num_strategy, cat_strategy):
    """
    Concatenate train_data and test data and treat missingness in one block
    :param train_data: pd.DataFrame
    :param test_data: pd.DataFrame
    :param label: str, the name of the outcome field
    :param num_strategy: str, method to impute for numeric features
    :param cat_strategy: str, method to impute for categorical features
    :return: X_out: pd.DataFrame, imputed dataset
    """
    if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
        raise Exception("Input ot preprocess_data() has to be a pandas df")

    # Print the missingness on the whoel dataset
    logging.debug('MISSINGNESS PERCENTAGES')
    logging.debug(calc_missingness_ratio(pd.concat([train_data, test_data])))

    # fit new imputers on training data
    train_data, train_num_imputer, train_cat_imputer = impute_data(data=train_data, num_imputer=None,
                                                                   num_strategy=num_strategy, cat_imputer=None,
                                                                   cat_strategy=cat_strategy, label=label)
    # use the imputers fitted ont he train data on the test data
    test_data, _, _ = impute_data(data=test_data, num_imputer=train_num_imputer,
                                   num_strategy=num_strategy, cat_imputer=train_cat_imputer,
                                   cat_strategy=cat_strategy, label=label)

    return train_data, test_data


def prepare_x_validate_data(train_data, test_data, x_validate):
    """
    Take the train and test data separately, concatenate, shuffle and
    output (train dataset, test dataset) tuples for each fold.
    :param train_data: pd.DataFrame for the training data with labels
    :param test_data: pd.DataFrame for the test data with labels
    :param x_validate: number of folds in cross-validation
    :return: a list of (train,test) tuples
    """
    data = pd.concat([train_data, test_data], ignore_index=True, axis=0)
    x_val_mod = np.mod(len(data), x_validate)
    perm_idxs = np.random.permutation(len(data))
    xval_idxs = perm_idxs[:-x_val_mod].reshape(x_validate, int(len(perm_idxs) / x_validate))
    # train and test tuples for each x-fold
    data_list = [(data.iloc[~data.index.isin(xval_idxs[i, :]), :],
                  data.iloc[data.index.isin(xval_idxs[i, :]), :]) for i in range(x_validate)]

    return data_list

def load_normal_data(train_data, test_data, label, x_validate):
    """
    Load normal (non-NLP) dataset
    :param train_data: pd.DataFrame that holds train input and label
    :param test_data: pd.DataFrame that holds test input and label#
    :param label: str, the name of the outcome field
    :param x_validate: number of x-validation if any
    :return:
    """
    # if x-validation prepare N folds, otherwise just one fold
    if x_validate is not None:
        data_list = prepare_x_validate_data(train_data, test_data, x_validate)
    else:
        data_list = [(train_data, test_data)]

    data_dict = dict()
    for i in range(len(data_list)):

        # The train and test data
        train_data, test_data = data_list[i]

        # Separate label column from inputs
        X_train = train_data.loc[:, [col != label for col in train_data.columns]]
        y_train = train_data[label]
        X_test = test_data.loc[:, [col != label for col in test_data.columns]]
        y_test = test_data[label]

        # calculate label priors
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        label_priors = label_counts / len(y_train)

        # Standard scale and encode X and y
        X_train, x_scaler, x_encoders = preprocess_data(X_train)  # fit scaler based on train set
        y_train, y_encoder = preprocess_labels(y_train, categorical=False)  # Encoder on entire label set
        X_test, _, _ = preprocess_data(X_test, x_scaler, x_encoders)  # normalise using the train set's
        y_test, _ = preprocess_labels(y_test, y_encoder, categorical=False)

        data_dict[i] = {'X_train': X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test,
                        'label_priors':label_priors, 'x_scaler':x_scaler, 'x_encoders':x_encoders,
                        'y_encoder': y_encoder, 'feature_names':train_data.columns}

    return data_dict


def preprocess_data(X, scaler=None, encoders=None):
    '''
    X should be a pandas df since we will need numeric and categorical column names.
    Reads in a raw training data, standard scales the columns that are numeric, and
    label encodes and then one-hot-encodes the fields that are categorical.
    StandardScaler can scale multip columns, while Label Encoder can only encode a single
    column. Therefore, an "encoders" dictionary is required to cover all categorical cols. Following
    label-encoding, the categorical columns are one-hot-encoded into separate dummy binary columns.
    '''
    if not isinstance(X, pd.DataFrame):
        raise Exception("Input to preprocess_data() has to be a pandas df")

    # Get the numeric and categorical column names to treat them differently
    numeric_cols = X.columns[[dt != 'object' for dt in X.dtypes]]
    cat_cols = X.columns[[dt == 'object' for dt in X.dtypes]]

    # 1- Standard scale Numeric features
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X[numeric_cols])
    X_num= pd.DataFrame(scaler.transform(X[numeric_cols]),columns=numeric_cols)

    # 2- Label encode Categorical features
    X_cat_list = list()
    if not encoders:
        encoders= dict()
        for col in cat_cols:
            encoders[col] = LabelEncoder()
            encoders[col].fit(X[col])
    for col in cat_cols:
        col_cat = encoders[col].transform(X[col]).astype(np.int32)
        X_cat_list.append(pd.DataFrame(np_utils.to_categorical(col_cat),
                          columns= [col + c for c in encoders[col].classes_]) )
        X_oh_cat = pd.concat(X_cat_list, axis=1)

    # 3- Concatenate Numeric and one-hot-encoded categorical cols
    X_out = pd.concat([X_num, X_oh_cat], axis=1)

    return X_out, scaler, encoders

def load_nlp_data(train_data, test_data, use_hashing, n_features, x_validate):
    """
    Generic load data method for nlp problems, using HashingVectorizer or
    TFifd Vectoriser
    :param train_data: pd.DataFrame for the training data with labels
    :param test_data: pd.DataFrame for the test data with labels
    :param use_hashing: boolean, if True use HashingVectorizer, else use TfidfVectorizer
    :param n_features: int, number of top features to print
    :param x_validate: int number of folds in cross-validation
    :return:
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    # if x-validation
    if x_validate is not None:
        data_list = prepare_x_validate_data(train_data, test_data, x_validate)
    else:
        data_list = [(train_data, test_data)]

    data_dict = dict()
    for i in range(len(data_list)):

        train_data, test_data = data_list[i]

        y_train = train_data['label'].values
        y_test = test_data['label'].values

        # calculate label priors
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        label_priors = label_counts / len(y_train)

        print('data loaded')

        print("Extracting features from the training data using a sparse vectorizer")
        if use_hashing:
            vectorizer = HashingVectorizer(stop_words='english', n_features=n_features)
            X_train = vectorizer.transform(train_data.subject.values)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
            X_train = vectorizer.fit_transform(train_data.subject.values)

        print("n_samples: %d, n_features: %d" % X_train.shape)
        print()

        print("Extracting features from the test data using the same vectorizer")

        X_test = vectorizer.transform(test_data.subject.values)

        print("n_samples: %d, n_features: %d" % X_test.shape)
        print()

        data_dict[i] = {'X_train': X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test,
                        'label_priors':label_priors, 'vectorizer':vectorizer}

    return data_dict