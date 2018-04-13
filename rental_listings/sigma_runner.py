import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn import model_selection
from enum import Enum
from feature_engineering_runner import operate_on_coordinates, perform_general_feature_engineering, add_cluster_column
import xgboost as xgb
import tqdm
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split, KFold
from sklearn.calibration import CalibratedClassifierCV


LOAD_STACK = True

class Objective(Enum):
    SUBM_XGB = 1
    GS_XGB = 2
    GS_RFC = 3
    GS_AB = 4
    CALL_RFC = 5
    CV_XGB = 6
    STACK = 7
    GS_FULL = 8
    TSNE = 9
    WEAK_CLF = 10
    CV_RFC = 11
    SUBM_RFC = 12
    STACK_SUBM = 13
    SVM = 14

def read_train_test(path='./'):
    return 0, 1
    data_path = "./"
    train_file = data_path + "train_tuned.json"
    test_file = data_path + "test_tuned.json"
    train_df = pd.read_json(train_file)
    test_df = pd.read_json(test_file)
    return train_df, test_df


def perform_fe(train_df, test_df):
    train_df, test_df = operate_on_coordinates(train_df, test_df)
    train_df, test_df = perform_general_feature_engineering(train_df, test_df)
    #train_df, test_df = add_cluster_column(train_df, test_df, 40)
    return train_df, test_df

def scale(train_df, test_df):

    dimensions_to_scale = ['price','log_price', 'bathrooms','bedrooms', 'longitude', 'latitude','num_rho', 'num_phi','num_rot45_X','num_rot45_Y','num_rot30_X','num_rot30_Y','num_rot15_X','num_rot15_Y','num_rot60_X','num_rot60_Y']

    mapper = DataFrameMapper([(dim, RobustScaler()) for dim in dimensions_to_scale])
    tmp = pd.concat([train_df[dimensions_to_scale], test_df[dimensions_to_scale]])


    tmp_mapped = mapper.fit_transform(tmp)
    train_n = tmp_mapped[:train_df.shape[0]]
    test_n = tmp_mapped[-test_df.shape[0]:]


    for ind, el in enumerate(dimensions_to_scale):
        train_df[el] = train_n[:,ind]
        test_df[el] = test_n[:, ind]
    return train_df, test_df



def transform_categorical(train_df, test_df):
    categorical = ['listing_id', "display_address", "building_id",'manager_id', "street_address"]
    for f in categorical:
        if train_df[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
    return train_df, test_df

def include_features(train_df, test_df, features_to_use):

    train_df['filtered_features'] = train_df["filtered_features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    test_df['filtered_features'] = test_df["filtered_features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

    tfidf = CountVectorizer(stop_words='english', max_features=200)
    tr_sparse = pd.DataFrame(tfidf.fit_transform(train_df["filtered_features"]).toarray(), index=train_df.index)
    te_sparse = pd.DataFrame(tfidf.transform(test_df["filtered_features"]).toarray(), index= test_df.index)


    train_X = pd.concat([train_df[features_to_use], tr_sparse], axis=1)
    test_X = pd.concat([test_df[features_to_use], te_sparse], axis=1)

    return train_X, test_X


def get_oof(clf, ntrain, ntest, x_train, y_train, name, x_test, SEED=0, n_classes = 3, NFOLDS = 5):

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    oof_train = np.zeros((ntrain,n_classes))
    oof_test = np.zeros((ntest,n_classes))
    oof_test_skf = np.zeros((ntest, NFOLDS*n_classes))

    for i, (train_index, test_index) in tqdm.tqdm(enumerate(kf), desc = name):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict_proba(x_te)
        oof_test_skf[:,3*i:3*i+3] = clf.predict_proba(x_test)
        print(oof_test_skf[:,3*i:3*i+3])

    for i in range(3):

        oof_test[:,i] = (oof_test_skf[:,i]+oof_test_skf[:,i+3]+oof_test_skf[:,i+6]+oof_test_skf[:,i+9]+oof_test_skf[:,i+12])/5
    return oof_train, oof_test

lightgbm_params = {
    'boosting_type': 'gbdt', 'objective': 'multiclass', 'nthread': -1, 'silent': True,
    'num_leaves': 2**4, 'learning_rate': 0.05, 'max_depth': -1,
    'max_bin': 255, 'subsample_for_bin': 50000,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.6, 'reg_alpha': 1, 'reg_lambda': 0,
    'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight': 1, 'num_class':3}


def get_oof_lightgbm(ntrain, ntest, x_train, y_train, name, x_test, SEED=0, n_classes = 3, NFOLDS = 5):

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    oof_train = np.zeros((ntrain,n_classes))
    oof_test = np.zeros((ntest,n_classes))
    oof_test_skf = np.zeros((ntest, NFOLDS*n_classes))

    for i, (train_index, test_index) in tqdm(enumerate(kf), desc = name):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        dset_oof = lgbm.Dataset(x_tr, y_tr, silent=True)


        bst = lgbm.train(lightgbm_params, dset_oof, 1250)
      
        oof_train[test_index] = bst.predict(x_te)

        oof_test_skf[:,3*i:3*i+3] = bst.predict(x_test)
        print(oof_test_skf[:,3*i:3*i+3])

    for i in range(3):

        oof_test[:,i] = (oof_test_skf[:,i]+oof_test_skf[:,i+3]+oof_test_skf[:,i+6]+oof_test_skf[:,i+9]+oof_test_skf[:,i+12])/5
    return oof_train, oof_test



if __name__ == '__main__':

    objectives_without_nan = [Objective.GS_RFC, Objective.GS_AB, Objective.CALL_RFC]
    run_objective = Objective.STACK_SUBM
    train_df, test_df = read_train_test()
    print(train_df.columns)
    train_df, test_df = perform_fe(train_df, test_df)
    print(train_df.columns)

    train_df, test_df = scale(train_df, test_df)
    train_df, test_df = transform_categorical(train_df, test_df)


    features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price", 'log_price', "num_rho", "num_phi", "num_rot45_X", "num_rot45_Y", "num_rot30_Y", "num_rot30_X","num_rot15_Y", "num_rot15_X","num_rot60_Y","num_rot60_X","display_address", "building_id",'manager_id', "street_address"]
    features_to_use.extend(['price_bed', 'price_t1', 'fold_t1', 'bath_room', 'num_nr_of_lines', 'num_redacted', 'num_phone_nr'])
    features_to_use.extend(["num_photos", "num_features", "num_description_words", 'num_email',
                        "listing_id", 'room_dif', 'room_sum', 'num_exc', "yearday", "Zero_building_id", 'num_cap_share','num_lev_rat','num_half_bathrooms']) # "created_year", "created_month", "created_day", "created_hour",])
    train_X, test_X = include_features(train_df, test_df, features_to_use)

    #TODO DELETE
    train_X, test_X = train_df[features_to_use], test_df[features_to_use]

    target_num_map = {'high': 0, 'medium': 1, 'low': 2}

    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

    elif run_objective is Objective.STACK_SUBM:
        import pickle
        x_train = pickle.load(open('./stack_predictions/first_ext', 'rb'))
        x_test = pickle.load(open('./stack_predictions/second_ext', 'rb'))

        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import normalize
        from sklearn.ensemble import BaggingClassifier

        xgb_cl = xgb.XGBClassifier(max_depth=3,learning_rate=0.02,n_estimators=525,
                                   objective='multi:softprob',subsample=0.8,colsample_bytree=0.8,reg_lambda=0.6)
        nn_cl = MLPClassifier(solver = 'lbfgs', alpha = 1e-05, max_iter = 200, hidden_layer_sizes = (20,20))


        bagging_xgb = BaggingClassifier(base_estimator=xgb_cl,n_estimators=5)
        bagging_nn = BaggingClassifier(base_estimator=nn_cl,n_estimators=20)



        bagging_xgb.fit(x_train,train_y)
        bagging_nn.fit(x_train,train_y)

        preds_nn = bagging_nn.predict_proba(x_test)
        preds_xgb = bagging_xgb.predict_proba(x_test)

        preds_avg = (preds_xgb**0.35)*(preds_nn**0.65)


        out_df_n = pd.DataFrame(preds_nn)
        out_df_n.columns = ["high", "medium", "low"]
        out_df_n["listing_id"] = test_df.listing_id.values
        out_df_n.to_csv("my_beautiful_stack_nn_only.csv", index=False)

        out_df_x = pd.DataFrame(preds_xgb)
        out_df_x.columns = ["high", "medium", "low"]
        out_df_x["listing_id"] = test_df.listing_id.values
        out_df_x.to_csv("my_beautiful_stack_xgb_only.csv", index=False)

        sys.exit(1)

        preds = normalize(preds_avg,axis=1, norm='l1')
        print(preds)

        out_df = pd.DataFrame(preds)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_df.listing_id.values
        out_df.to_csv("my_beautiful_stack.csv", index=False)

    elif run_objective is Objective.STACK:

        train_X = train_X.as_matrix()

        if LOAD_STACK:
            import pickle
            print('Running stack with loading predictions from base models')
            x_train = pickle.load(open('./stack_predictions/first_layer', 'rb'))
            print('xgb model CV: {}'.format(log_loss(train_y, x_train[:,0:3])))
            print('rfc-callibrated model CV: {}'.format(log_loss(train_y, x_train[:,3:6])))
            print('ada model CV: {}'.format(log_loss(train_y, x_train[:,6:9])))
            print('linear regression model CV: {}'.format(log_loss(train_y, x_train[:,9:12])))

            target_reversed_num_map = {'high': 2, 'medium': 1, 'low': 0}

            train_y_rev = np.array(train_df['interest_level'].apply(lambda x: target_reversed_num_map[x]))

            print('nn model CV: {}'.format(log_loss(train_y_rev, x_train[:,12:15])))

            print('light gbm model CV: {}'.format(log_loss(train_y, x_train[:,15:18])))



        else:
            # RANDOM FOREST
            rf_first_level_non_call = RandomForestClassifier(max_features = 0.3, n_estimators = 500, min_samples_split = 3, max_depth = 17)
            rf_first_level = CalibratedClassifierCV(rf_first_level_non_call, method='isotonic', cv=5)

            # # XGB
            xgb_first_level = xgb.XGBClassifier(max_depth = 10, learning_rate = 0.07, n_estimators = 1025, objective = 'multi:softprob',
                                    gamma = 3.14, min_child_weight = 2, subsample = 0.98, colsample_bytree = 0.61 )

            # LOGISTIC REGRESSION
            lr_first_level = LogisticRegression(solver='newton-cg', C = 0.1, max_iter = 350)


            #ADA BOOST WITH RF AS WEAK LEARNERS
            ada_weak_learner = RandomForestClassifier(max_depth = 17, min_samples_split = 8, n_estimators = 360, max_features = 0.3)
            ada_first_level = AdaBoostClassifier(n_estimators = 5, base_estimator = ada_weak_learner, learning_rate = 0.03)


            xg_oof_train, xg_oof_test = get_oof(xgb_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'xgb')
            print("XGB - CV: {}".format(log_loss(train_y, xg_oof_train)))

            rf_oof_train, rf_oof_test = get_oof(rf_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'rfc - callibrated', test_X)
            print("Random forest - CV: {}".format(log_loss(train_y, rf_oof_train)))

            ada_oof_train, ada_oof_test = get_oof(ada_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'ada', test_X)
            print("Ada Boost - CV: {}".format(log_loss(train_y, ada_oof_train)))

            lr_oof_train, lr_oof_test = get_oof(lr_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'lr', test_X)
            print("Logistic regression - CV: {}".format(log_loss(train_y, lr_oof_train)))

            lg_oof_train, lg_oof_test = get_oof_lightgbm(xgb_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'xgb')
            print("LGBM - CV: {}".format(log_loss(train_y, lg_oof_train)))



            x_train = np.concatenate((xg_oof_train, rf_oof_train, ada_oof_train, lr_oof_train), axis=1)
            x_test = np.concatenate((xg_oof_test, rf_oof_test, ada_oof_test, lr_oof_test), axis=1)


            print('x_train shape: {}'.format(x_train.shape))
            print('x_test shape: {}'.format(x_test.shape))

            import pickle
            pickle.dump(x_train, open('first_lvl_clfs', 'wb+'))
            pickle.dump(x_test, open('second_lvl_predictions', 'wb+'))
            import sys
            sys.exit(0)
  