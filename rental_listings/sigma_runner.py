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

    #f = features_to_use[:]
    #f.append('interest_level')

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



    if run_objective is Objective.SUBM_XGB:
        #from xgb_runner import runXGB

        print("running XGB model")
        #preds, model = runXGB(train_X, train_y, test_X)

        from sklearn.ensemble import BaggingClassifier, VotingClassifier

        base_clf = xgb.XGBClassifier(max_depth = 6, learning_rate = 0.03, n_estimators = 1100, objective = 'multi:softprob',
                                    gamma = 3.5, min_child_weight = 2, subsample = 0.8, colsample_bytree = 0.7, reg_lambda=0.8, seed=0)


        #base_clf = xgb.XGBClassifier(max_depth = 10, learning_rate = 0.07, n_estimators = 1025, objective = 'multi:softprob',
                                  #  gamma = 3.14, min_child_weight = 2, subsample = 0.98, colsample_bytree = 0.61, seed=0)
        #
        # base_clf2 = xgb.XGBClassifier(max_depth = 4, learning_rate = 0.058, n_estimators = 1025, objective = 'multi:softprob',
        #                             gamma = 0.62, min_child_weight = 2, subsample = 0.92, colsample_bytree = 0.82, reg_lambda=0.83)
        #
        # clf = BaggingClassifier(base_estimator=base_clf, n_estimators=2)
        #
        # clf2 = BaggingClassifier(base_estimator=base_clf2, n_estimators=2)
        #
        #
        #
        # ens = VotingClassifier(estimators=[('first',base_clf), ('second',base_clf2)], voting='soft')

        base_clf.fit(train_X, train_y)
        out_df = pd.DataFrame(base_clf.predict_proba(test_X))
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_df.listing_id.values
        out_df.to_csv("xgb_tuned_old.csv", index=False)

    elif run_objective is Objective.SUBM_RFC:

        print('running RFC submission')

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=500,max_depth=16,min_samples_split=5,max_features=0.38)
        clf.fit(train_X, train_y)
        out_df = pd.DataFrame(clf.predict_proba(test_X))
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_df.listing_id.values
        out_df.to_csv("rfc_subm.csv", index=False)


    elif run_objective is Objective.GS_XGB:
        from bo import run_bay_rfc
        print('used features:\n')
        print(features_to_use)
        print('running bayesian xgb optimization')
        run_bay_rfc(train_X, train_y, run='xgb')

    elif run_objective is Objective.GS_RFC:
        from bo import run_bay_rfc
        print('used features:\n')
        print(features_to_use)
        print('running bayesian rfc optimization')
        run_bay_rfc(train_X, train_y, run='rfc')

    elif run_objective is Objective.GS_AB:
        from adaboost import bayesian_adaptive_boost
        print('running bayesian adaptive boost optimization')
        bayesian_adaptive_boost(train_X, train_y)

    elif run_objective is Objective.CALL_RFC:
        from callibration_rfc import test_callibration_vs_rf
        print("running rfc callibration on chosen parameters set in callibration_rfc.py")
        test_callibration_vs_rf(train_X, train_y, test_df=test_df, submit=test_X)

    elif run_objective is Objective.CV_RFC:

        from xgb_runner import rfccv

        #def rfccv(n_estimators, min_samples_split, max_features, max_depth, min_leaf)
        #(max_features = 0.38, n_estimators = 500, min_samples_split = 5, max_depth = 16)
        import itertools

        for depth, esti, max_f, min_s in itertools.product([17,18,16],[500,600], [0.3,0.5],[3,5]):
            print(depth, esti, max_f, min_s)
            print(rfccv(d=train_X, t=train_y, n_estimators=esti,min_samples_split=min_s, max_features=max_f,max_depth=depth, min_leaf=1))


    elif run_objective is Objective.CV_XGB:
        from xgb_runner import xgbccv
        # Running for lr: 0.03,  depth: 6, gamma: 2, reg_lam: 0.7, child: 3

        scores, iterations = xgbccv(d=train_X, t=train_y, n_est = 2000, learning_rate=0.03, min_child=3, cols = 0.7,
           max_depth = 6)

        #scores, iterations = xgbccv(d=train_X, t=train_y, n_est = 2000, learning_rate=0.03, min_child=3, cols = 0.7,
         #  max_depth = 6)

    #
    #     kf = model_selection.KFold(n_splits = 5, shuffle = True)
    #     cv_scores = []
    #     iterations = []
    #     d = train_X
    #     t = train_y
    #
    # #
    # elif run_objective is Objective.CV_RFC:
    #
    #
    #
    #
    #     for dev_index, val_index in kf.split(range(d.shape[0])):
    #         train_X, test_X = d.iloc[dev_index,:], d.iloc[val_index,:]
    #         train_y, test_y = t[dev_index], t[val_index]
    #
    #         base_clf = xgb.XGBClassifier(max_depth = 10, learning_rate = 0.07, n_estimators = 1025, objective = 'multi:softprob',
    #                                 gamma = 3.14, min_child_weight = 2, subsample = 0.98, colsample_bytree = 0.61 )
    #         from sklearn.ensemble import BaggingClassifier, VotingClassifier
    #         clf = BaggingClassifier(base_estimator=base_clf, n_estimators=5)
    #
    #         base_clf2 = xgb.XGBClassifier(max_depth = 4, learning_rate = 0.058, n_estimators = 1025, objective = 'multi:softprob',
    #                                 gamma = 0.62, min_child_weight = 2, subsample = 0.92, colsample_bytree = 0.82, reg_lambda=0.83)
    #
    #         clf2 = VotingClassifier(estimators=[('deep',base_clf), ('shallow',base_clf2)], voting='soft')
    #
    #         base_clf.fit(train_X, train_y)
    #         print('Voting classifier CV score: {}'.format(log_loss(test_y,clf2.predict_proba(test_X))))
    #         #clf.fit(train_X, train_y)
    #         #print('Bagging classifier CV score: {}'.format(log_loss(test_y,clf.predict_proba(test_X))))

    elif run_objective is Objective.STACK_SUBM:
        import pickle
        x_train = pickle.load(open('./stack_predictions/first_ext', 'rb'))
        x_test = pickle.load(open('./stack_predictions/second_ext', 'rb'))

        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import normalize
        from sklearn.ensemble import BaggingClassifier
        #bagging_clf = BaggingClassifier(base_estimator=base_clf,n_estimators=50)

        #bagging_clf = LogisticRegression(solver='newton-cg', C = 0.1, max_iter = 350)

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

# nn
        out_df_n = pd.DataFrame(preds_nn)
        out_df_n.columns = ["high", "medium", "low"]
        out_df_n["listing_id"] = test_df.listing_id.values
        out_df_n.to_csv("my_beautiful_stack_nn_only.csv", index=False)

# xgb

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
            # xgb_first_level = xgb.XGBClassifier(max_depth = 10, learning_rate = 0.07, n_estimators = 1025, objective = 'multi:softprob',
            #                         gamma = 3.14, min_child_weight = 2, subsample = 0.98, colsample_bytree = 0.61 )

            # LOGISTIC REGRESSION
            lr_first_level = LogisticRegression(solver='newton-cg', C = 0.1, max_iter = 350)


            #ADA BOOST WITH RF AS WEAK LEARNERS
            ada_weak_learner = RandomForestClassifier(max_depth = 17, min_samples_split = 8, n_estimators = 360, max_features = 0.3)
            ada_first_level = AdaBoostClassifier(n_estimators = 5, base_estimator = ada_weak_learner, learning_rate = 0.03)


           # xg_oof_train, xg_oof_test = get_oof(xgb_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'xgb')
           # print("XGB - CV: {}".format(log_loss(train_y, xg_oof_train)))

            rf_oof_train, rf_oof_test = get_oof(rf_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'rfc - callibrated', test_X)
            print("Random forest - CV: {}".format(log_loss(train_y, rf_oof_train)))

            ada_oof_train, ada_oof_test = get_oof(ada_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'ada', test_X)
            print("Ada Boost - CV: {}".format(log_loss(train_y, ada_oof_train)))

            lr_oof_train, lr_oof_test = get_oof(lr_first_level, train_X.shape[0], test_X.shape[0], train_X, train_y, 'lr', test_X)
            print("Logistic regression - CV: {}".format(log_loss(train_y, lr_oof_train)))



            x_train = np.concatenate((rf_oof_train, ada_oof_train, lr_oof_train), axis=1)
            x_test = np.concatenate((rf_oof_test, ada_oof_test, lr_oof_test), axis=1)


            print('x_train shape: {}'.format(x_train.shape))
            print('x_test shape: {}'.format(x_test.shape))

            import pickle
            pickle.dump(x_train, open('first_lvl_clfs', 'wb+'))
            pickle.dump(x_test, open('second_lvl_predictions', 'wb+'))
            import sys
            sys.exit(0)

        dtrain = xgb.DMatrix(x_train, label=train_y)
        #dtest = xgb.DMatrix(x_test)
        import itertools

        ans = []

        #
        # params = {
        #  'alpha': [1e-4, 1e-5, 1e-6],
        #  'activation': ['tanh', 'sigmoid'], # 'relu', 'sigmoid'....
        #  'max_iter' : [170, 250],
        #  'hidden_layer_sizes': [(10, 6), (16, 10), (20, 20), (10,8,6)]#, (64,32), (32,16), (40,20), (32,16,6),(6,24,5)]
        # }

#         for alpha, iters, layers in itertools.product([1e-4, 1e-5, 1e-6],[170,250] ,[(10, 6), (16, 10), (20, 20), (10,8,6)]):
#             kf = model_selection.KFold(n_splits = 5, shuffle = True)
#             cv_scores = []
#             from sklearn.neural_network import MLPClassifier
#
#             clf_b = MLPClassifier(solver = 'lbfgs', alpha = alpha, max_iter = iters, hidden_layer_sizes = layers)
#
#             for dev_index, val_index in kf.split(range(x_train.shape[0])):
#                 dev_X, val_X = x_train[dev_index,:], x_train[val_index,:]
#                 dev_y, val_y = train_y[dev_index], train_y[val_index]
#
#                 clf_b.fit(dev_X, dev_y)
#
#                 preds = clf_b.predict_proba(val_X)
#
#                 #preds, model = runXGB(dev_X, dev_y, val_X, val_y)
# #
# #
#                 cv_scores.append(log_loss(val_y, preds))
#
#             print('alpha: {}, iters: {}, layers: {}'.format(alpha, iters, layers))
#             print(cv_scores)
#             print(np.mean(cv_scores))
#             print('\n\n')
#
#         sys.exit(1)
        # 0.02 3 0.8 0.6

        xgb_cl = xgb.XGBClassifier(max_depth=3,learning_rate=0.02,n_estimators=530,
                                   objective='multi:softprob',subsample=0.8,colsample_bytree=0.8,reg_lambda=0.6
        )
        #
        from sklearn.neural_network import MLPClassifier

        nn_cl = MLPClassifier(solver = 'lbfgs', alpha = 1e-05, max_iter = 200, hidden_layer_sizes = (20,20))
        from sklearn.preprocessing import normalize
        for xgb_w in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:

            xgb_scores = []
            nn_scores = []
            geom_avg_scores = []

            kf = model_selection.KFold(n_splits = 5, shuffle = True)
            for dev_index, val_index in kf.split(range(x_train.shape[0])):
                dev_X, val_X = x_train[dev_index,:], x_train[val_index,:]
                dev_y, val_y = train_y[dev_index], train_y[val_index]

                nn_cl.fit(dev_X, dev_y)
                xgb_cl.fit(dev_X, dev_y)


                preds_xgb = xgb_cl.predict_proba(val_X)
                preds_nn = nn_cl.predict_proba(val_X)


                xgb_scores.append(log_loss(val_y, preds_xgb))
                nn_scores.append(log_loss(val_y, preds_nn))

                preds_avg = (preds_xgb**xgb_w)*(preds_nn**(1-xgb_w))

                geom_avg_scores.append(log_loss(val_y,normalize(preds_avg,axis=1, norm='l1')))

                print('result for xgb weight {}, : xgb: {}, nn: {}, geom: {}'.format(xgb_w, xgb_scores[-1], nn_scores[-1], geom_avg_scores[-1]))


        sys.exit(0)
                #preds, model = runXGB(dev_X, dev_y, val_X, val_y)
#
#
                #cv_scores.append(log_loss(val_y, preds))

#alpha: 1e-05, iters: 170, layers: (20, 20)



        for eta, depth, colsample, lam in itertools.product([0.02,0.03,0.05],[2,3,4],[0.8, 0.9], [0.6, 0.8, 0.95]):
            xgb_params = {
                'objective': 'multi:softprob',
                'eta': eta,
                'max_depth' : depth,
                'num_class' : 3,
                'eval_metric': 'mlogloss',
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': colsample,
                'reg_lambda' : lam,
                'silent' : 1
            }

            kf = KFold(x_train.shape[0], n_folds=5, shuffle=True, random_state=1)

            res = xgb.cv(xgb_params, dtrain, num_boost_round=900, folds=kf, early_stopping_rounds=15, show_stdv=False)


            print('CV results for parameters:')
            print(eta, depth, colsample, lam)
            print('Stack CV: {0}, {1}'.format(res.iloc[-1,0], res.iloc[-1, 1]))
            print('best rounds: {}'.format(res.shape[0]-1))

            ans.append([depth, colsample, lam, res.iloc[-1,0], res.iloc[-1, 1], res.shape[0]-1 ])


        import pickle

        pickle.dump(ans, open('five_models_stack_cv', 'wb+'))
        #pickle.dump(x_train, open('stack_train', 'wb+'))
        #pickle.dump(x_test, open('stack_test', 'wb+'))

    elif run_objective is Objective.GS_FULL:

        from adaboost import bayesian_adaptive_boost
        print('running bayesian adaptive boost optimization')
        bayesian_adaptive_boost(train_X, train_y)

    # elif run_objective is Objective.TSNE:
    #     #tsne = manifold.TSNE(n_components=2, init='pca', random_state=2, method='barnes_hut', n_iter=600, verbose=1, angle=0.4, learning_rate = 1000, perplexity = 60)
    #     features_for_tsne = features_to_use[:]
    #     features_for_tsne.remove('manager_id')
    #     features_for_tsne.remove('building_id')
    #     features_for_tsne.remove('display_address')
    #     features_for_tsne.remove('street_address')
    #     features_for_tsne.remove('listing_id')
    #
    #
    #     #spectral = manifold.SpectralEmbedding(n_components=2, n_jobs=-1)
    #
    #
    #     #print(features_for_tsne)
    #
    #     X_Sel = train_df[features_for_tsne]
    #     #X_Sel = train_X
    #
    #     X_norm = normalize(X_Sel, axis=0)
    #
    #     target_num_map = {'high': 0, 'medium': 2, 'low': 1}
    #
    #
    #     y_test = train_df['interest_level'].apply(lambda x: target_num_map[x])
    #     #print(y)
    #     #X_train, X_test, y_train, y_test = train_test_split(X_norm, y, random_state=5, stratify=y, test_size=0.75)
    #     print('fitting rte')
    #
    #     hasher = RandomTreesEmbedding(n_estimators=10, random_state=7, max_depth=10)
    #     X_transformed = hasher.fit_transform(X_norm)
    #     print('shape after rte: {}'.format(X_transformed.shape))
    #     #print('fitting pca')
    #     pca = PCA(n_components=5, copy=True, whiten=False)
    #
    #     #X_pca = pca.fit_transform(X_transformed.toarray())
    #     #print('shape after rte: {}'.format(X_pca.shape))
    #     print('fitting t-sne')
    #     X_tsne = pca.fit_transform(X_transformed.toarray())
    #     print(pca.explained_variance_ratio_)
    #     print(pca.explained_variance_)
    #     print('shape after t-sne: {}'.format(X_tsne.shape))
    #     plt.figure()
    #
    #     plt.scatter(X_tsne[np.where(y_test == 2), 0],
    #                X_tsne[np.where(y_test == 2), 1],
    #                color='b',label='low', s=20)
    #
    #     plt.scatter(X_tsne[np.where(y_test == 1), 0],
    #                X_tsne[np.where(y_test == 1), 1],
    #                color='g',label='medium',s=5)
    #
    #     plt.scatter(X_tsne[np.where(y_test == 0), 0],
    #                X_tsne[np.where(y_test == 0), 1],
    #                color='r',label='high',s=5)
    #
    #     plt.xlabel('Dim 1')
    #     plt.ylabel('Dim 2')
    #     plt.title('T-SNE on 15% train samples')
    #     plt.legend(loc='best')
    #     plt.savefig('1.png')
    #     plt.show()




    elif run_objective is Objective.SVM:
        from sklearn import svm
        svm_model = svm.SVC(decision_function_shape='ovo', tol=0.00000001, probability=True)
        #svm_model = svm_model.fit(X_train, y_train)
        #print(log_loss())
        cv_scores = []
        train_X = train_X.as_matrix()

        kf = KFold(train_X.shape[0], n_folds=5, shuffle=True, random_state=1)


        for dev_index, val_index in kf:
                dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
                dev_y, val_y = train_y[dev_index], train_y[val_index]

                svm_model.fit(dev_X, dev_y)

                preds = svm_model.predict_proba(val_X)

                #preds, model = runXGB(dev_X, dev_y, val_X, val_y)
#
#
                cv_scores.append(log_loss(val_y, preds))
                print(cv_scores[-1])
        print(cv_scores)

    elif run_objective is Objective.WEAK_CLF:
        import pickle
        x_train = pickle.load(open('./stack_predictions/second_layer', 'rb'))

        call = x_train[:,3:6]
        ada = x_train[:,6:9]
        lr = x_train[:,9:12]
        light = x_train[:,15:18]



        out_df = pd.DataFrame(call)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_df.listing_id.values
        out_df.to_csv("thesis_call.csv", index=False)

        out_df = pd.DataFrame(ada)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_df.listing_id.values
        out_df.to_csv("thesis_ada.csv", index=False)

        out_df = pd.DataFrame(lr)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_df.listing_id.values
        out_df.to_csv("thesis_lr.csv", index=False)

        out_df = pd.DataFrame(light)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_df.listing_id.values
        out_df.to_csv("thesis_light.csv", index=False)