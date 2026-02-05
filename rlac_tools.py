import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize   
from sklearn.metrics import r2_score



def _dedup(X):
    dup = X.columns.duplicated()
    if dup.any():
        print(f"Drop {dup.sum()} duplicate columns: {X.columns[dup].tolist()}")
        X = X.loc[:, ~dup]
    return X




def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type,return_sorted_features=False):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0, n_jobs=128)
        f1_list = []
        importance_list = [] 
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            importance_list.append(clf.feature_importances_)
        avg_importance = np.mean(importance_list, axis=0)
        sorted_idx = np.argsort(avg_importance)[::-1]
        sorted_features = X.columns[sorted_idx].tolist()
        sorted_importance = avg_importance[sorted_idx].tolist()
        if return_sorted_features:
            return np.mean(f1_list), sorted_features
        else:
            return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0, n_jobs=128)
        rae_list = []
        importance_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            importance_list.append(reg.feature_importances_)
        avg_importance = np.mean(importance_list, axis=0)
        sorted_idx = np.argsort(avg_importance)[::-1]
        sorted_features = X.columns[sorted_idx].tolist()
        sorted_importance = avg_importance[sorted_idx].tolist()
        if return_sorted_features:
            return np.mean(rae_list), sorted_features
        else:
            return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=128)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=128))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    elif task_type == 'rank':
        pass
    else:
        return -1

# 'RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'LASSO', 'DT'
def downstream_task_by_method(data, task_type, method):
    X = data.iloc[:, :-1]
    #print(f"Type of X_train: {type(X_train)}")
    #print(f"Shape of X_train: {X_train.shape}")
    y = data.iloc[:, -1].astype(float)
    X = _dedup(X)

    #X = data.iloc[:, :-1].to_numpy()
    #y = data.iloc[:, -1].astype(float).to_numpy()
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso(alpha=1)
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    else:
        return -1


def downstream_task_by_method_std(data, task_type, method):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso(alpha=0.1)
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list), np.std(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list), np.std(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list), np.std(f1_list)
    else:
        return -1



def test_task_wo_cv(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            auc_roc_score.append(roc_auc_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list = [], [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
            break
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            ras.append(roc_auc_score(y_test, y_predict))
            recall.append(recall_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='macro'))
            rec_list.append(recall_score(y_test, y_predict, average='macro'))
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
            maf1_list.append(f1_score(y_test, y_predict, average='macro'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1

def test_task_new(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list, auc_roc_score, pr_auc_list = [], [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)    

            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))

            n_classes = len(clf.classes_)
            if n_classes == 2:            
                pr_auc = average_precision_score(y_test, y_prob[:, 1])
            else:                         
                y_test_bin = label_binarize(y_test, classes=clf.classes_)
                pr_auc = average_precision_score(
                    y_test_bin,
                    y_prob,
                    average='weighted'    
                )
            pr_auc_list.append(pr_auc)

        #return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(pr_auc_list)
    
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list,r2_list = [], [], [], [],[]
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            
            '''mse_list.append(1 - mean_squared_error(y_test, y_predict))'''
            
            r2_list.append(r2_score(y_test, y_predict))
            
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            #rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
            rmse_list.append(1 - np.sqrt(mean_squared_error(y_test, y_predict)))


        #return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
        return np.mean(mae_list), np.mean(r2_list), np.mean(rae_list), np.mean(rmse_list)
    
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            ras.append(roc_auc_score(y_test, y_predict))
            recall.append(recall_score(y_test, y_predict, average='weighted'))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='macro'))
            rec_list.append(recall_score(y_test, y_predict, average='macro'))
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
            maf1_list.append(f1_score(y_test, y_predict, average='macro'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1


def adjust_perf(record, fe, task_type='cls',
                alpha=40.0, beta=0.05, lambda_min=0.96,
                entropy_thr=0.10):
    perf = record.performance          

    if task_type == 'cls':
        labels = fe.original["label"].to_numpy()
        counts = np.bincount(labels)
        probs = counts / counts.sum()
        entropy = -np.sum([p * log(p) for p in probs if p > 0])
        norm_entropy = entropy / log(len(probs)) if len(probs) > 1 else 0.0

        if norm_entropy < entropy_thr:         
            perf *= 0.95
    else:
        y = fe.original["label"].to_numpy().astype(float)
        cv = y.std() / (abs(y.mean()) + 1e-12)
        logistic = 1.0 / (1.0 + np.exp(-alpha * (cv - beta)))
        factor = lambda_min + (1.0 - lambda_min) * logistic
        perf *= factor

    return perf


def downstream_task_new_score(data, task_type,return_sorted_features=False):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0, n_jobs=128)
        f1_list = []
        importance_list = [] 
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            importance_list.append(clf.feature_importances_)
        avg_importance = np.mean(importance_list, axis=0)
        sorted_idx = np.argsort(avg_importance)[::-1]
        sorted_features = X.columns[sorted_idx].tolist()
        sorted_importance = avg_importance[sorted_idx].tolist()
        if return_sorted_features:
            feat_imp_pairs = list(zip(sorted_features, sorted_importance))
            return np.mean(f1_list), feat_imp_pairs
        else:
            return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0, n_jobs=128)
        rae_list = []
        importance_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            importance_list.append(reg.feature_importances_)
        avg_importance = np.mean(importance_list, axis=0)
        sorted_idx = np.argsort(avg_importance)[::-1]
        sorted_features = X.columns[sorted_idx].tolist()
        sorted_importance = avg_importance[sorted_idx].tolist()
        if return_sorted_features:
            feat_imp_pairs = list(zip(sorted_features, sorted_importance))
            #return np.mean(rae_list), sorted_features
            return np.mean(rae_list), feat_imp_pairs
        else:
            return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=128)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=128))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    elif task_type == 'rank':
        pass
    else:
        return -1