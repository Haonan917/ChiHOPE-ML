from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, clone
from xgboost import XGBClassifier
import numpy as np
from scipy import interp
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import shap
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings('ignore', category=DataConversionWarning)

# Search best params by five-fold cross validation for each algorithm
def optimise_all_models(X, y, splits):
    splits_copy = splits.copy()
    svc_result = optimise_svc_without_fs(X, y, cv=splits_copy)
    logres_result = optimise_logres_without_fs(X, y, cv=splits_copy)
    xgb_result = optimise_xgb_without_fs(X, y, cv=splits_copy)
    
    averaged_models = EnsembleModels(models = (xgb_result, svc_result, logres_result))

    results = {}
    results['svc'] = svc_result
    results['xgb'] = xgb_result
    results['lr'] = logres_result
    results['avg'] = averaged_models
    return results

def refit_all_models(X,y,results,splits, omicsComb, save_dir, random_state):
    splits_copy = splits.copy()
    refit = {}
    for model in results.keys():
        try:
            refit[model] = model_refit(X, y, model=results[model].best_estimator_, cv=splits_copy, algo=model, omicsComb=omicsComb, save_dir=save_dir, random_state=random_state)
        except:
            refit[model] = model_refit(X, y, model=results[model], cv=splits_copy, algo=model, omicsComb=omicsComb, save_dir=save_dir, random_state=random_state)
    return refit

def model_refit(X, y, model, cv, algo, omicsComb, save_dir, random_state):
    cv_models, cv_aucs, cv_ypreds, cv_yreals, cv_tprs, total_ypreds, total_yreals, total_indexes = ([] for _ in range(8))
    mean_fpr = np.linspace(0, 1, 10)
    # record model performance during cross-validation
    for i, (tr,ts) in enumerate(cv):
        model.fit(X.iloc[tr,:], y[tr])
        cv_models.append(deepcopy(model))
        
        y_pred = model.predict_proba(X.iloc[ts,:])[:,1]
        y_test = y[ts]
        total_indexes.extend(ts)
        # Precision
        total_ypreds.extend(y_pred)
        total_yreals.extend(y_test)
        # total_index.extend(ts)
        cv_ypreds.append(y_pred)
        cv_yreals.append(y_test)

        # AUC
        roc_auc = roc_auc_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        cv_tprs.append(interp(mean_fpr, fpr, tpr)) # interp(container, x, y)
        cv_aucs.append(roc_auc)
        cv_tprs[-1][0] = 0.0 # ensure curve start from 0
    
    # Internal prediction
    df_data = pd.DataFrame({'ytruth': total_yreals, 'ypred': total_ypreds}, index=total_indexes)
    df_data.sort_index(inplace=True)
    df_data.to_csv(save_dir / 'valid_ypred_{}_{}.csv'.format(algo, omicsComb), index=False)

    valid_auc = roc_auc_score(df_data['ytruth'], df_data['ypred'])
    mean_auc = np.mean(cv_aucs)
    print(f"fold avg: {mean_auc}; all auc: {valid_auc}")

    file_path = save_dir / 'valid_output_{}.csv'.format(omicsComb)
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a') as f:
        if not file_exists:
            f.write('data_type,algo,seed,train.val,mean_auc,valid_auc\n')
        
        f.write('{},{},{},{},{},{}\n'.format(omicsComb,algo,random_state,1,mean_auc,valid_auc))

    print('{},refit,{},{},{}\n'.format( omicsComb, algo, mean_auc, valid_auc))
    f.close()

    cv_file_path = save_dir / 'cv_valid_output_{}.csv'.format(omicsComb)
    cv_file_exists = os.path.exists(cv_file_path)
    with open(cv_file_path, 'a') as f:
        if not cv_file_exists:
            f.write('data_type,algo,seed,train.val,fold,auc\n')
        for i in range(0, len(cv_aucs)):
            f.write('{},{},{},{},{},{}\n'.format(omicsComb,algo,random_state,1,i+1,cv_aucs[i]))

    print('{},cv,{},{}\n'.format( omicsComb, algo, mean_auc))
    f.close()

    ### Refit
    model.fit(X,y)
    return [model, cv_models]


def test_all_models(X,y,results,omicsComb,save_dir,random_state):
    test_result = {}
    for model in results.keys():
        test_result[model] = model_test(X,y,results[model][0],algo=model, omicsComb=omicsComb, save_dir=save_dir, random_state=random_state)
    return test_result


def save_SHAP(shap_values, origin_values, feature_list, output_dir, file_prefix):
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_list,
        'feature_importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values(by='feature_importance', key=lambda x: abs(x),
                                                            ascending=False)
    
    shap_df =  pd.DataFrame(shap_values, columns= feature_list)
    origin_df = pd.DataFrame(origin_values, columns= feature_list)
    
    feature_importance_df.to_csv(output_dir / "{}feature_importance.tsv".format(file_prefix), sep='\t', index=False)
    shap_df.to_csv(output_dir / "{}shap_values.tsv".format(file_prefix), sep='\t', index=False)
    origin_df.to_csv(output_dir / "{}origin_values.tsv".format(file_prefix), sep='\t', index=False)

def analyze_feature_importance(X, y, model, algo, omicsComb, save_dir, random_state):
    if algo == 'avg':
        return
    selected_features = X.columns.tolist()
    observations = model.named_steps['scaler'].transform(model.named_steps['imputer'].transform(X))
    if  algo in model.named_steps and isinstance(model.named_steps[algo], SVC):
        explainer = shap.LinearExplainer(model.named_steps[algo], observations, feature_dependence="independent")
        shap_values = explainer.shap_values(observations)
    elif algo in model.named_steps and isinstance(model.named_steps[algo], XGBClassifier):
        # XGBoost
        explainer = shap.TreeExplainer(model.named_steps[algo])
        shap_values = explainer.shap_values(observations)
    elif algo in model.named_steps and isinstance(model.named_steps[algo], LogisticRegression):
        # LR
        explainer = shap.LinearExplainer(model.named_steps[algo], observations)
        shap_values = explainer.shap_values(observations)
    else:
        # raise ValueError(f"Unsupported model type or algorithm {algo} not recognized.")
        print(f"Unsupported model type or algorithm {algo} not recognized.")
        return

    save_SHAP(shap_values=shap_values, 
            origin_values=observations, 
            feature_list=selected_features, 
            output_dir=save_dir,
            file_prefix='{}_{}_'.format(algo, omicsComb))


def model_test(X, y, model, algo, omicsComb, save_dir, random_state):
    y_pred = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_pred)

    # AUC
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    df_roc = pd.DataFrame().from_dict({'fpr':fpr, 'tpr':tpr})
    df_roc.to_csv(save_dir / 'roc_{}_{}.csv'.format(algo, omicsComb), index=False)

    df_data = pd.DataFrame().from_dict({'ytruth':y, 'ypred':y_pred})
    df_data.to_csv(save_dir / 'test_ypred_{}_{}.csv'.format(algo, omicsComb), index=False)

    file_path = save_dir / 'test_output_{}.csv'.format(omicsComb)
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a') as f:
        if not file_exists:
            f.write('data_type,algo,seed,train.val,auc\n')
        
        f.write('{},{},{},{},{}\n'.format(omicsComb,algo,random_state,0,roc_auc))

    print("{}, {}, test auc: {}".format(omicsComb, algo, roc_auc))
    analyze_feature_importance(X, y, model, algo, omicsComb, save_dir, random_state)

def optimise_logres_without_fs(X, y,  cv):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    logres = LogisticRegression(random_state=0, penalty='elasticnet', solver='saga', max_iter=1000, class_weight='balanced')
    pipe = Pipeline(steps=[('imputer', imputer), ('scaler', scaler), ('lr', logres)])

    param_grid = { 
            'lr__C': np.logspace(-3,3,30),
            'lr__l1_ratio': np.arange(0.1,1.1,0.1) 
            }
        

    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', return_train_score=True, n_jobs=-1, verbose=0, n_iter=500, random_state=0)
    search.fit(X,y)

    print("Best params:", search.best_params_)
    return search


def optimise_svc_without_fs(X, y, cv=5):
    # Pipeline components
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    svc = SVC(random_state=0, max_iter=1000, probability=True,verbose=0, kernel='linear')
    pipe = Pipeline(steps=[('imputer', imputer), ('scaler', scaler), ('svc', svc)])

    param_grid = { 
        'svc__C': np.logspace(-6,3,100)
    }
    
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0, n_iter=500, random_state=0)
    search.fit(X,y)

    print("Best params:", search.best_params_)
    return search

def optimise_xgb_without_fs(X, y, cv=5):
    # Pipeline components
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
    pipe = Pipeline(steps=[('imputer', imputer), ('scaler', scaler), ('xgb', xgb)])

    # Parameter grid for XGBoost
    param_grid = {
        "xgb__max_depth": [1, 2, 3],
        "xgb__n_estimators": [5, 10, 25, 50, 100],
        "xgb__learning_rate": [0.01, 0.1, 0.2, 0.3],
        "xgb__subsample": [0.6, 0.7, 0.8, 0.9, 1],
        "xgb__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1]
    }

    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', return_train_score=True, 
                                n_jobs=-1, verbose=0, n_iter=500, random_state=0)
    
    # Fit the model
    search.fit(X, y)
    
    print("Best params:", search.best_params_)
    return search


# Ensemble models from different algorithms
# Cited from https://github.com/ cclab-brca/neoadjuvant-therapy-response-predictor
class EnsembleModels(BaseEstimator):
    def __init__(self, models, is_classifier=True):
        self.models = models
        self.is_classifier = is_classifier

    def fit(self, X, y):
        self.models_ = [clone(x.best_estimator_) if hasattr(x, 'best_estimator_') else clone(x) for x in self.models]
        # training
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'models_')
        # classfication
        if self.is_classifier:
            predictions = np.column_stack([model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X) for model in self.models_])
            return np.mean(predictions, axis=1)

        # regression
        else:
            predictions = np.column_stack([model.predict(X) for model in self.models_])
            return np.mean(predictions, axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, 'models_')
        if not self.is_classifier:
            raise AttributeError("predict_proba is not available for regression models")
        predictions_0 = np.column_stack([
            model.predict_proba(X)[:,0] for model in self.models_
        ])

        predictions_1 = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_
        ])
        means_0 = np.mean(predictions_0, axis=1)
        means_1 = np.mean(predictions_1, axis=1)
        return np.column_stack([means_0, means_1])
