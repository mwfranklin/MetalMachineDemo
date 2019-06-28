#general requirements
import pandas as pd
import numpy as np
import sys
import warnings
import pickle
from collections import Counter
#preprocessing stuff
import ExtraMLFxns as mlf #my personal extra fxns in CustomModules
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit, StratifiedShuffleSplit, cross_validate
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.combine import SMOTEENN, SMOTETomek
#classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
#process results
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score, confusion_matrix, make_scorer, matthews_corrcoef, hamming_loss
warnings.filterwarnings(action="ignore")

name_index = int(sys.argv[1]) #from job array
over_samp = str(sys.argv[2]) #OverSamp, OvUnEEN, OvUnTomek; has to be OverSampNC because of the categorical features
cv_name = str(sys.argv[3]) #GSS for groupShuffleSplit, SSS for StratifiedShuffleSplit
group_name = str(sys.argv[4]) #select from 'cath_class', 'cath_arch', 'scop_class', 'scop_fold', 'ECOD_arch', 'ECOD_x_poshom', 'ECOD_hom'; we settled on using just ECOD_arch
data_subset = str(sys.argv[5]) #All_Sph, All_Shell, Gen, etc 
opt_type = str(sys.argv[6]) #Prec Acc MCC Multi
relaxed = str(sys.argv[7]) #Relaxed or UnRelaxed

print(name_index, over_samp, cv_name, group_name, data_subset, opt_type, relaxed)

def scale_resample(X_train, y_train, X_test, over_samp_strat = "OverSampNC"):
    #split for scaling into categorical and not categorical
    not_cat_electro = ("geom_gRMSD", "geom_MaxgRMSDDev","geom_val", "geom_nVESCUM","geom_AtomRMSD")
    geom = [name for name in X_train if name.startswith("geom")]
    cat_data = [x for x in geom if not x in not_cat_electro]
    cat_data.extend(['MetalCodes', 'MetalAtoms', 'SEPocket', "groupID"])
    #print(cat_data)
    
    X_train_scale = preprocessing.scale( X_train[X_train.columns.difference(cat_data)] )
    scaler = preprocessing.StandardScaler().fit( X_train[X_train.columns.difference(cat_data)] ) #scale the training set
    X_test_scale = scaler.transform( X_test[X_test.columns.difference(cat_data)] ) #scale the test set based on the scale of the training set
    num_norm_cols = X_test_scale.shape[1]
    #add categorical data back
    #print(X_train[[ entry for entry in X_train.columns if entry in cat_data]])
    X_train_scale = np.concatenate((X_train_scale, X_train[[ entry for entry in X_train.columns if entry in cat_data]]), axis = 1)
    X_test_scale = np.concatenate((X_test_scale, X_test[[ entry for entry in X_test.columns if entry in cat_data]]), axis = 1)
    #print(X_test_scale)
    X_test_scale = X_test_scale[:,:-1] #drop out the foldID b/c not needed for X_test
    #print(X_test_scale[0:2,:])

    #can only use oversampling for canonicals
    if over_samp_strat == "OverSampNC":
        smote = SMOTENC(categorical_features= np.arange(num_norm_cols, X_train_scale.shape[1], 1) )
    else:
        print("Invalid oversampling term because of categoricals")
        sys.exit()

    X_train_new, y_train_new = smote.fit_resample(X_train_scale, y_train.Catalytic)
    #print(X_train_new.shape, Counter(y_train_new))
    fold_ids = X_train_new[:,-1] #last column is the foldID converted to numeric to assign to groups
    X_train_new = X_train_new[:,:-1] #drop the foldID out for final training

    print("Finished processing data")
    return(X_train_new, y_train_new, X_test_scale, fold_ids)


prec_score = make_scorer(mlf.prec_score_custom, greater_is_better = True)
mcc_score = make_scorer(mlf.mcc_score, greater_is_better = True)
tnr_score = make_scorer(mlf.tnr_score, greater_is_better = True)
hamm_score = make_scorer(mlf.hamming_score, greater_is_better = True)
if opt_type == "Prec":
    this_scoring = prec_score
elif opt_type == "Acc":
    this_scoring = "accuracy"
elif opt_type == "MCC":
    this_scoring = mcc_score
elif opt_type == "Multi":
    this_scoring = {"Acc":'accuracy', "MCC": mcc_score, "Hamm": hamm_score }
else:
    print("Invalid scoring term")
    sys.exit()

if cv_name == "GSS":
    cv_type = GroupShuffleSplit(n_splits=10)
elif cv_name == "SSS":
    cv_type = StratifiedShuffleSplit(n_splits=10)
else:
    cv_type = 10 #uses KFold by default in the GridSearchCV

fold_names = ['cath_class', 'cath_arch', 'scop_class', 'scop_fold', 'ECOD_arch', 'ECOD_x_poshom', 'ECOD_hom'] #there is data available relatively easily to use another fold type, but we settled on ECOD_arch as the best 
if group_name != "ECOD_arch":
    print("Invalid fold selection, you can only pick ECOD_arch")
    sys.exit()

names = ["LogRegr", "Ridge", "PassAggr",
        "QDA", "LDA", "NaiveBayes",
        "NearNeigh",
        "LinSVM", "RBFSVM", "SigSVM",
        "RandomForest", "ExtraTrees", "GradBoost", 
        "NeurNet"]

classifiers = [
    LogisticRegression(solver="liblinear", class_weight = "balanced", tol = 1e-4), 
    RidgeClassifier(solver="auto", class_weight = "balanced", tol = 1e-4),
    PassiveAggressiveClassifier(class_weight = "balanced", tol = 1e-4),
    QuadraticDiscriminantAnalysis(priors=None), #also almost no real parameters to adjust
    LinearDiscriminantAnalysis(solver = "lsqr"),
    GaussianNB(priors=None), #only one real parameter to adjust! 
    KNeighborsClassifier(algorithm = "ball_tree", weights = 'distance'), 
    SVC(kernel="linear", max_iter=10000, class_weight = "balanced", tol = 1e-4), 
    SVC(kernel="rbf", class_weight = "balanced", tol = 1e-4), 
    SVC(kernel="sigmoid", class_weight = "balanced", tol = 1e-4),
    RandomForestClassifier(class_weight = "balanced", min_samples_split = 5, bootstrap = True),
    ExtraTreesClassifier(class_weight = "balanced", min_samples_split = 5, bootstrap = True), 
    GradientBoostingClassifier(subsample = 0.8, criterion = 'friedman_mse', min_samples_split = 5, validation_fraction=0.2, n_iter_no_change=5, tol=0.01,),
    MLPClassifier(learning_rate_init = 0.01, early_stopping = True, n_iter_no_change = 5), 
]

#read features
wp_scores = pd.read_table("WholeScores_%s.txt"%relaxed) #this now includes only ECOD_arch as the foldID
#Catalytic = 0 are enzymatic, Catalytic = 1 are non-catalytic
wp_scores['Catalytic'] = wp_scores['Catalytic'].apply(lambda x: 0 if x == True else 1) #flipped so that Catalytic are in the "positive" position of CFM
wp_scores = wp_scores.dropna(subset = ["SITE_ID"])
wp_scores.drop_duplicates(inplace = True)
#pull the validation set out separately
valid_set = wp_scores[wp_scores.ValidSet == True]
wp_scores = wp_scores[wp_scores.ValidSet == False]

#split into features and classification
X, y = mlf.subset_data(wp_scores, data_subset, group_name)
valid_set, y_valid = mlf.subset_data(valid_set, data_subset, group_name)
print(len(X), len(valid_set))

parameter_space = [
    { "C": np.logspace(0, 3, 4), "penalty": ["l1", "l2"] }, #LogRegr
    { "alpha": np.logspace(-4, 1, 6) }, #Ridge
    { "C": np.logspace(-1, 3, 5) }, #PassAggr
    { "reg_param": np.linspace(0.5, 1, 7) }, #QDA
    { "shrinkage": ["auto", 0, 0.1, 0.25, 0.5, 0.75, 1] }, #LDA
    { "var_smoothing": np.logspace(-9, 0, 10) }, #Gauss
    [ {"metric": ["minkowski"], "p":[2, 3], "n_neighbors": [5, 8, 10, 15]}, {"metric": ["chebyshev"], "n_neighbors": [5, 8, 10, 15]} ], #kNN
    { "C": np.logspace(-3, 1, 5) }, #SVC lin
    { "C": np.logspace(-3, 1, 5), "gamma": ["scale", "auto"] }, #SVC rbf
    { "C": np.logspace(-3, 1, 5),"gamma": ["scale", "auto"] }, #SVC sig
    { "max_features": ["auto", "sqrt", "log2"], "max_depth": [3, 4, 5, 8, 9, 10], "n_estimators": [100, 500, 1000, 2500, 5000] }, #RF
    { "max_features": ["auto", "sqrt", "log2"], "max_depth": [3, 4, 5, 8, 9, 10], "n_estimators": [100, 500, 1000, 2500, 5000], 'criterion': ["gini", 'entropy'] }, #ExtraTrees
    { "loss": ['deviance', 'exponential'], 'learning_rate': np.logspace(-4, 0, 10), "n_estimators": [100, 500, 1000, 2500, 5000], "max_features": ["auto", "sqrt", "log2"] },#GBClassifier
    { "hidden_layer_sizes": [(50,), (75,), (100,), (150,), (200,), (250,)], "activation": ["relu", "logistic", "tanh"], "alpha": np.logspace(-5, -1, 6) } #MLPClass
]

name = names[name_index]
this_clf = classifiers[name_index]
these_params = parameter_space[name_index]
#if we didn't have a separate validation group
"""outer_cv_results = []
for i, (train_idx, test_idx) in enumerate(cv_type.split(X,y.Catalytic, groups = X["groupID"])):
    print(i)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    #print(train_idx[0:10], X_train.head(10))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    #print(train_idx, y_train.head(10))
    
    #scale and sample features 
    X_train_new, y_train_new, X_test_scale, fold_ids = scale_resample(X_train, y_train, X_test, over_samp)
    #print(X_train_new[0:10], y_train_new[0:10])
    #print(X_train_new)
    #run feature selection and CV
    clf = GridSearchCV(estimator = this_clf, n_jobs = -1, cv=cv_type, param_grid = these_params, scoring = this_scoring, iid = True, refit = False)
    clf.fit(X_train_new, y_train_new, groups = fold_ids)
    results = clf.cv_results_
    #somehow get best combination of multiple scoring terms
    #print(results)
    ranks = []
    for key in results:
        if "rank_test_" in key:
            ranks.append(results[key])
    best_params = results['params'][np.argmin(np.sum(np.asarray(ranks), axis = 0))]
    #print(best_params) #best params will have to be identified for full data set for final model building after best model is selected
    outer_clf = this_clf.set_params(**best_params) #set the new classifier to these parameters
    outer_clf.fit(X_train_new, y_train_new) #fit on all training data - this is what GridSearchCV(refit = True) will do anyways
    y_test = y_test.assign(Prediction=outer_clf.predict( X_test_scale)) #predict based on fitted outer CV model
    accuracy = accuracy_score(y_test.Catalytic, y_test.Prediction)#, sample_weight = y_test.SampleWeights)
    recall = recall_score(y_test.Catalytic, y_test.Prediction, pos_label=0) #calculates for catalytic = 1 which is now neg
    precision = precision_score(y_test.Catalytic, y_test.Prediction, pos_label=0)#, sample_weight = y_test.SampleWeights)
    true_neg_rate = len( y_test[(y_test.Catalytic == 1) & (y_test.Prediction == 1)] )/ len(y_test[(y_test.Catalytic == 1)])
    hamming = hamming_loss(y_test.Catalytic, y_test.Prediction)
    mcc = matthews_corrcoef(y_test.Catalytic, y_test.Prediction)
    dist_rand = (recall + -1*(1-true_neg_rate)) / np.sqrt(2)

    outer_cv_results.append([ accuracy, precision, recall, true_neg_rate, hamming, mcc, dist_rand ])
    #print([ accuracy, precision, recall, true_neg_rate, hamming, mcc, dist_rand ])
outer_cv_results = np.asarray(outer_cv_results)
print("Not applied to valid set:", outer_cv_results)
#print(np.mean(outer_cv_results, axis = 0))
#print(np.std(outer_cv_results, axis = 0))"""

outer_cv_results = []
for x in range(0,10):
    X_train_new, y_train_new, valid_scale, fold_ids = scale_resample(X, y, valid_set, over_samp)
    #print(X_train_new[0:10], y_train_new[0:10])
    #print(X_train_new)
    #run feature selection and CV
    clf = GridSearchCV(estimator = this_clf, n_jobs = -1, cv=cv_type, param_grid = these_params, scoring = this_scoring, iid = True, refit = False)
    clf.fit(X_train_new, y_train_new, groups = fold_ids)
    results = clf.cv_results_
    #somehow get best combination of multiple scoring terms
    #print(results)
    ranks = []
    for key in results:
        if "rank_test_" in key:
            ranks.append(results[key])
    best_params = results['params'][np.argmin(np.sum(np.asarray(ranks), axis = 0))]
    print(best_params) #best params will have to be identified for full data set for final model building after best model is selected
    outer_clf = this_clf.set_params(**best_params) #set the new classifier to these parameters
    outer_clf.fit(X_train_new, y_train_new) #fit on all training data - this is what GridSearchCV(refit = True) will do anyways, but its selection of params is not necessary mine
    
    y_valid = y_valid.assign(Prediction=outer_clf.predict( valid_scale)) #predict based on fitted outer CV model
    accuracy = accuracy_score(y_valid.Catalytic, y_valid.Prediction)#, sample_weight = y_valid.SampleWeights)
    recall = recall_score(y_valid.Catalytic, y_valid.Prediction, pos_label=0) #calculates for catalytic = 1 which is now neg
    precision = precision_score(y_valid.Catalytic, y_valid.Prediction, pos_label=0)#, sample_weight = y_valid.SampleWeights)
    true_neg_rate = len( y_valid[(y_valid.Catalytic == 1) & (y_valid.Prediction == 1)] )/ len(y_valid[(y_valid.Catalytic == 1)])
    hamming = hamming_loss(y_valid.Catalytic, y_valid.Prediction)
    mcc = matthews_corrcoef(y_valid.Catalytic, y_valid.Prediction)
    dist_rand = (recall + -1*(1-true_neg_rate)) / np.sqrt(2)
    outer_cv_results.append([ accuracy, precision, recall, true_neg_rate, hamming, mcc, dist_rand ])
    #print([ accuracy, precision, recall, true_neg_rate, hamming, mcc, dist_rand ])
outer_cv_results = np.asarray(outer_cv_results)
#print("No outer CV:", outer_cv_results)


with open("ClassifierScores/%s/ClassifierScores_%s_%s_%s_%s_%s_%s_%s.txt"%(over_samp, relaxed, data_subset, cv_name, group_name[0:4], over_samp, opt_type, name), "w+") as outData:
    outData.write("Classifier\tAccuracy\tPrecision\tRecall\tTrueNegRate\tHammingLoss\tMCC\tDistRand\tAcc_std\tPrec_std\tRecall_std\tTrueNegRate_std\tHammingLoss_std\tMCC_std\tDistRand_std\n")
    outData.write(name + "\t" + "\t".join(map(str, np.mean(outer_cv_results, axis = 0))) + "\t" + "\t".join(map(str, np.std(outer_cv_results, axis = 0))) + "\n")