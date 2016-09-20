from __future__ import print_function
import pandas as pd
import numpy as np
import itertools
import random
import xgboost as xgb
import sys
from sklearn.metrics import roc_auc_score


##############################################################
# Basic options
##############################################################

# choose whether to use local validation 
UseValidation=True
    
DataDir='../data/'
if ~UseValidation:
    SolutionToOutput='xgb_v4_valid_withtest.csv'

# a subset of features selected in the stage of EDA. 
FeatSelections=set(['group_1', 'char_2_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x',
                    'char_13', 'char_17', 'char_22', 'char_25', 'char_34', 'char_36', 'char_37',
                    'char_38', # numerical
                    'char_10_y', 'year_y', 'month_y', 'weekno'])
##############################################################
## Auxilary functions
##############################################################

# compute mutual entropy with outcome 0/1
def mutual_entropy(a, l):
    n = len(a)
    a_ent = -np.sum([x*np.log(x) for x in a.value_counts()/n])
    l_ent = -np.sum([x*np.log(x) for x in l.value_counts()/n])
    al_ent = -np.sum([x*np.log(x) for x in a[l==0].value_counts()/n]) - \
             np.sum([x*np.log(x) for x in a[l==1].value_counts()/n])
    return a_ent + l_ent - al_ent


# create leave-one-person-out counting/proportion features 
def create_counting_features(thistrain, field='people_id'):
    c=dict()
    people_list=thistrain[field].unique()
    people_group=thistrain.groupby(field)
    
    for name in set(['people_id', field]):
        print(name, end=',')
        sys.stdout.flush()
        c[name]=thistrain[name].value_counts(dropna=False)-1
    for name in thistrain.columns.values:
        if name not in set(['people_id', field, 'activity_id', 'fold', 'char_38', 'outcome', 'date_x', 'date_y']) and not name.startswith('extrapolate') and name in FeatSelections:
                print(name, end=',')
                sys.stdout.flush()
                c_total=thistrain[name].value_counts(dropna=False)
                count_total=sum(c_total)
                c_people=people_group[name].value_counts(dropna=False)
                c_people=c_people.reset_index(name='counts')
                people_count_total=c_people.groupby(field)['counts'].sum()
                c_people['proportion']=c_people.apply(lambda x: (c_total[x[name]] - x['counts']) / (count_total - people_count_total[x[field]]), axis=1)
                c[name]=c_people.groupby([field, name])['proportion'].mean()
    print('... [Done]')
    return c


def create_train_data(train, test=None, c=None):
    train_feats=np.empty([len(train), 0])
    train_label=np.array(train['outcome'])
    feature_names=[];
    if test is not None:
        test_feats=np.empty([len(test), 0])
        
    for name in train.columns.values:
        if name not in set(['people_id', 'activity_id', 'fold', 'char_38', 'outcome', 'date_x', 'date_y']) and name in FeatSelections:                        
            print(name, end=',')
            sys.stdout.flush()
            feature_names=feature_names + [name]
            new_feat=np.array([c[name][train[['people_id', name]].to_records(index=False)]]).T            
            train_feats=np.append(train_feats, new_feat, 1)
            if test is not None:
                new_feat=np.array([c[name][test[['people_id', name]].to_records(index=False)]]).T          
                test_feats=np.append(test_feats, new_feat, 1)
        if name in set(['char_38', 'extrapolate2', 'extrapolate3', 'extrapolate4']):
            print(name, end=',')
            sys.stdout.flush()
            feature_names=feature_names + [name]
            new_feat=np.array([train[name]], dtype=np.float).T
            train_feats=np.append(train_feats, new_feat, 1)
            if test is not None:
                new_feat=np.array([test[name]], dtype=np.float).T
                test_feats=np.append(test_feats, new_feat, 1)
        if False and name in set(['people_id']):
            print(name, end=',')
            feature_names=feature_names + [name]
            new_feat=np.array([c[name][train[name]]]).T
            train_feats=np.append(train_feats, new_feat, 1)
            if test is not None:
                new_feat=np.array([c[name][test[name]]]).T
                test_feats=np.append(test_feats, new_feat, 1)                
    print(' ... [Done]')
    if test is None:
        return feature_names,train_feats,train_label
    else:
        return feature_names,train_feats,train_label,test_feats


def create_train_data2(train, test=None, c=None, field='people_id'):
    train_feats=np.empty([len(train), 0])
    train_label=np.array(train['outcome'])
    feature_names=[];
    if test is not None:
        test_feats=np.empty([len(test), 0])
        
    for name in train.columns.values:
        if name not in set(['people_id', field, 'activity_id', 'fold', 'char_38', 'outcome', 'date_x', 'date_y']) and name in FeatSelections:                        
            print(name, end=',')
            sys.stdout.flush()
            feature_names=feature_names + [name]
            new_feat=np.array([c[name][train[[field, name]].to_records(index=False)]]).T            
            train_feats=np.append(train_feats, new_feat, 1)
            if test is not None:
                new_feat=np.array([c[name][test[[field, name]].to_records(index=False)]]).T          
                test_feats=np.append(test_feats, new_feat, 1)
        if name in set(['char_38']):
            print(name, end=',')
            sys.stdout.flush()
            feature_names=feature_names + [name]
            new_feat=np.array([train[name]], dtype=np.float).T
            train_feats=np.append(train_feats, new_feat, 1)
            if test is not None:
                new_feat=np.array([test[name]], dtype=np.float).T
                test_feats=np.append(test_feats, new_feat, 1)
        if False and name in set(['people_id']):
            print(name, end=',')
            feature_names=feature_names + [name]
            new_feat=np.array([c[name][train[name]]]).T
            train_feats=np.append(train_feats, new_feat, 1)
            if test is not None:
                new_feat=np.array([c[name][test[name]]]).T
                test_feats=np.append(test_feats, new_feat, 1)
    print('... [Done]')
    if test is None:
        return feature_names,train_feats,train_label
    else:
        return feature_names,train_feats,train_label,test_feats  
    

    
    
##############################################################
## Start here
##############################################################

# load and basic preprocessing 
print("Read people.csv...")
people=pd.read_csv(DataDir + 'people.csv',
                   dtype={'char_38': np.int32},
                   parse_dates=['date'])

print("Load act_train.csv...")
act_train=pd.read_csv(DataDir + 'act_train.csv',
                      parse_dates=['date'])

if UseValidation:
    print("Split train into mytrain and mytest...")
    mask = pd.read_csv(DataDir + "my_test_mask.csv")['mask']
    act_test=act_train[mask]
    act_train=act_train[~mask]
else:
    print("Load act_test.csv...")
    act_test=pd.read_csv(DataDir + 'act_test.csv',
                         parse_dates=['date'])


print("Process tables...")
for table in [act_train, act_test]:
    table['year'] = table['date'].dt.year
    table['month'] = table['date'].dt.month
    table['day'] = table['date'].dt.day
    table['weekday'] = table['date'].dt.weekday
    table['weekno'] = table['date'].dt.date.apply(lambda x: x.isocalendar()[1])
    ##table.drop('date', axis=1, inplace=True)
    table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
    for i in range(1, 11):
        table['char_' + str(i)].fillna('type -999', inplace=True)
        table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    table['people_id'] = table['people_id'].str.lstrip('ppl_').astype(np.float).astype(np.int32)

people['year'] = people['date'].dt.year
people['month'] = people['date'].dt.month
people['day'] = people['date'].dt.day
people['weekday'] = people['date'].dt.weekday

#people.drop('date', axis=1, inplace=True)
people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
people['people_id'] = people['people_id'].str.lstrip('ppl_').astype(np.float).astype(np.int32)
for i in range(1, 10):
    people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
for i in range(10, 38):
    people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)



# merge people data into activities
print('Merge people and train/test ... ')
train=pd.merge(people, act_train, on='people_id')
test=pd.merge(people, act_test, on='people_id')
del(act_train)
del(act_test)


# create 5-fold split by 'people_id'
print('Create fold split for stacking ... ')
np.random.seed(0) # IMPORTANT, DON'T CHANGE SEED
folds_people=pd.DataFrame({'fold':np.random.randint(0, 5, len(people['people_id'].unique())), 'people_id': people['people_id'].unique()})
cv=pd.merge(folds_people, train, on='people_id')


# load rules and interpolations
print('Load pre-computed interpolations ... ')
if UseValidation:
    cv['extrapolate']=pd.Series.from_csv('mycv:extrapolate.csv')
    train['extrapolate']=pd.Series.from_csv('mytrain:extrapolate.csv')
    test['extrapolate']=pd.Series.from_csv('mytest:extrapolate.csv')

    cv['extrapolate2']=pd.Series.from_csv('mycv:extrapolate2.csv')
    train['extrapolate2']=pd.Series.from_csv('mytrain:extrapolate2.csv')
    test['extrapolate2']=pd.Series.from_csv('mytest:extrapolate2.csv')

    cv['extrapolate3']=pd.Series.from_csv('mycv:extrapolate3.csv')
    train['extrapolate3']=pd.Series.from_csv('mytrain:extrapolate3.csv')
    test['extrapolate3']=pd.Series.from_csv('mytest:extrapolate3.csv')

    cv['extrapolate4']=pd.Series.from_csv('mycv:extrapolate4.csv')
    train['extrapolate4']=pd.Series.from_csv('mytrain:extrapolate4.csv')
    test['extrapolate4']=pd.Series.from_csv('mytest:extrapolate4.csv')        

else:
    cv['extrapolate']=pd.Series.from_csv('cv:extrapolate.csv')
    train['extrapolate']=pd.Series.from_csv('train:extrapolate.csv')
    test['extrapolate']=pd.Series.from_csv('test:extrapolate.csv')

    cv['extrapolate2']=pd.Series.from_csv('cv:extrapolate2.csv')
    train['extrapolate2']=pd.Series.from_csv('train:extrapolate2.csv')
    test['extrapolate2']=pd.Series.from_csv('test:extrapolate2.csv')

    cv['extrapolate3']=pd.Series.from_csv('cv:extrapolate3.csv')
    train['extrapolate3']=pd.Series.from_csv('train:extrapolate3.csv')
    test['extrapolate3']=pd.Series.from_csv('test:extrapolate3.csv')        

    cv['extrapolate4']=pd.Series.from_csv('cv:extrapolate4.csv')
    train['extrapolate4']=pd.Series.from_csv('train:extrapolate4.csv')
    test['extrapolate4']=pd.Series.from_csv('test:extrapolate4.csv')        

test2train=test[test['extrapolate']!=-999]
test2train['outcome']=test2train['extrapolate']
train_new=pd.concat((train,test2train))
# print('Compute Mutual Entropy for feature filtering ... ')
# a=dict()
# b=dict()
# gain=dict()
# newtrain=cv[cv['extrapolate2']!=-999]
# for name in newtrain.columns.values:
#     if name not in set(['people_id', 'activity_id', 'char_38', 'fold', 'date', 'extrapolate', 'extrapolate2']):
# #        print(name, end=': ')
#         gain[name] = mutual_entropy(newtrain[name], newtrain['outcome'])
# #        print(gain[name])
# #        a[name], b[name] = get_log_posterior(newtrain, name)


# compute all leave-one-out related statistics
#print('Creating LOO counting features for train ... ')
#c=create_counting_features(train)
print('Creating LOO counting features for all ... ')
c_all=create_counting_features(pd.concat((train, test)))




print('Prepare feature set1 ... ')        
#feature_names,cv_feats,cv_label = create_train_data(cv, None, c)
feature_names,train_feats,train_label,test_feats = create_train_data(train, test, c_all)

print('Prepare feature set2 ... ')        
#feature_names2,cv2_feats,cv2_label = create_train_data2(train, None, c)
feature_names2,train2_feats,train2_label,test2_feats = create_train_data2(train_new, test, c_all)



# create weight vectors
print('Compute Sample weights')
weight2v=np.array(train_new.groupby('group_1')['people_id'].unique().apply(lambda x: len(x)).apply(lambda x: 5*x*0.2**x)[train_new['group_1']])
weight2= np.array(1./train_new['group_1'].value_counts()[train_new['group_1']])
weight3=np.array(train_new.groupby('group_1')['people_id'].unique().apply(lambda x: len(x)).apply(lambda x: 5*x*0.2**x)[train_new['group_1']])
weight3=weight3 * np.array(train_new['people_id'].value_counts().apply(lambda x: 1/x)[train_new['people_id']])



# create prediction for leaky data
submission0=pd.DataFrame({'activity_id':test['activity_id'][test['extrapolate']!=-999],'outcome': test['extrapolate'][test['extrapolate']!=-999]})



# create prediction for semi-leaky data
print('train and create prediction for semi-leaky data ... ')
params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": ["auc", "logloss"],
        "eta": 0.02,
        "gamma":100,
        "max_depth": 3,
#        "subsample": subsample,
#        "colsample_bytree": colsample_bytree,
        "silent": 1,
        'nthread': 4
}


filter=np.array(train['extrapolate']==-999) & np.array(train['extrapolate2']!=-999)
dtrain=xgb.DMatrix(train_feats[filter],
                   train_label[filter], missing=-999)

filter_test=np.array(test['extrapolate']==-999) & np.array(test['extrapolate2']!=-999)
dtest=xgb.DMatrix(test_feats[filter_test], missing=-999)

gbm1 = xgb.train(params, dtrain, num_boost_round=201, evals=[(dtrain, 'train')], verbose_eval=15)
print(gbm1.get_fscore())

submission1=pd.DataFrame({'activity_id':test['activity_id'][filter_test],'outcome': gbm1.predict(dtest)})

# create prediction for leaky free data
print('train and create prediction for leaky free data ... ')
params2 = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": ["auc", "logloss"],
        "eta" : 0.05,
        "gamma" : 10,
        "max_depth" : 8,
#        "subsample": subsample,
#        "colsample_bytree": colsample_bytree,
        "silent": 1,
        'nthread': 4
}

dtrain2=xgb.DMatrix(train2_feats,
                    train2_label, weight=weight2, missing=-999)

filter_test2=np.array(test['extrapolate2']==-999)
dtest2=xgb.DMatrix(test2_feats[filter_test2], missing=-999)

gbm2 = xgb.train(params2, dtrain2, num_boost_round=81, evals=[(dtrain2, 'train')], verbose_eval=15)
print(gbm2.get_fscore())

submission2=pd.DataFrame({'activity_id':test['activity_id'][filter_test2],'outcome': gbm2.predict(dtest2)})


# output final solution
submission=pd.concat((submission0, submission1, submission2))
if UseValidation:
    a=pd.merge(test, submission, on='activity_id')
    print('Local AUC: ', roc_auc_score(a['outcome_x'].as_matrix().astype(np.int),
                                       a['outcome_y'].as_matrix()))
    a[['activity_id', 'outcome_x', 'outcome_y']].to_csv('local_' + SolutionToOutput, index=False)
else:
    print('Write solution ... ')
    submission.to_csv('submit_' + SolutionToOutput, index=False)

print('Done')
