import numpy as np
import pandas as pd
import gc
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split
import os
from catboost import CatBoostClassifier

os.chdir("C:/Munish/av/mck")

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

label=train['renewal']

t_len=train.shape[0]

train.drop(['renewal'],axis=1,inplace=True)

comb=train.append(test)

comb['age']=round(comb['age_in_days']/365)

comb['inc_age']=comb['Income']/comb['age']

comb['inc_prem']=comb['Income']/comb['premium']
comb['ratio1']=comb['no_of_premiums_paid']/comb['age']

comb['ratio2']=comb['Count_3-6_months_late']/comb['no_of_premiums_paid']
comb['ratio3']=comb['Count_6-12_months_late']/comb['no_of_premiums_paid']
comb['ratio4']=comb['Count_more_than_12_months_late']/comb['no_of_premiums_paid']
comb['ratio5']=comb['Count_3-6_months_late']/comb['Count_6-12_months_late']
comb['ratio6']=comb['Count_6-12_months_late']/comb['Count_more_than_12_months_late']

ls1=['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late']
ls2=['perc_premium_paid_by_cash_credit','age','Income','application_underwriting_score','no_of_premiums_paid','premium']
ls3=ls1+ls2

for i in range(len(ls1)):
    for j in range(len(ls2)):
        comb['var_avg'+str(i)+str(j)]=comb.groupby(ls1[i])[ls2[j]].transform('mean')
        comb['var_min'+str(i)+str(j)]=comb.groupby(ls1[i])[ls2[j]].transform('min')
        comb['var_max'+str(i)+str(j)]=comb.groupby(ls1[i])[ls2[j]].transform('max')
        comb['var_med'+str(i)+str(j)]=comb.groupby(ls1[i])[ls2[j]].transform('median')

comb['prem_by_cc']=comb['premium']*comb['perc_premium_paid_by_cash_credit']
comb['diff_cc']=comb['premium']-comb['prem_by_cc']
comb['tot_paid']=comb['premium']*comb['no_of_premiums_paid']
comb['tot_paid_ratio']=comb['Income']/comb['tot_paid']
comb['application_underwriting_score']=comb['application_underwriting_score'].apply(lambda x:-99 if pd.isnull(x) else x)
comb['comb1']=str(comb['residence_area_type'])+str('_')+str(comb['sourcing_channel'])
comb['sourcing_channel']=comb['sourcing_channel'].astype('category')
comb['residence_area_type']=comb['residence_area_type'].astype('category')
comb['comb1']=comb['comb1'].astype('category')

for j in range(len(ls3)):
        comb['var2_avg'+str(j)]=comb.groupby('comb1')[ls3[j]].transform('mean')
        comb['var2_min'+str(j)]=comb.groupby('comb1')[ls3[j]].transform('min')
        comb['var2_max'+str(j)]=comb.groupby('comb1')[ls3[j]].transform('max')
        comb['var2_med'+str(j)]=comb.groupby('comb1')[ls3[j]].transform('median')
        
ls4=list(set(ls3)-set(['no_of_premiums_paid']))

for j in range(len(ls4)):
        comb['var3_avg'+str(j)]=comb.groupby('no_of_premiums_paid')[ls4[j]].transform('mean')
        comb['var3_min'+str(j)]=comb.groupby('no_of_premiums_paid')[ls4[j]].transform('min')
        comb['var3_max'+str(j)]=comb.groupby('no_of_premiums_paid')[ls4[j]].transform('max')
        comb['var3_med'+str(j)]=comb.groupby('no_of_premiums_paid')[ls4[j]].transform('median')

train_df = comb.iloc[:t_len]
test_df = comb.iloc[t_len:]

gc.collect()

train_df['label']=label

v1=list(set(train_df.columns)-set(['id','label']))

folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

sub_preds = np.zeros(test_df.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[v1], train_df['label'])):
    train_x, train_y = train_df[v1].iloc[train_idx], train_df['label'].iloc[train_idx]
    valid_x, valid_y = train_df[v1].iloc[valid_idx], train_df['label'].iloc[valid_idx]

    clf = LGBMClassifier(
    boosting_type= 'gbdt',
    objective= 'binary',
    metric= 'binary_logloss',
    learning_rate= 0.02,
    max_depth= 4,
    num_leaves= 10,
    feature_fraction= 0.7,
    bagging_fraction= 1,
    bagging_freq= 20,
    nthread=4,
    n_estimators=5000)

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            verbose= 100, early_stopping_rounds= 200)

    sub_preds += clf.predict_proba(test_df[v1], num_iteration=clf.best_iteration_)[:,1]/ folds.n_splits

sub_preds1 = np.zeros(test_df.shape[0])

cat_cols=[i for i,x in enumerate(v1) if x in ['sourcing_channel','residence_area_type','comb1']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[v1], train_df['label'])):
    train_x, train_y = train_df[v1].iloc[train_idx], train_df['label'].iloc[train_idx]
    valid_x, valid_y = train_df[v1].iloc[valid_idx], train_df['label'].iloc[valid_idx]

    model=CatBoostClassifier(iterations=5000,learning_rate=0.02,depth=4,loss_function='Logloss',od_type='Iter')

    model.fit(train_x, train_y, eval_set=(valid_x, valid_y), 
            verbose= 100, use_best_model=True,cat_features=cat_cols)

    sub_preds1 += model.predict_proba(test_df[v1])[:,1]/ folds.n_splits

sub_pred_final=0.5*sub_preds+0.5*sub_preds1  

sub=pd.DataFrame({'id':test['id'],'renewal':sub_pred_final,'prem':test['premium']})


def funcx(a,b,c):
    t1=-5*np.log(1-(100*b/20))
    d=-400*np.log(1-(t1/10))
    e=((a+a*b)*c-d)
#    if (a+a*b)>1:
#        e=0
    return d,e


ls1=list(range(1,18,1))
ls2=[i/100 for i in ls1]+[0.005]



inc=[]

for i,row in sub.iterrows():
       print ('row-%s'%i)
       prem=0
       obj=0
       for i in range(len(ls2)):
           v1,v2=funcx(row['renewal'],ls2[i],row['prem'])
           if v2>obj:
               obj=v2
               prem=v1
       inc.append(prem)

sub['incentives']=inc
       

sub.drop(['prem'],inplace=True,axis=1)
sub.to_csv("sub.csv",index=False)