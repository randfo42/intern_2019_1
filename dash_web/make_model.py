
import pandas as pd
import time
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split,KFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN,SMOTETomek

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier , ExtraTreesClassifier, GradientBoostingClassifier ,BaggingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score,roc_curve, auc,make_scorer
from imblearn.pipeline import Pipeline
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

#stat_test_array=['CO_NAME_new','REP_PHONE_NUM_new','ADDR_new','ROAD_ADDR_new']

#stat_test_array=['CO_NAME_new','REP_PHONE_NUM_new','ADDR_new','ADDR_else','ROAD_ADDR_new','ROAD_ADDR_else']
stat_test_array=['CO_mean','RE_mean','AD_mean','RO_mean']
#stat_test_array=['CO_NAME_new_qg','CO_NAME_new_jr','CO_NAME_new_jrw','CO_NAME_new_lev','CO_NAME_new_cos','CO_NAME_new_sw','CO_NAME_new_lcs',
#                 'REP_PHONE_NUM_new_qg','REP_PHONE_NUM_new_jr','REP_PHONE_NUM_new_jrw','REP_PHONE_NUM_new_lev','REP_PHONE_NUM_new_cos','REP_PHONE_NUM_new_sw','REP_PHONE_NUM_new_lcs',
#                 'ADDR_new_qg','ADDR_new_jr','ADDR_new_jrw','ADDR_new_lev','ADDR_new_cos','ADDR_new_sw','ADDR_new_lcs',
#                 'ROAD_ADDR_new_qg','ROAD_ADDR_new_jr','ROAD_ADDR_new_jrw','ROAD_ADDR_new_lev','ROAD_ADDR_new_cos','ROAD_ADDR_new_sw','ROAD_ADDR_new_lcs'
#                  ,'SOURCE_1','SOURCE_2']

set_one_hot_incoder=False
enc=OneHotEncoder(handle_unknown='ignore')

def insert_history_query(cv,model_name,sample_n,same_label,confusion,score,status,used_var,sampling,param,PT,source_1,source_2,db):

    query = (""" INSERT INTO ci_dev.Model_history """+
                    """ VALUES (NULL,%d,'%s',%d,%d,%f,%f,%f,%f,%f,%f,%f,
                    %d,%d,%d,%d,'%s','%s',now(),"%s",%d,'%s','%s') """
                    %(cv,model_name,sample_n,same_label,
                    confusion[1][1],confusion[0][0],confusion[0][1],confusion[1][0],
                    score[0],score[1],score[2],status[0],status[1],
                    status[2],status[3],used_var,sampling,param,PT,source_1,source_2)
            )
    db.execute(query)

#TODO 통계 돌릴 시 가져오는 조건 변화
def get_data_to_db_for_statistic(db,mean_table,label_table):
    query=(""" select A.*,B.label """+
            """ from %s A """%(mean_table)+
            """ INNER JOIN %s B """%(label_table)+
            """ on A.SOURCE_ID_1 = B.SOURCE_ID_1 and A.SOURCE_ID_2=B.SOURCE_ID_2 """+
            #""" and A.SOURCE_1=B.SOURCE_1 """+ #and A.source_2=B.source_2"""+
            """ where B.label !=2 and B.source_1!=B.source_2 and A.pair_source!='intersecting_set' """) #and B.source_1!=B.source_2
    res_df=pd.read_sql_query(query,db)
    
    return res_df

def make_set_df_cleaning(df):
    label_col=list(df.columns)
    label_col=[x for x in label_col if x[0:2]=='CO' or x[0:2]=='RE' or x[0:2]=='AD' or x[0:2]=='RO'] 
    label_col.remove('COMPUTED_DT')

    new_label_col=[]
    for idx in label_col:
        if idx[-3:]=='_qg' or idx[-3:]=='_jr' or idx[-3:]=='cos' or idx[-3:]== '_sw' or idx[-3:]=='lcs':
            new_label_col.append(idx)

    co=[x for x in new_label_col if x[0:2]=='CO'] 
    re=[x for x in new_label_col if x[0:2]=='RE'] 
    ad=[x for x in new_label_col if x[0:2]=='AD']
    ro=[x for x in new_label_col if x[0:2]=='RO']

    co_list=list(df[co].mean(axis=1))
    re_list=list(df[re].mean(axis=1))
    ad_list=list(df[ad].mean(axis=1))
    ro_list=list(df[ro].mean(axis=1))

    res_df=df.copy()

    res_df['CO_mean']=co_list
    res_df['RE_mean']=re_list
    res_df['AD_mean']=ad_list
    res_df['RO_mean']=ro_list
    
    return res_df


def make_set(df):
    global set_one_hot_incoder
    global stat_test_array

    if 'CO_mean' not in df.columns.values.tolist():
        df=make_set_df_cleaning(df)

    
    each_len=1

    CO_NAME_l=[]
    REP_PHONE_l=[]
    ADDR_l=[]
    ROAD_ADDR_l=[]

    #source_1=list(df['SOURCE_1'])
    #source_2=list(df['SOURCE_2'])
    source_o=[]
    
    for idx in list(df['pair_source']):
        source_o.append([idx])

    #for idx in range(0,len(source_2)):
    #    source_o.append([source_1[idx]+source_2[idx]])

    if set_one_hot_incoder==False:
        enc.fit(source_o)
        set_one_hot_incoder=True

    source=enc.transform(source_o).toarray()

    print(source)

    
    for idx in stat_test_array:
        
        if idx[0:2]=='CO':
            CO_NAME_l.append(list(df[idx]))
        if idx[0:2]=='RE':
            REP_PHONE_l.append(list(df[idx]))
        if idx[0:2]=='AD':
            ADDR_l.append(list(df[idx]))
        if idx[0:2]=='RO':
            ROAD_ADDR_l.append(list(df[idx]))


    res_list=[]
    for idx in range(0,len(CO_NAME_l[0])):

        add=[]

        for k in range(0,each_len):
            add.append(CO_NAME_l[k][idx])
        
        for k in range(0,each_len):
            add.append(REP_PHONE_l[k][idx])

        if ROAD_ADDR_l[0][idx] is None:
            for k in range(0,each_len):     
               add.append(ADDR_l[k][idx])
        elif math.isnan(ROAD_ADDR_l[0][idx]) :
            for k in range(0,each_len):     
               add.append(ADDR_l[k][idx])
        else:
            aver_a=0
            aver_r=0
            for k in range(0,each_len):
                aver_a=aver_a+ADDR_l[k][idx]
                aver_r=aver_r+ROAD_ADDR_l[k][idx]        
            if aver_a>aver_r:
                for k in range(0,each_len):
                    add.append(ADDR_l[k][idx])
            else:
                for k in range(0,each_len):
                    add.append(ROAD_ADDR_l[k][idx])
        
        for k in source[idx]:
            add.append(k)

        res_list.append(add)

   
    for idx in range(0,len(res_list)):
        for i in range(0,len(res_list[idx])): 
            if math.isnan(res_list[idx][i]):
                res_list[idx][i]=0
            
    print(res_list[10])

    return res_list


def history_set(model,test_y,test_x):
    
    predicted_y=model.predict(test_x)

    fpr,tpr,auc=roc_cur(model,test_y,test_x)

    confusion=confusion_matrix(test_y,predicted_y)

    score=[0,0,0,0,0]

    score[0]=accuracy_score(test_y,predicted_y)
    score[1]=precision_score(test_y,predicted_y,average='binary')
    score[2]=recall_score(test_y,predicted_y,average='binary')
    score[3]=auc
    score[4]=f1_score(test_y,predicted_y)

    param=str(model.get_params())

    return confusion,score,fpr,tpr,param




def statistic_set(model,df,test_set,k_fold):
    start_time=time.time()

    #input_x,test_x,input_y,test_y=train_test_split(input_x,input_y,
    #                                test_size=0.25,stratify=input_y,random_state=43)
    input_y= list(df['label'])
    input_x= make_set(df)

    if not test_set.empty:
        src_df_res=test_set
        test_x=make_set(src_df_res)
        test_y=list(src_df_res['label'])
        df=df[~df[['SOURCE_ID_1','SOURCE_ID_2']].apply(tuple,1).isin(src_df_res[['SOURCE_ID_1','SOURCE_ID_2']].apply(tuple,1))]
        input_y= list(df['label'])
        input_x= make_set(df)
    else:
        input_x,test_x,input_y,test_y=train_test_split(input_x,input_y,
                                    test_size=0.25,stratify=input_y,random_state=43)
    


    if k_fold==1:
        cv= KFold(5,shuffle=True,random_state=43)
        for i,(idx_train,idx_test) in enumerate(cv.split(input_x,input_y)):
            x_train_list=[]
            y_train_list=[]
            x_test_list=[]
            y_test_list=[]

            for idx in idx_train:
                x_train_list.append(input_x[idx])
                y_train_list.append(input_y[idx])

            for idx in idx_test:
                x_test_list.append(input_x[idx])
                y_test_list.append(input_y[idx])
            
            x_train_list,y_train_list=SMOTEENN(random_state=0).fit_sample(x_train_list,y_train_list)

            clf=model.fit(x_train_list,y_train_list)

            print("score = %.8f"%(clf.score(x_test_list,y_test_list)))
    
    input_x,input_y = SMOTEENN(random_state=0).fit_sample(input_x,input_y)

    fin_clf=model.fit(input_x,input_y)
    fin_score=fin_clf.score(test_x,test_y)

    print('final_score')
    print(fin_score)
    res_time=time.time()-start_time
    return fin_clf,fin_score,test_y,fin_clf.predict(test_x),test_x,res_time

def grid_serch_set(df,test_set,estimator,params):
    start_time=time.time()

    input_y= list(df['label'])
    input_x= make_set(df)

    scoring = {'Precision': make_scorer(precision_score)}

    if not test_set.empty:
        src_df_res=test_set
        test_x=make_set(src_df_res)
        test_y=list(src_df_res['label'])
        df=df[~df[['SOURCE_ID_1','SOURCE_ID_2']].apply(tuple,1).isin(src_df_res[['SOURCE_ID_1','SOURCE_ID_2']].apply(tuple,1))]
        input_y= list(df['label'])
        input_x= make_set(df)
    else:
        input_x,test_x,input_y,test_y=train_test_split(input_x,input_y,
                                    test_size=0.25,stratify=input_y,random_state=43)
    
    grid=GridSearchCV(estimator=estimator,param_grid=params, refit='Precision',scoring=scoring,
                   cv=5)
    
    grid=grid.fit(input_x,input_y)

    res_time=time.time()-start_time

    print('best!')
    print(grid.best_params_)
    print(grid.best_score_)

    return grid.best_estimator_,test_y,test_x,res_time

def logistic_score(df_dict,test_set,k_fold):

    model= LogisticRegression(random_state=0,solver='liblinear')

    return statistic_set(model,df_dict,test_set,k_fold)

#random forest
def random_forest(df_dict,test_set,k_fold):

    model = RandomForestClassifier(bootstrap=True,class_weight=None,max_depth=100,
                                    n_estimators=2,random_state=43)
    
    return statistic_set(model,df_dict,test_set,k_fold)

def GBC(df,test_set,k_fold):
    model = GradientBoostingClassifier(n_estimators=200,learning_rate=1
                                            ,max_depth=1,random_state=43)
    
    return statistic_set(model,df,test_set,k_fold)

def ENSE(df,test_set,k_fold):
    model1=LogisticRegression(random_state=43)
    model2=QuadraticDiscriminantAnalysis()
    model3=GaussianNB()
    ensemble = VotingClassifier(estimators=[('lr', model1), 
                                            ('qda', model2), 
                                            ('gnb', model3)], 
                                            voting='soft')
    return statistic_set(ensemble,df,test_set,k_fold)

def ECLF(df,test_set,k_fold):
    
    rf = RandomForestClassifier(bootstrap=True,class_weight=None, max_depth=100, n_estimators=2,random_state=43)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=43)
    knn = KNeighborsClassifier()
    svc = SVC(probability=True)
    eclf = VotingClassifier(estimators=[('Random Forests', rf), 
                            ('Extra Trees', et), ('KNeighbors', knn), 
                            ('SVC', svc)], voting='soft')
    
    return statistic_set(eclf,df,test_set,k_fold)

def KNN_RF(df,test_set,k_fold):
    rf = RandomForestClassifier(bootstrap=False,max_depth=200,n_estimators=2)
    knn=KNeighborsClassifier(n_neighbors=500)

    knn_rf=VotingClassifier(estimators=[('Random Forests',rf),('KNeigbors',knn)],voting='soft')

    return statistic_set(knn_rf,df,test_set,k_fold)

def grid_serch(df,test_set):


    p1 = Pipeline([['RF', RandomForestClassifier()]])
    p2 = Pipeline([['KNN', KNeighborsClassifier()]])

    model= Pipeline([('sampling', SMOTEENN()),
        ('eclf', VotingClassifier(estimators=[("p1",p1), ("p2",p2)],voting='soft'))])

    params = {'eclf__p1__RF__n_estimators': [150,175,200],
          'eclf__p1__RF__bootstrap': [False], 
          'eclf__p1__RF__max_depth' : [1,5,3,15], 
         'eclf__p2__KNN__n_neighbors': [70,75,450,400]}

    return grid_serch_set(df,test_set,model,params)

def grid_serch_gbc(df,test_set):


    model= Pipeline([('sampling', SMOTEENN()),
        ('gbc', GradientBoostingClassifier())])

    params= {'gbc__n_estimators':[400,300,200,100],
            'gbc__learning_rate':[1,0.7,0.05,0.03],
            'gbc__max_depth':[1,2],
            'gbc__min_samples_leaf':[1,0.1],
            'gbc__random_state':[43]
             }
    return grid_serch_set(df,test_set,model,params)

def roc_cur(model,y_test,x_test):
    y_predict=model.predict_proba(x_test)[:,1]

    fpr,tpr,thresh=roc_curve(y_test,y_predict)
    roc_auc =auc(fpr,tpr)

    return fpr,tpr,roc_auc
