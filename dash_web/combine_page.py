import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input,Output,State
import matplotlib.pylab as plt
import seaborn as sns
import plotly.graph_objs as go
import json

from server import app 
import global_var as gv
import base64
import make_model


def  label_density(df):
    grouped = df.groupby('label')
    f1 = plt.figure(figsize = (30, 15))

    for label_no,group in grouped:
        print('LABEL ', label_no, 'Histogram')
        x1 = group['ADDR_new'].tolist()
       
        sns.distplot(x1.copy(), hist = False, label = 'LABEL ' + str(label_no),
                kde_kws = {'shade': True})
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
        plt.title('ADDR_new',fontsize=40)
    f1.savefig('1features_by_label.png')

    f2 = plt.figure(figsize = (30, 15))
    for label_no, group in grouped:
        x2 = group['REP_PHONE_NUM_new'].tolist()
      
        sns.distplot(x2, hist = False, label = 'LABEL ' + str(label_no),
                kde_kws = {'shade': True})
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
        plt.title('REP_PHONE_NUM_new',fontsize=40)
    f2.savefig('2features_by_label.png')

    f3 = plt.figure(figsize = (30, 15))
    for label_no, group in grouped:
        x3 = group['ADDR_else'].tolist()
      
        sns.distplot(x3, hist = False, label = 'LABEL ' + str(label_no),
                kde_kws = {'shade': True})
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
        plt.title('ADDR_else',fontsize=40)
    f3.savefig('3features_by_label.png')

    f4 = plt.figure(figsize = (30, 15))
    for label_no, group in grouped:
        x4 = group['ROAD_ADDR_else'].tolist()
        x4 = [x for x in x4 if str(x) != 'nan']
       
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
        
        sns.distplot(x4, hist = False, kde = True, label = 'LABEL ' + str(label_no),
                kde_kws = {'shade': True})
        plt.title('ROAD_ADDR_else',fontsize=40)
            
    f4.savefig('4features_by_label.png')
    
    f5 = plt.figure(figsize = (30, 15))
    for label_no, group in grouped:
        x4 = group['CO_NAME_new'].tolist()
        x4 = [x for x in x4 if str(x) != 'nan']
       
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
        
        sns.distplot(x4, hist = False, kde = True, label = 'LABEL ' + str(label_no),
                kde_kws = {'shade': True})
        plt.title('CO_NAME_new',fontsize=40)
            
    f5.savefig('5features_by_label.png')

    f6 = plt.figure(figsize = (30, 15))
    for label_no, group in grouped:
        x4 = group['ROAD_ADDR_new'].tolist()
        x4 = [x for x in x4 if str(x) != 'nan']
       
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
        
        sns.distplot(x4, hist = False, kde = True, label = 'LABEL ' + str(label_no),
                kde_kws = {'shade': True})
        plt.title('ROAD_ADDR_new',fontsize=40)
            
    f6.savefig('6features_by_label.png')


def make_score_cf_table_col_row(model_name,score,cf):

    sc_col=[{"name":'','id':' '},{"name":model_name,'id':model_name}]
    sc_data=[{' ':'Accuracy',model_name:'{0:0.3f}'.format(score[0])},
            {' ':'Precision',model_name:'{0:0.3f}'.format(score[1])},
            {' ':'Recall',model_name:'{0:0.3f}'.format(score[2])},
            {' ':'roc curve auc',model_name:'{0:0.3f}'.format(score[3])},
            {' ':'F1 score',model_name:'{0:0.3f}'.format(score[4])}]
    
    cf_col=[{"name":' ','id':' '},{"name":'predict 0','id':'predict 0'},{"name":'predict 1','id':'predict 1'}]
    cf_data=[{' ':'Observed 0','predict 0':cf[0][0],'predict 1':cf[0][1]},
             {' ':'Observed 1','predict 0':cf[1][0],'predict 1':cf[1][1]}]

    return sc_col,sc_data,cf_col,cf_data


# data에 대한 logistic 과 random forest의 결과 화면
combine_page=html.Div([
    
    html.A('this page show model`s score'),
    html.Br(),
    html.Br(),
    html.A('현재 가지고 있는 data로 모델을 만들고 score를 보려면 create model 버튼을,'),
    html.Br(),
    html.Br(),
    html.A('모델로 data를 predict 하여 labeling 하려면 adapt model버튼을 누르세요 '),
    
    html.Br(),

    html.Div(id='model_score_show',style = {'display':'none'}),
    
    html.Div(id='model_score_show_roc',style = {'width': '49%','height':400,'float':'left','display':'inline-block'}),

    html.Div(id='model_score_table',style = {'width': '49%','height':400,'display':'inline-block','float':'left'}),
    
    html.Div([
        dcc.Tabs(id='cfs_tabs' , value='1', children=[
            dcc.Tab(label='logistic_regression',value='1'),
            dcc.Tab(label='random_forest',value='2'),
            dcc.Tab(label='GBC',value='3'),
            dcc.Tab(label='log,qda,gn',value='4'),
            dcc.Tab(label='ECLF',value='5'),
            dcc.Tab(label='rf knn grid',value='6'),
            dcc.Tab(label='gbc grid',value='7')
        ],style= {'position':'relative'}),
        html.Div(id='cfs_content')
    ],style = {'width': '49%','height':400,'display':'inline-block','float':'left'}),

    #html.Div(id='model_etc_table',style = {'width': '49%','height':400,'display':'inline-block'}),
    html.Div([
    dcc.Tabs(id='label_image_tab',children=[
                        dcc.Tab(label='ADDR_new',value='1'),
                        dcc.Tab(label='ADDR_else',value='2'),
                        dcc.Tab(label='ROAD_ADDR_new',value='3'),
                        dcc.Tab(label='ROAD_ADDR_else',value='4'),
                        dcc.Tab(label='REP_PHONE_NUM_new',value='5'),
                        dcc.Tab(label='CO_NAME_new',value='6'),
                ]),
                html.Div(id='label_image_div',style={'width': '100%','height':600,'display':'inline-block','float':'left'})
    ],style= {'width': '49%','height':600,'display':'inline-block','float':'left'}),
    
    
    html.Br(),
    html.Br(),
    
    html.Button('create model',id='create_model_btn'),
    html.A(html.Button('adapt model',id='adapt_btn'),href='/adpt'),
    html.Br(),

])


# db에 있는 data에 대해 model을 평가하고 화면을 띄우고, output 저장.
@app.callback(
    Output('model_score_show','children'),
    [Input('create_model_btn','n_clicks')]
)
def create_model(n_clicks):

    

   
    

    if n_clicks is not None:
        print('x')
        
        #var_query=""" select * from %s limit 1 """%(mean_table)
        #var_df=pd.read_sql_query(var_query,db_0_50)
        
        variable_name = make_model.stat_test_array
        variable_name_string=",".join((str(n) for n in variable_name))
        print(variable_name_string)

        

        mean_df_con=make_model.get_data_to_db_for_statistic(gv.db_0_50,gv.mean_table,gv.label_table)

        label_col=list(mean_df_con.columns)
        label_col=[x for x in label_col if x[0:2]=='CO' or x[0:2]=='RE' or x[0:2]=='AD' or x[0:2]=='RO'] 
        label_col.remove('COMPUTED_DT')
        

        new_label_col=[]
        for idx in label_col:
            if idx[-3:]=='_qg' or idx[-3:]=='_jr' or idx[-3:]=='cos' or idx[-3:]== '_sw' or idx[-3:]=='lcs':
                new_label_col.append(idx)
        
        print(new_label_col)

        co=[x for x in new_label_col if x[0:2]=='CO'] 
        re=[x for x in new_label_col if x[0:2]=='RE'] 
        ad=[x for x in new_label_col if x[0:2]=='AD']
        ro=[x for x in new_label_col if x[0:2]=='RO']

        co_list=list(mean_df_con[co].mean(axis=1))
        re_list=list(mean_df_con[re].mean(axis=1))
        ad_list=list(mean_df_con[ad].mean(axis=1))
        ro_list=list(mean_df_con[ro].mean(axis=1))


        mean_df_con['CO_mean']=co_list
        mean_df_con['RE_mean']=re_list
        mean_df_con['AD_mean']=ad_list
        mean_df_con['RO_mean']=ro_list
        ###

        

        same_label=list(mean_df_con['label']).count(1)


        sample_n=len(mean_df_con)

        print(sample_n)

        STATUS_N=[0,0,0,0,0]
        STATUS_N[0]=0#list(mean_df_con['STATUS']).count(1)
        STATUS_N[1]=0#list(mean_df_con['STATUS']).count(2)
        STATUS_N[2]=0#list(mean_df_con['STATUS']).count(3)
        STATUS_N[3]=0#list(mean_df_con['STATUS']).count(4)
        STATUS_N[4]=sample_n

        #label_density(mean_df_con)



        
        #test_set=pd.read_pickle("card_sim_features.pkl")
        #src_id_1=list(test_set['SOURCE_ID_1'])
        #src_id_2=list(test_set['SOURCE_ID_2'])

        #src_id=[]
        #for idx in range(0,len(src_id_1)):
        #    src_id.append((src_id_1[idx],src_id_2[idx]))
    
        #query=(""" select A.*,B.label
        #   from ci_dev.SIM_FEATURES_MEAN_test A
        #   INNER JOIN ci_dev.features_lable B
        #   on A.SOURCE_ID_1 = B.SOURCE_ID_1 and A.SOURCE_ID_2=B.SOURCE_ID_2
        #   and A.SOURCE_1=B.SOURCE_1 and A.source_2=B.source_2
        #   where B.label !=2 and (A.source_id_1,A.source_id_2) in {}""".format(tuple(src_id)))
        #src_df_res=pd.read_sql_query(query,db)    

        test_set=pd.DataFrame()

        # 통계 돌림
        log_model,score,log_y_test,log_y_score,log_x_test,log_time=make_model.logistic_score(mean_df_con,test_set,1)
        
        rf_model,rf_fin_score,rf_y_test,rf_y_score,rf_x_test,rf_time=make_model.random_forest(mean_df_con,test_set,1)

        gbc_model,gbc_fin_score,gbc_y_test,gbc_y_predict,gbc_x_test,gbc_time=make_model.GBC(mean_df_con,test_set,1)

        ens_model,ens_fin_score,ens_y_test,ens_y_predict,ens_x_test,ens_time=make_model.ENSE(mean_df_con,test_set,1)
        
        eclf_model,eclf_fin_score,eclf_y_test,eclf_y_predict,eclf_x_test,eclf_time=make_model.ECLF(mean_df_con,test_set,1)

        knn_rf_model,knn_rf_fin_score,knn_rf_y_test,knn_rf_y_predict,knn_rf_x_test,knn_rf_time=make_model.KNN_RF(mean_df_con,test_set,1)
        
        #########
        logistic_confution,log_score,log_fpr,log_tpr,log_parm=make_model.history_set(log_model,log_y_test,log_x_test)

        rf_confusion,rf_score,rf_fpr,rf_tpr,rf_parm=make_model.history_set(rf_model,rf_y_test,rf_x_test)

        gbc_confusion,gbc_score,gbc_fpr,gbc_tpr,gbc_parm=make_model.history_set(gbc_model,gbc_y_test,gbc_x_test)

        ens_confusion,ensemble_score,ens_fpr,ens_tpr,ens_parm=make_model.history_set(ens_model,ens_y_test,ens_x_test)

        eclf_confusion,eclf_score,eclf_fpr,eclf_tpr,eclf_parm=make_model.history_set(eclf_model,eclf_y_test,eclf_x_test)

        knn_rf_confusion,knn_rf_score,knn_rf_fpr,fnn_rf_tpr,knn_rf_parm=make_model.history_set(knn_rf_model,knn_rf_y_test,knn_rf_x_test)

        #######
        grid_model,grid_test_y,grid_test_x,grid_time=make_model.grid_serch(mean_df_con,test_set)

        grid_confusion,grid_score,grid_fpr,grid_tpr,grid_parm=make_model.history_set(grid_model,grid_test_y,grid_test_x)

        grid_gbc_model,grid_gbc_test_y,grid_gbc_test_x,grid_gbc_time=make_model.grid_serch_gbc(mean_df_con,test_set)

        grid_gbc_confusion,grid_gbc_score,grid_gbc_fpr,grid_gbc_tpr,grid_gbc_parm=make_model.history_set(grid_gbc_model,grid_gbc_test_y,grid_gbc_test_x)

        ######_____________________________
        
        SPONE=" , ".join( str(n) for n in mean_df_con['SOURCE_1'].unique())
        SPTWO=" , ".join( str(n) for n in mean_df_con['SOURCE_2'].unique())


        print('importance')

        print(rf_model.feature_importances_)


        #history 기록
        make_model.insert_history_query(5,'logistic regression',sample_n,same_label,
                                    logistic_confution,log_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',log_parm,log_time,SPONE,SPTWO,gv.db_0_50)
        
        make_model.insert_history_query(5,'random forest',sample_n,same_label,
                                   rf_confusion,rf_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',rf_parm,rf_time,SPONE,SPTWO,gv.db_0_50)

        make_model.insert_history_query(5,'GBC',sample_n,same_label,
                                    gbc_confusion,gbc_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',gbc_parm,gbc_time,SPONE,SPTWO,gv.db_0_50)
        
        make_model.insert_history_query(5,'ensemble',sample_n,same_label,
                                    ens_confusion,ensemble_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',ens_parm,ens_time,SPONE,SPTWO,gv.db_0_50)
        
        make_model.insert_history_query(5,'ECLF',sample_n,same_label,
                                    eclf_confusion,eclf_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',eclf_parm,eclf_time,SPONE,SPTWO,gv.db_0_50)

        make_model.insert_history_query(5,'KNN_RF',sample_n,same_label,
                                    knn_rf_confusion,knn_rf_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',knn_rf_parm,knn_rf_time,SPONE,SPTWO,gv.db_0_50)
        make_model.insert_history_query(5,'grid_rf_knn',sample_n,same_label,
                                    grid_confusion,grid_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',grid_parm,grid_time,SPONE,SPTWO,gv.db_0_50)

        make_model.insert_history_query(5,'grid_gbc',sample_n,same_label,
                                    grid_gbc_confusion,grid_gbc_score,STATUS_N,
                                    variable_name_string,'SMOTEENN',grid_gbc_parm,grid_gbc_time,SPONE,SPTWO,gv.db_0_50)
     
        fpr_tpr=[[log_fpr,log_tpr],[rf_fpr,rf_tpr],[gbc_fpr,gbc_tpr],[ens_fpr,ens_tpr],[eclf_fpr,eclf_tpr]]

        res_list=[fpr_tpr,
                [log_score,rf_score,gbc_score,ensemble_score,eclf_score,grid_score,grid_gbc_score],
                [logistic_confution,rf_confusion,gbc_confusion,ens_confusion,eclf_confusion,grid_confusion,grid_gbc_confusion],
                STATUS_N]        

        return pd.Series(res_list).to_json(orient='values') 
        #pd.Series(fpr_tpr).to_json(orient='values')

#tab table 반환
@app.callback(
    Output('cfs_content','children'),
    [Input('cfs_tabs','value')],
    [State('model_score_show','children')]
)
def show_cfs_table(value,dff):
    res=pd.read_json(dff,orient='value')

    if res is not None:

        if value=='1':
            sc_col,sc_data,cf_col,cf_data=make_score_cf_table_col_row('logistic',res[0][1],res[0][2])
        elif value=='2':
            sc_col,sc_data,cf_col,cf_data=make_score_cf_table_col_row('random forest',res[1][1],res[1][2])
        elif value=='3':
            sc_col,sc_data,cf_col,cf_data=make_score_cf_table_col_row('GBC',res[2][1],res[2][2])
        elif value=='4':
            sc_col,sc_data,cf_col,cf_data=make_score_cf_table_col_row('log,qda,gn',res[3][1],res[3][2])
        elif value=='5':
            sc_col,sc_data,cf_col,cf_data=make_score_cf_table_col_row('ECLF',res[4][1],res[4][2])
        elif value=='6':
            sc_col,sc_data,cf_col,cf_data=make_score_cf_table_col_row('rf knn grid',res[5][1],res[5][2])
        elif value=='7':
            sc_col,sc_data,cf_col,cf_data=make_score_cf_table_col_row('gbc grid',res[6][1],res[6][2])

        return html.Div([
            dt.DataTable(
            columns=cf_col,
            data=cf_data
            ),
            dt.DataTable(
            columns=sc_col,
            data=sc_data
            )
            ])

@app.callback(
    Output('model_score_show_roc','children'),
    [Input('model_score_show','children')]
)
def show_roc_graph(value):
    res=pd.read_json(value,orient='value')

    return dcc.Graph(
            figure = {
                'data' : [
                    go.Scatter(
                        x=res[0][0][0],
                        y=res[0][0][1],
                        mode='lines+markers',
                        name='logistic'
                    ),
                    go.Scatter(
                        x=res[1][0][0],
                        y=res[1][0][1],
                        mode='lines+markers',
                        name='random forest'
                    ),
                    go.Scatter(
                        x=[0,1],
                        y=[0,1],
                        mode='lines'
                    ),
                    go.Scatter(
                        x=res[2][0][0],
                        y=res[2][0][1],
                        mode='lines+markers',
                        name='GBC'
                    ),
                    go.Scatter(
                        x=res[3][0][0],
                        y=res[3][0][1],
                        mode='lines+markers',
                        name='ensemble'
                    ),
                    go.Scatter(
                        x=res[4][0][0],
                        y=res[4][0][1],
                        mode='lines+markers',
                        name='ECLF'
                    )
            ]})


#score table 반환
@app.callback(
    Output('model_score_table','children'),
    [Input('model_score_show','children')]
)
def show_score_table(value):
    
    res=pd.read_json(value,orient='value')
    if res is not []:
        print('xx')

        return html.Div([ 
            dt.DataTable(
                columns=[{"name":'STATUS',"id":"STATUS"},{"name":'n','id':'n'}],
                data=[{'STATUS':'1','n':res[0][3]},{'STATUS':'2','n':res[1][3]},
                {'STATUS':'3','n':res[2][3]},{'STATUS':'4','n':res[3][3]},
                {'STATUS':'sample_num','n':res[4][3]}]
            )])

            
@app.callback(
    Output('label_image_div','children'),
    [Input('label_image_tab','value')]
)
def show_image_tab(value):
    #addr
    #road
    #phon
    #coname
    if value is not None:
        if value=='1':
            res_name='1features_by_label.png'
        elif value=='2':
            res_name='3features_by_label.png'
        elif value=='3':
            res_name='6features_by_label.png'
        elif value=='4':
            res_name='4features_by_label.png'
        elif value=='5':
            res_name='2features_by_label.png'
        elif value=='6':
            res_name='5features_by_label.png'


        encoded_image = base64.b64encode(open(res_name, 'rb').read())
        return html.Div([ 
        html.Img(id='test_img',src = 'data:image/png;base64,{}'.format(encoded_image.decode())
        ,style={'display':'inline-block','width':'50%','height':'50%'}) ])
