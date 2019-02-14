import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input,Output,State
import pandas as pd
import sqlalchemy
import random 

from server import app 
from global_var import label_table,mean_table,db_0_50,get_source_show_table,get_show_table_id,get_show_db
import make_model
from listing_page import transe_df_col 



# TODO model 적용 뒤, query에 쓰기 위해 사용될 column 
def source_id_table_pair(df):
    id_f=list(df['SOURCE_ID_1'])
    id_s=list(df['SOURCE_ID_2'])
    source_f=list(df['SOURCE_1'])
    source_s=list(df['SOURCE_2'])
    stat=list(df['SIM_LEVEL'])
    

    res=[]

    for idx in range(0,len(id_f)):
        add=[]
        add.append(id_f[idx])
        add.append(id_s[idx])
        add.append(source_f[idx])
        add.append(source_s[idx])
        add.append(stat[idx])
        res.append(add)

    return res

adapt_page=html.Div([
        
        html.A(' 1 . select model to adapt'),
        dcc.Dropdown(
            id='adapt_model_select_drop',
            options=[{'label':'logistic_regression','value':'log'},
                    {'label':'random_forest','value':'rf'},
                    {'label':'GBC','value':'gbc'},
                    {'label':'ensemble','value':'ens'},
                    {'label':'ECLF','value':'eclf'}
                    ]
        ),
        html.Button('select model',id='adapt_model_sel_btn'),
        html.Div(id='selected_model'),
        html.Div(id='adapt_process',style={'display':'none'}),
        html.Br(),
        html.A(' 2. threshold 를 정하세요 '),
        html.Br(),
        html.A(' same class는 100%에서 (100-threshold)% 사이의 pair이고,  '),
        html.Br(),
        html.A(' unsure class는 (100-threshold) 와 threshold 사이,  '),
        html.Br(),
        html.A(' differernt 는 threshold보다 작은 class 입니다.  '),
        html.Br(),
        dcc.Input(id='adapt_thresh',type='number'),
        html.Button('adapt thresh',id='adapt_thresh_btn'),
        html.Div(id='adapt_thresh_var',style={'display':'none'}),
        html.Br(),
        html.A(' 3. select class want to see, and click select class button'),
        dcc.Dropdown(
            id='select_class',
            options=[{'label':'same','value':1},
                     {'label':'different','value':0},
                     {'label':'unsure','value':2}
                     ]
        ),
        html.Button('select class',id='select_class_btn'),
        html.Div(id='seleced_class_var'),
        html.Br(),
        html.Div(id='adapt_show_table'),
        html.Br(),
        html.Button('yes',id='adapt_yes',n_clicks_timestamp=0),
        html.Button('no',id='adapt_no',n_clicks_timestamp=0),
        html.Button('unsure',id='adapt_unsure',n_clicks_timestamp=0),
        html.Button('back',id='adapt_back',n_clicks_timestamp=0),
        html.Button('reset',id='adapt_reset',n_clicks_timestamp=0),
        
        
        html.Div(id='adapt_list_var',style={'display':'none'}),
        html.Br(),
        html.Button('save to db',id='save_to_db'),
        html.Div(id='save_db_div'),
        html.Div(id='predict_var',style={'display':'none'})

])

# labeled data로 model fitting 뒤, model이 다른 data(평가되지 않은 mean table data)를 preict
@app.callback(
    Output('adapt_process','children'),
    [Input('selected_model','children')]
)
def adapting_model(value):

    #print(value)
    if value is not None:
        
        #df_set=make_model.get_data_to_db_for_statistic(db_0_50,mean_table,label_table)

        test_set=pd.DataFrame()


        query=(""" select A.SOURCE_ID_1,A.SOURCE_ID_2
           from ci_dev.SIM_FEATURES_test A
           left JOIN ci_dev.features_lable B
           on A.SOURCE_ID_1 = B.SOURCE_ID_1 and A.SOURCE_ID_2=B.SOURCE_ID_2
           where B.source_id_1 is null and A.source_1!=A.source_2 and A.pair_source='phonenum_B' """)
           #           and A.SOURCE_1=B.SOURCE_1 and A.source_2=B.source_2


        print('get id!')
        
        test_set_id=pd.read_sql_query(query,db_0_50)

        test_zip=zip(list(test_set_id['SOURCE_ID_1']),list(test_set_id['SOURCE_ID_2']))

        test_zip_list=list(test_zip)


        test_pair=random.sample(test_zip_list,100)

        #test_pair=test_pair[0:100]
        
        test_set_query=(""" select * from ci_dev.SIM_FEATURES_test where (source_id_1,source_id_2) in {} and source_1!=source_2""".format(tuple(test_pair)))

        test_set=pd.read_sql(test_set_query,db_0_50)


        test_set=source_id_table_pair(test_set)
        
        print(test_set)

        #print(test_set[1])
        res=[value,test_set]

        res=pd.Series(res).to_json(orient='values')

        print(res)

        return res

@app.callback(
    Output('predict_var','children'),
    [Input('adapt_process','children')]
)
def predict_set(value):
    if value is not None:
        df=pd.read_json(value,orient='values')
        val=df[0][0]
        test_set=pd.DataFrame()
        df_set=make_model.get_data_to_db_for_statistic(db_0_50,mean_table,label_table)
        
        if val == 'rf':
            model = make_model.random_forest(df_set,test_set,0)[0]
        elif val == 'log':
            model = make_model.logistic_score(df_set,test_set,0)[0]
        elif val == 'gbc':
            model = make_model.GBC(df_set,test_set,0)[0]
        elif val == 'ens':
            model = make_model.ENSE(df_set,test_set,0)[0]
        elif val == 'eclf':
            model = make_model.ECLF(df_set,test_set,0)[0]


        src_id=[]
        for idx in range(0,len(df[0][1])):
            src_id.append((df[0][1][idx][0],df[0][1][idx][1],df[0][1][idx][2],df[0][1][idx][3]))

        #TODO
        query=(""" select * from ci_dev.SIM_FEATURES_test where """+
                """ (source_id_1,source_id_2,source_1,source_2) in {} """.format(tuple(src_id)))

        src_df=pd.read_sql_query(query,db_0_50)

        res_test=make_model.make_set(src_df)


        res=model.predict_proba(res_test)[:,1]
        return pd.Series(res).to_json(orient='values')

@app.callback(
    Output('adapt_thresh_var','children'),
    [Input('adapt_thresh_btn','n_clicks')],
    [State('adapt_thresh','value'),
     State('predict_var','children'),
     State('adapt_process','children')]
)
def set_thresh(n_clicks,value,pre,src):
    print('??')

    if n_clicks is not None:
        thresh_var=value/100
        source=pd.read_json(src,orient='values')[0][1]
        predict=pd.read_json(pre,orient='values')[0]

        class_true=[]
        class_false=[]
        class_unsure=[]
        
        yes_adapt_arr=[]
        no_adapt_arr=[]
        unsure_adapt_arr=[]


        for idx in range(0,len(source)):
            add=idx

            if predict[idx]>(1-thresh_var):
                class_true.append(add)
            elif predict[idx]<thresh_var:
                class_false.append(add)
            else:
                class_unsure.append(add)        

        print('true')
        print(len(class_true))
        print('false')
        print(len(class_false))
        print('unsure')
        print(len(class_unsure))

        res=[class_true,class_false,class_unsure]

        return pd.Series(res).to_json(orient='values')

@app.callback(
    Output('seleced_class_var','children'),
    [Input('select_class_btn','n_clicks')],
    [State('select_class','value')]
)
def set_class(n_clicks,value):
    if n_clicks is not None:
        return value



@app.callback(
    Output('adapt_show_table','children'),
    [Input('seleced_class_var','children'),
    Input('adapt_list_var','children')],
    [State('predict_var','children'),
     State('adapt_process','children'),
     State('adapt_thresh_var','children')]   
)
def make_adapt_table(select_cls,arr,pre,src,clas):

    if select_cls is not None:
        
        if arr is not None:
            init=pd.read_json(arr,orient='values')
            yes_adapt_arr=list(init.loc[0])
            yes_adapt_arr=[x for x in yes_adapt_arr if x ==1 or x==0 or x==2]
            no_adapt_arr=list(init.loc[1])
            no_adapt_arr=[x for x in no_adapt_arr if x ==1 or x==0 or x==2]
            unsure_adapt_arr=list(init.loc[2])
            unsure_adapt_arr=[x for x in unsure_adapt_arr if x ==1 or x==0 or x==2]
        else:
            yes_adapt_arr=[]
            no_adapt_arr=[]
            unsure_adapt_arr=[]
    
        dff=pd.read_json(clas,orient='value')
        pred=list(pd.read_json(pre,orient='value')[0])
        sourc=pd.read_json(src,orient='values')[0][1]

        tr=list(dff.loc[0])
        ne=list(dff.loc[1])
        un=list(dff.loc[2])

        if select_cls==1:
            page_len=len(yes_adapt_arr)
        elif select_cls==0:
            page_len=len(no_adapt_arr)
        else:
            page_len=len(unsure_adapt_arr)
        
        print(page_len)

        if select_cls ==1:
            flag=tr[page_len]
        elif select_cls == 0:
            flag=ne[page_len]
        else:
            flag=un[page_len]
        
        flag=int(flag)

        src=sourc[flag]
        prob=pred[flag]
         
        #print(src)
        #print(prob)
        #print(predict)

        source_f=get_source_show_table(src[2])
        source_id_f=get_show_table_id(source_f)
        f_db=get_show_db(source_f)

        source_s=get_source_show_table(src[3])
        source_id_s=get_show_table_id(source_s)
        s_db=get_show_db(source_s)

        adapt_table_query_f=""" select * from %s where %s = %d """%(source_f,source_id_f,src[0])
        adapt_table_query_s=""" select * from %s where %s = %d """%(source_s,source_id_s,src[1])

        adapt_table_f=pd.read_sql_query(adapt_table_query_f,f_db)
        adapt_table_f=transe_df_col(adapt_table_f,source_f)

        adapt_table_s=pd.read_sql_query(adapt_table_query_s,s_db)
        adapt_table_s=transe_df_col(adapt_table_s,source_s)

        adapt_table=pd.concat([adapt_table_f,adapt_table_s],sort=False)

        style_list=[{
                "if" : {"row_index":0},
                "backgroundColor": "#00cc00",
        }]

        if prob>=0.5:
            style_list.append({
                "if" : {"row_index":1},
                "backgroundColor": "#ccffcc"
                })
        elif prob<0.5:
            style_list.append({
                "if" : {"row_index":1},
                "backgroundColor": "#ff4d4d"
            })

        return html.Div([
                    html.A('SOURCE 1 = %s'%(source_f)),
                    html.Br(),
                    html.A('SOURCE 2 = %s'%(source_s)),
                    dt.DataTable(
                    columns= [{"name":n,"id":n} for n in adapt_table.columns],
                    data= adapt_table.to_dict('rows'),
                    style_data_conditional=style_list,
                    style_table={'overflowX': 'scroll'}
                )
                ])
        

        


@app.callback(
    Output('adapt_list_var','children'),
    [Input('adapt_yes','n_clicks_timestamp'),
    Input('adapt_no','n_clicks_timestamp'),
    Input('adapt_back','n_clicks_timestamp'),
    Input('adapt_reset','n_clicks_timestamp'),
    Input('adapt_unsure','n_clicks_timestamp')
    ],
    [State('seleced_class_var','children'),
     State('adapt_list_var','children')]
)
def adapt_list_set(yes,no,back,reset,unsure,clas,sel):
    if (yes != 0 or no !=0 or unsure!=0) and clas is not None :
                
        if sel is not None:
            init=pd.read_json(sel,orient='values')
            yes_adapt_arr=list(init.loc[0])
            yes_adapt_arr=[x for x in yes_adapt_arr if x ==1 or x==0 or x==2]
            no_adapt_arr=list(init.loc[1])
            no_adapt_arr=[x for x in no_adapt_arr if x ==1 or x==0 or x==2]
            unsure_adapt_arr=list(init.loc[2])
            unsure_adapt_arr=[x for x in unsure_adapt_arr if x ==1 or x==0 or x==2]
        else:
            yes_adapt_arr=[]
            no_adapt_arr=[]
            unsure_adapt_arr=[]

        if sel is not None:
            init=pd.read_json(sel,orient='values')
            yes_adapt_arr=list(init.loc[0])
            yes_adapt_arr=[x for x in yes_adapt_arr if x ==1 or x==0 or x==2]
            no_adapt_arr=list(init.loc[1])
            no_adapt_arr=[x for x in no_adapt_arr if x ==1 or x==0 or x==2]
            unsure_adapt_arr=list(init.loc[2])
            unsure_adapt_arr=[x for x in unsure_adapt_arr if x ==1 or x==0 or x==2]
        else:
            yes_adapt_arr=[]
            no_adapt_arr=[]
            unsure_adapt_arr=[]


        if clas ==1:
            res=yes_adapt_arr
        elif clas==0:
            res=no_adapt_arr
        else:
            res=unsure_adapt_arr
    
        print(res)

        flag=max(yes,no,back,reset,unsure)
        if flag==0:
            flag=-1

        if flag==yes:
            res.append(1)
        elif flag==no:
            res.append(0)
        elif flag==back:
            res.pop()
        elif flag==unsure:
            res.append(2)
        else:
            res=[]
        

        if clas ==1:
            yes_adapt_arr=res
        elif clas==0:
            no_adapt_arr=res
        else:
            unsure_adapt_arr=res
        
        res_re=[yes_adapt_arr,no_adapt_arr,unsure_adapt_arr]

        return pd.Series(res_re).to_json(orient='values')
        
@app.callback(
    Output('save_db_div','children'),
    [Input('save_to_db','n_clicks')],
    [
     State('adapt_process','children'),
     State('adapt_thresh_var','children'),
     State('adapt_list_var','children')]   
)
def save_adapt_listing_to_db(n_clicks,src,clas,arr):
    if n_clicks is not None:
        dff=pd.read_json(clas,orient='value')
        sourc=list(pd.read_json(src,orient='values')[0][1])
                
        if arr is not None:
            init=pd.read_json(arr,orient='values')
            yes_adapt_arr=list(init.loc[0])
            yes_adapt_arr=[x for x in yes_adapt_arr if x ==1 or x==0 or x==2]
            no_adapt_arr=list(init.loc[1])
            no_adapt_arr=[x for x in no_adapt_arr if x ==1 or x==0 or x==2]
            unsure_adapt_arr=list(init.loc[2])
            unsure_adapt_arr=[x for x in unsure_adapt_arr if x ==1 or x==0 or x==2]
        else:
            yes_adapt_arr=[]
            no_adapt_arr=[]
            unsure_adapt_arr=[]

        tr=list(dff.loc[0])
        ne=list(dff.loc[1])
        un=list(dff.loc[2])


        if yes_adapt_arr != []:
            for idx in range(0,len(yes_adapt_arr)):
                flag=int(tr[idx])
                insert_ad_query_yes= (""" INSERT INTO %s """%(label_table)+
                                """ VALUES (%s,%s,'%s','%s',%d,%d) """
                                %(sourc[flag][0],sourc[flag][1],
                                str(sourc[flag][2]),str(sourc[flag][3]),
                                yes_adapt_arr[idx],sourc[flag][4]))
                print(insert_ad_query_yes)
                try:
                    db_0_50.execute(insert_ad_query_yes)
                except sqlalchemy.exc.IntegrityError:
                    pass
                
        
        if no_adapt_arr !=[]:
            for idx in range(0,len(no_adapt_arr)):
                flag=int(ne[idx])
                insert_ad_query_no=(""" INSERT INTO %s """%(label_table)+
                                """ VALUES (%s,%s,'%s','%s',%d,%d) """
                                %(sourc[flag][0],sourc[flag][1],
                                str(sourc[flag][2]),str(sourc[flag][3]),
                                no_adapt_arr[idx],sourc[flag][4]))
                print(insert_ad_query_no)
                try:
                    db_0_50.execute(insert_ad_query_no)
                except sqlalchemy.exc.IntegrityError:
                    pass
        
        if unsure_adapt_arr !=[]:
            for idx in range(0,len(unsure_adapt_arr)):
                flag=int(un[idx])
                insert_ad_query_un=(""" INSERT INTO %s """%(label_table)+
                                """ VALUES (%s,%s,'%s','%s',%d,%d) """
                                %(sourc[flag][0],sourc[flag][1],
                                str(sourc[flag][2]),str(sourc[flag][3]),
                                unsure_adapt_arr[idx],sourc[flag][4]))
                print(insert_ad_query_un)
                try:
                    db_0_50.execute(insert_ad_query_un)
                except sqlalchemy.exc.IntegrityError:
                    pass

