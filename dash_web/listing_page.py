import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input,Output,State
from sqlalchemy import create_engine
import sqlalchemy
import time

import global_var
from server import app 
from db_select_page import db_table,db_select_page
#use distict id query output



#값이 같은지 평가된 값 list
evaluated_list=[]

#listing 2값
listing_selected_source_2=[]

features_mean_dict=[]


#radio버튼을 만든다. id와 예측된 값이 parameter로 들어감

def radio_item_create(id_in,prevalue):
    
    return dcc.RadioItems(
            id=id_in,
            options=[
                {'label': 'Yes' , 'value' : 1 },
                {'label': 'No' , 'value' : 0 },
                {'label': 'Unsure' , 'value' : 2 }
            ],
            value=prevalue,
            labelStyle={'display': 'inline-block'},
            style={'margin-top':'6px'}
        )


# dataframe과 table 명을 주면 column명을 변화해서 반환(same_columns_dictionary 수정하면 됨)
def transe_df_col(df,table_Na):
    
    res_d=df
    
    for idx in range(0,len(global_var.same_columns_dictionary)):
        if global_var.same_columns_dictionary[idx]['table']==table_Na:
            res_list=global_var.same_columns_dictionary[idx]['origin_col']
    col={}
    for idx in range(0,len(res_list)):
        col[res_list[idx]]=global_var.transe_same_col_dict[idx]
    
    res_d.rename(columns=col,inplace=True)


    return res_d   



# listing page 화면
listing_page=html.Div([
    html.A('data를 labeling 하기위해 보기원하시는 colum을 dropdown에서 선택하고 버튼을 눌러주세요.'),
    html.Br(),
    html.A('만일 dropdown에서 값을 선택하지 않고 버튼을 누르시면 default 값으로 선택됩니다.'),
    html.Br(),
    html.A('table을 보고, 맨 위의 row와 같은지 선택해 주세요. '),
    html.Br(),
    html.A('labeling 도중, 혹은 중간에 save to db를 누르면 db에 저장됩니다. '),
    html.Br(),
    html.Div(id='column_select_drop'),
    #html.A('do these two records refer to same thing?'),
    html.Div(id='show_stat'),
    html.Div(id='listing_page_table',style={'display':'none'}),
    html.Div(id='listing_btn_var'),
    html.Div(id='listing_drop_te'),
    html.Div([ 
            
                dt.DataTable(
                    id='listing_show_table',
                   
                    style_table={'maxWidth': '82%','display':'inline-block','float':'left',
                                'padding-right':0,'border':'1px solid red','overflowX': 'scroll'
                               }
                ),
                html.Div(
                    id='radio_item_3',
                    style={'width': '17%','display':'inline-block','border':'1px solid red',
                            'margin-top':'65px'}
                )
            ],style={'width':'100%'}),
    html.Br(),
    html.Br(),
    html.Button('Back',id='listing_back',n_clicks_timestamp=0),
    html.Button('Next',id='listing_next',n_clicks_timestamp=0),
    html.Br(),
    html.Div(id='for_loading'),
    html.Br(),
    html.Button('save to db',id='save_list_to_db'),
    html.Div(id='check_list_save'),
    html.Div(id='ref'),
    html.Div(id='listing_table_one_dv',style={'display':'none'}),
    html.Div(id='features_mean_dict_dv',style={'display':'none'})
])

#보기 원하는 column을 선택하는 dropdown return 및 default column setting
@app.callback(
    Output('column_select_drop','children'),
    [Input('ref','children')]
)
def show_Stat(done):
    print('????????')
    col_option=[]
    for idx in db_table:
        query=""" select * from %s limit 1"""%(idx['value'])
        get_df=pd.read_sql_query(query,global_var.get_show_db(idx['value']))
        for k in list(get_df.columns):
            col_option.append(k)
    
    for idx in global_var.same_columns_dictionary:
        for k in idx['origin_col']:
            col_option.remove(k)
    

    return html.Div([
                dcc.Dropdown(
                    id='listing_column_select_drop',
                    options=[{'label':n,'value':n} for n in col_option],
                    multi=True
                    ),
                html.Button('submit',id='listing_columns_set')            
            ])




@app.callback(
    Output('show_stat','children'),
    [Input('listing_columns_set','n_clicks')],
    [State('listing_column_select_drop','value')]
)
def set_listing_selected_col(n_clicks,value):
    print('??')
    listing_selected_col=[]

    for idx in global_var.transe_same_col_dict:
        listing_selected_col.append(idx)
    
    if value is not None:
        for idx in value:
            listing_selected_col.append(idx)
  
    return listing_selected_col

@app.callback(
    Output('listing_page_table','children'),
    [Input('show_stat','children'),
     Input('listing_back','n_clicks_timestamp'),
     Input('listing_next','n_clicks_timestamp')],
    [State('db_set_done','children')]
)
def show_listing_table(init,back,next_b,distinct_source_id):
    
    global features_mean_dict

    print('o')
    global listing_selected_source_2

    if init !=[]:
        page_num=len(evaluated_list)
        

        dist_source_f=global_var.get_show_table_source(global_var.show_table)


        #SOURCE_ID_1에 대해 평가된 dataframe을 meantable에서 가져옴
        get_df_query="""select * from %s where SOURCE_ID_1='%d' and SOURCE_1='%s' """%(global_var.mean_table,distinct_source_id[page_num],dist_source_f)
        get_df_row=pd.read_sql_query(get_df_query,global_var.db_0_50).to_dict("rows")

        features_mean_dict = get_df_row

        find_id_set=[]
        find_source_set=[]
        find_stat_set=[]
        find_fir_source=global_var.show_table  

        print('source_id_1')
        print(distinct_source_id[page_num])
        
        print('source_id_2')
        for idx in get_df_row:
            find_id_set.append(idx['SOURCE_ID_2'])
            find_source_set.append(global_var.get_source_show_table(idx['SOURCE_2']))
            find_stat_set.append(idx['STATUS'])


        add=[]
        add.append(find_id_set)
        add.append(find_source_set)
        add.append(find_stat_set)
        listing_selected_source_2.append(add)


        db_show=global_var.get_show_db(global_var.show_table)

        use_id=global_var.get_show_table_id(global_var.show_table)

        # SOURCE_ID_1의 원본 data를 가져옴
        sql_get_show_table_fir=""" SELECT * FROM %s WHERE %s = %d"""%(find_fir_source,use_id,distinct_source_id[page_num]) 
        
        get_show_table_df=pd.read_sql_query(sql_get_show_table_fir,db_show)
        
        get_show_table_df=transe_df_col(get_show_table_df,find_fir_source)

        


        # SOURCE_ID_2의 원본 data들을 가져옴
        get_show_table_df_else=pd.DataFrame()
        for i in range(0,len(find_id_set)):
            use_id=global_var.get_show_table_id(find_source_set[i])
            db_show=global_var.get_show_db(find_source_set[i])
            sql_get_show_table_else="""SELECT * FROM %s WHERE %s = %d """%(find_source_set[i],use_id,find_id_set[i])
            sql_get_df=pd.read_sql_query(sql_get_show_table_else,db_show)
            sql_get_df=transe_df_col(sql_get_df,find_source_set[i])
            if get_show_table_df_else.empty:
                get_show_table_df_else=sql_get_df
            else:
                get_show_table_df_else = pd.concat([get_show_table_df_else,sql_get_df])

        

        fir_df_select_col=list(set(init)&set(list(get_show_table_df.columns)))
        use_fir_df=get_show_table_df[fir_df_select_col]
        else_df_select_col=list(set(init)&set(list(get_show_table_df_else.columns)))
        
        use_else_df=get_show_table_df_else[else_df_select_col]

        listing_table_one=pd.concat([use_fir_df,use_else_df])
    
        return listing_table_one.to_json(date_format='iso', orient='split')

@app.callback(
    Output('listing_show_table','columns'),
    [Input('listing_page_table','children')]
)
def one_col_set(value):
    return [{"name": n,"id":n} for n in pd.read_json(value,orient='split').columns]
    
@app.callback(
    Output('listing_show_table','data'),
    [Input('listing_page_table','children')]
)
def one_data_set(value):
    print('y')
    return pd.read_json(value,orient='split').to_dict('rows')


#listing table 의 강조 될 부분 칠해줌 
@app.callback(
    Output('listing_show_table','style_data_conditional'),
    [Input('listing_page_table','children')],
    [State('db_set_done','children')]
)
def one_style_condition_set(value,distinct_source_id):
    page_num=len(evaluated_list)
    display_one=[]
    

    #source_id_1의 원본 row 칠해줌
    use_id=global_var.get_show_table_id(global_var.show_table)
    try:
        if distinct_source_id[page_num] == pd.read_json(value,orient='split').to_dict('rows')[0][use_id]:
            display_one.append({
                'if' : {"row_index":0},
                "backgroundColor": "#00cc00",
                })
    except IndexError:
        pass
    except KeyError:
        pass
    
    #source_id_2 에서 하나라도 mean이 0.8이 넘으면 칠해줌
    for idx in range(0,len(features_mean_dict)):
        flag=0
        check_list=list(features_mean_dict[idx].values())
        check_list.pop(0)
        check_list.pop(0)

        for con in check_list:
            if isinstance(con,int) or isinstance(con,float):
                if con>0.8:
                    flag=1
                    break
        
        if flag==1:
            display_one.append({
                'if' : {"row_index": idx+1},
                "backgroundColor": "#ccffcc",
            })
    return display_one

#radio item 생성기
# yes,no,unsure의 평가를 list 에 넣는 것과 병렬적으로 이루어 지기 때문에 sleep 사용.
@app.callback(
    Output('radio_item_3','children'),
    [Input('listing_page_table','children')]
)
def show_radio_item_3(init):
    time.sleep(5)
    radio_num=len(pd.read_json(init,orient='split').to_dict('rows'))
    print(evaluated_list)
    return html.Div([radio_item_create('radio_'+str(n),0) for n in range(1,radio_num)])

@app.callback(
    Output('for_loading','children'),
    [Input('listing_page_table','children')]
)
def show_loading_stat(init):
    time.sleep(5)
    page_num=len(evaluated_list)
    return html.A("current page load = "+str(page_num))


# 평가된 항목 listing의 initiate
@app.callback(
    Output('test0','children'),
    [Input('listing_back','n_clicks_timestamp'),
     Input('listing_next','n_clicks_timestamp')]
)
def radio_value_init(back,next_b):
    global evaluated_list
    
    flag = max(back,next_b)
    page_num=len(evaluated_list)

    if flag == 0:
        flag=-1
    
    to=[0]

    if flag == back:
        if page_num>0:
            evaluated_list.pop()
            return -1
    elif flag == next_b:
        evaluated_list.append(to)
        return 1

# dynamic callback 생성이 안되기에 만들어진 최대 25 개의 radio button에 대한 callback
@app.callback(
    Output('test1','children'),
    [Input('test0','children')],
    [State('radio_1','value')]
)
def radio_value(children,value):
    
    global evaluated_list

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1



@app.callback(
    Output('test2','children'),
    [Input('test1','children')],
    [State('radio_2','value')],
)
def radio_value_a(children,value):
    
    global evaluated_list

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


@app.callback(
    Output('test3','children'),
    [Input('test2','children')],
    [State('radio_3','value')],
)
def radio_value_b(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


@app.callback(
    Output('test4','children'),
    [Input('test3','children')],
    [State('radio_4','value')],
)
def radio_value_c(children,value):
    global evaluated_list
    
    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


@app.callback(
    Output('test5','children'),
    [Input('test4','children')],
    [State('radio_5','value')],
)
def radio_value_d(children,value):
    global evaluated_list
  

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


@app.callback(
    Output('test6','children'),
    [Input('test5','children')],
    [State('radio_6','value')],
)
def radio_value_e(children,value):
    global evaluated_list
   

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


@app.callback(
    Output('test7','children'),
    [Input('test6','children')],
    [State('radio_7','value')],
)
def radio_value_f(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


@app.callback(
    Output('test8','children'),
    [Input('test7','children')],
    [State('radio_8','value')]
)
def radio_value_g(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


@app.callback(
    Output('test9','children'),
    [Input('test8','children')],
    [State('radio_9','value')]
)
def radio_value_h(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test10','children'),
    [Input('test9','children')],
    [State('radio_10','value')]
)
def radio_value_i(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test11','children'),
    [Input('test10','children')],
    [State('radio_11','value')]
)
def radio_value_j(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test12','children'),
    [Input('test11','children')],
    [State('radio_12','value')]
)
def radio_value_k(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test13','children'),
    [Input('test12','children')],
    [State('radio_13','value')]
)
def radio_value_l(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test14','children'),
    [Input('test13','children')],
    [State('radio_14','value')]
)
def radio_value_m(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test15','children'),
    [Input('test14','children')],
    [State('radio_15','value')]
)
def radio_value_n(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test16','children'),
    [Input('test15','children')],
    [State('radio_16','value')]
)
def radio_value_o(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test17','children'),
    [Input('test16','children')],
    [State('radio_17','value')]
)
def radio_value_p(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test18','children'),
    [Input('test17','children')],
    [State('radio_18','value')]
)
def radio_value_q(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test19','children'),
    [Input('test18','children')],
    [State('radio_19','value')]
)
def radio_value_r(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test20','children'),
    [Input('test19','children')],
    [State('radio_20','value')]
)
def radio_value_s(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test21','children'),
    [Input('test20','children')],
    [State('radio_21','value')]
)
def radio_value_t(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test22','children'),
    [Input('test21','children')],
    [State('radio_22','value')]
)
def radio_value_u(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test23','children'),
    [Input('test22','children')],
    [State('radio_23','value')]
)
def radio_value_v(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test24','children'),
    [Input('test23','children')],
    [State('radio_24','value')]
)
def radio_value_w(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1

@app.callback(
    Output('test25','children'),
    [Input('test24','children')],
    [State('radio_25','value')]
)
def radio_value_x(children,value):
    global evaluated_list
    

    if children == 1:
        to=evaluated_list.pop()
        to.append(value)
        evaluated_list.append(to)
        return 1


#TODO 평가된 값 db에 저장
@app.callback(
    Output('check_list_save','children'),
    [Input('save_list_to_db','n_clicks')],
    [State('db_set_done','children')]
)
def save_list_btn_func(n_clicks,distinct_source_id):
    global listing_selected_source_2

    if n_clicks is not None:        
        for idx in range(0,len(evaluated_list)):

            source_1=global_var.get_show_table_source(global_var.show_table)


            #TODO label table column 변경시 조건 변경 필수
            for idx_t in range(0,len(listing_selected_source_2[idx][0])):
               
                insert_data_query=(""" INSERT INTO %s """%(global_var.label_table)+
                                    """ VALUES (%s,%s,'%s','%s',%d,%d) """
                                    %(distinct_source_id[idx],listing_selected_source_2[idx][0][idx_t],
                                    str(source_1),str(global_var.get_show_table_source(listing_selected_source_2[idx][1][idx_t])),
                                    evaluated_list[idx][idx_t+1]
                                    ,listing_selected_source_2[idx][2][idx_t]))

                print(insert_data_query)
                try:
                    global_var.db_0_50.execute(insert_data_query)
                except sqlalchemy.exc.IntegrityError:
                    print('inte')
                    pass

        print('over')

        return html.A('test!!!!!!')