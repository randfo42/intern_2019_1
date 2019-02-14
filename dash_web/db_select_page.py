import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import dash_table as dt
import pandas as pd
import global_var as gv
import random

from server import app 


#TODO 첫 page dropdown에서 선택하는 table option label에는 table alias, value에는 table 원본 명을 쓴다.
# source_1에 대한 table 
db_table=[#{'label':'PROCESSED_MEUMS_COMPANY','value':'ci_dev.PROCESSED_MEUMS_COMPANY'},
            #{'label':'MWS_SR_COLT_BSC_DATA','value':'wspider.MWS_SR_COLT_BSC_DATA'},
            {'label':'MEUMS_COMPANY','value':'`eums-shared`.MEUMS_COMPANY'},
            #{'label':'PROCESSED_CARD_LOGIN','value':'ci_dev.PROCESSED_CARD_LOGIN'},
            {'label':'MEUMS_COLL_CARD_MER','value':'eums.MEUMS_COLL_CARD_MER'}]


#처음 보여주는 화면
db_select_page = html.Div([

    html.A('평가를 원하는 table을 dropdown 에서 선택하고 select 버튼을 눌러 정한 뒤, listing 버튼을 누르거나, '),
    html.Br(),
    html.A('combine 버튼을 눌러 모델을 만들 수 있습니다.'),
    dcc.Dropdown(
        id='db_select_drop',
        options=db_table
    ),
    html.Button('select',id='db_submit'),
    html.Br(),
    html.Div(id='show_select_stat'),
    html.Br(),
    html.Br(),
    html.A(' for labeling click listing button, to create model and combine data click combine button  '),
    html.Br(),
    html.Button('listing',id='to_listing_page',n_clicks_timestamp=0),
    html.A(html.Button('combine',id='to_combine_page',n_clicks_timestamp=0),href='/combine'),
    
    #,style={'display': 'none'}
])

@app.callback(
    Output('db_set_done','children'),
    [Input('db_submit','n_clicks')],
    [State('db_select_drop','value')]
)
def set_db(n_clicks,value):
    
    distinct_source_id=[]
    show_table_use=gv.get_show_table_source(gv.show_table)

    
    #TODO 가져올 source_id_1에 대한 조건.   
    get_dist_id_query=("""select distinct A.SOURCE_ID_1 """ +
                        """ from %s A """%(gv.mean_table) +
                        """ LEFT JOIN %s B """%(gv.label_table)+
                        """ ON A.SOURCE_ID_1 = B.SOURCE_ID_1 """+
                        """ where B.SOURCE_ID_1 IS NULL """+
                        """  and A.SOURCE_1='%s' and A.SOURCE_1!=A.SOURCE_2 limit %d,100 """%(show_table_use,random.randrange(1,1000)))
                        #and A.STATUS=1


    #TODO 두 table에 대한 db
    dist_id_dt=pd.read_sql_query(get_dist_id_query,gv.db_0_50)


    dist_id_dict=dist_id_dt.to_dict("rows")

    for idx in dist_id_dict:
        distinct_source_id.append(list(idx.values())[0])

    distinct_source_id=list(set(distinct_source_id))


    
    return distinct_source_id

@app.callback(
    Output('db_set_done_show','children'),
    [Input('db_set_done','children')]
)
def db_source_id_set(value):
    print('A')
    print('db_selet')
    print(value)
    return html.A('table_selected')

#선택한 table 적용
@app.callback(
    Output('show_select_stat','children'),
    [Input('db_submit','n_clicks')],
    [State('db_select_drop','value')]
)
def show_select_info(n_clicks,value):
    global show_table
    
    if value is None:
        return html.A('please select table want to edit at listing page')
    else :
        gv.show_table=value

    return html.A('table selected = %s' %value)

