import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

from dash.dependencies import Input,Output,State


import listing_page
import combine_page
import db_select_page
import adapt_page
from server import app 


# TODO 주석은 조건이 변할 때 고쳐야 할 code위에 있다.   
# mean table은 원본 table 페어 2개의 string이나 값에 대해 특정 값으로 평가된 값이 들어 있는 table 로써, SOURCE에 대한 id, 위치(alias) 기타 조건이 들어있다.
# label table은 pair 2개에 대해 yes,no,unsure로 평가된 값이 들어있다.
#원본 table pair는 source 1 source 2로 구분한다.


app.layout = html.Div([
    
    dcc.Location(id='url',refresh=False),
    html.Div(id='page-content'),

    #db info 저장
    html.Div(id='db_info',style={'display': 'none'}),
    
    #radio button의 callback을 dynamic 하게 할당하지 못해서 static하게 
    # 최대 25개 까지 지원하게 만들기 위한 div
    html.Div(id='test1',style={'display': 'none'}),
    html.Div(id='test2',style={'display': 'none'}),
    html.Div(id='test3',style={'display': 'none'}),
    html.Div(id='test4',style={'display': 'none'}),
    html.Div(id='test5',style={'display': 'none'}),
    html.Div(id='test6',style={'display': 'none'}),
    html.Div(id='test7',style={'display': 'none'}),
    html.Div(id='test8',style={'display': 'none'}),
    html.Div(id='test9',style={'display': 'none'}),
    html.Div(id='test10',style={'display': 'none'}),
    html.Div(id='test11',style={'display': 'none'}),
    html.Div(id='test12',style={'display': 'none'}),
    html.Div(id='test13',style={'display': 'none'}),
    html.Div(id='test14',style={'display': 'none'}),
    html.Div(id='test15',style={'display': 'none'}),
    html.Div(id='test16',style={'display': 'none'}),
    html.Div(id='test17',style={'display': 'none'}),
    html.Div(id='test18',style={'display': 'none'}),
    html.Div(id='test19',style={'display': 'none'}),
    html.Div(id='test20',style={'display': 'none'}),
    html.Div(id='test21',style={'display': 'none'}),
    html.Div(id='test22',style={'display': 'none'}),
    html.Div(id='test23',style={'display': 'none'}),
    html.Div(id='test24',style={'display': 'none'}),
    html.Div(id='test25',style={'display': 'none'}),
    html.Div(id='test0',style={'display': 'none'}),

    html.Div(id='db_set_done',style={'display':'none'}),
    html.Div(id='db_set_done_show'),

])

#model adapt 한 뒤, db data에 적용하여 평가하는 페이지의 model dropdown 
@app.callback(
    Output('selected_model','children'),
    [Input('adapt_model_sel_btn','n_clicks')],
    [State('adapt_model_select_drop','value')]
)
def select_model_to_adapt(n_clicks,var):
    return var 



@app.callback(
    Output('url','pathname'),
    [Input('to_listing_page','n_clicks')
    ]
)
def go_to_listing(to_listing):

    if to_listing is not None:
        return '/listing'

@app.callback(
    Output('page-content','children'),
    [Input('url','pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return db_select_page.db_select_page
    elif pathname == '/listing':
        return listing_page.listing_page
    elif pathname == '/combine':
        return combine_page.combine_page
    elif pathname == '/adpt':
        return adapt_page.adapt_page



if __name__=='__main__':
    app.run_server(debug=True)