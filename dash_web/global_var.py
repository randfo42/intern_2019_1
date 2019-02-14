from sqlalchemy import create_engine


#TODO 다른 db의 table을 추가할 떄 db 추가 
#db_0_50은 mean table과 label table의 주소. mean table과 label table은 같은 주소를 쓴다고 가정한다.
db_0_50=create_engine("mysql://",
                            encoding = 'utf8',
                            pool_size=20,
                            pool_recycle=3600,
                            connect_args={'connect_timeout':1000000})

db_143_151=create_engine("mysql://",
                            encoding = 'utf8',
                            pool_size=20,
                            pool_recycle=3600,
                            connect_args={'connect_timeout':1000000})

db_135_21 = create_engine("mysql://", 
                            encoding = 'utf8' ,
                            pool_size=20,
                            pool_recycle=3600,
                            connect_args={'connect_timeout':1000000})

#TODO if use another table, fill the form below
#TODO 각각 table 당 설정된 id.  
each_table_identifier=[#{'table':'ci_dev.PROCESSED_MEUMS_COMPANY','id':'SOURCE_ID'},
                        #{'table':'wspider.MWS_SR_COLT_BSC_DATA','id':'None'},
                        {'table':'`eums-shared`.MEUMS_COMPANY','id':'ID'},
                        #{'table':'ci_dev.PROCESSED_CARD_LOGIN','id':'SOURCE_ID'},
                        {'table':'eums.MEUMS_COLL_CARD_MER','id':'ID'}
                        ]

#TODO 실제 table당 mean table에 있는 source_1 과 sourec_2 값  
table_source_var=[#{'table':'ci_dev.PROCESSED_MEUMS_COMPANY','source':'PROCESSED_MEUMS_COMPANY'},
                  {'table':'`eums-shared`.MEUMS_COMPANY','source':'PROCESSED_MEUMS_COMPANY'},
                  #{'table':'ci_dev.PROCESSED_CARD_LOGIN','source':'PROCESSED_CARD_LOGIN'},
                  {'table':'eums.MEUMS_COLL_CARD_MER','source':'PROCESSED_CARD_LOGIN'}]

#TODO source가 들어오면 default로 어떤 table을 찾는가 ](source_2에 대해 찾는 option)
show_table_default=[{'source':'PROCESSED_MEUMS_COMPANY_recleaning','table':'`eums-shared`.MEUMS_COMPANY'},
                    {'source':'PROCESSED_PROCESSED_CARD_LOGIN_recleaning','table':'eums.MEUMS_COLL_CARD_MER'}]

label_table="ci_dev.features_lable"#TODO source 두개에 대해 label 할 table.

mean_table="ci_dev.SIM_FEATURES_test" #TODO 특정 pair의 유사도를 평가한 mean table 수정 필요
show_table="ci_dev.PROCESSED_MEUMS_COMPANY" #data table에 보여줄 db table. dropdown으로 바뀔 수 있어서 수정 안해도 됨.
get_SOURCE_var=""

#TODO 아래 두 list는 여러 sourece table 에 대해 같은 column name으로 변경하여
#병합하기 위해 만든 list 로써, same_colums_dictionary에 실제 table의 column이 있고
#transe same col dict에는 실제 column을 변경 후의 name이 들어있다.
same_columns_dictionary=[
                        {'table':'`eums-shared`.MEUMS_COMPANY','origin_col':['ID','CO_NAME','REP_PHONE_NUM','ADDR','ROAD_ADDR']},
                        {'table':'eums.MEUMS_COLL_CARD_MER','origin_col':['ID','MER_NAME','MER_MSISDN','MER_ADDR']}
                        ]

transe_same_col_dict=['ID','NAME','PHONE_NUM','ADDR','ROAD_ADDR']

#TODO source_1 반영 one hot 

def get_show_db(par):
    if par == "ci_dev.PROCESSED_MEUMS_COMPANY" or par=="ci_dev.PROCESSED_CARD_LOGIN":
        return db_0_50
    elif par == "wspider.MWS_SR_COLT_BSC_DATA" or par =="`eums-shared`.MEUMS_COMPANY":
        return db_143_151
    elif par == "eums.MEUMS_COLL_CARD_MER":
        return db_135_21

#table명에 대해 id 반환
def get_show_table_id(par):
    for idx in each_table_identifier:
        if idx['table']==par:
            res=idx['id']
            return res

def get_source_show_table(par):
    for idx in show_table_default:
        if idx['source']==par:
            res=idx['table']
    return res

def get_show_table_source(par):
    for idx in table_source_var:
        if idx['table']==par:
            res=idx['source']
    return res

