
import mysql.connector
import pymysql
from sqlalchemy import create_engine


def getdb():
    mydb = pymysql.connect(
        host = '40.2.195.1',
        port = 3306,
        user = 'cmbcdbop',
        password = '1qaz@WSX',
        db = 'paasdb'
    )
    # mydb = mysql.connector.connect(
    #     host = '40.2.195.1',
    #     port = '3306',
    #     user = 'cmbcdbop',
    #     password = '1qaz@WSX',
    #     database = 'paasdb'
    # )

def getdata_bysql(*args):
    result = []
    db = getdb()
    cursor = db.cursor()
    try:
        for sql in args:
            cursor.execute(sql)
            result.append(cursor.fetchall())
        cursor.close()
        db.close()
        return result
    except:
        return 0

def sql_df2sql(dbname,tablename,df):
    engine = create_engine(f'mysql+mysqlconnector://cmbcdbop:1qaz@WSX40.2.159.1:3306/{dbname}')
    df.to_sql(name=tablename,con=engine,if_exists='append',index=False)
    engine.dispose()