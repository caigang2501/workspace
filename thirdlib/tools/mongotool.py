import pymongo

def getclient():
    db = 'FW'
    port = '27017'
    host = '40.2.159.7'
    username = 'cstools'
    password = '!QAZ2wsx#EDC'

    MONGO_URL = 'mongodb://{}:{}@{}/{}'.format(username,password,host,port,db)
    client = pymongo.MongoClient(MONGO_URL)
    return client

pipline = [
    {
        '$match':{
            'name':{'$ne':''},
            'class':{'$regex':'^[^0-9]*$'}
        }
    },
    {
        '$group':{
            '_id':'strname',
            'newname':{
                '$max':{
                    '$concat':['$time','$addrees']
                }
            }
        }
    }
]