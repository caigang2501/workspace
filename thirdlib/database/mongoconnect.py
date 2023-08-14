import pymongo

# 连接到 MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")

# 创建或选择数据库
db = client["mydatabase"]

# 创建或选择集合
collection = db["customers"]

# 插入一条数据
data = {"name": "John", "address": "Highway 37"}
insert_result = collection.insert_one(data)
print("Inserted ID:", insert_result.inserted_id)

# 查询数据
query = {"address": "Highway 37"}
result = collection.find(query)
for document in result:
    print(document)

# 更新数据
query = {"address": "Highway 37"}
new_values = {"$set": {"address": "Park Lane 38"}}
update_result = collection.update_one(query, new_values)
print("Modified Count:", update_result.modified_count)

# 删除数据
delete_query = {"address": "Park Lane 38"}
delete_result = collection.delete_one(delete_query)
print("Deleted Count:", delete_result.deleted_count)
