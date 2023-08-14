import mysql.connector
import 

# 连接到 MySQL 数据库
mydb = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  password="yourpassword",
  database="mydatabase"
)

# 创建一个游标对象
cursor = mydb.cursor()

# 执行 SQL 查询
cursor.execute("SELECT * FROM customers")

# 获取查询结果
result = cursor.fetchall()
for row in result:
  print(row)

# 插入数据
sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
cursor.execute(sql, val)
mydb.commit()
print("1 record inserted, ID:", cursor.lastrowid)

# 更新数据
sql = "UPDATE customers SET address = 'Park Lane 12' WHERE address = 'Highway 21'"
cursor.execute(sql)
mydb.commit()
print(cursor.rowcount, "record(s) affected")

# 删除数据
sql = "DELETE FROM customers WHERE address = 'Park Lane 12'"
cursor.execute(sql)
mydb.commit()
print(cursor.rowcount, "record(s) deleted")

# 关闭游标和数据库连接
cursor.close()
mydb.close()
pd.da
