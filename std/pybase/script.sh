name="Alice"

if [ "$name" = "Alice" ]; then
  echo "Condition is true"
elif [ another_condition ]; then
  echo "Another condition is true"
else
  echo "Condition is false"
fi

# 循环
for i in 2 3 5 7 11; do
  echo "Number $i"
done

for i in {1..5}; do
  echo "Number $i"
done

for file in $(ls *.txt); do
  echo "Processing $file"
done

count=1
while [ $count -le 5 ]; do
  echo "Count: $count"
  count=$((count + 1))
done

count=1
until [ $count -gt 5 ]; do
  echo "Count: $count"
  count=$((count + 1))
done

# 函数
my_function() {
  echo "Hello from my_function"
}
my_function

greet() {
  echo "Hello, $1"
}
greet "Alice"

# 数学运算
a=$((3 + 5))
echo "a = $a"  # 输出 8
b=$(expr 3 \* 5)
echo "b = $b"  # 输出 15

# 用户输入
echo "Enter your name:"
read name
echo "Hello, $name"


# $0: 当前脚本的文件名。
# $1, $2, ...: 脚本的第一个、第二个参数。
# $@: 所有参数的列表。
# $#: 参数的个数。
# $?: 上一个命令的退出状态。
# $$: 当前脚本的进程 ID。

# 数据结构
my_array=(1 2 3 4 5)
my_array[0]=value0
valuen=${array_name[n]} # 读取所有元素: ${array_name[n]}

declare -A associative_array
associative_array["name"]="John"
associative_array["age"]=30

string="runoob is a great site"
echo ${string:1:4} # 输出 unoo
echo `expr index "$string" io`  # 输出 4

