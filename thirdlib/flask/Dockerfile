# 使用一个基础的 Python 镜像
FROM tensorflow/tensorflow

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到工作目录
COPY . .

# 安装依赖
#--ignore-installed
RUN pip install -r requirements.txt

# 暴露服务运行的端口
EXPOSE 5000

# 定义环境变量
ENV FLASK_APP=app.app.py

# 运行 Flask 服务
CMD ["flask", "run", "--host=0.0.0.0"]

# docker build -t your-image-name .
# docker save -o your-image-name.tar your-image-name
# scp your-image-name.tar username@10.83.190.87:/path/to/destination
# docker load -i /path/to/destination/your-image-name.tar
# docker run -d -p 5000:5000 your-image-name

#删除指定容器
#docker rm <container_id_or_name>
#删除所有停止的容器
#docker container prune
#删除指定镜像
#docker rmi <image_id_or_name>
#删除所有未被使用的镜像
#docker image prune
## 删除所有容器
#docker rm $(docker ps -aq)
## 删除所有镜像
#docker rmi $(docker images -q)




