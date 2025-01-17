echo "正在复制"
rm -rf /tmp/caigang/green_energy
unzip /tmp/green_energy.zip -d /tmp/caigang/

echo "正在删除"
cid=$(docker ps | grep mygreen_energy | awk '{print $1}')
if [ -n "$cid" ]; then
  docker stop "$cid"
else
  echo "No container found for mygreen_energy."
fi
docker rm "${cid}"
docker rmi green_energy

echo "构建并启动"
docker build -t green_energy /tmp/caigang/green_energy/
docker run -d -p 8088:5000 -v /tmp/python_logs:/green_energy/example/output --name mygreen_energy green_energy
echo "完成!!!"

# .7 chmod +x /gcph/pythonData/green_energy.sh
# .6 chmod +x /tmp/green_energy.sh
# http://10.30.107.5:9980/dev/green_energy.git

