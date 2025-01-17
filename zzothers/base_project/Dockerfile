FROM flask2501

WORKDIR /green_energy

COPY . .

# COPY requirements.txt .
# RUN pip install -r requirements.txt
# RUN pip install numpy scipy matplotlib pandas tqdm openpyxl

EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0"]

# gunicorn --workers 4 --threads 2 --log-level debug --bind 0.0.0.0:8000 wsgi:app
# ENV FLASK_APP=wsgi.py
# ENV FLASK_ENV=production
# CMD ["gunicorn", "--workers", "3","--log-level","debug","--bind","0.0.0.0:5000","wsgi:app"]

# docker build -t green_energy .
# docker save -o green_energy.tar green_energy
# scp danger_env.tar root@10.83.40.175:caigang
# docker load -i green_energy.tar
# unzip /tmp/green_energy.zip -d /gcph/pythonData/

# docker run -d -p 8088:5000  --name mygreen_energy green_energy
# docker run
#    -d -p 8088:5000
#    -v /tmp/python_logs:/green_energy/example/output
#    --name mygreen_energy green_energy
#    --network gp_network

# docker exec -it mygreen_energy /bin/sh            进入容器
# exit or ctrl+d                                    退出

