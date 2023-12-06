import requests
r = requests.get('http://192.168.0.109:5000/user/dssd')
print(r.status_code)
print(r.content)