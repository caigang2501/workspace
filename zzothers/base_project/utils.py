import os,json,shutil
import zipfile
from datetime import datetime


def clear_dir(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            clear_dir(file_path)

def save_dict_to_folder(data_dict):
    shutil.rmtree(os.getcwd()+'/example/input')
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')
    
    folder_path = os.path.join(os.getcwd()+'/example/input', current_date)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, f'{current_time}.json')
    
    with open(file_path, 'w',encoding="utf-8") as f:
        json.dump(data_dict, f,ensure_ascii=False)

def zip_folder(folder_path):
    
    if not os.path.isdir(folder_path):
        return 'Folder not found'

    zip_path = os.path.dirname(folder_path)+'/logs.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

    return zip_path

