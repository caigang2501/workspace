import os
import shutil,random,string

def move_file(new_dir):
    current_directory = os.getcwd()
    # 遍历当前目录及其子目录下的所有文件和文件夹
    for foldername, subfolders, filenames in os.walk(current_directory):
        for filename in filenames:
            # 获取文件的完整路径
            file_path = os.path.join(foldername, filename)
            
            # 检查文件大小是否大于100M (100 * 1024 * 1024字节)
            if os.path.getsize(file_path) > 100 * 1024 * 1024:
                target_folder = os.path.join(current_directory, new_dir)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                shutil.move(file_path, os.path.join(target_folder, filename))

def rename_(path):
    for file_name in os.listdir(path):
        new_name = ''.join(random.choices(string.ascii_letters + string.digits, k=5))+'.jpg'
        old_path = os.path.join(path, file_name)
        new_path = os.path.join(path, new_name)
        if os.path.isfile(old_path):
            os.rename(old_path,new_path)
        else:
            rename_(old_path)

def clear_dir(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            clear_dir(file_path)

if __name__=='__main__':
    move_file('njs')
