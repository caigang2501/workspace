import os
import shutil

# 获取当前目录
current_directory = os.getcwd()

# 遍历当前目录及其子目录下的所有文件和文件夹
for foldername, subfolders, filenames in os.walk(current_directory):
    for filename in filenames:
        # 获取文件的完整路径
        file_path = os.path.join(foldername, filename)
        
        # 检查文件大小是否大于100M (100 * 1024 * 1024字节)
        if os.path.getsize(file_path) > 100 * 1024 * 1024:
            target_folder = os.path.join(current_directory, 'njs')
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            shutil.move(file_path, os.path.join(target_folder, filename))

print("移动完成。")
