import os
import csv,shutil,re
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    width = root.find('size/width').text
    height = root.find('size/height').text

    bboxes = ''
    obj_names = []
    i = -1
    for obj in root.findall('object'):
        obj_names.append(obj.find('name').text)
        obj_name = obj.find('name').text
        if obj_name=='head':
            xmin = obj.find('bndbox/xmin').text
            ymin = obj.find('bndbox/ymin').text
            xmax = obj.find('bndbox/xmax').text
            ymax = obj.find('bndbox/ymax').text
            bboxes += xmin+' '+ymin+' '+xmax+' '+ymax +' 1;'
    if 'head' in obj_names and 'helmet' in obj_names:
        i += 1
        return [filename,bboxes[:-1],width, height]


    return False

def parse_all_xml(root):
    xml_folder = root + 'original_lables' # 修改为你的xml文件所在的文件夹路径
    output_csv = root + 'annotations.csv' # 修改为你的输出CSV文件路径

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['imgname', 'bboxes', 'width', 'height'])

        i = 0
        for xml_file in os.listdir(xml_folder):
            xml_path = os.path.join(xml_folder, xml_file)
            annotation = parse_xml(xml_path)
            if annotation:
                source_file = root+'all_imgs/'+annotation[0]
                annotation[0] = re.sub(r'(\D+)(\d+)(\D+)', rf'\g<1>{i}\g<3>', annotation[0])
                destination_folder = root+'nohelmet_imgs/'+annotation[0]
                
                shutil.copy(source_file,destination_folder)
                csv_writer.writerow(annotation)
                i += 1

if __name__ == "__main__":
    root = 'data/traindata/objdetection/helmet/'  
    parse_all_xml(root)

