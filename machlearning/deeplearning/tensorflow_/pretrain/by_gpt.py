import numpy as np
import tensorflow as tf
from tf_slim.nets import inception
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

model_name = 'ssd_inception_v2_coco_2017_11_17'  # 选择一个预训练的模型
detection_model = tf.saved_model.load(model_name)

label_map_path = 'path/to/label_map.pbtxt'  # 替换为实际的标签映射文件路径
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

image_path = 'path/to/your/image.jpg'  # 替换为实际的图像路径
image_np = np.array(tf.keras.preprocessing.image.load_img(image_path))
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

detections = detection_model(input_tensor)
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np[0], detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(np.int32),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,  # 可视化的最大框数
    min_score_thresh=.5,  # 可视化的最小置信度
    agnostic_mode=False)


