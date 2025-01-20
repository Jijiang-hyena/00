import fiftyone as fo
import fiftyone.utils.ultralytics as fou
import fiftyone.zoo as foz
import os
from fiftyone import ViewField as F
# 导出目录
EXPORT_DIR = "/home/bird/yolo_dataset_1"

# 加载 COCO 2017 验证集
dataset = fo.zoo.load_zoo_dataset("coco-2017", split="validation")

# 过滤出包含 "person" 标签的目标
person_view = dataset.filter_labels("ground_truth", F("label") == "person")

# 进一步筛选包含至少一个人目标的样本
person_samples = person_view.match(F("ground_truth.detections") != [])

# 手动指定类别列表，只有 "person" 类别
# classes = ["person"]
classes = person_view.default_classes

# 删除现有的 dataset.yaml 文件（如果已存在）
yaml_path = os.path.join(EXPORT_DIR, "dataset.yaml")
if os.path.exists(yaml_path):
    os.remove(yaml_path)  # 删除原来的 dataset.yaml

# 导出训练集
person_samples.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,  # 这里是YOLOv5格式，兼容YOLOv8
    label_field="ground_truth",
    split="train",  # 导出为训练集
    classes=classes,
    
)

# 导出验证集
validation_samples = person_samples[:len(person_samples)//5]  # 选择一部分作为验证集（比如20%）

validation_samples.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="val",  # 导出为验证集
    classes=classes,  # 确保使用相同的类别列表
    
)
