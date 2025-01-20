import fiftyone as fo
from fiftyone import ViewField as F

# 加载 COCO 2017 验证集（或其他版本、分割）
# dataset = fo.load_dataset("coco-2017", split="validation")
dataset = fo.zoo.load_zoo_dataset("coco-2017", split="validation")

# 假设 COCO 标注存储在字段 "ground_truth"，其中每个检测结果都有 "label" 属性
# 这里通过过滤“ground_truth”字段中 label 为 "person" 的标注，得到包含人类目标的视图
person_view = dataset.filter_labels("ground_truth", F("label") == "person")

# 进一步只保留包含至少一个人目标的样本
person_samples = person_view.match(F("ground_truth.detections") != [])

# 可视化筛选后的数据
session = fo.launch_app(person_samples)
session.wait()  # 保持应用界面
