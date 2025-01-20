from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
# import comet_ml
# import clearml



if __name__ == '__main__':
    # comet_ml.init(api_key="a4SXVv36icYUUjVBuy5G2keEi")
    # clearml.browser_login()

    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO("/home/user1/hyena_workspace/py_workspace/yolov11_train_1/yolo11m.pt")
    model.train(data=r'/home/user1/hyena_workspace/yolo_dataset_1/dataset.yaml',
                imgsz=640,
                epochs=50,
                batch=0.80,
                workers=0,
                device='0',
                optimizer='AdamW',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='train_1',
                single_cls=False,
                cache=False,
                )
