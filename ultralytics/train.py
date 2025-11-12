from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/model_pretrain/yolov8m.pt")  # load a pretrained model (recommended for transfer learning)

    results = model.train(
        data="ultralytics/data.yaml",
        epochs=200,         
        imgsz=640,
        batch=4,
        degrees=30,     
        project="ultralytics/detection_results",  
        deterministic=False,
        save_period=5,
        device=1,
        workers=4,
        name="yolov8n_CCPD2019_plate_detection",
        single_cls=True,
        close_mosaic=10,
    )