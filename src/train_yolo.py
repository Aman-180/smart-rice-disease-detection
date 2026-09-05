from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/detect/rice_disease_yolo-3/weights/last.pt")  # starts from epoch 10 weights, but as a NEW run
    results = model.train(
        data="dataset_yolo/data.yaml",
        epochs=20,          # this WILL be honored since resume=True is not set
        imgsz=640,
        batch=5,
        patience=15,
        device=0,
        workers=2,
        project="runs/detect",
        name="rice_disease_yolo_continued"
    )
    print("Training complete.")
    print(f"Best weights saved at: {results.save_dir}")

if __name__ == "__main__":
    main()