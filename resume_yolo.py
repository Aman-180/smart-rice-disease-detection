from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/detect/rice_disease_yolo-3/weights/last.pt")
    results = model.train(resume=True, epochs=40)

    print("Training complete.")
    print(f"Best weights saved at: {results.save_dir}")

if __name__ == "__main__":
    main()