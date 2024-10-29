from ultralytics import YOLO

# Carrega o modelo pré-treinado YOLOv8 para fine-tuning
model = YOLO('yolov8n.pt')  # escolha a versão adequada para o seu hardware (n: nano, s: small, m: medium, etc.)

# Treina o modelo com os dados fornecidos
model.train(data='data.yaml', epochs=100, imgsz=640, batch=8)
