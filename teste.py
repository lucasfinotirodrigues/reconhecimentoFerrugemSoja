from ultralytics import YOLO
import cv2
import os

# Carrega o modelo YOLOv8 treinado
model = YOLO('/home/ciag/Desktop/reconhecimentoFerrugemSoja/runs/detect/train9/weights/best.pt')  # substitua pelo caminho do seu modelo treinado

# Caminho para as imagens de teste (pode ser uma única imagem ou um diretório)
image_path = '/home/ciag/Desktop/reconhecimentoFerrugemSoja/val/images/ferrugem24.jpg'

# Pasta de saída para salvar as imagens com detecção (opcional)
output_dir = 'detections_output'
os.makedirs(output_dir, exist_ok=True)

def detect_and_save(image_path):
    # Realiza a detecção
    results = model(image_path)
    
    # Pega o primeiro resultado da lista
    result = results[0]
    
    # Verifica se houve alguma detecção
    if result.boxes is not None and len(result.boxes) > 0:
        # Salva a imagem com as bounding boxes
        output_file = os.path.join(output_dir, os.path.basename(image_path))
        result.plot(save=True, filename=output_file)
        
        # Carrega a imagem salva para exibir
        img = cv2.imread(output_file)
        cv2.imshow("Detecções", img)
        cv2.waitKey(0)  # Aguarda até que uma tecla seja pressionada
        cv2.destroyAllWindows()  # Fecha a janela de visualização

        print(f"Detecção salva e exibida em: {output_file}")
    else:
        print("Nenhuma detecção encontrada na imagem.")

# Verifica se o caminho é uma única imagem ou uma pasta de imagens
if os.path.isfile(image_path):
    detect_and_save(image_path)
elif os.path.isdir(image_path):
    # Se for um diretório, processa cada imagem na pasta
    for img_file in os.listdir(image_path):
        img_path = os.path.join(image_path, img_file)
        if img_file.endswith(('.jpg', '.jpeg', '.png')):  # Verifica formatos de imagem
            detect_and_save(img_path)
else:
    print("Caminho inválido. Certifique-se de fornecer uma imagem ou pasta de imagens.")
