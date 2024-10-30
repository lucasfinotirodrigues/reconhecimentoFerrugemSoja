from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2
from teste import detectImage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DETECTIONS_OUTPUT_FOLDER'] = 'static/detections_output'
os.makedirs(app.config['DETECTIONS_OUTPUT_FOLDER'], exist_ok=True)

# Carrega o modelo YOLO treinado
model = YOLO('yolov8n.pt')  # Atualize o caminho se necessário

@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/detecao')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Salvar imagem carregada
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        filename = detectImage(filepath)

        # # Fazer a predição na imagem
        # results = model(filepath)

        # # Carregar a imagem original com OpenCV
        # img = cv2.imread(filepath)

        # # Verifique se o resultado contém detecções
        # highest_confidence = 0
        # bounding_boxes_drawn = 0

        # if results and len(results) > 0:
        #     for result in results:
        #         if result.boxes is not None and len(result.boxes) > 0:
        #             for detection in result.boxes:
        #                 x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Coordenadas do bounding box
        #                 confidence = detection.conf[0].item()  # Confiança da detecção

        #                 if confidence > 0.2:  # Threshold de confiança
        #                     bounding_boxes_drawn += 1
        #                     if confidence > highest_confidence:
        #                         highest_confidence = confidence

        #                     # Desenhar bounding boxes na imagem
        #                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #                     cv2.putText(img, f'Ferrugem {confidence:.2f}', (x1, y1 - 10), 
        #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        #     if bounding_boxes_drawn == 0:
        #         print("Nenhuma área detectada com confiança suficiente.")
        #     else:
        #         print(f"{bounding_boxes_drawn} bounding boxes desenhadas.")

            # Salvar a imagem com bounding boxes
        #     output_path = os.path.join(app.config['DETECTIONS_OUTPUT_FOLDER'], f"output_{file.filename}")
        #     cv2.imwrite(output_path, img)

        #     # Exibir resultado
        print(str(filename))
        return render_template('index.html', filename=str(filename), disease="Ferrugem")
        # else:
            # print("Nenhum resultado retornado do modelo.")
            # return redirect(url_for('index'))

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
