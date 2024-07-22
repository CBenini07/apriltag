''''
Como usar:
    python -m venv myenv
    source myenv/bin/activate
    pip install opencv-python apriltag
    python detector_2d_video.py
'''
import cv2
import apriltag

# Definir cores para a plotagem
LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

# Função para desenhar uma cruz nos cantos das tags
def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

# Função para plotar texto na imagem
def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)

# Inicializar a câmera USB (substitua 0 pelo índice da sua câmera se necessário)
cap = cv2.VideoCapture(1)

# Verificar se a câmera foi aberta com sucesso
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit(1)

# Inicializar o detector de AprilTags
detector = apriltag("tag36h11")
detector = apriltag.Detector()

while True:
    ret, frame = cap.read() # Capturar o quadro atual da câmera

    if not ret:
        print("Erro ao capturar o quadro")
        break

    # Converter o quadro para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar AprilTags no quadro
    detections = detector.detect(gray_frame)

    print(f"Número de tags detectadas: {len(detections)}")
    # Exibir as coordenadas das tags detectadas
    if len(detections) >= 1:
        for i, detection in enumerate(detections):
            # Acessar e imprimir atributos específicos da detecção
            detection_id = detection.tag_id
            detection_center = detection.center
            detection_corners = detection.corners
            print(f"Detecção {i}: ID da Tag = {detection_id}, Centro = {detection_center}")

    # Imprimir as coordenadas dos cantos
    for j, corner in enumerate(detection_corners):
        print(f"Canto {j}: {corner}")

    # Desenhar as detecções na imagem
    for detection in detections:
        # Desenhar o centro da tag
        frame = plotPoint(frame, detection.center, CENTER_COLOR)
        # Adicionar texto com o ID da tag
        frame = plotText(frame, detection.center, CENTER_COLOR, detection.tag_id)
        # Desenhar os cantos da tag
        for corner in detection.corners:
            frame = plotPoint(frame, corner, CORNER_COLOR)

    # Exibir o quadro com as detecções desenhadas
    cv2.imshow('Detections', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
