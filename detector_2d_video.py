import cv2
import numpy as np
import apriltag

# Carregar os parâmetros de calibração da câmera
with np.load('camera_calibration/camera_calibration_params.npz') as X:
    K, dist = [X[i] for i in ('K', 'dist')]

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

id_camera = 0
# Tentar abrir a camera com o backend DirectShow
cap = cv2.VideoCapture(id_camera, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Erro: Nao foi possivel abrir a camera com o backend DirectShow, tentando o padrão...")
    cap = cv2.VideoCapture(id_camera)
    if not cap.isOpened():
        print("Erro: Nao foi possivel abrir a camera.")
        exit()

# Inicializar o detector de AprilTags
options = apriltag.DetectorOptions(families='tag36h11',
                                 border=1,
                                 nthreads=4,
                                 quad_decimate=1.0,
                                 quad_blur=0.0,
                                 refine_edges=True,
                                 refine_decode=False,
                                 refine_pose=False,
                                 debug=False,
                                 quad_contours=True)
detector = apriltag.Detector(options)

# Def. o tamanho da tag (em metros)
tag_size = 0.165  

# Função para converter coordenadas de pixels para metros usando solvePnP
def convert_to_real_world_coords(corners, tag_size, K, dist):
    # Define os pontos 3D da tag no mundo real
    object_points = np.array([
        [-tag_size/2, tag_size/2, 0],
        [tag_size/2, tag_size/2, 0],
        [tag_size/2, -tag_size/2, 0],
        [-tag_size/2, -tag_size/2, 0]
    ], dtype=np.float32)

    # Estima a pose da tag em relação à câmera
    success, rvec, tvec = cv2.solvePnP(object_points, corners, K, dist)

    if success:
        return tvec
    else:
        return None

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

            # Converter coordenadas para o mundo real
            real_center = convert_to_real_world_coords(detection_corners, tag_size, K, dist)
            if real_center is not None:
                print(f"Coordenadas do Centro (metros): {real_center.flatten()}")

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
