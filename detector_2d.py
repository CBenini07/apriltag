''''
Como usar:
    python -m venv myenv
    source myenv/bin/activate
    pip install opencv-python apriltag
    cd ws_apriltag/apriltag
    python detector_2d.py
'''

# referência do plot: https://blog.fixermark.com/posts/2022/april-tags-python-recognizer/
# --------------------------------------------------------------------------
LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

# Plota cruz nos cantos das tags
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

def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)
# --------------------------------------------------------------------------
import cv2
import apriltag

# Caminho para a imagem contendo a AprilTag
image_path = '../tags/images.png'  # Certifique-se de que este caminho está correto

# Carregar a imagem em escala de cinza
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print(f"Erro ao carregar a imagem: {image_path}")
    exit(1)
                               # PARAMETROS
detector = apriltag.Detector() # Detector(searchpath=['apriltags'],
                               #          families='tag36h11',
                               #          nthreads=1,
                               #          quad_decimate=1.0,
                               #          quad_sigma=0.0,
                               #          refine_edges=1,
                               #          decode_sharpening=0.25,
                               #          debug=0)

detections = detector.detect(image)

print(f"Número de tags detectadas: {len(detections)}")

# Exibir as coordenadas das tags detectadas
for i, detection in enumerate(detections):
    # Acessar e imprimir atributos específicos da detecção
    detection_id = detection.tag_id
    detection_center = detection.center
    detection_corners = detection.corners
    print(f"Detecção {i}: ID da Tag = {detection_id}, Centro = {detection_center}")

    # Imprimir as coordenadas dos cantos
    for j, corner in enumerate(detection_corners):
        print(f"Canto {j}: {corner}")

# Converter a imagem para BGR para desenhar em cores
image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Desenhar as detecções na imagem
for detection in detections:
    image_bgr = plotPoint(image_bgr, detection.center, CENTER_COLOR)
    image_bgr = plotText(image_bgr, detection.center, CENTER_COLOR, detection.tag_id)
    for corner in detection.corners:
        image_bgr = plotPoint(image_bgr, corner, CORNER_COLOR)

# Exibir a imagem com as detecções desenhadas
cv2.imshow('Detections', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
