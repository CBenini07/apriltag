import cv2
import os
import time

# Nome da pasta onde as imagens serão salvas
output_folder = 'camera_calibration/calibration_images'

# Cria a pasta se ela Nao existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

id_camera = 0
# Tentar abrir a camera com o backend DirectShow
cap = cv2.VideoCapture(id_camera, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Erro: Nao foi possivel abrir a camera com o backend padrao. Tentando com DirectShow...")
    cap = cv2.VideoCapture(id_camera)
    if not cap.isOpened():
        print("Erro: Nao foi possivel abrir a camera.")
        exit()
        
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro: Nao foi possivel capturar o quadro.")
        break

    cv2.imshow('frame', frame)
    # Salva a imagem na pasta especificada
    cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count}.png'), frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.3) # def o framerate 
    
cap.release()
cv2.destroyAllWindows()
