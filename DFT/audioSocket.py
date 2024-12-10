import socket
import pyaudio
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import style

nWindow = 2**10
modelo = load_model('cavitacion.h5') 
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lastvalues = []
ent_list = []  # Lista para guardar los vectores de "ent"
hanning_window = np.hanning(nWindow)

### Configuración del gráfico con subplots
style.use('fivethirtyeight')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Crear dos subplots

# ax1: Gráfico de resultados del modelo
# ax2: Gráfico de la FFT en tiempo real

label_mapping = {
    0: 'chill de cojones',
    1: 'CAVITACION CTM CORRE!!'
}
y_values = [0, 1]
y_labels = [label_mapping[value] for value in y_values]
aux11 = list(label_mapping.keys())
aux12 = list(label_mapping.values())

# Configuración de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 8192 * 2

# Configuración del socket UDP
UDP_IP = "0.0.0.0"
UDP_PORT = 5000

# Inicializar PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK)

# Configurar el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Escuchando en {UDP_IP}:{UDP_PORT}...")

def procesarData(audio_data):
    global ent_list

    ffts = np.fft.fft(audio_data)
    fft_magnitude = np.abs(ffts[:len(ffts)//2])  
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)

    # Reordenar datos para pasarlos como entradas
    rows = []
    for i in range(nWindow):
        rows.append(abs(ffts[i]))  
    ent = np.reshape(rows, (1, -1))

    # Guardar el vector "ent"
    ent_list.append(ent.flatten())  # Aplanar para guardar como una fila en el CSV

    # Usar modelo para predecir el resultado
    resultado = modelo.predict(ent, verbose=0).astype(float)
    lastvalues.append(resultado)

    # Cuando se superan los 10 valores, se elimina el último valor y se realiza el gráfico de la data 
    if len(lastvalues) > 10:
        lastvalues.pop(0)  # Eliminar dato #11
        # Graficar resultados del modelo (ax1)
        ax1.clear()
        ax22 = np.reshape(lastvalues, (10, -1))
        plt.sca(ax1)
        plt.yticks(aux11, aux12)
        plt.ylim([-0.1, 1.5])
        ax1.plot(ax22)
        ax1.set_title('Resultados del Modelo')
        ax1.set_ylabel('Predicción')

        # Graficar FFT en tiempo real (ax2)
        ax2.clear()
        ax2.plot(fft_magnitude)
        ax2.set_title('FFT en Tiempo Real')
        ax2.set_ylabel('Magnitud')
        ax2.set_xlabel('Frecuencia (bin)')
        ax2.set_xlim([0, len(fft_magnitude)])  # Ajustar el eje X según la longitud de la FFT

        plt.pause(0.0001)  # Sin el pause no grafica!


try:
    while True:
        # Recibir datos desde Android
        data, addr = sock.recvfrom(CHUNK * 2)
        
        # Convertir los datos de bytes a un arreglo de números
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Normalizar datos  
        audio_data = audio_data[:nWindow]

        # Procesar el audio_data aquí
        procesarData(audio_data)
        
        # Reproducir el audio
        stream.write(data)
except KeyboardInterrupt:
    print("\nCerrando la escucha...")
finally:
    # Guardar los vectores "ent" en un archivo CSV
    ent_df = pd.DataFrame(ent_list)
    ent_df.to_csv('datos_audio_no_cavitacion.csv', index=False,header=None)
    print("Vectores 'ent' guardados en 'ent_vectors.csv'.")

    # Liberar recursos
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sock.close()
    print("Recursos liberados.")
