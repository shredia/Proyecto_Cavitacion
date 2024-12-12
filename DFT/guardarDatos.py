import socket
import pyaudio
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import style

nWindow = 2**13

archivo = "cavitacion.csv"
#archivo = "noCavitacion.csv"


lastvalues=[]
accumulated_data = np.array([], dtype=np.float32)
hanning_window = np.hanning(nWindow)


# Configuración del gráfico para la FFT
style.use('fivethirtyeight')
fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))  # Crear un solo subplot

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


def grabarData(datos):
    rows = []
    with open(archivo, 'a', newline='') as file:
        writer = csv.writer(file)
        rows = [datos[i] for i in range(len(datos))]
        writer.writerow(rows)
        #file.flush()



def procesarData(audio_data):
    ffts = np.fft.fft(audio_data)
    fft_magnitude = np.abs(ffts[:len(ffts)//2])  
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)
    
    # Calcular frecuencias correspondientes a los bins
    freqs = np.fft.fftfreq(len(audio_data), d=1/RATE)[:len(ffts)//2]

    # Guardar datos en el archivo CSV
    grabarData(fft_magnitude)

    # Graficar FFT en tiempo real (ax2)
    ax2.clear()
    ax2.plot(freqs, fft_magnitude)  # Usar frecuencias en lugar de bins
    ax2.set_title('FFT en Tiempo Real')
    ax2.set_ylabel('Magnitud')
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_xlim([0, RATE / 2])  # Limitar el eje X a la mitad de la frecuencia de muestreo

    plt.pause(0.001)  # sin el pause no grafica

try:
    while True:
        # Recibir datos desde Android
        data, addr = sock.recvfrom(CHUNK * 2)
        
        # Convertir los datos de bytes a un arreglo de números
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32)
        accumulated_data = np.append(accumulated_data, audio_data)

        # Normalizar ventana para evitar que las DNN aprendan por la intensidad del sonido
        accumulated_data = accumulated_data / np.max(np.abs(accumulated_data))

        if len(accumulated_data) >= nWindow:
            audio_data = accumulated_data[:nWindow]
            procesarData(audio_data)

            # Limpiar buffer
            accumulated_data = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("\nCerrando la escucha...")
finally:
    # Liberar recursos
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sock.close()
    print("Recursos liberados.")
