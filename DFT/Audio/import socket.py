import socket
import pyaudio

# Configuración de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 8192*2

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

try:
    while True:
        # Recibir datos desde Android
        data, addr = sock.recvfrom(CHUNK * 2)
        stream.write(data)  # Reproducir el audio
except KeyboardInterrupt:
    print("\nCerrando la escucha...")
finally:
    # Liberar recursos
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sock.close()
    print("Recursos liberados.")

