import json
import websocket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nWindow = 10  # Tamaño de la ventana de datos
acceleration_data = {'x': [], 'y': [], 'z': [], 's': []}  # Datos de aceleración
cont = 0  # Contador para la cantidad de datos recopilados

# Configuración del gráfico en tiempo real
fig, ax1 = plt.subplots()
ax1.set_title("Gráfico en Tiempo Real")
ax1.set_xlabel("Muestras")
ax1.set_ylabel("Amplitud")
ax1.set_ylim([0, 1.5])  # Límite en el eje Y
lines = [ax1.plot([], [], label=label)[0] for label in ['X', 'Y', 'Z', 'S']]  # Líneas del gráfico
ax1.legend()

def procesarData():
    global acceleration_data

    # Aplicar ventana de Hanning
    window = np.hanning(nWindow)
    x = np.array(acceleration_data['x']) * window
    y = np.array(acceleration_data['y']) * window
    z = np.array(acceleration_data['z']) * window
    s = np.array(acceleration_data['s']) * window

    # Calcular la FFT con zero padding
    padded_length = 2 ** np.ceil(np.log2(len(x))).astype(int)
    fft_x = np.abs(np.fft.fft(x, n=padded_length))
    fft_y = np.abs(np.fft.fft(y, n=padded_length))
    fft_z = np.abs(np.fft.fft(z, n=padded_length))
    fft_s = np.abs(np.fft.fft(s, n=padded_length))

    # Actualizar datos del gráfico
    for line, fft_data in zip(lines, [fft_x, fft_y, fft_z, fft_s]):
        line.set_data(range(len(fft_data[:nWindow])), fft_data[:nWindow])

    acceleration_data['x'] = []
    acceleration_data['y'] = []
    acceleration_data['z'] = []
    acceleration_data['s'] = []

def actualizarGrafico(frame):
    if len(acceleration_data['x']) == nWindow:
        procesarData()
    return lines

def on_message(ws, message):
    global cont
    cont += 1
    values = json.loads(message)['values']
    x, y, z, s = values[0], values[1], values[2], values[3]
    acceleration_data['x'].append(x)
    acceleration_data['y'].append(y)
    acceleration_data['z'].append(z)
    acceleration_data['s'].append(s)

def on_error(ws, error):
    print("Ocurrió un error:")
    print(error)

def on_close(ws, close_code, reason):
    print("Conexión cerrada")
    print("Código de cierre:", close_code)
    print("Razón:", reason)

def on_open(ws):
    print("Conectado al servidor")

def connect(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()

# Configurar animación para graficar
ani = FuncAnimation(fig, actualizarGrafico, interval=100)

# Cambiar IP según la configuración del dispositivo
connect("ws://192.168.1.7:8080/sensor/connect?type=android.sensor.linear_acceleration")
plt.show()
