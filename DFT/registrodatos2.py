import websocket
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import threading
import tkinter as tk
import queue

nWindow = 10  # tamaño de la ventana de datos

# Arreglo donde se guardan los datos obtenidos
acceleration_data = {'x': [], 'y': [], 'z': []}

# Variables auxiliares
cont = 0  # contador de datos recopilados
lastvalues = []  # últimos resultados para graficar

# Estilo y configuración del gráfico
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# Crear una cola para comunicar los datos del hilo de WebSocket con el hilo principal
data_queue = queue.Queue()

# Función para procesar los datos y graficarlos
def procesarData():
    global lastvalues
    while not data_queue.empty():
        data = data_queue.get()  # Obtener datos de la cola
        # Crear ventana de Hanning
        window = np.hanning(nWindow)

        # Aplicar ventana a los datos
        x = np.array(data['x']) * window
        y = np.array(data['y']) * window
        z = np.array(data['z']) * window

        # Zero padding para realizar la FFT
        padded_length = 2 ** np.ceil(np.log2(len(x))).astype(int)

        # Calcular la FFT de los datos
        fft_x = np.fft.fft(x, n=padded_length)
        fft_y = np.fft.fft(y, n=padded_length)
        fft_z = np.fft.fft(z, n=padded_length)

        # Reordenar datos para pasarlos como entradas
        rows = []
        for i in range(nWindow):
            rows.append(np.abs(fft_x[i]))
            rows.append(np.abs(fft_y[i]))
            rows.append(np.abs(fft_z[i]))

        ent = np.reshape(rows, (1, -1))
        lastvalues.append(ent[0])  # Agregar los nuevos valores procesados

        # Limpiar datos procesados
        acceleration_data['x'] = []
        acceleration_data['y'] = []
        acceleration_data['z'] = []

        # Graficar los datos en tiempo real
        if len(lastvalues) > 10:
            lastvalues.pop(0)  # Eliminar el dato más antiguo

        ax1.clear()  # Limpiar gráfico
        ax22 = np.reshape(lastvalues, (len(lastvalues), -1))  # Reordenar data para graficar
        ax1.plot(ax22)  # Graficar
        plt.draw()  # Dibujar gráfico
        plt.pause(0.0001)  # Sin el pause no grafica

    # Volver a programar la actualización
    root.after(100, procesarData)  # Ejecutar nuevamente después de 100ms

# Función de conexión WebSocket
def on_message(ws, message):
    global cont
    cont += 1  # contador de datos recopilados
    values = json.loads(message)['values']
    x, y, z = values[0], values[1], values[2]  # Solo 3 datos (x, y, z)
    acceleration_data['x'].append(x)
    acceleration_data['y'].append(y)
    acceleration_data['z'].append(z)

    # Al recopilar nWindow datos, se procesa la info y se realiza la predicción
    if cont == nWindow:
        # Colocar los datos en la cola para que el hilo principal los procese
        data_queue.put({'x': acceleration_data['x'], 'y': acceleration_data['y'], 'z': acceleration_data['z']})
        cont = 0

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_code, reason):
    print("Conexión cerrada:", close_code, reason)

def on_open(ws):
    print("Conectado al servidor WebSocket")

# Función para iniciar la conexión
def connect(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# Función para iniciar la conexión en un hilo
def iniciar_conexion():
    connect("ws://192.168.1.7:8080/sensor/connect?type=android.sensor.accelerometer")

# Función para manejar el evento del botón de detener
def detener_peticion(conexion_thread):
    print("Deteniendo la conexión WebSocket...")
    # Detener el hilo si es posible (mejor forma de hacerlo depende de la implementación del WebSocket)
    conexion_thread.join()  # Espera a que el hilo termine

# Función para manejar el evento del botón de iniciar
def iniciar_peticion():
    global conexion_thread
    print("Iniciando petición WebSocket...")
    # Crear un hilo para la conexión WebSocket
    conexion_thread = threading.Thread(target=iniciar_conexion)
    conexion_thread.start()

# Crear la ventana gráfica
def crear_grafico():
    plt.ion()  # Habilitar modo interactivo
    plt.show()

# Crear ventana de interfaz gráfica
root = tk.Tk()
root.title("Control de Conexión WebSocket")

# Crear un botón para empezar la recepción de datos
start_button = tk.Button(root, text="Iniciar Conexión WebSocket", command=iniciar_peticion)
start_button.pack(pady=20)

# Crear un botón para detener la recepción de datos
stop_button = tk.Button(root, text="Detener Conexión WebSocket", command=lambda: detener_peticion(conexion_thread))
stop_button.pack(pady=20)

# Crear la ventana gráfica
crear_grafico()

# Iniciar la interfaz gráfica y el procesamiento de los datos
root.after(100, procesarData)  # Iniciar la primera llamada para procesar los datos
root.mainloop()