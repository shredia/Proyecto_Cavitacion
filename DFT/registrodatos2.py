import json
import websocket
import numpy as np 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
from matplotlib import style 


nWindow = 10 #tamaño de la ventana de datos

#arreglo dnde se guardan los datos obstenidos
acceleration_data = {'x': [], 'y': [], 'z': [], 's': []}

#cargar modelo
modelo = load_model('cavitacion.h5')  

#variables auxiliares
cont=0 #contador de data recopilada
promedio=[] #arreglo para realizar promedio entre las ultimas mediciones
lastvalues=[] #ultimos resultados para graficar

###estilo y confiugracion del grafico
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
label_mapping = {
    0: 'Sin cavitacion',
    1: 'Cavitacion'
}
y_values = [0, 1]
y_labels = [label_mapping[value] for value in y_values] 
aux11= list(label_mapping.keys())
aux12= list(label_mapping.values())


def procesarData():
    #crear ventana de hanning
    window = np.hanning(nWindow)

    x = acceleration_data['x']
    y = acceleration_data['y']
    z = acceleration_data['z']
    s = acceleration_data['s']

    #aplicar ventana a los datos
    x = np.array(x) * window
    y = np.array(y) * window
    z = np.array(z) * window
    s = np.array(s) * window
    #x = np.convolve(x, window, 'same')
    #y = np.convolve(y, window, 'same')
    #z = np.convolve(z, window, 'same')
    #s = np.convolve(s, window, 'same')

    padded_length = 2 ** np.ceil(np.log2(len(x))).astype(int) #zero padding

    #calcular la fft de los datos
    fft_x = np.fft.fft(x, n=padded_length)
    fft_y = np.fft.fft(y, n=padded_length)
    fft_z = np.fft.fft(z, n=padded_length)
    fft_s = np.fft.fft(s, n=padded_length)

  
    #reordenar datos para pasarlos como entradas
    rows = []

    for i in range(nWindow):
        rows.append(np.abs(fft_x[i]))
        rows.append(np.abs(fft_y[i]))
        rows.append(np.abs(fft_z[i]))
        rows.append(np.abs(fft_s[i]))
    ent= np.reshape(rows, (1, -1))

    #usar modelo para predecir el resultado
    resultado = np.argmax(modelo.predict(ent, verbose=0), axis=-1)[0]
    #resultado = modelo.predict(ent,verbose=0).astype(int)

    lastvalues.append(resultado)

    if resultado == 1:
        print('Cavitacion')
    elif resultado == 0:
        print('Sin cavitacion')
    else:
        print('Error')

     #cuando se superan los 10 valores, se elimina el ultimo valor y se realiza el grafico de la data 
    if(len(lastvalues)> 10):
        lastvalues.pop(0) #eliminar dato #11
        #graficar data
        ax1.clear()#limpiar grafico
        ax22=np.reshape(lastvalues,(10,-1))#reordenar data para poder ser usada en la grafica                       
        plt.yticks(aux11,aux12 ) #configuracion de lables en el eje Y
        plt.ylim([0, 1.5])#limite superior e inferior
        ax1.plot(ax22)#graficar  
        plt.draw()    #dibujar grafico
        plt.pause(.0000000001) #sin el pause no grafica!


def on_message(ws, message):
    global cont 
    cont +=1 #contador de datos recopilados
    values = json.loads(message)['values']    
    x, y, z, s= values[0], values[1], values[2], values[3]        
    acceleration_data['x'].append(x)
    acceleration_data['y'].append(y)
    acceleration_data['z'].append(z)
    acceleration_data['s'].append(s)
    #al recopilar nWindow datos se procesa la info y se realiza la prediccion
    if(cont == nWindow):              
        procesarData()
        acceleration_data['x'] = []
        acceleration_data['y'] = []
        acceleration_data['z'] = []
        acceleration_data['s'] = []
        cont = 0

def on_error(ws, error):
    print("error occurred")
    print(error)

def on_close(ws, close_code, reason):
    print("connection close")
    print("close code : ", close_code)
    print("reason : ", reason  )

def on_open(ws):
    print("connected")

def connect(url):
    ws = websocket.WebSocketApp(url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever()

# Llamar a la función connect
#cambiar ip segun la indicada en el celular    
connect("ws://192.168.1.112:8080/sensor/connect?type=android.sensor.linear_acceleration")