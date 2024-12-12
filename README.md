Este es un proyecto enfocado en detectar la cavitación de una bomba mediante el uso de un micrófono de celular android (también soporta vibraciones) integrandolo con una IA.
El enfoque que permite unir a la IA con la comunicación es el uso de una STFT, la cual es una ventana de tamaño fijo que permite aplicar una FFT y analizar una variación de frecuencia o de tiempo dependiendo de la resolución de la ventana.

Existen 3 carpetas principales:
  Sensor-server-Main (APK android): Basado en el proyecto de: https://github.com/umer0586/SensorServer , se modificó para enviar streaming de audio a una dirección IP mediante un thread.
    Se puede modificar mediante Android Studio, ó, si solo se quiere ejecutar se puede ir a la carpeta SensorServer-main\app\build\outputs\apk\debug, donde estará el archivo apk. Instalar en el dispositivo.
    Es necesario darles permisos para grabar audio directamente, puesto que la aplicación no los pedirá. Al abrir la aplicación hay que insertar la dirección IP (LAN), para posteriormente "iniciar" y dejar el botón de abajo en "iniciar audio".

  IA (No es necesario tocar esta carpeta):En esta carpeta se encuentran algunos modelos de entrenamiento de Cavitación y no-cavitación, aparte de el modelo de la IA (.h5), adicionalmente de RN_normal que sirve para el entrenamiento de las redes con otro archivo que le permite modificar sus capas personalizadas con python.

  DFT(Python e IA):
    audiosocket: Es el encargado de establecer la comunicación en tiempo real y comunicarse con la IA mediante ventanas para que esta decida si está cavitando o no.
    GuardarDatos: Este programa establece comunicación en tiempo real y empieza a guardar la información con un estado de "cavitando" o "no está cavitando", para entrenar a la IA.
  
