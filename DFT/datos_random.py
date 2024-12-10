import pandas as pd
import numpy as np

def generar_datos_csv(nombre_archivo, num_datos=100, dimension=10):
    """
    Genera un archivo CSV con datos aleatorios usando pandas.
    
    Args:
        nombre_archivo (str): Nombre del archivo CSV a crear.
        num_datos (int): Cantidad de datos a generar.
        dimension (int): Dimensión de cada dato (número de características).
    """
    # Generar datos aleatorios
    datos = np.random.rand(num_datos, dimension)
    
    # Crear un DataFrame
    columnas = [f"feature_{i+1}" for i in range(dimension)]
    df = pd.DataFrame(datos, columns=columnas)
    
    # Guardar el DataFrame en un archivo CSV
    df.to_csv(nombre_archivo, index=False,header=None   )
    print(f"Archivo '{nombre_archivo}' generado con {num_datos} datos y {dimension} características.")

# Ejemplo de uso
generar_datos_csv("datos_audio_no_cavitacion.csv", num_datos=25, dimension=2**10)
