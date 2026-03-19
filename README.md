# LABORATORIO VOZ
En esta práctica de Procesamiento Digital de Señales se realiza el análisis espectral de señales de voz humana mediante herramientas de procesamiento en Python. A partir de grabaciones de voces masculinas y femeninas, se aplica la Transformada de Fourier para estudiar su comportamiento en el dominio de la frecuencia y extraer características relevantes como frecuencia fundamental, frecuencia media, brillo, intensidad, jitter y shimmer. El objetivo es identificar y comparar diferencias entre ambos tipos de voz, comprendiendo cómo estas características permiten analizar propiedades acústicas del habla y su posible aplicación en áreas como reconocimiento de voz y evaluación biomédica.  <br> 
## OBJETIVOS
### General 
 Emplear técnicas de análisis espectral para la diferenciación
o clasificación de señales de voz según el género.<br> 
### Especificos 
- Capturar y procesar señales de voz masculinas y femeninas.
- Aplicar la Transformada de Fourier como herramienta de análisis espectral
de la voz.
- Extraer parámetros característicos de la señal de voz: frecuencia
fundamental, frecuencia media, brillo, intensidad, jitter y shimmer.
- Comparar las diferencias principales entre señales de voz de hombres y
mujeres a partir de su análisis en frecuencia.
Desarrollar conclusiones sobre el comportamiento espectral de la voz
humana en función del género.
## PARTE A

En la Parte A de la práctica se realiza la adquisición y el análisis inicial de señales de voz. Para ello se graban seis muestras de audio correspondientes a tres voces masculinas y tres voces femeninas pronunciando la misma frase. Posteriormente, las señales se importan en Python para ser analizadas en el dominio del tiempo y en el dominio de la frecuencia mediante la Transformada de Fourier. A partir de este análisis se identifican y calculan diferentes características de la señal de voz, como la frecuencia fundamental, la frecuencia media, el brillo y la intensidad, con el fin de estudiar su comportamiento espectral y preparar la comparación entre ambos géneros.<br> 

#### FRASE: 
> ***El arcoiris brilla en el cielo con colores suaves y mucha luz***
 
### Algoritmo 
<img width="375" height="564" alt="image" src="https://github.com/user-attachments/assets/920dcbf8-cd33-4c6f-b6b1-73718d2ea418" />

### Codigo 
```
ruta = r"C:\Users\aleja\Downloads\audios"
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fft import fft, fftfreq
import os
import pandas as pd
archivos = sorted([f for f in os.listdir(ruta) if f.endswith(".wav")])
print("Archivos encontrados:")
for a in archivos:
    print(a)
```
Este código sirve para organizar todo antes de empezar el análisis de los audios. Primero se define la carpeta donde están guardados los archivos `.wav`. Luego se importan las librerías que se van a usar: <br>
- NumPy para cálculos <br>
- Matplotlib para graficar<br>
- SciPy para trabajar con audio <br>
- os para manejar archivos<br>
-  pandas por si se necesitan tablas<br>
Después, el programa busca todos los archivos `.wav` dentro de esa carpeta, los ordena y los guarda en una lista. Por último, imprime los nombres de los archivos encontrados para asegurarse de que sí se están leyendo correctamente. <br>

```
def analizar_audio(nombre_archivo):
    fs, señal = wav.read(os.path.join(ruta, nombre_archivo))
    if len(señal.shape) > 1:
        señal = señal[:, 0]
    señal = señal / np.max(np.abs(señal))
    t = np.arange(len(señal)) / fs
    plt.figure(figsize=(10,4))
    plt.plot(t, señal)
    plt.title(f"Señal en el tiempo - {nombre_archivo}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.show()
    
```
Define una función que se encarga de analizar un archivo de audio. Primero, carga el archivo `.wav` usando su nombre y la ruta definida. Luego, verifica si el audio está en estéreo (dos canales) y, si es así, toma solo uno para trabajar en mono y simplificar el análisis. Después, normaliza la señal dividiéndola por su valor máximo, para que la amplitud quede entre -1 y 1.<br>
A continuación, crea un vector de tiempo usando la frecuencia de muestreo, lo que permite ubicar cada punto de la señal en segundos. Finalmente, grafica la señal en el dominio del tiempo, mostrando cómo cambia la amplitud a lo largo del tiempo, lo cual ayuda a visualizar la forma de la señal de audio.<br>

```
    N = len(señal)
    fft_vals = fft(señal)
    freqs = fftfreq(N, 1/fs)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    magnitud_pos = np.abs(fft_vals[mask])
    plt.figure(figsize=(10,4))
    plt.plot(freqs_pos, magnitud_pos)
    plt.title(f"Espectro de Frecuencia (0 a +Hz) - {nombre_archivo}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.xlim(0, np.max(freqs_pos))
    plt.grid()
    plt.show()
```
Analiza la señal en el dominio de la frecuencia. Primero, calcula la Transformada Rápida de Fourier (FFT) para obtener las componentes de frecuencia de la señal, junto con el vector de frecuencias asociado. Luego, se filtran únicamente las frecuencias positivas, ya que la FFT es simétrica y esta parte es suficiente para el análisis. También se calcula la magnitud de la señal en frecuencia, que indica qué tan fuerte es cada componente. Finalmente, se grafica el espectro de frecuencia, donde el eje x representa la frecuencia en Hz y el eje y la magnitud (intensidad de cada frecuencia), permitiendo identificar qué frecuencias predominan en el audio.<br>

```
    idx_f0 = np.argmax(magnitud_pos)
    f0 = freqs_pos[idx_f0]
    f_media = np.sum(freqs_pos * magnitud_pos) / np.sum(magnitud_pos)
    brillo = f_media
    rms = np.sqrt(np.mean(señal**2))

    return {
        "Archivo": nombre_archivo,
        "F0 (Hz)": f0,
        "Frecuencia media (Hz)": f_media,
        "Brillo (Hz)": brillo,
        "RMS": rms
    }

```
Este fragmento realiza varios cálculos importantes para caracterizar el audio.Primero, encuentra la frecuencia fundamental (F0), que corresponde a la frecuencia con mayor magnitud en el espectro, es decir, la más dominante en la señal. Luego, calcula la frecuencia media o centroide espectral, que representa el “promedio” de las frecuencias presentes y se asocia con el brillo del sonido (qué tan agudo o grave se percibe). Por eso, el brillo se toma igual a este valor. También se calcula la intensidad de la señal mediante el valor RMS, que indica qué tan fuerte es el audio en promedio. Finalmente, todos estos resultados se guardan en un diccionario, junto con el nombre del archivo, para poder organizarlos y usarlos después de forma más fácil.<br>

```
resultados = []

for archivo in archivos:
    res = analizar_audio(archivo)
    resultados.append(res)

df = pd.DataFrame(resultados)

print("\nResultados:")
print(df)

df.to_excel("resultados_parte_A.xlsx", index=False)
```
Este código procesa todos los audios llamando la función de análisis para cada archivo y guardando los resultados en una lista. Luego, convierte esos datos en una tabla con pandas, la muestra en pantalla y la exporta a un archivo de Excel para tener los resultados organizados.<br>

### GRAFICAS
#### SEÑAL EN TIEMPO 
|  AUDIO   |    SEÑAL EN TIEMPO    |  ANALISIS     |
|----------|-----------------------|----------------|
| HOMBRE 1 | <img width="551" height="226" alt="image" src="https://github.com/user-attachments/assets/2ee6898a-e506-4460-adc5-b574974d54b1" /> | Se puede ver que al inicio (cerca de 0 s) la señal es muy pequeña, lo que indica silencio o ruido muy bajo. Luego, a partir de más o menos 0.3 s, empiezan picos grandes y constantes: eso corresponde a la voz hablando. Esos picos altos representan mayor amplitud, o sea, mayor intensidad (cuando la persona habla más fuerte), mientras que las partes más pequeñas son sonidos más suaves o pausas entre palabras. También se nota que la señal es bastante irregular, lo cual es normal en la voz humana porque no es una señal constante sino que cambia dependiendo de las sílabas, el tono y la pronunciación. Además, hacia el final (después de ~3.5 s) la amplitud vuelve a disminuir, lo que indica que la persona deja de hablar o baja la intensidad. <br> |
| HOMBRE 2 |<img width="524" height="226" alt="image" src="https://github.com/user-attachments/assets/8c2236e7-e44a-4d96-8151-be393ba42a97" />|se puede observar que inicia con una amplitud baja, indicando un pequeño silencio antes de empezar a hablar. Luego aparecen picos altos y bien marcados, lo que indica una voz con buena intensidad. A lo largo de la señal se ven variaciones fuertes en la amplitud, lo que refleja cambios en la entonación y en la fuerza al hablar. También se notan varias pausas cortas donde la amplitud disminuye, lo que sugiere separación entre palabras o frases. Hacia la mitad hay algunos picos más altos que en otras partes, indicando momentos de mayor intensidad. Finalmente, al final la señal se vuelve más pequeña y uniforme.|
| HOMBRE 3 |<img width="530" height="227" alt="image" src="https://github.com/user-attachments/assets/cdc2bee9-bf18-4a63-96e6-aa6fabf101c3" />|La gráfica muestra la señal de voz en el dominio del tiempo para aproximadamente 5 segundos. Al inicio se observa un breve silencio (amplitud cercana a cero), seguido de varios segmentos con picos de alta amplitud que corresponden a la pronunciación de palabras. También se identifican pausas donde la señal disminuye, reflejando el ritmo natural del habla. La señal presenta un comportamiento no periódico pero con patrones característicos de la voz humana, y su amplitud está bien normalizada entre -1 y 1, lo que indica que no hay saturación y que la grabación tiene buena calidad.|
| MUJER 1  |<img width="529" height="227" alt="image" src="https://github.com/user-attachments/assets/c22a70b9-e19a-4a51-bb39-9ecd2c86401e" /> |La gráfica muestra la señal de la voz en el tiempo y se puede observar que inicia con un pico bastante alto, lo que indica que la persona comienza hablando con una intensidad fuerte. Luego, la amplitud varía constantemente, reflejando los cambios normales en la voz durante el habla, como la entonación y la pronunciación. Aquí se notan más pausas pequeñas entre palabras, ya que hay momentos donde la amplitud baja significativamente. La señal sigue siendo irregular, como es típico en la voz humana, pero con una intensidad un poco más moderada en varias partes. Finalmente, hacia el final la amplitud disminuye, indicando que la persona deja de hablar o baja el volumen de su voz.|
| MUJER 2  |<img width="529" height="229" alt="image" src="https://github.com/user-attachments/assets/93230747-65b0-4cd3-b829-cfc04e0553e1" />|se observa que inicia con una amplitud baja, indicando un pequeño silencio o entrada suave, pero rápidamente aparecen picos altos que representan el inicio del habla. A lo largo de la señal se evidencian variaciones constantes en la amplitud, lo que refleja cambios en la intensidad y entonación de la voz. Se notan varias zonas con picos pronunciados, lo que indica momentos donde la persona habla con mayor fuerza, y también pequeñas pausas donde la amplitud disminuye. En general, la señal es continua pero con variaciones marcadas, mostrando un habla dinámica.|
| MUJER 3  |<img width="533" height="229" alt="image" src="https://github.com/user-attachments/assets/4b8b31c7-5e8d-4d34-9a5c-698b22de8bc4" /> |aquí casi no hay silencio inicial, ya que desde el comienzo la amplitud es alta, indicando que la persona empieza a hablar de inmediato. A lo largo del tiempo se observan picos bastante grandes y frecuentes, lo que sugiere una voz con buena intensidad y energía. La señal es muy variable e irregular, lo cual es característico del habla, pero además se nota que hay menos pausas largas, es decir, el discurso es más continuo. También se pueden identificar cambios en la amplitud que reflejan variaciones en el tono y la pronunciación. Finalmente, hacia el último tramo la amplitud disminuye progresivamente, indicando que la persona va terminando de hablar.|

#### FFT
|  AUDIO   |    FFT    |  ANALISIS     |
|----------|-----------------------|----------------|
| HOMBRE 1 |<img width="535" height="226" alt="image" src="https://github.com/user-attachments/assets/7ef46a35-dc8b-41d0-b520-4f2eb8aa566a" /> |La gráfica indica que la señal corresponde a una voz masculina, ya que la mayor parte de la energía se concentra en bajas frecuencias y presenta un pico dominante asociado a la frecuencia fundamental (F0), que en hombres suele ser más baja. Esto significa que el tono de la voz es grave. Además, la rápida disminución de la magnitud en altas frecuencias muestra que la señal está bien definida y sin mucho ruido, lo que indica una buena calidad de grabación. En conjunto, el espectro confirma que la información más relevante de la voz se encuentra en bajas frecuencias y permite caracterizar el timbre y tono del hablante. |
| HOMBRE 2 |<img width="515" height="226" alt="image" src="https://github.com/user-attachments/assets/a737231a-bfda-49c8-a478-1f6201773d28" />|La gráfica muestra que la señal presenta una alta concentración de energía en bajas frecuencias, con un pico dominante que corresponde a la frecuencia fundamental (F0), lo cual es característico de una voz masculina y sugiere un tono grave. Esto indica que la mayor parte de la información relevante de la voz se encuentra en esa zona del espectro. A medida que aumenta la frecuencia, la magnitud disminuye notablemente, evidenciando una menor contribución de componentes de alta frecuencia. Además, la ausencia de picos significativos en altas frecuencias sugiere que la señal tiene poco ruido y una buena calidad de grabación. En conjunto, el espectro permite confirmar las características tonales de la voz y su adecuada captura. |
| HOMBRE 3 |<img width="521" height="228" alt="image" src="https://github.com/user-attachments/assets/a863c1cc-f144-48db-becc-0e652e2b5017" />|La gráfica muestra que la energía de la señal se concentra en bajas frecuencias, con un pico dominante que corresponde a la frecuencia fundamental (F0), indicando una voz masculina de tono grave. A medida que aumenta la frecuencia, la magnitud disminuye, evidenciando menor aporte de altas frecuencias. Además, la ausencia de picos en altas frecuencias sugiere buena calidad de grabación y bajo nivel de ruido. |
| MUJER 1 |<img width="1060" height="453" alt="image" src="https://github.com/user-attachments/assets/c293a7df-062f-4c0e-8e3f-691d5a7b3e96" /> |La gráfica muestra el espectro de frecuencia de una señal de voz femenina, donde se observa que la mayor parte de la energía está concentrada en las frecuencias bajas, especialmente por debajo de los 1000 Hz. Los picos más altos en esa zona corresponden a la frecuencia fundamental y a sus primeros armónicos, que son característicos de la voz humana. A medida que aumenta la frecuencia, la magnitud disminuye considerablemente, lo que indica que las componentes de alta frecuencia tienen menor energía y aportan más a detalles finos del sonido que a su intensidad principal. Además, después de aproximadamente los 5000 Hz la señal se vuelve casi despreciable, lo cual es normal en señales de voz, ya que la mayor información relevante se encuentra en bajas y medias frecuencias. En general, este comportamiento confirma que se trata de una señal de voz bien definida, con un contenido espectral típico donde predominan las frecuencias graves y medias |
| MUJER 2 | <img width="1039" height="459" alt="image" src="https://github.com/user-attachments/assets/08b35bb2-c4e9-4588-8b23-728247b532e6" />|El espectro muestra una señal de voz con alta concentración de energía en bajas frecuencias, donde aparecen picos muy pronunciados que indican una frecuencia fundamental clara y armónicos fuertes. Esto sugiere una voz con buena intensidad y definición. A medida que aumenta la frecuencia, la energía disminuye rápidamente, evidenciando que las componentes de alta frecuencia tienen poca influencia. En conjunto, la señal presenta un contenido espectral típico de voz humana, pero con mayor amplitud en los picos principales, lo que puede asociarse a una voz más fuerte o con mayor proyección. |
| MUJER 3 |<img width="1050" height="448" alt="image" src="https://github.com/user-attachments/assets/e8c05bff-7e90-4eab-aa1f-2f703ccf61ca" /> |El espectro muestra una fuerte concentración de energía en bajas frecuencias con picos muy altos, lo que indica una frecuencia fundamental bien definida y armónicos claros. Sin embargo, a diferencia de los anteriores, aquí se observa una mayor presencia de energía en frecuencias medias (alrededor de 5 kHz a 10 kHz), lo que sugiere una señal con más contenido en detalles o mayor riqueza tonal. Aunque la energía disminuye en altas frecuencias, esta señal presenta una distribución más amplia, lo que puede asociarse a una voz con mayor brillo o claridad. |

#### ANALISIS
<img width="700" height="169" alt="image" src="https://github.com/user-attachments/assets/586659ed-6ccf-4a65-8d41-bc492eb7f094" />

|N |          Archivo |    F0 (Hz) | Frecuencia media (Hz) |  Brillo (Hz)    |   RMS|
|----|---------------|-----------|----------------|------------|------------|
| 1 | Hombre1.m4a.wav | 192.360476  |  1904.812069 | 1904.812069 | 0.148978 |
| 2 | Mujer-3.m4a.wav | 226.017201  |  4937.410048  |4937.410048 | 0.240340 |
| 3 |  Mujer1.m4a.wav | 212.619914  |  2319.054167  |2319.054167 | 0.181220 |
| 4 |  Mujer2.m4a.wav | 240.414508  |  3476.629405  |3476.629405 | 0.212984 | 
| 5 | hombre2.m4a.wav | 133.217065  |  2978.085298  | 2978.085298 | 0.183491 |
| 6 | hombre3.m4a.wav | 296.715939  |  1512.656158  | 1512.656158 | 0.177155 |

Primero, al observar la frecuencia fundamental (F0), se nota que las voces femeninas (Mujer1, Mujer2 y Mujer3) presentan valores más altos (entre ~212 Hz y 240 Hz) en comparación con la mayoría de las voces masculinas (por ejemplo, Hombre1 con ~192 Hz y Hombre2 con ~133 Hz). Esto es coherente con la teoría, ya que las cuerdas vocales más cortas y tensas producen frecuencias más altas. Sin embargo, el caso de Hombre3 (~296 Hz) destaca porque tiene una F0 inusualmente alta para una voz masculina, lo que podría indicar una voz aguda, un tono elevado al hablar o incluso alguna variación en la grabación.<br>

En cuanto a la frecuencia media y el brillo, se observa que en todos los casos ambos valores coinciden, lo que indica que el centro de energía del espectro está bien representado por estas medidas. La voz de Mujer3 resalta significativamente con valores cercanos a 4937 Hz, lo que indica una mayor presencia de componentes de alta frecuencia, asociadas a una voz más brillante, clara o con mayor nitidez. Por otro lado, voces como Hombre3 (~1512 Hz) presentan menor brillo, lo que sugiere un tono más opaco o grave.<br>

Finalmente, el RMS permite analizar la energía o intensidad de la señal. Aquí se observa que Mujer3 tiene el valor más alto (~0.24), lo que indica que su señal es más intensa o fuerte en comparación con las demás. En contraste, Hombre1 presenta el valor más bajo (~0.15), lo que sugiere una menor energía en la grabación.<br>

En conjunto, los datos evidencian diferencias claras entre las voces: las femeninas tienden a ser más agudas y brillantes, mientras que las masculinas son más graves y con menor contenido en altas frecuencias. Además, se identifican variaciones individuales importantes que muestran que cada voz tiene características propias, lo cual es clave en el análisis de señales de voz.<br>

## PARTE B

Para esta Parte del laboratorio, es necesario realizar  la Medición de Jitter y Shimmer, el procediemiento fue seleccionar una de las grabaciones realizadas en la Parte A por cada género (una voz de hombre y una de mujer), posteriormente se busco aplicar un filtro pasa-banda en el rango de la voz (80–400 Hz para hombres, 150–500 Hz para mujeres) para eliminar ruido no deseado.
Se escogió al sujeto 1 masculino y femenino para aplicar el filtro pasa banda.


### Algoritmo 
### Codigo 


```
# =========================
# PARTE B - SOLO FILTRO PASA BANDA
# =========================

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import os

# =========================
# FUNCION FILTRO PASA BANDA
# =========================
def filtro_pasabanda(audio, fs, lowcut, highcut, order=4):
    """Aplica filtro Butterworth pasa-banda"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Verificar longitud del audio
    min_length_required = 3 * (order + 1)
    if len(audio) <= min_length_required:
        print(f"Warning: Audio muy corto ({len(audio)}). Reduciendo orden.")
        new_order = max(1, (len(audio) // 3) - 1)
        order = new_order

    if order <= 0:
        print("Error: Audio demasiado corto para filtrar.")
        return np.zeros_like(audio)

    b, a = butter(order, [low, high], btype='band')
    audio_filtrado = filtfilt(b, a, audio)
    return audio_filtrado

# =========================
# CONFIGURACION INICIAL
# =========================
print("="*60)
print("PARTE B - FILTRO PASA BANDA")
print("="*60)

# =========================
# ESPECIFICAR TUS ARCHIVOS
# =========================
print("\nVerificando archivos...")

# Directorio actual
directorio = os.getcwd()
print(f"Directorio: {directorio}")

# Tus archivos especificos
archivo_fem = "Paula_Mujer_2"
archivo_masc = "Ralf_Hombre_1"

# Posibles extensiones
extensiones = ['.wav', '.m4a', '.mp3', '']

# Buscar el archivo femenino
ruta_fem = None
for ext in extensiones:
    prueba = os.path.join(directorio, archivo_fem + ext)
    if os.path.exists(prueba):
        ruta_fem = prueba
        print(f"Encontrado FEMENINO: {os.path.basename(prueba)}")
        break

# Buscar el archivo masculino
ruta_masc = None
for ext in extensiones:
    prueba = os.path.join(directorio, archivo_masc + ext)
    if os.path.exists(prueba):
        ruta_masc = prueba
        print(f"Encontrado MASCULINO: {os.path.basename(prueba)}")
        break

if ruta_fem is None or ruta_masc is None:
    print("\nERROR: No se encontraron los archivos")
    print("\nArchivos en el directorio:")
    for archivo in os.listdir(directorio):
        print(f"   - {archivo}")
    exit()

# =========================
# PROCESAR VOZ FEMENINA (Paula)
# =========================
print("\n" + "="*60)
print("PROCESANDO VOZ FEMENINA - Paula")
print("="*60)

try:
    fs_fem, audio_fem = wavfile.read(ruta_fem)
    print(f"Cargado: {len(audio_fem)} muestras, {fs_fem} Hz")
    
    # Convertir a float32
    audio_fem = audio_fem.astype(np.float32)
    if len(audio_fem.shape) > 1:
        audio_fem = audio_fem[:, 0]
        print("Stereo convertido a Mono")
    
    # Normalizar
    max_val = np.max(np.abs(audio_fem))
    if max_val > 0:
        audio_fem = audio_fem / max_val
    
    # Filtrar (150-500 Hz para femenino)
    print("\nAplicando filtro pasa banda (150-500 Hz)...")
    audio_fem_filtrado = filtro_pasabanda(audio_fem, fs_fem, 150, 500)
    print("Filtro aplicado correctamente")
    
except Exception as e:
    print(f"Error: {e}")

# =========================
# PROCESAR VOZ MASCULINA (Ralf)
# =========================
print("\n" + "="*60)
print("PROCESANDO VOZ MASCULINA - Ralf")
print("="*60)

try:
    fs_masc, audio_masc = wavfile.read(ruta_masc)
    print(f"Cargado: {len(audio_masc)} muestras, {fs_masc} Hz")
    
    # Convertir a float32
    audio_masc = audio_masc.astype(np.float32)
    if len(audio_masc.shape) > 1:
        audio_masc = audio_masc[:, 0]
        print("Stereo convertido a Mono")
    
    # Normalizar
    max_val = np.max(np.abs(audio_masc))
    if max_val > 0:
        audio_masc = audio_masc / max_val
    
    # Filtrar (80-400 Hz para masculino)
    print("\nAplicando filtro pasa banda (80-400 Hz)...")
    audio_masc_filtrado = filtro_pasabanda(audio_masc, fs_masc, 80, 400)
    print("Filtro aplicado correctamente")
    
except Exception as e:
    print(f"Error: {e}")

# =========================
# GRAFICA 1: VOZ FEMENINA
# =========================
print("\nGenerando grafica 1 - Voz femenina...")

plt.figure(figsize=(12, 5))

# Señal original
plt.subplot(1, 2, 1)
muestras_orig = min(5000, len(audio_fem))
tiempo_orig = np.arange(muestras_orig) / fs_fem
plt.plot(tiempo_orig, audio_fem[:muestras_orig], color='gray', linewidth=1)
plt.title('Voz Femenina - Original (Paula)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

# Señal filtrada
plt.subplot(1, 2, 2)
muestras_filt = min(5000, len(audio_fem_filtrado))
tiempo_filt = np.arange(muestras_filt) / fs_fem
plt.plot(tiempo_filt, audio_fem_filtrado[:muestras_filt], color='magenta', linewidth=1.5)
plt.title('Voz Femenina - Filtrada (150-500 Hz)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================
# GRAFICA 2: VOZ MASCULINA
# =========================
print("Generando grafica 2 - Voz masculina...")

plt.figure(figsize=(12, 5))

# Señal original
plt.subplot(1, 2, 1)
muestras_orig = min(5000, len(audio_masc))
tiempo_orig = np.arange(muestras_orig) / fs_masc
plt.plot(tiempo_orig, audio_masc[:muestras_orig], color='gray', linewidth=1)
plt.title('Voz Masculina - Original (Ralf)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

# Señal filtrada
plt.subplot(1, 2, 2)
muestras_filt = min(5000, len(audio_masc_filtrado))
tiempo_filt = np.arange(muestras_filt) / fs_masc
plt.plot(tiempo_filt, audio_masc_filtrado[:muestras_filt], color='blue', linewidth=1.5)
plt.title('Voz Masculina - Filtrada (80-400 Hz)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =========================
# GRAFICA 3: COMPARACION (opcional)
# =========================
print("Generando grafica 3 - Comparacion...")

plt.figure(figsize=(12, 5))

# Ambas señales filtradas
plt.subplot(1, 2, 1)
tiempo_fem = np.arange(min(5000, len(audio_fem_filtrado))) / fs_fem
tiempo_masc = np.arange(min(5000, len(audio_masc_filtrado))) / fs_masc
plt.plot(tiempo_fem, audio_fem_filtrado[:5000], color='magenta', alpha=0.7, label='Femenina')
plt.plot(tiempo_masc, audio_masc_filtrado[:5000], color='blue', alpha=0.7, label='Masculina')
plt.title('Comparacion - Señales Filtradas')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True, alpha=0.3)

# Informacion de filtros
plt.subplot(1, 2, 2)
plt.axis('off')
info_text = (
    f"INFORMACION DE FILTROS\n"
    f"{'='*30}\n\n"
    f"Voz Femenina (Paula):\n"
    f"• Filtro: 150 - 500 Hz\n"
    f"• Muestras: {len(audio_fem)}\n"
    f"• Fs: {fs_fem} Hz\n\n"
    f"Voz Masculina (Ralf):\n"
    f"• Filtro: 80 - 400 Hz\n"
    f"• Muestras: {len(audio_masc)}\n"
    f"• Fs: {fs_masc} Hz"
)
plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', fontfamily='monospace')
plt.title('Resumen de Procesamiento')

plt.tight_layout()
plt.show()

# =========================
# RESUMEN FINAL
# =========================
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

print(f"\nVoz Femenina (Paula):")
print(f"   • Archivo: {os.path.basename(ruta_fem)}")
print(f"   • Frecuencia de muestreo: {fs_fem} Hz")
print(f"   • Muestras originales: {len(audio_fem)}")
print(f"   • Filtro aplicado: 150 - 500 Hz")
print(f"   • Muestras filtradas: {len(audio_fem_filtrado)}")

print(f"\nVoz Masculina (Ralf):")
print(f"   • Archivo: {os.path.basename(ruta_masc)}")
print(f"   • Frecuencia de muestreo: {fs_masc} Hz")
print(f"   • Muestras originales: {len(audio_masc)}")
print(f"   • Filtro aplicado: 80 - 400 Hz")
print(f"   • Muestras filtradas: {len(audio_masc_filtrado)}")

print("\n" + "="*60)
print("PROCESAMIENTO COMPLETADO")
print("="*60)

```

Posterior a la ejecucicon del codigo se obtuvueron las siguentes graficas
<img src= https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/Vmujer.png/>
<img src= https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/Vhombre.png/>
<img src= https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/Vcomparacion.jpeg/>






Para el Shimmer Y Jittler se obtuvo
### Mujer 1 Anita:
<img src=https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/anita.png/>

```
Anita_Mujer_1 (femenino):
  F0=185.8Hz, Jitter=50.544% (Elevado), Shimmer=18.671%
```
### Mujer 2 Paula:
<img src=https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/Pau.png/>

```
Paula_Mujer_2 (femenino):
  F0=196.7Hz, Jitter=32.397% (Elevado), Shimmer=15.582% 
```
### Mujer 3 Lina:
<img src=https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/Lina.png/>
```
Lina_Mujer_3 (femenino):
  F0=159.6Hz, Jitter=47.617% (Elevado), Shimmer=11.331%
```

### Hombre 1 Ralph:
<img src=https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/Ralph.png/>

```
Ralf_Hombre_1 (masculino):
  F0=129.8Hz, Jitter=64.381% (Elevado), Shimmer=83.363% 
```
### Hombre 2 Laboratorista:

<img src=https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/Lab.png/>

```
Laboratorista_Hombre_2 (masculino):
  F0=131.9Hz, Jitter=74.440% 
```
### Hombre 3 Dani:

<img src=https://github.com/Maria-Alejandra-Luque/LABORATORIO3-2026/blob/main/dani.png/>

```
Dani_Hombre_3 (masculino):
  F0=130.8Hz, Jitter=82.150% 
```


## PARTE C 
