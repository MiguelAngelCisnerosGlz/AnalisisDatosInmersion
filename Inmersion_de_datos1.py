"""""Recuerda instalar pip install pandas en la terminal
El .csv esta dentro de la misma carpeta por ella no se necesita mas
Importamos nuestra libreria de pandas para empezarla a ocupar
Si estas en google colab de tiene que montar con drive para que jale
from google.colab import drive
drive.mount('/content/drive')
pip install matplotlib
importamos matplotlib para los graficos y lo instalamos"""

import pandas as pd
import matplotlib.pyplot as plt

#aqui se abre el archivo csv para empezar a trabajar con el
inmuebles = pd.read_csv('inmuebles_bogota.csv')

#visualizemos el data set como estoy trabajando en pycharm utilizo print
#para ver en consola
print(inmuebles.head())
#ahora vemos cuantos registros tiene y cuantas columnas
print(inmuebles.shape)
#ahora solo veremos solo las columnas
print(inmuebles.columns)
#contiene la columna baños y área cambiaremos el nombre por medio de un diccionario
columnas = {'Baños' : 'Banos','Área' : 'Area'}
inmuebles = inmuebles.rename(columns=columnas)

#verifiquemos pero ahora con una muestra de (n) cantidad
print(inmuebles.sample(10))

#veamos la informacion del dataframe como nulos tipo etc
print(inmuebles.info())
#localizar el indice por indice recordemos que le indice empieza en 0
print(inmuebles.iloc[300])
#sacando varios inmuebles con rango
print(inmuebles.iloc[300 : 305])
#tomando un inmueble en la columna valor y el numero 300
print(inmuebles['Valor'] [300])
#con un rango de valor
print(inmuebles['Valor'] [300 : 305])
#vamos a hacer operaciones ya con el dataframe
#Sacando el promedio de la columna area
Area_inmuebles=inmuebles.Area.mean()
print(Area_inmuebles)
#Filtrando cantidad de inmuebles con el nombre del barrio
cantidad_inmuebles_chico = sum((inmuebles.Barrio == 'Chico Reservado'))
#este es la mascara de solo los que coinciden con el barrio chico
inmuebles_chico = (inmuebles.Barrio == 'Chico Reservado')
#lista falsos o verdaderos
print(inmuebles_chico)
#dataframe de puros inmuebles de barrio chico reservado de todos globales
chico_reservado = inmuebles[inmuebles_chico]
print(chico_reservado)
#promedio de chico reservado en la columna area
promedio_chico = chico_reservado.Area.mean()
print(promedio_chico)
#Barrios totales
#value counts conteo de cada un valor de los barrios
barrios_totales = inmuebles.Barrio.value_counts()
print(barrios_totales)
#Veamos que upz estan
UPZ_conteo = inmuebles.UPZ.value_counts()
print(UPZ_conteo)
#vamos a hacer una grafica(barras) de cantidad de inmuebles por barrios
#plot bar carga la grafica para despues plt show mostrarla todas las graficas cargadas antes de que se mande a llamar
grafica_barrios = barrios_totales.head(10).plot.bar(x='Nombre del Barrio',y='Valor')
#Muestra el grafico lo comentare para que no salga la imprimir en pantalla
#plt.show()


#Calcular promedio de area de todos los barrios un grafico
#conteo ,media,valor minimo,valor maximo
promedio_area_barrios = inmuebles.Area.mean()
print(promedio_area_barrios)

#Miramos la cantidad que hay
cantidad_inmuebles_kennedy = sum((inmuebles.Barrio == 'Kennedy'))
#este es la mascara de solo los que coinciden con el barrio Kennedy
inmuebles_kennedy = (inmuebles.Barrio == 'Kennedy')
#dataframe de puros inmuebles de barrio kennedy reservado de todos globales
dataframe_kennedy = inmuebles[inmuebles_kennedy]
#minimo y maximo en area
minimo_area = dataframe_kennedy['Area'].min()
maximo_area = dataframe_kennedy['Area'].max()
print(f'El valor minimo de area de los barrios en Kennedy es : {minimo_area}')
print(f'El valor maximo de area de los barrios de kenedy es: {maximo_area}')

