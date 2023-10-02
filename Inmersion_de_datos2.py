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
import seaborn as sns

#aqui se abre el archivo csv para empezar a trabajar con el
inmuebles = pd.read_csv('inmuebles_bogota.csv')
#contiene la columna baños y área cambiaremos el nombre por medio de un diccionario
columnas = {'Baños' : 'Banos','Área' : 'Area'}
inmuebles = inmuebles.rename(columns=columnas)

print(inmuebles.columns)
print(inmuebles.info())

#veamos que tipo es el valor ya que aparece como object y vemos que es str(string)
print(type(inmuebles.Valor[0]))

#transformando la columna valor str a int
#La funcion split separa el espacio de lo demas
#ojo inmuebles.valor es un panda series o una lista por eso ponemos str.split
#ponemos expand = True para que lo regrese como dataframe no como series
valor = inmuebles.Valor.str.split(expand = True)
#estamos agregando dos columnas mas
inmuebles['Moneda'] = valor[0]
inmuebles['Precio'] = valor[1]
print(inmuebles.sample(3))
#en este punto tenemos las dos nuevas columnas pero precio aun tiene '.'
#el cual es un caracter especial quitemoslo para poder convertirlo en entero
print(inmuebles.info())
#aqui ya eliminamos los puntos recordemos que . es un caracter especial por eso
#ponemos la \\
inmuebles['Precio'] = inmuebles['Precio'].str.replace('\\.','',regex=True)
#Ahora tomaremos solo el precio ya sin punto y el barrio en un data set por eso
#el doble corchete
print(inmuebles[['Precio','Barrio']])
#sigue siendo objeto precio pasemoslo a float en una nueva columna
inmuebles['Precio_Millon'] = inmuebles.Precio.astype('float')/1000000
print(inmuebles.info())
#veamos la media min y maximo rapido un escaneo rapido
print(inmuebles.describe())
#quiremos 0 para que sea ams facil leer
pd.set_option('display.precision',2)
pd.set_option('display.float_format',lambda x : '%.2f' % x)
print((inmuebles.describe()))

#veamos por que el max es 110 de habitaciones de dos formas
#metodo loc esta mal este dato ya que no puede tener una casa 110 habitaciones
habitaciones=inmuebles.loc[inmuebles.Habitaciones==110]
print(habitaciones)
#este igual el area como tiene 2 es incorrecto error de typeo
Area=inmuebles.loc[inmuebles.Area==2]
print(Area)
#grafica de histograma de pecio millon establece el rango de 50
#es para ver su valor como estan distribuidos
#distribucion de frecuencias
"""inmuebles['Precio_Millon'].plot.hist(bins=50)
la comento para que no salga recordar plt show manda todas las graficas hechas hata donde esta la llamada"""

#graficos con otras herramientas comp matploblib y seaborns
#primero crear un lienzo para poder hacer la garfica
plt.figure(1,figsize=(10,8))
grafica = sns.histplot(data= inmuebles,x='Precio_Millon',kde=True,hue='Tipo')
grafica.set_title('Distribucion de Valores de los inmuebles en Bogota')
plt.xlim((50,1000))
#Esto se hace para guardar a tabla
plt.savefig('C:\\Users\\jerry\\Documents\\pythonProject\\Inmersion_De_datos\\valor_inmuebles.png',format='png')

#vamos a seleccionar 3 tipos de inmuebles y haremos otros grafico de el
plt.figure(2,figsize=(10,6))
plt.xlim((50,1000))
tipos_inmuebles=['Apartamento','Casa','Local']
inmuebles_seleccionados = inmuebles[inmuebles['Tipo'].isin(tipos_inmuebles)]
grafica2= sns.histplot(data=inmuebles_seleccionados,x='Precio_Millon',kde=True,hue='Tipo')

plt.show()
