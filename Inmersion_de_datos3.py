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

#para lso colores de las graficas
colores_graficas = ["#00FFFF", "#FFA500", "#FF1493", "#8A2BE2", "#800080","#FF0000","#FFFF00","#00FF00","#0000FF","#FF00FF"]

#aqui se abre el archivo csv para empezar a trabajar con el
inmuebles = pd.read_csv('inmuebles_bogota.csv')
#contiene la columna baños y área cambiaremos el nombre por medio de un diccionario
columnas = {'Baños' : 'Banos','Área' : 'Area'}
inmuebles = inmuebles.rename(columns=columnas)


#transformando la columna valor str a int
#La funcion split separa el espacio de lo demas
#ojo inmuebles.valor es un panda series o una lista por eso ponemos str.split
#ponemos expand = True para que lo regrese como dataframe no como series
valor = inmuebles.Valor.str.split(expand = True)
#estamos agregando dos columnas mas
inmuebles['Moneda'] = valor[0]
inmuebles['Precio'] = valor[1]
#en este punto tenemos las dos nuevas columnas pero precio aun tiene '.'
#el cual es un caracter especial quitemoslo para poder convertirlo en entero
#aqui ya eliminamos los puntos recordemos que . es un caracter especial por eso
#ponemos la \\
inmuebles['Precio'] = inmuebles['Precio'].str.replace('\\.','',regex=True)
#Ahora tomaremos solo el precio ya sin punto y el barrio en un data set por eso
#el doble corchete
#sigue siendo objeto precio pasemoslo a float en una nueva columna
inmuebles['Precio_Millon'] = inmuebles.Precio.astype('float')/1000000
#quiremos 0 para que sea ams facil leer
pd.set_option('display.precision',2)
pd.set_option('display.float_format',lambda x : '%.2f' % x)

#aca iniciamos otra sesion
#realizando una nueva columna haciendo lo que nos interesa
inmuebles['Valor_m2_Millon'] = inmuebles['Precio_Millon']/inmuebles['Area']
#ahora agrupamos por barrio y vemos la media
#print(inmuebles.columns)
columnas_numericas = inmuebles.select_dtypes(include=['int', 'float'])
agrupados  = columnas_numericas.groupby(inmuebles['Barrio']).mean()

print(agrupados.head(3))

#esto de arriba esta mal por que es media de media o promedio de promedio
#vamos a sumar todos los inmuebles que pertenecen aun barrio y lo dividimos entre
#todas las superficies que pertenecen a ese mismo barrio y obtendremos el valro especifico por metro2
columnas_numericas['Barrio'] = inmuebles['Barrio']
datos_barrio = columnas_numericas.groupby('Barrio').sum()
print(datos_barrio.head())

datos_barrio['Valor_m2_Barrio'] = datos_barrio['Precio_Millon']/datos_barrio['Area']
print(datos_barrio.head(5))
#guarderemos esto es nuestro dataframe real

m2_barrio = dict(datos_barrio['Valor_m2_Barrio'])


inmuebles['Valor_m2_Barrio'] = inmuebles['Barrio']
#en esta linea con el map cambia el nombre de barrio por si valor en el dic
inmuebles['Valor_m2_Barrio'] = inmuebles['Valor_m2_Barrio'].map(m2_barrio)
print(inmuebles.head(5))
#los barrios que tienen mas inmuebles en venta los top 10
top_barrios = inmuebles['Barrio'].value_counts()[:10].index

datos_barrio.reset_index(inplace=True)
print(datos_barrio.query('Barrio in @top_barrios'))
print(inmuebles.shape)
"""
**********Estas son graficas *************** del top 10 barrios coon inmuebles mas vendidos
plt.figure(figsize=(10,8))
#top barrios con la query hue='x' para que le de mas enfasis a cada barrita
ax=sns.barplot(x='Barrio',y='Valor_m2_Barrio',data = datos_barrio.query('Barrio in @top_barrios'),hue='Barrio',palette=colores_graficas,legend=
               False)
#cada punto que queremos representar
ax.tick_params(axis='x',rotation = 45)
#hagamos otro que sea bloxplot pero con el dataset de inmuebles del top de barrios
plt.figure(figsize=(10,8))
ax=sns.boxplot(x='Barrio',y='Valor_m2_Millon',data = inmuebles.query('Barrio in @top_barrios'),hue='Barrio',palette=colores_graficas,legend=
               False)
ax.tick_params(axis='x',rotation = 45)
#otro con condicion
plt.figure(figsize=(10,8))
ax=sns.boxplot(x='Barrio',y='Valor_m2_Millon',data = inmuebles.query('Barrio in @top_barrios & Valor_m2_Millon < 15'),hue='Barrio',palette=colores_graficas,legend=
               False)
ax.tick_params(axis='x',rotation = 45)
#Uno mas
plt.figure(figsize=(10,8))
ax=sns.boxplot(x='Barrio',y='Area',data = inmuebles.query('Barrio in @top_barrios & Area < 500'),hue='Barrio',palette=colores_graficas,legend=
               False)
ax.tick_params(axis='x',rotation = 45)
#Uno mas
plt.figure(figsize=(10,8))
ax=sns.boxplot(x='Barrio',y='Precio_Millon',data = inmuebles.query('Barrio in @top_barrios & Precio_Millon < 2000'),hue='Barrio',palette=colores_graficas,legend=
               False)
ax.tick_params(axis='x',rotation = 45)

plt.show()"""



#Utilizaremos mas datos para poder hacer un modelado mejor ya que
#Son poco datos para hacer el modelado lo agregaremos  en la linea de arriba
#encoding el codigo en que esta la base de datos
datos_raw = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/Identificación (Capítulo A).csv",
                        sep=';',encoding='latin-1')

print(datos_raw.head(5))
print(datos_raw.shape)
#Vamos a quitar los municipios solo los del distrito ya que esta base trae mas datos de loos municipios aledaños
#Solo nos interesan los del centrod e bogota

datos_raw = datos_raw.loc[datos_raw.MPIO == 11001]
print(datos_raw.shape)

datos_b = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/Datos de la vivenda y su entorno (Capítulo B).csv",sep=';',encoding='latin-1')
datos_c = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/Condiciones habitacionales del hogar (Capítulo C).csv",sep=';',encoding='latin-1')
datos_e = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/Composición del hogar y demografía (Capítulo E).csv",sep=';',encoding='latin-1')
datos_h = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/Educación (Capítulo H).csv",sep=';',encoding='latin-1')
datos_l = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/Percepción sobre las condiciones de vida y el desempeño institucional (Capítulo L).csv",sep=';',encoding='latin-1')
datos_k = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/Fuerza de trabajo (Capítulo K).csv",sep=';',encoding='latin-1')

#combinemos todas las tablas directorio es la que los une y left pa conservar todas las tablas
datos_dane = pd.merge(datos_raw,datos_b,on='DIRECTORIO',how='left')

datos_dane = pd.merge(datos_raw,datos_c,on='DIRECTORIO',how='left')

datos_dane = pd.merge(datos_raw,datos_e,on='DIRECTORIO',how='left')

datos_dane = pd.merge(datos_raw,datos_h,on='DIRECTORIO',how='left')

datos_dane = pd.merge(datos_raw,datos_l,on='DIRECTORIO',how='left')

datos_dane = pd.merge(datos_raw,datos_k,on='DIRECTORIO',how='left')

print(datos_dane.shape)

#mirada y ver variables
#organizar el noteboo 
#importamos inmuebles para su uso posterior ya tratado
inmuebles.to_csv('datos_inmueble_clase3.csv', index=False)