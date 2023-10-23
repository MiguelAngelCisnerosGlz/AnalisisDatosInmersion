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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score


inmuebles = pd.read_csv('datos_inmueble_clase3.csv')
print(inmuebles.shape)

#Aqui ya se hizo el analisis de la clase anterior segun el criterio que vamos a ocupar
datos_dane = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/datos_dane.csv")

#Cambiando el nombre para entender mejor las variables
dic_dane = {
       'NVCBP4':'CONJUNTO_CERRADO',
       'NVCBP14A':'FABRICAS_CERCA', 'NVCBP14D':'TERMINALES_BUS', 'NVCBP14E':'BARES_DISCO',
       'NVCBP14G':'OSCURO_PELIGROSO', 'NVCBP15A':'RUIDO', 'NVCBP15C':'INSEGURIDAD',
       'NVCBP15F':'BASURA_INADECUADA', 'NVCBP15G':'INVASION','NVCBP16A3':'MOV_ADULTOS_MAYORES',
       'NVCBP16A4':'MOV_NINOS_BEBES',
       'NPCKP17':'OCUPACION','NPCKP18':'CONTRATO','NPCKP23':'SALARIO_MES',
       'NPCKP44A':'DONDE_TRABAJA', 'NPCKPN62A':'DECLARACION_RENTA',
       'NPCKPN62B':'VALOR_DECLARACION', 'NPCKP64A':'PERDIDA_TRABAJO_C19',
       'NPCKP64E':'PERDIDA_INGRESOS_C19',
       'NHCCP3':'TIENE_ESCRITURA', 'NHCCP6':'ANO_COMPRA', 'NHCCP7':'VALOR_COMPRA', 'NHCCP8_1':'HIPOTECA_CRED_BANCO',
       'NHCCP8_2':'OTRO_CRED_BANCO', 'NHCCP8_3':'CRED_FNA', 'NHCCP8_6':'PRESTAMOS_AMIGOS',
       'NHCCP8_7':'CESANTIAS', 'NHCCP8_8':'AHORROS', 'NHCCP8_9':'SUBSIDIOS',
       'NHCCP9':'CUANTO_PAGARIA_MENSUAL', 'NHCCP11':'PLANES_ADQUIRIR_VIVIENDA',
       'NHCCP11A':'MOTIVO_COMPRA', 'NHCCP12':'RAZON_NO_ADQ_VIV', 'NHCCP41':'TIENE_CARRO','NHCCP41A':'CUANTOS_CARROS',
       'NHCCP47A':'TIENE_PERROS', 'NHCCP47B':'TIENE_GATOS', 'NHCLP2A':'VICTIMA_ATRACO', 'NHCLP2B':'VICTIMA_HOMICIDIO',
       'NHCLP2C':'VICTIMA_PERSECUSION',
       'NHCLP2E':'VICTIMA_ACOSO', 'NHCLP4':'COMO_VIVE_ECON', 'NHCLP5':'COMO_NIVEL_VIDA',
       'NHCLP8AB':'REACCION_OPORTUNA_POLICIA', 'NHCLP8AE':'COMO_TRANSPORTE_URBANO', 'NHCLP10':'SON_INGRESOS_SUFICIENTES',
       'NHCLP11':'SE_CONSIDERA_POBRE', 'NHCLP29_1A':'MED_C19_TRABAJO',
       'NHCLP29_1C':'MED_C19_CAMBIO_VIVIENDA', 'NHCLP29_1E':'MED_C19_ENDEUDAMIENTO',
       'NHCLP29_1F':'MED_C19_VENTA_BIENES','NPCHP4':'NIVEL_EDUCATIVO'
       }

datos_dane = datos_dane.rename(columns=dic_dane)
#print(datos_dane.columns)
#print(datos_dane.info())

#agrupar y ver el promedio aqui nos sale mas de 1 por que el otro valor es no segun la documentacion
print(datos_dane.groupby('NOMBRE_ESTRATO')[['CONJUNTO_CERRADO','INSEGURIDAD','TERMINALES_BUS','BARES_DISCO','RUIDO',
                    'OSCURO_PELIGROSO','SALARIO_MES','TIENE_ESCRITURA','PERDIDA_TRABAJO_C19','PERDIDA_INGRESOS_C19',
                    'PLANES_ADQUIRIR_VIVIENDA']].mean().head())
#ahora acomodaremos para que solos ea binario es decir 1 si 2 no
datos = datos_dane[['NOMBRE_ESTRATO','CONJUNTO_CERRADO','INSEGURIDAD','TERMINALES_BUS','BARES_DISCO','RUIDO',
                    'OSCURO_PELIGROSO','SALARIO_MES','TIENE_ESCRITURA','PERDIDA_TRABAJO_C19','PERDIDA_INGRESOS_C19',
                    'PLANES_ADQUIRIR_VIVIENDA']].replace(2,0)
#print(datos)
datos_tratados=datos.groupby('NOMBRE_ESTRATO')[['CONJUNTO_CERRADO','INSEGURIDAD','TERMINALES_BUS','BARES_DISCO','RUIDO',
                    'OSCURO_PELIGROSO','SALARIO_MES','TIENE_ESCRITURA','PERDIDA_TRABAJO_C19','PERDIDA_INGRESOS_C19',
                    'PLANES_ADQUIRIR_VIVIENDA']].mean()
print(datos_tratados)
datos_ml=pd.merge(inmuebles,datos_tratados,left_on='UPZ',right_on='NOMBRE_ESTRATO',how='left')
print(datos_ml.info())

#trae nombre estrato y upz
upz = pd.read_csv("C:/Users/jerry/Documents/pythonProject/Inmersion_de_datos/bases/cod_upz.csv")
#merge para traer el codigo del upz pusimon inner para que jale solo los que coincidan entre los dos bases
datos_ml = pd.merge(datos_ml,upz,left_on='UPZ',right_on='NOMBRE_ESTRATO',how='inner')

print(datos_ml.info())
#########graficas##########
"""
Para ver outlayer
plt.figure(figsize=(10,8))
sns.boxplot(data=datos_ml , y= 'Precio_Millon')
plt.show()
"""
#quitemos outlayer
datos_ml = datos_ml.query('Precio_Millon < 1200 & Precio_Millon > 60')
print(datos_ml)
datos_ml['SALARIO_ANUAL_MI'] = datos_ml['SALARIO_MES'] * 12/1000000
print(datos_ml['SALARIO_ANUAL_MI'])
"""
SIN OUTLAYER
plt.figure(figsize=(10,8))
sns.boxplot(data=datos_ml ,x='SALARIO_ANUAL_MI', y= 'Valor_m2_Millon')
plt.show()

plt.figure(figsize=(10,8))
sns.scatterplot(data=datos_ml ,x='SALARIO_ANUAL_MI', y= 'Valor_m2_Millon')
plt.ylim((0,15))
plt.show()
"""

#mapa de coorelacion
datos_ml2 = datos_ml.select_dtypes(include=['number'])
datos_correlacion = datos_ml2.corr()

print(datos_correlacion)
"""
plt.figure(figsize=(18, 8))
#https://www.tylervigen.com/spurious-correlations
#mascara = np.triu(np.ones_like(datos_ml.corr(), dtype=bool)) mask=mascara,
heatmap = sns.heatmap(datos_ml2.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlacion de las variables', fontdict={'fontsize':18}, pad=16);
plt.show()
"""
#regresion lineal predecir
####Aqui machin learning con sklearn
X = datos_ml[['COD_UPZ_GRUPO']]
y = datos_ml['Precio_Millon']

#split 4 variables X enytrenamiento x prueba y entrenamiento y prueba y ahcer split
#test size la cantidad de datos para la prueba
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=99)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

#instanciar el modelo y predecir
modelo = LinearRegression()
#ajustamos los datos para entrenarlo
modelo.fit(X_train,y_train)
#aplicamos machine learning
y_predict_test = modelo.predict(X_test)
#Aplicamos metricas para ver el modelo
#Error absoluto medio y ER^2
baseline_mae = mean_absolute_error(y_test,y_predict_test)
baseline_r2 = r2_score(y_test,y_predict_test)
#Esta es la variacion que tiene en las predcciones
print(baseline_mae,baseline_r2)

###Como no esta bien entrenado vamos a seguir entrenado al modelo
X = datos_ml[['COD_UPZ_GRUPO','Habitaciones','Banos','CONJUNTO_CERRADO','SALARIO_ANUAL_MI','TIENE_ESCRITURA']]

y = datos_ml["Precio_Millon"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 99)
modelo_1 = LinearRegression()
modelo_1.fit(X_train, y_train)
y_predict_test = modelo_1.predict(X_test)
y_predict_train = modelo_1.predict(X_train)
mae_test = mean_absolute_error(y_test, y_predict_test)
r2_test = r2_score(y_test, y_predict_test)
mae_train = mean_absolute_error(y_train, y_predict_train)
r2_train = r2_score(y_train, y_predict_train)
print('*'*20)
print(mae_test,r2_test)
print(mae_train,r2_train)
#Vamos hacer un test importante pasarlo como data frame cuanot vale n inmubele
#segun las varibles upz,habitaciones,baños,conjutnos cerrado etc

ejemplo1=modelo_1.predict([[816,3,2,1,50,1]])
print(ejemplo1)#434.2112 millones de pesos valdria