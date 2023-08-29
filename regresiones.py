# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

regresion = linear_model.LinearRegression()
entrada = [[0.1],[0.15],[0.2]]
datos = [0.00089,0.0008,0.0041]
datos_teo = [0.000507553,0.002041312,0.005153374,0.013023256]
x  = np.linspace(0.1, 0.2)
datos_2 = [0.0012,0.0032,0.0061]
datos_3 = [0.0061,0.0069,0.014]
datos_4 = [0.0092,0.018,0.03]

modelo = regresion.fit(entrada,datos)
predic = modelo.predict(entrada)
prediccion = []

for every in predic:
    prediccion.append(every)

print("b:", modelo.intercept_)
print("m:", modelo.coef_)
print("Max_error:", max_error(datos,prediccion))
print("Mean_absolute_error: ", mean_absolute_error(datos,prediccion))
print("Mean_squared_error:", mean_squared_error(datos,prediccion))
print("Suma de los cuadrados de los residuos:", (mean_squared_error(datos,prediccion)*len(prediccion)))
print("Raiz cuadrada del error cuadratico:", mean_squared_error(datos,prediccion,squared = False))
print("Coeficiente de determinación [R^2]:" , r2_score(datos, prediccion))
def funct(yay):
    return yay*1.68*(10**(-8))/(3.31*(10**(-6)))
def funct_2(yay):
    return yay*1.68*(10**(-8))/(0.823*(10**(-6)))
def funct_3(yay):
    return yay*1.68*(10**(-8))/(0.326*(10**(-6)))
def funct_4(yay):
    return yay*1.68*(10**(-8))/(0.129*(10**(-6)))

ok = funct(x)


plt.xlabel("Longitud [m]")
plt.ylabel("Resistencia [Ohms]")
plt.title("")
#plt.plot(entrada,modelo.predict(entrada))
plt.plot(x, ok, color = "darkmagenta", label = "Curva de valores teoricos AWG = 12 ", linestyle= ":")
plt.scatter(entrada,datos, color = "orchid", label = "Valores experimentales AWG = 12 ")
plt.plot(x, funct_2(x), color = "palevioletred", label = "Curva de valores teoricos AWG = 18 ", linestyle= ":")
plt.scatter(entrada,datos_2, color="crimson", label = "Valores experimentales AWG = 18 ")
plt.plot(x, funct_3(x), color = "steelblue", label = "Curva de valores teoricos AWG = 22 ", linestyle= ":")
plt.scatter(entrada,datos_3, color="turquoise", label = "Valores experimentales AWG = 22 ")
plt.plot(x, funct_4(x), color = "darkseagreen", label = "Curva de valores teoricos AWG = 26 ", linestyle= ":")
plt.scatter(entrada,datos_4, color="lightgreen", label = "Valores experimentales AWG = 26 ")
plt.legend(fontsize = 6)
plt.show()


