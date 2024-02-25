
"""
                                   MODELAMIENTO SOFTCLIPPER MEDIANTE DEEP LEARNING

                                              Nicolas Marin Ruiz
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

"""
                                                     DATABASE
"""

X = torch.arange(0, 2 * 3.14, 0.01227).view(-1, 1)
X_in = torch.sin(X)
Y_no_noise = 0.4 * torch.tanh(X_in / 0.4)               #SE SINTETIZAN LOS DATOS DE ENTRENAMIENTO
noise = torch.randn(X.size())                           #SE CREA UN TENSOR DE NUMEROS ALEATORIOS
Y = Y_no_noise + 0.002*noise                            #SE APLICA EL TENSOR ALEATORIO A LOS DATOS SINTETIZADOS PARA GENERAR VARIANZA

X_test = torch.arange(0, 2 * 3.14,
                      0.01227).view(-1, 1)              #SE CREAN LOS TENSORES A UTILIZAR
X_in_test = torch.sin(X_test)
Y_test= (0.4*torch.tanh(X_in_test/0.4)+ 0.002*noise)

print("DATOS SINTETIZADOS")
plt.plot(X_in.numpy(), Y.numpy(), "r+", label = "Y")    #SE MUESTRAN LOS DATOS SINTETIZADOS 
plt.xlabel("x")
plt.ylabel("y")
plt.legend("Modelo Deep Learning Regresion No lineal")
plt.show()


# %%

"""
                                         ENSAMBLE DE LAS CAPAS DE DEEP LEARNING
"""

from torch import nn
class NonLinearModel(nn.Module):
    
    def __init__(self, input_size, H1, output_size):    #SE INICIALIZA EL MODELO
        super(NonLinearModel,self).__init__()           #SE INICIALIZAN EL PESO Y EL SESGO PARA W*X+B
        self.linear1 = nn.Linear(input_size, H1)        #SE CARGA EL MODELO DEEP LEARLING W*X+B
        self.act1 = nn.ReLU()                           #SE AÑADE UNA CAPA OCULTA DE ACTIVACION NO LINEAL PARA CUMPLIR CON APROX UNIVERSAL
        self.linear2 = nn.Linear(H1,output_size)        #SE AÑADE UNA CAPA OCULTA LINEAL PARA CUMPLIR CON APROX UNIVERSAL
        
    def forward(self, x):                               #SE DEFINE EL METODO QUE EVALUA EL MODELO W*X+B
        x= self.linear1(x)                              #PRIMERA CAPA
        x= self.act1(x)                      
        yhat=self.linear2(x)                            #SEGUNDA CAPA
        
        return yhat                                     #SE DEVUELVE EL MODELO PROCESADO

model =  NonLinearModel(1, 100, 1)                      #SE HACE USO DE LA CLASE CREADA


# %%

"""
                                             ENTRENAMIENTO DEL MODELO
"""

from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.001)     # SE ESTABLECE EL METODO DE OPTI DECENSO DE GRADIENTE ESTOCASTICA

#Minimos cuadrados
criterion = nn.MSELoss()                                #SE ESTABLECE LA FUNCION DE PERDIDA CON MINIMOS CUADRADOS

epochs = 200                                            #SE ESTABLECE EL NUMERO DE EPOCAS DE ENTRENAMIENTO
LOSS =[]                                                #SE CREA EL VECTOR DONDE SE ALMACENAN LOS VALORES DE LA FUNCION DE PERDIDA
epoch_count = []
test_loss_values =[]
def train(epochs):
    for epoch in range (epochs):                        #SE ITERA SOBRE CADA TENSOR 
        
        Yhat = model(X_in)                              #SE REALIZAN LAS INFERENCIAS GENERALES A PARTIR DEL MODELO (GRAFICA)
        loss_batch = criterion(Yhat,Y)                  #SE EVALUA LA FUNCION DE PERDIDA GENERAL
        LOSS.append(loss_batch.item())                  #SE ALMACENAN LOS VALORES DE FUNCION DE PERDIDA PARA GRAFICAR
        
        for x, y in zip(X_in,Y):                        #SE ITERA SOBRE CADA LOTE PARA GRADIENTE DESCENDENTE
            yhat= model(x)
            loss = criterion(yhat,y)
            optimizer.zero_grad()                       #SE REESTABLECEN LOS VALORES DE LOS TENSORES  CERO PARA EVITAR ACUMULACION
            loss.backward()                             #SE ACTUALIZAN LOS PESOS Y CESGOS MEDIANTE LA GRADIENTE 
            optimizer.step()                            #SE MINIMIZAN LAS PERDIDAS PARA APROXIMAR LOS PESOS Y CESGOS
        Yhat = model(X_in)                              #NO RECUERDO POR QUE SE IMPLEMENTA EL MODELO NUEVAMENTE
        if epoch % 10==0:
            model.eval()
            with torch.inference_mode():
                test_pred = model(X_in_test)
                test_loss = criterion(test_pred,
                                      Y_test)
                epoch_count.append(epoch)
                test_loss_values.append(test_loss.item())
    
        
        plt.subplot(2, 2, 1)
        plt.plot(X_in.numpy(), Y.numpy(), 'r+',
                 label='DATOS SINTETIZADOS')           #SE GRAFICA LA SEÑAL SINTETIZADA
        plt.plot(X_in.numpy(),
                 Yhat.detach().numpy(),
                 label="MODELO DEEP LEARNING")         #SE GRAFICA LOS DATOS PREDECIDOS POR EL MODELO
        plt.title("funcion de transferencia")
        plt.xlabel("Input")
        plt.ylabel("Output")        
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot( Y.numpy(), 'r+',
                 label='DATOS SINTETIZADOS')           #SE GRAFICA LA SEÑAL SINTETIZADA
        plt.plot( Yhat.detach().numpy(),
                 label="MODELO DEEP LEARNING")         #SE GRAFICA LOS DATOS PREDECIDOS POR EL MODELO
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud") 
        plt.title("Waveform")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(LOSS, label='Loss function (train)')
        plt.plot(test_loss_values,
                 label='Loss function (test)')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss function")
        plt.legend()
        plt.grid(True)
        plt.ylim([0,0.02])
        plt.tight_layout()
        plt.show()
        
        
train(epochs)                                          #SE INSTANCIA LA FUNCION DE ENTRENAMIENTO
print("ENTRENAMIENTO LISTO")



# %%

"""
                                          TEST DEL MODELO A PARTIR DE SEÑAL SINUSOIDAL
"""

modelo_cargado = NonLinearModel(1, 100, 1)           
modelo_cargado.load_state_dict(torch.load('D:\\PLUGINCOMPLEMENT\\PYTORCH IA\\MODELO\\SoftClipper_god.pth'))
modelo_cargado.eval()  # Cambia al modo de evaluación (inactiva la aleatoriedad de Dropout y BatchNorm)

t = torch.arange(0, 2 * 3.14, 0.01).view(-1, 1)
X_nuevas = torch.sin(t)


with torch.inference_mode():                           # Pasar la señal sinusoidal de entrada a través del modelo
    salida_modelo_cargado = modelo_cargado(X_nuevas)


plt.figure(figsize=(10, 5))                            # Graficar las señales sinusoidales de entrada y salida del modelo
plt.plot(t.numpy(), X_nuevas.numpy(),
         label='SEÑAL DE ENTRADA')
plt.plot(t.numpy(), salida_modelo_cargado.numpy(),
         label='SALIDA DEL MODELO ')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('SOFTCLIPPER (WAVESHAPING)')
plt.legend()
plt.grid(True)
plt.show()


# %%

"""
                                                 GUARDADO DEL MODELO
"""

ruta_modelo = os.path.join('D:', 'PLUGINCOMPLEMENT',
                           'PYTORCH IA', 'MODELO',
                           'SoftClipper_god.pth')      #SE ESTABLECE LA RUTA DONDE SE GUARDARA EL MODELO

torch.save(model.state_dict(), ruta_modelo)            #SE GUARDA EL MODELO ENTRENADO
print("MODELO GUARDADO")