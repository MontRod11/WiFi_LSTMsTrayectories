#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:26:51 2021

@author: laura

"""

import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
import random
import re
import haversine as hs
from haversine import Unit
from pytorchtools import EarlyStopping


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from torch.utils.data.sampler import Sampler
from prepare_data import separate_data, min_max_norm, min_max_norm_test, coordenates_norm
from prepare_data import coordenates_norm_test, coordenates_denorm
from prepare_OverlapSequences import completeseq_consolape, num_seq_tot


sche = False

# In[2]: Semilla para reproducibilidad
seed = 15
torch.manual_seed(seed)    
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# In[3]: Uso de GPU o CPU (is cuda available?)
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()
print(device)

# In[5.1]: Preprocesado de los datos de entrenamiento

# Apertura del documento que contiene los datos del conjunto de entrenamiento
txt ="conjuntoTrain_numTrayectorias170_Nexus_difGPSfreqcon0_numTrayectoriasTest7_numTrayectoriasValid57_distValidTrain.csv"
coordenadas_train = pd.read_csv("/home/laura/DATASETS/train_dataset/Dataset/Nexus/"+txt,header=None) 


# Extracción del número de trayectorias que conforman el conjunto de entrenamiento para facilitar su guardado y localización para tratar los datos
# Realmente, por el tratamiento previo en MATLAB realmente este 'numTrayect' que indica el nombre es la suma de las trayectorias de entrenamiento y de validación. 
# La separación de ambas se hace mediante MATLAB, y las trayectorias de validación son un conjunto del total que se elige de manera aleatoria cada vez que se crea el conjunto.
numeros_nombre_train = [float(s) for s in re.findall(r'-?\d+\.?\d*', txt)]
numTrayect = int(numeros_nombre_train[0])
# numTrayect = int(''.join(filter(str.isdigit, txt)))
# numTrayect = int(numTrayect/10) # se hace esto porque la función anterior multiplica por 10 al añadir el 0 de 'con0'

# A continuación se extraen los índices de los inicios de las distintas trayectorias. Gracias a esto se puede hacer que las secuencias que se forman de las distintas
# trayectorias no cojan posiciones de trayectorias distintas. Es decir, todas las posiciones de una secuencia solo pertenecen a una misma trayectoria.
# En este dataset las nulas no están puestas a -200, por eso se ven muchísimos 1's en vez de 0's
listado = coordenadas_train[coordenadas_train[0]==0].index.tolist()  # https://www.delftstack.com/es/howto/python-pandas/how-to-get-index-of-all-rows-whose-particular-column-satisfies-given-condition-in-pandas/
listado = np.asarray(listado)

coordenadas_nptrain = np.asarray(coordenadas_train)

# Separación de los datos de entrada de los 'target', datos objetivo. 
trayect_data_train, trayect_labels_train = separate_data(coordenadas_train)

# Indica el número total de posiciones, calculado como la suma de las posiciones de todas las trayectorias
numpositionstrain = trayect_labels_train.shape[0]

# Normalización de los datos: Se devueven los valores máximos y mínimos en la normalización para usarlos posteriormente a la hora
# de normalizar los datos de 'test' y validación. 
trayect_data_train_normalized, minvalue, maxvalue = min_max_norm(trayect_data_train)

# Normalización de las coordenadas (datos objetivo o 'targets'), devolviendo los valores máximos y mínimos para la normalización 
# posterior de los datos de validación y test con respecto a los datos de entrenamiento. Al igual que se normalizan los datos para 
# su entrenamiento y prueba, se deben normalizar las etiquetas que para este caso serán la latitud y la longitud de las coordenadas 
# que se pretenden predecir.
trayect_labels_train_norm = np.zeros(trayect_labels_train.shape)
trayect_labels_train_norm[:,0], trayect_labels_train_norm[:,1], minmaxlat_train, minmaxlon_train = coordenates_norm(trayect_labels_train)

# Conversión de los datos en tensores
trayect_data_train_normalized = torch.tensor(trayect_data_train_normalized)#[:,None,:]
trayect_labels_train_norm = torch.tensor(trayect_labels_train_norm, dtype= torch.float32)#[:,None,:]

# In[5.2]: Preprocesado de los datos de validación

# Apertura del documento que contiene los datos del conjunto de validación
txt_valid = 'conjuntoValid_numTrayectorias170_numTrayectoriasValid57_Nexus_difGPSfreqcon0_numTrayectoriasTest7_distValidTrain.csv'
coordenadas_valid= pd.read_csv("/home/laura/DATASETS/valid_dataset/Dataset/Nexus/"+txt_valid,header=None) 

numeros_nombre_valid = [float(s) for s in re.findall(r'-?\d+\.?\d*', txt_valid)]
numTrayects_valid = int(numeros_nombre_valid[1])

# Estas dos sentencias siguientes no son realmente necesarias ya que no se ha extraido el conjunto de validación del conjunto de entrenamiento sino que es un conjunto propio.
# Lo que hacen realmente es reinicial a '0' la cuenta perteneciente a la columna de 'Index' de un tipo de dato 'Series' de Pandas, ya que cuando se divide el conjunto de entrenamiento
# y se saca el de validación de este, la columna 'Index' contiene el índice que tenía dentro de entrenamiento, a pesar de ser un dato distinto, y para crear correctamente el 
# 'Sampler' que se explica a continuación, se requiere que esta columna comience su cuenta en '0'.
cero_to_len_coordenadas_valid = pd.Series(range(0,len(coordenadas_valid)))
coordenadas_valid = coordenadas_valid.set_index([cero_to_len_coordenadas_valid])

# A continuación se extraen los índices de los inicios de las distintas trayectorias. Gracias a esto se puede hacer que las secuencias que se forman de las distintas
# trayectorias no cojan posiciones de trayectorias distintas. Es decir, todas las posiciones de una secuencia solo pertenecen a una misma trayectoria.
# En este dataset las nulas no están puestas a -200, por eso se ven muchísimos 1's en vez de 0's
listado_valid = coordenadas_valid[coordenadas_valid[0]==0].index.tolist()  # https://www.delftstack.com/es/howto/python-pandas/how-to-get-index-of-all-rows-whose-particular-column-satisfies-given-condition-in-pandas/
listado_valid = np.asarray(listado_valid)

coordenadas_npvalid = np.asarray(coordenadas_valid)

# Separación de los datos de entrada de sus etiquetas correspondientes (o 'target')
trayect_data_valid, trayect_labels_valid = separate_data(coordenadas_valid)
# Indica el número total de posiciones, calculado como la suma de las posiciones de todas las trayectorias
numpositionsvalid= trayect_labels_valid.shape[0]

# Normalización de los datos: se usan los valores máximos y mínimos de los datos de entrenamiento para normalizar los datos.
trayect_data_valid_normalized = min_max_norm_test(trayect_data_valid, minvalue, maxvalue)
    
# Normalización de las coordenadas (datos objetivo o 'targets'), devolviendo los valores máximos y mínimos para la normalización 
# posterior de los datos de validación y test con respecto a los datos de entrenamiento. Al igual que se normalizan los datos para 
# su entrenamiento y prueba, se deben normalizar las etiquetas que para este caso serán la latitud y la longitud de las coordenadas 
# que se pretenden predecir.
trayect_labels_valid_norm = np.zeros(trayect_labels_valid.shape)
trayect_labels_valid_norm[:,0], trayect_labels_valid_norm[:,1] = coordenates_norm_test(trayect_labels_valid,minmaxlat_train, minmaxlon_train)

# Conversión de los datos en tensores
trayect_data_valid_normalized = torch.tensor(trayect_data_valid_normalized)#[:,None,:]
trayect_labels_valid_norm = torch.tensor(trayect_labels_valid_norm, dtype= torch.float32)#[:,None,:]

# In[6]: Creación del conjunto de datos muestreador de trayectorias - Permite posteriormente dividir en secuencias las distintas trayectorias sin unir el final de una con el comienzo de otra

# Esta clase devuelve con el método 'iter' los distintos índices en los que comienza una trayectoria, es decir los distintos valores que componen la variable
# 'listado' creada previamente. Recordar que la variable listado contiene los índices donde se encuentra la posición '0' de cada una de las trayectorias que 
# componen el conjunto de datos. De esta manera, cuando se use el muestreador ('Sampler') este devolverá con 'iter' el valor del índice donde comienza una
# trayectoria y con 'len' la longitud de esa trayectoria en términos de posiciones (o muestras) que la componen.
class indexSampler(Sampler):
    # https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#Custom-Sampler
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#Sampler
    def __init__(self, listado, numpostot):
        self.lista_indices = list(listado)
        self.listado = listado
        self.indice_final = numpostot-1
        
        
    def __iter__(self):
        return iter(self.lista_indices)
    
    def __len__(self,posicion):
        ind = self.lista_indices.index(posicion)
        if ind == len(self.lista_indices) -1:
            return self.indice_final-self.listado[ind]+1
        else:
            return self.listado[ind+1]-self.listado[ind]

# Creación de los distintos muestreadores para los distintos conjuntos de datos.
trainSampler = indexSampler(listado,numpositionstrain)
validSampler = indexSampler(listado_valid,numpositionsvalid)

# In[7]: Creación de los dataset donde la primera dimension de los datos es el número total de trayectorias, la segunda la longitud de la trayectoria y la tercera el número de puntos de acceso.

cols = trayect_data_train_normalized.shape[1] # Representa el número de puntos de acceso que están proporcionando información.
cols_target = trayect_labels_train_norm.shape[1] # Representa las dos columnas asociadas a longitud y latitud de las coordenadas.

# Longitud de la secuencia (número de posiciones que conforman una secuencia y en las que se va a dividir una trayectoria).
sequence_length_train = 16 
sequence_length_valid = 16

# El factor de solape indica que porcentaje de posiciones (empezando por el final) se van a solapar entre secuencias de una misma trayectoria.
# Es decir, si tenemos una trayectoria: 1 2 3 4 5 6 7 8 9 10 11 12, donde la longitud de secuencia es 5, y el factor_solape es 1/3 de la longitud
# de secuencia, esto implica que el solape es: 1/3*5 = 1.66. Como no podemos coger 1.66 posiciones, cogemos el entero: int(1/3*5) = 1.
# De esta manera las secuencias que se crearían son:
#   Secuencia 1: 1 2 3 4 5
#   Secuencia 2: 5 6 7 8 9
#   Secuencia 3: 9 10 11 12
# Donde la última posición de la secuencia anterior se ve solapada con la secuencia siguiente.    
factor_solape = 0.24941425634282657

# solape = int(factor_solape*config.sequence_length)
solape = int(factor_solape*sequence_length_train)

# A continuación se dividen los conjuntos de datos de entrenamiento, validación y test en las distintas secuencias. Para entrenamiento y validación
# da realmente igual que secuencia pertenece a que trayectoria y a que día, mientras que para test se debe dstinguir que trayectorias pertenecen a que
# día para poder hacer un estudio de como se degrada la localización.

  ##################################  TRAIN  #############################################

# La variable numsecuencias_tot_train indica cuanta secuencias de sequence_length posiciones se han creado a partir de las trayectorias que componen
# el conjunto de entrenamiento, mientas que numsecuencias_cadatrayect_train es un 'array' que contiene cuantas secuencias de sequence_length
# posiciones se han creado por cada trayectoria. Esta segunda depede del número de posiciones que componga cada trayectoria. La suma de todos los 
# valones de numsecuencias_cadatrayect_train debe coincidir con numsecuencias_tot_train.
numsecuencias_tot_train, numsecuencias_cadatrayect_train = num_seq_tot(trainSampler, sequence_length_train, trayect_data_train_normalized, solape)

# Creación del array que va a contener todas las secuencias creadas y el array que contendár su objetivo asociado.
trayectoria_ensecuencia_train = torch.empty(size=(numsecuencias_tot_train, sequence_length_train, cols))
targets_ensecuencia_train = torch.empty(size=(numsecuencias_tot_train, sequence_length_train, cols_target))

# Almacenar en los distintos arrays las secuencias y objetivos. El tamaño final será [num_secuencias_totales, longitud_secuencia, num_aps]
# para el 'array' de datos y de [num_secuencias_totales, longitud_secuencia, 2 (latitud y longitud)]
trayectoria_ensecuencia_train, targets_ensecuencia_train = completeseq_consolape(trayectoria_ensecuencia_train, targets_ensecuencia_train, 
                                                                              trainSampler, trayect_data_train_normalized, 
                                                                              sequence_length_train, trayect_labels_train_norm, solape)

  ##################################  VALID  #############################################
  
# La variable numsecuencias_tot_valid indica cuanta secuencias de sequence_length posiciones se han creado a partir de las trayectorias que componen
# el conjunto de validación, mientas que numsecuencias_cadatrayect_valid es un 'array' que contiene cuantas secuencias de sequence_length
# posiciones se han creado por cada trayectoria. Esta segunda depede del número de posiciones que componga cada trayectoria. La suma de todos los 
# valones de numsecuencias_cadatrayect_valid debe coincidir con numsecuencias_tot_valid.
numsecuencias_tot_valid, numsecuencias_cadatrayect_valid = num_seq_tot(validSampler, sequence_length_valid, trayect_data_valid_normalized, solape)

# Creación del array que va a contener todas las secuencias creadas y el array que contendár su objetivo asociado.
trayectoria_ensecuencia_valid = torch.empty(size=(numsecuencias_tot_valid, sequence_length_valid, cols))
targets_ensecuencia_valid = torch.empty(size=(numsecuencias_tot_valid, sequence_length_valid, cols_target))

# Almacenar en los distintos arrays las secuencias y objetivos. El tamaño final será [num_secuencias_totales, longitud_secuencia, num_aps]
# para el 'array' de datos y de [num_secuencias_totales, longitud_secuencia, 2 (latitud y longitud)]
trayectoria_ensecuencia_valid, targets_ensecuencia_valid = completeseq_consolape(trayectoria_ensecuencia_valid, targets_ensecuencia_valid, 
                                                                              validSampler, trayect_data_valid_normalized, 
                                                                              sequence_length_valid, trayect_labels_valid_norm, solape)

  ############################### CREACIÓN DATASET #######################################
  
train_set = TensorDataset(trayectoria_ensecuencia_train, targets_ensecuencia_train)
valid_set = TensorDataset(trayectoria_ensecuencia_valid, targets_ensecuencia_valid)

# In[8]:  Creación de los dataloaders

# Tamaño de 'batch_size' para el conjunto de entrenamiento y de 'tbatch_size' para el conjunto de test y el de validación
batch_size = 47
tbatch_size = 1

# Creación de los 'dataloaders' sobre los que se va a iterar. 
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = False, drop_last=True)  
valid_loader = DataLoader(valid_set, batch_size=tbatch_size, shuffle = False, drop_last=True)  

 ############################### Comprobación funcionamiento #############################
 
for step, (data, targets) in enumerate(train_loader):
  print(data)
  print(data.shape)
  print(targets)
  print(targets.shape)
  break
for step, (data, targets) in enumerate(valid_loader):
  print(data)
  print(data.shape)
  print(targets)
  print(targets.shape)
  break

# In[9]:  Creación de la arquitectura del modelo (WifiLSTM_withLinear = LSTM con Cells y una fully connected; Wifi_fullLSTMCells = LSTM hecha solo con cells)

# Establecimiento del tamaño de entrada de la red = Nº de puntos de acceso vistos. Establecimiento de parámetrros de la red como
# tamaño de salida, número de capas (solo válido si se usa el modelo de WifiLSTM cuya arquitectura usa el módulo nn.LSTM en vez de 
# nn.LSTMCell) y número de nodos de cada capa oculta.
in_size = trayect_data_train_normalized.shape[1] 
out_dim = 2
num_lay = 2
hidd_d = 116


# Esta clase define una arquitectura donde se tienen dos capas de LSTM definidas de manera manual usando LSTMCell. Tras eso se le aplica
# una capa 'fully_connected' para transformar el tamaño de salida de 'hidd_d' a 'out_dim'.
class WifiLSTM_withLinear(nn.Module):
    def __init__(self, i_size, hidden_size, num_classes):
        super(WifiLSTM_withLinear, self).__init__()
        self.input_size = i_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.wifi_lstm_lay1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)#, num_layers=self.num_layers,
                             #   batch_first=True)
        self.wifi_lstm_lay2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.out = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)

    def forward(self, x_in, h0,c0, h1,c1):
        (hid_lay1, c_lay1) = self.wifi_lstm_lay1(x_in, (h0,c0))
        (hid_lay2, c_lay2) = self.wifi_lstm_lay2(hid_lay1,(h1,c1))
        out = self.out(hid_lay2)
        return out, (hid_lay1, c_lay1), (hid_lay2, c_lay2)

# Esta clase define una arquitectura donde se tienen dos capas de LSTM definidas de manera manual usando LSTMCell. En este caso no se 
# aplica una 'fully_conneccted' a la salida sino que se usa otra LSTMCell para reducir la dimensionalidad a la de la salida. 
class Wifi_fullLSTMCells(nn.Module):
    def __init__(self, i_size, hidden_size, num_classes):
        super(Wifi_fullLSTMCells, self).__init__()
        self.input_size = i_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.wifi_lstm_lay1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)#, num_layers=self.num_layers,
                             #   batch_first=True)
        self.wifi_lstm_lay2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.lstm_out = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.num_classes)

    def forward(self, x_in, h0,c0, h1,c1, h2,c2):
        (hid_lay1, c_lay1) = self.wifi_lstm_lay1(x_in, (h0,c0))
        (hid_lay2, c_lay2) = self.wifi_lstm_lay2(hid_lay1,(h1,c1))
        (hid_out, c_out) = self.lstm_out(hid_lay2,(h2,c2))
        return (hid_out,c_out), (hid_lay1, c_lay1), (hid_lay2, c_lay2)


# Esta clase define una arquitectura donde se tienen dos capas de LSTM definidas de manera manual usando LSTMCell. En este caso no se 
# aplica una 'fully_conneccted' a la salida sino que se usa otra LSTMCell para reducir la dimensionalidad a la de la salida. 
class Wifi_fullLSTMCells2Lay(nn.Module):
    def __init__(self, i_size, hidden_size, num_classes):
        super(Wifi_fullLSTMCells2Lay, self).__init__()
        self.input_size = i_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.wifi_lstm_lay1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)#, num_layers=self.num_layers,
                             #   batch_first=True)
        self.wifi_lstm_out = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.num_classes)

    def forward(self, x_in, h0,c0, h1,c1):
        (hid_lay1, c_lay1) = self.wifi_lstm_lay1(x_in, (h0,c0))
        (hid_out, c_out) = self.wifi_lstm_out(hid_lay1,(h1,c1))
        return (hid_out,c_out), (hid_lay1, c_lay1)


# Declaración para el uso de la arquitectura que acaba con una capa 'fully_connected':
# lstm = WifiLSTM_withLinear(in_size,hidd_d, out_dim).to(device)
# Declaración para el uso de la arquitectura que acaba con una capa 'LSTMCell':
if num_lay == 3:
    lstm = Wifi_fullLSTMCells(in_size,hidd_d, out_dim).to(device)
elif num_lay == 2:
    lstm = Wifi_fullLSTMCells2Lay(in_size,hidd_d, out_dim).to(device)
else:
    print('Error al declarar la arquitectura')
    exit()

# Creación del modelo:
model = lstm
print(model)

# Cálculo del número de parámetros entrenables totales del modelo:
pytorch_total_params = sum(p.numel() for p in model.parameters())

# In[10]: Train function with Teacher Forcing and Valid Function for sequences of trayectories

# Se define un factor de aleatoriedad mediante el cuál se pretende usar la técnica 'Teacher Forcing'. Esta técnica consiste en que cuando
# se calcule un número aleatorio y este sea inferior al valor umbral establecido como 'teacher_forcing_ratio' en lugar de pasar como recurrencia
# el estado (salida) anterior, se pasa como recurrencia el valor objetivo ('target').
# Es una especie de 'engaño' a la hora de entrenar la red.
teacher_forcing_ratio = 0.046792982444183215
    
# import efemarai as ef

# Definición de la función de entreamiento de secuencias para una arquitectura de 3 layers:
def train_seq(train_load, net, loss_function, epoch, col, seq_len, b_size, device, optimizer):
    train_loss = []
    for step, (data, targets) in enumerate(train_load):
        c0 = torch.zeros(batch_size, hidd_d).to(device)
        hidden_0 = torch.zeros(batch_size, hidd_d).to(device)
        c1 = torch.zeros(batch_size, hidd_d).to(device)
        hidden_1 = torch.zeros(batch_size, hidd_d).to(device)
        c2 = torch.zeros(batch_size, out_dim).to(device)
        hidden_2 = torch.zeros(batch_size, out_dim).to(device)
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        # with ef.scan():
        output = torch.zeros(targets.shape).to(device)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for i in range(targets.shape[1]):
                (hidden_2,c2), (hidden_0, c0), (hidden_1, c1) = net(data[:,i], hidden_0, c0, hidden_1, c1, hidden_2, c2) 
                output[:,i,:] = hidden_2
                hidden_2 = targets[:,i]
        else:    
            for i in range(targets.shape[1]):
                (hidden_2,c2), (hidden_0, c0), (hidden_1, c1) = net(data[:,i], hidden_0, c0, hidden_1, c1, hidden_2, c2) 
                output[:,i,:] = hidden_2
        loss = loss_function(output, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()
    
        train_loss.append(loss.item())

    return sum(train_loss) / len(train_loss) 

# Definición de la función de validación de secuencias. Nótese que ahora no se aplica el Teacher Forcing para una arquitectura de 3 layers:
def valid_seq(valid_load, seq_len, net, loss_function, device, optimizer, minmaxlat_train, minmaxlon_train, prueba):  # , state):
    valid_loss_per_batch = []
    net.eval()
    contador = 0
    with torch.no_grad():
        for step, (data, targets) in enumerate(valid_load):
            c0 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_0 = torch.zeros(tbatch_size, hidd_d).to(device)
            c1 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_1 = torch.zeros(tbatch_size, hidd_d).to(device)
            c2 = torch.zeros(tbatch_size, out_dim).to(device)
            hidden_2 = torch.zeros(tbatch_size, out_dim).to(device)
            optimizer.zero_grad()
            data, targets = data.to(device), targets.to(device)
            output = torch.zeros(targets.shape).to(device)
            for i in range(targets.shape[1]):
                (hidden_2,c2), (hidden_0, c0), (hidden_1, c1) = net(data[:,i], hidden_0, c0, hidden_1, c1, hidden_2, c2) 
                output[:,i,:] = hidden_2

            loss = loss_function(output, targets)
            # if contador == 12:
            #     check_accuracy_gifs_validTraining(output, targets, minmaxlat_train, minmaxlon_train, prueba, contador)
            valid_loss_per_batch.append(loss.item())
            contador = contador + 1
    return sum(valid_loss_per_batch) / len(valid_loss_per_batch)




#############################################################################################################################################



# Definición de la función de entreamiento de secuencias para una arquitectura de 2lay:
def train_seq2lay(train_load, net, loss_function, epoch, col, seq_len, b_size, device, optimizer):
    train_loss = []
    for step, (data, targets) in enumerate(train_load):
        c0 = torch.zeros(batch_size, hidd_d).to(device)
        hidden_0 = torch.zeros(batch_size, hidd_d).to(device)
        c1 = torch.zeros(batch_size, out_dim).to(device)
        hidden_1 = torch.zeros(batch_size, out_dim).to(device)
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        # with ef.scan():
        output = torch.zeros(targets.shape).to(device)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for i in range(targets.shape[1]):
                (hidden_1, c1), (hidden_0, c0) = net(data[:,i], hidden_0, c0, hidden_1, c1) 
                output[:,i,:] = hidden_1
                hidden_1 = targets[:,i]
        else:    
            for i in range(targets.shape[1]):
                (hidden_1, c1), (hidden_0, c0) = net(data[:,i], hidden_0, c0, hidden_1, c1) 
                output[:,i,:] = hidden_1
        loss = loss_function(output, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()
    
        train_loss.append(loss.item())

    return sum(train_loss) / len(train_loss)  

# Definición de la función de validación de secuencias. Nótese que ahora no se aplica el Teacher Forcing para una arquitectura de 2 lay:
def valid_seq2lay(valid_load, seq_len, net, loss_function, device, optimizer, minmaxlat_train, minmaxlon_train, prueba):  # , state):
    valid_loss_per_batch = []
    net.eval()
    contador = 0
    with torch.no_grad():
        for step, (data, targets) in enumerate(valid_load):
            c0 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_0 = torch.zeros(tbatch_size, hidd_d).to(device)
            c1 = torch.zeros(tbatch_size, out_dim).to(device)
            hidden_1 = torch.zeros(tbatch_size, out_dim).to(device)
            optimizer.zero_grad()
            data, targets = data.to(device), targets.to(device)
            output = torch.zeros(targets.shape).to(device)
            for i in range(targets.shape[1]):
                (hidden_1, c1), (hidden_0, c0) = net(data[:,i], hidden_0, c0, hidden_1, c1) 
                output[:,i,:] = hidden_1

            loss = loss_function(output, targets)
            # if contador == 12:
            #     check_accuracy_gifs_validTraining(output, targets, minmaxlat_train, minmaxlon_train, prueba, contador)
            valid_loss_per_batch.append(loss.item())
            contador = contador + 1
    return sum(valid_loss_per_batch) / len(valid_loss_per_batch)


# In[11]:  Entrenamiento y validación

# Definición de párametros para el optimizador, la función de pérdidas y el entrenamiento:
lr = 0.005637669307211288
momentum =  0.9699 # NOT USED
num_epochs = 200
loss_func = nn.MSELoss() 


optimizador = 1 # config.optimizer # 1 # 1 = adam's optimizer, 2 = SGD optimizer
if optimizador == 1:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Using Adam's optimizer")
elif optimizador == 2:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)#config.momentum)
    print("Using SGD optimizer")
else:
    print("Number of optimizer must be either 0 or 1")
    
valid_loss_min = np.Inf
training_losses,  validation_losses = [], []
loss_per_epoch_valid = np.Inf



# Definición del 'scheduler' para cambiar de valor el 'learning_rate' tras un número determinado de epochs.
if sche == True:
    stepsize = int(1/4*num_epochs)#50
    gamma = 0.2609678219930688	# config.gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.2)

patience = 20

if sche == True:
    if optimizador ==1:
        dirpath = 'models/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)+'.pt'
    elif optimizador ==2:
        dirpath = 'models/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_momentum'+str(momentum)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)+'.pt'
else:
    if optimizador ==1:
        dirpath = 'models/no_scheduler/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'NoScheduler.pt'
    elif optimizador ==2:
        dirpath = 'models/no_scheduler/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_momentum'+str(momentum)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'NoScheduler.pt'


h_state = torch.zeros(num_lay, batch_size, hidd_d, dtype=torch.float).to(device)
early_stopping = EarlyStopping(patience=patience, verbose=True,delta=0.00000000000000001,path=dirpath)

# Entrenamiento como tal para un número determinado de épocas. La validación se hace cada 3 epocas:
    
if num_lay == 3:
    for epochs in range(1, num_epochs + 1):
        print("Wifi LSTM training, epoch " + str(epochs))
        model.train()
    
        loss_per_epoch_train = train_seq(train_loader, model, loss_func, epochs, in_size, sequence_length_train, batch_size, device, optimizer)
        training_losses.append(loss_per_epoch_train)
        if epochs % 3 == 0:
            loss_per_epoch_valid = valid_seq(valid_loader, sequence_length_valid, model, loss_func, device, optimizer, minmaxlat_train, minmaxlon_train, epochs)
            validation_losses.append(loss_per_epoch_valid)
            if loss_per_epoch_valid <= valid_loss_min:
                torch.save(model.state_dict(), dirpath)
                valid_loss_min = loss_per_epoch_valid

        early_stopping(valid_loss_min, model)
        
        if early_stopping.early_stop:
             print("Early stopping")
             break
        if sche == True:
            scheduler.step()
        print("Minimum validation loss for epoch " + str(epochs) + " is " + str(valid_loss_min))
        print("Training loss for epoch is " + str(loss_per_epoch_train))
elif num_lay == 2:
    for epochs in range(1, num_epochs + 1):
        print("Wifi LSTM training, epoch " + str(epochs))
        model.train()
    
        loss_per_epoch_train = train_seq2lay(train_loader, model, loss_func, epochs, in_size, sequence_length_train, batch_size, device, optimizer)
        training_losses.append(loss_per_epoch_train)
        if epochs % 3 == 0:
            loss_per_epoch_valid = valid_seq2lay(valid_loader, sequence_length_valid, model, loss_func, device, optimizer, minmaxlat_train, minmaxlon_train, epochs)
            validation_losses.append(loss_per_epoch_valid)
            if loss_per_epoch_valid <= valid_loss_min:
                torch.save(model.state_dict(), dirpath)
                valid_loss_min = loss_per_epoch_valid
        if sche == True:
            scheduler.step()
        early_stopping(valid_loss_min, model)
        
        if early_stopping.early_stop:
             print("Early stopping")
             break
        print("Minimum validation loss for epoch " + str(epochs) + " is " + str(valid_loss_min))
        print("Training loss for epoch is " + str(loss_per_epoch_train))
else:
    print('Error al entrenar')
    exit()

# Representación de lasé prdidas de valóidacin y de entrenamiento
plt.figure()
plt.plot(training_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.legend(frameon=False)
if sche == True:
    plt.savefig('models/train_valid_graph/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)+'.png',format='png', dpi=600)
else:
    plt.savefig('models/train_valid_graph/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'NoScheduler.png',format='png', dpi=600)    


""" Añadir los ficheros creados para indicar los días a los que pertenecen las distintas secuencias / trayectorias"""

numtrayectsTrain_txt = "numTrayectoriasTrain170_Nexus_numTrayectoriasTest7_numTrayectoriasValid57_numerodetrayectoriaspordia_test_distValidTrain.csv"
numtrayectsValid_txt = "numTrayectoriasValid57_Nexus_numTrayectoriasTrain170_numTrayectoriasTest7_numerodetrayectoriaspordia_test_distValidTrain.csv"

archivo_trayectoriasxdia_train = pd.read_csv("/home/laura/DATASETS/train_dataset/Dataset/Nexus/"+numtrayectsTrain_txt,header=None) 
archivo_trayectoriasxdia_train = np.asarray(archivo_trayectoriasxdia_train).reshape(-1)

num_dias_que_se_midieron_trayectorias_train = len(archivo_trayectoriasxdia_train)

lista_dias_train = []

if sum(archivo_trayectoriasxdia_train) == len(numsecuencias_cadatrayect_train):
    for dia in range(num_dias_que_se_midieron_trayectorias_train):
        trayectorias_xdia = archivo_trayectoriasxdia_train[dia]
        secuencias_xdia = sum(numsecuencias_cadatrayect_train[np.sum(archivo_trayectoriasxdia_train[0:dia]):np.sum(archivo_trayectoriasxdia_train[0:dia]) +trayectorias_xdia])
        print(np.sum(archivo_trayectoriasxdia_train[0:dia]))
        print(np.sum(archivo_trayectoriasxdia_train[0:dia]) +trayectorias_xdia)
        print(numsecuencias_cadatrayect_train[np.sum(archivo_trayectoriasxdia_train[0:dia]):np.sum(archivo_trayectoriasxdia_train[0:dia]) +trayectorias_xdia])
        for j in range(secuencias_xdia):
            lista_dias_train.append("Trayectoria del día "+str(dia))
else:
    print('Error con el numero de secuencias de Train')
    exit()
    
archivo_trayectoriasxdia_valid = pd.read_csv("/home/laura/DATASETS/valid_dataset/Dataset/Nexus/"+numtrayectsValid_txt,header=None) 
archivo_trayectoriasxdia_valid = np.asarray(archivo_trayectoriasxdia_valid).reshape(-1)

num_dias_que_se_midieron_trayectorias_valid = len(archivo_trayectoriasxdia_valid)

lista_dias_valid = []

if sum(archivo_trayectoriasxdia_valid) == len(numsecuencias_cadatrayect_valid):
    for dia in range(num_dias_que_se_midieron_trayectorias_valid):
        trayectorias_xdia = archivo_trayectoriasxdia_valid[dia]
        secuencias_xdia = sum(numsecuencias_cadatrayect_valid[np.sum(archivo_trayectoriasxdia_valid[0:dia]):np.sum(archivo_trayectoriasxdia_valid[0:dia]) +trayectorias_xdia])
        print(np.sum(archivo_trayectoriasxdia_valid[0:dia]))
        print(np.sum(archivo_trayectoriasxdia_valid[0:dia]) +trayectorias_xdia)
        print(numsecuencias_cadatrayect_valid[np.sum(archivo_trayectoriasxdia_valid[0:dia]):np.sum(archivo_trayectoriasxdia_valid[0:dia]) +trayectorias_xdia])
        for j in range(secuencias_xdia):
            lista_dias_valid.append("Trayectoria del día "+str(dia))
else:
    print('Error con el numero de secuencias de Validación')
    exit()    

# In[13]: Definición de check_accuracy_imgs_seq():
    
import os
from numpy import savetxt
    
# Definición de la función que comprueba la precisión mediante una representación gráfica de las posiciones en el mapa. Además permite la
# exportación de los datos para un posterior cálculo en MATLAB del error medio.
def check_accuracy_imgs_seq(loader, net, dataset, name, loss_function, h_state, seq_len, col, test_batch_size, device, minmax, propiedades, nameprop, listado_titulo): #, num_classes, cols):
    if dataset == 'valid':
        print("Checking accuracy on validation data")
    else:
        print("Checking accuracy on train data")

    # Creación de la carpeta donde se van a almacenar los datos.
    namedir = '../results/'+dataset+'/'+name+'/'+nameprop+'/export'
    namedir_imgs = '../results/'+dataset+'/'+name
    os.makedirs(namedir, exist_ok=True) 
    os.makedirs(namedir_imgs, exist_ok=True) 
    namedir_props = namedir_imgs+'/'+nameprop
    os.makedirs(namedir_props, exist_ok=True) 
    # Almacenamiento de las propiedades tanto del conjunto de datos como de los valores usados para entrenar y que estos sean replicables.
    f = open(namedir_props+'/Propiedades.txt','wt')
    f.write(propiedades)
    f.close()
    
    pred_coordenatestot, coordenatestot = [], []
    net.eval()
    minmaxlat_train = minmax[0]
    minmaxlon_train = minmax[1]

    with torch.no_grad():
        test_loss = []
        test_losses = []
        total = []
        for step, (data, target) in enumerate(loader):
            c0 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_0 = torch.zeros(tbatch_size, hidd_d).to(device)
            c1 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_1 = torch.zeros(tbatch_size, hidd_d).to(device)
            c2 = torch.zeros(tbatch_size, out_dim).to(device)
            hidden_2 = torch.zeros(tbatch_size, out_dim).to(device)         
            data = data.to(device=device)
            target = target.to(device=device)
            pred_coordenates, coordenates = [], []
                       
            # Paso de las secuencias de prueba por la red una vez ya entrenada:
            scores = torch.zeros(target.shape).to(device)
            for i in range(target.shape[1]):
                (hidden_2,c2), (hidden_0, c0), (hidden_1, c1) = net(data[:,i], hidden_0, c0, hidden_1, c1, hidden_2, c2) 
                scores[:,i,:] = hidden_2
            # scores, _ = net(data)
            loss = loss_function(scores, target)
            test_loss.append(loss.item())
            
            scores = scores.reshape(-1,2)
            target = target.reshape(-1,2)
            
            # Desnormalización de los datos para su representación y para su exportación para calcular el error medio en distancia:
                
            lat_denorm, lon_denorm = coordenates_denorm(target, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            coord_denorm = np.zeros((target.shape[0],2))
            coord_denorm[:,0] = lat_denorm[:].cpu().numpy()  # lat_denorm[:].numpy() # if cuda is not available = without .cpu()
            coord_denorm[:,1] = lon_denorm[:].cpu().numpy()  # lon_denorm[:].numpy() # if cuda is not available = without .cpu()

            predlat_denorm, predlon_denorm = coordenates_denorm(scores, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            predcoord_denorm = np.zeros((scores.shape[0],2))
            predcoord_denorm[:,0] = predlat_denorm[:].cpu().numpy() # predlat_denorm[:].numpy() # if cuda is not available = without .cpu()
            predcoord_denorm[:,1] = predlon_denorm[:].cpu().numpy() # predlon_denorm[:].numpy() # if cuda is not available = without .cpu()

            # Exportación de las coordenadas objetivo y de las predichas para el cálculo del error en distancia entre unas y otras:
            savetxt(namedir+'/coordscores_step'+str(step)+'.csv',coord_denorm, delimiter=',')
            savetxt(namedir+'/coordtarget_step'+str(step)+'.csv',predcoord_denorm, delimiter=',') 

            distancia_coords = np.zeros((scores.shape[0]))
            for i in range(0,len(scores)):
                # distancia_coords[i] = hs.haversine(scores[i,:],target[i,:],unit=Unit.METERS)
                  distancia_coords[i] = hs.haversine(coord_denorm[i,:],predcoord_denorm[i,:],unit=Unit.METERS)
                  total.append(distancia_coords[i])
            error_medio = sum(distancia_coords)/len(distancia_coords)
            max_error = max(distancia_coords)
            min_error = min(distancia_coords)
            mse = sum(distancia_coords**2)/len(distancia_coords)
            Info = 'Para la secuencia '+str(step)+' de la trayectoria del día del conjunto de '+dataset+', el error entre cada una de las posiciones es:\n'+str(distancia_coords)+' metros.'
            Info = Info + '\nEl error medio es de: '+str(error_medio)+' metros. \nEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n El error cuadrático medio es de: '+str(mse)+' metros.'
            
            f = open(namedir+'/informacion_error_secuencia'+str(step)+'.txt','wt')
            f.write(Info)
            f.close()
            print('coord_target = {}  -> predicted coord = {},'.format(coord_denorm, predcoord_denorm))
            pred_coordenates.append(predcoord_denorm)
            coordenates.append(coord_denorm)
            pred_coordenatestot.append(predcoord_denorm)
            coordenatestot.append(coord_denorm)
                
            print(step)
            coordenadas_obj = np.asarray(coordenates)
            coordenadas_obj = coordenadas_obj.reshape(coordenadas_obj.shape[1],-1)
            coordenadas_pred = np.asarray(pred_coordenates)
            coordenadas_pred = coordenadas_pred.reshape(coordenadas_pred.shape[1],-1) 
            
            
            # # Representación en el mapa de las coordenadas objetivo y las predichas, con lineas uniendo puntos:
            # BBox = (-3.15210, -3.14706, 40.64149, 40.64465)      
            # plt.figure()
            # map_image = plt.imread('../imgs/map_GuadaAfueras2.png')
            # fig, ax = plt.subplots(figsize = (8,7))
            # ax.scatter(coordenadas_obj[:,1],coordenadas_obj[:,0], zorder=1, alpha= 0.6, c='b', s=10)
            # ax.scatter(coordenadas_pred[:,1],coordenadas_pred[:,0], zorder=1, alpha= 0.7, c='r', s=10)  
            
            # ax.set_title(listado_titulo[step])
            # ax.set_xlim(BBox[0],BBox[1]) # EJE DE LONGITUD
            # ax.set_ylim(BBox[2],BBox[3]) # EJE DE LATITUD
            # plt.annotate("Inicio", (coordenadas_obj[0,1],coordenadas_obj[0,0]))
            # plt.plot([coordenadas_obj[:,1],coordenadas_pred[:,1]],[coordenadas_obj[:,0],coordenadas_pred[:,0]],'-g',linewidth=0.5)
            # ax.imshow(map_image, zorder=0, extent = BBox, aspect= 'equal')
            # plt.savefig(namedir_export_test+'/Pruebamap_tray'+str(step)+'_predGuadaAfuerasTrain.png',format='png', dpi=600)

        test_losses.append(sum(test_loss) / len(test_loss))
        total = np.array(total)
        error_medio = sum(total)/len(total)
        max_error = max(total)
        min_error = min(total)
        mse = sum(total**2)/len(total)
        Info = 'Cogiendo todas las secuencias que salen de las trayectorias tomadas para el conjunto de '+dataset+', se obtiene:'
        Info = Info + '\n\tEl error medio es de: '+str(error_medio)+' metros. \n\tEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n\tEl error cuadrático medio es de: '+str(mse)+' metros.'
        
        f = open(namedir+'/informacion_error_total_secuencias.txt','wt')
        f.write(Info)
        f.close()   
        return pred_coordenatestot, coordenatestot



#############################################################################################################################################################




# Definición de la función que comprueba la precisión mediante una representación gráfica de las posiciones en el mapa. Además permite la
# exportación de los datos para un posterior cálculo en MATLAB del error medio, en este caso para la arquitectura de 2 layers:
def check_accuracy_imgs_seq2lay(loader, net, dataset, name, loss_function, h_state, seq_len, col, test_batch_size, device, minmax, propiedades, nameprop, listado_titulo): #, num_classes, cols):
    if dataset == 'valid':
        print("Checking accuracy on validation data")
    else:
        print("Checking accuracy on train data")

    # Creación de la carpeta donde se van a almacenar los datos.
    namedir = '../results/'+dataset+'/'+name+'/'+nameprop+'/export'
    namedir_imgs = '../results/'+dataset+'/'+name
    os.makedirs(namedir, exist_ok=True) 
    os.makedirs(namedir_imgs, exist_ok=True) 
    namedir_props = namedir_imgs+'/'+nameprop
    os.makedirs(namedir_props, exist_ok=True) 
    # Almacenamiento de las propiedades tanto del conjunto de datos como de los valores usados para entrenar y que estos sean replicables.
    f = open(namedir_props+'/Propiedades.txt','wt')
    f.write(propiedades)
    f.close()
    
    pred_coordenatestot, coordenatestot = [], []
    net.eval()
    minmaxlat_train = minmax[0]
    minmaxlon_train = minmax[1]

    with torch.no_grad():
        test_loss = []
        test_losses = []
        total = []
        for step, (data, target) in enumerate(loader):
            c0 = torch.zeros(tbatch_size, hidd_d).to(device)
            hidden_0 = torch.zeros(tbatch_size, hidd_d).to(device)
            c1 = torch.zeros(tbatch_size, out_dim).to(device)
            hidden_1 = torch.zeros(tbatch_size, out_dim).to(device)
            data = data.to(device=device)
            target = target.to(device=device)
            pred_coordenates, coordenates = [], []
                       
            # Paso de las secuencias de prueba por la red una vez ya entrenada:
            scores = torch.zeros(target.shape).to(device)
            for i in range(target.shape[1]):
                (hidden_1, c1), (hidden_0, c0) = net(data[:,i], hidden_0, c0, hidden_1, c1) 
                scores[:,i,:] = hidden_1
            # scores, _ = net(data)
            loss = loss_function(scores, target)
            test_loss.append(loss.item())
            
            scores = scores.reshape(-1,2)
            target = target.reshape(-1,2)
            
            # Desnormalización de los datos para su representación y para su exportación para calcular el error medio en distancia:
                
            lat_denorm, lon_denorm = coordenates_denorm(target, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            coord_denorm = np.zeros((target.shape[0],2))
            coord_denorm[:,0] = lat_denorm[:].cpu().numpy()  # lat_denorm[:].numpy() # if cuda is not available = without .cpu()
            coord_denorm[:,1] = lon_denorm[:].cpu().numpy()  # lon_denorm[:].numpy() # if cuda is not available = without .cpu()

            predlat_denorm, predlon_denorm = coordenates_denorm(scores, minmaxlat_train, minmaxlon_train) # donde ~ se debe pasar la posición predicha
            predcoord_denorm = np.zeros((scores.shape[0],2))
            predcoord_denorm[:,0] = predlat_denorm[:].cpu().numpy() # predlat_denorm[:].numpy() # if cuda is not available = without .cpu()
            predcoord_denorm[:,1] = predlon_denorm[:].cpu().numpy() # predlon_denorm[:].numpy() # if cuda is not available = without .cpu()

            # Exportación de las coordenadas objetivo y de las predichas para el cálculo del error en distancia entre unas y otras:
            savetxt(namedir+'/coordscores_step'+str(step)+'.csv',coord_denorm, delimiter=',')
            savetxt(namedir+'/coordtarget_step'+str(step)+'.csv',predcoord_denorm, delimiter=',') 

            distancia_coords = np.zeros((scores.shape[0]))
            for i in range(0,len(scores)):
                # distancia_coords[i] = hs.haversine(scores[i,:],target[i,:],unit=Unit.METERS)
                  distancia_coords[i] = hs.haversine(coord_denorm[i,:],predcoord_denorm[i,:],unit=Unit.METERS)
                  total.append(distancia_coords[i])
            error_medio = sum(distancia_coords)/len(distancia_coords)
            max_error = max(distancia_coords)
            min_error = min(distancia_coords)
            mse = sum(distancia_coords**2)/len(distancia_coords)
            Info = 'Para la secuencia '+str(step)+' de la trayectoria del día del conjunto de '+dataset+', el error entre cada una de las posiciones es:\n'+str(distancia_coords)+' metros.'
            Info = Info + '\nEl error medio es de: '+str(error_medio)+' metros. \nEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n El error cuadrático medio es de: '+str(mse)+' metros.'
            
            f = open(namedir+'/informacion_error_secuencia'+str(step)+'.txt','wt')
            f.write(Info)
            f.close()
            
            print('coord_target = {}  -> predicted coord = {},'.format(coord_denorm, predcoord_denorm))
            pred_coordenates.append(predcoord_denorm)
            coordenates.append(coord_denorm)
            pred_coordenatestot.append(predcoord_denorm)
            coordenatestot.append(coord_denorm)
                
            print(step)
            coordenadas_obj = np.asarray(coordenates)
            coordenadas_obj = coordenadas_obj.reshape(coordenadas_obj.shape[1],-1)
            coordenadas_pred = np.asarray(pred_coordenates)
            coordenadas_pred = coordenadas_pred.reshape(coordenadas_pred.shape[1],-1) 
            
            
            # # Representación en el mapa de las coordenadas objetivo y las predichas, con lineas uniendo puntos:
            # BBox = (-3.15210, -3.14706, 40.64149, 40.64465)      
            # plt.figure()
            # map_image = plt.imread('../imgs/map_GuadaAfueras2.png')
            # fig, ax = plt.subplots(figsize = (8,7))
            # ax.scatter(coordenadas_obj[:,1],coordenadas_obj[:,0], zorder=1, alpha= 0.6, c='b', s=10)
            # ax.scatter(coordenadas_pred[:,1],coordenadas_pred[:,0], zorder=1, alpha= 0.7, c='r', s=10)  
            
            # ax.set_title(listado_titulo[step])
            # ax.set_xlim(BBox[0],BBox[1]) # EJE DE LONGITUD
            # ax.set_ylim(BBox[2],BBox[3]) # EJE DE LATITUD
            # plt.annotate("Inicio", (coordenadas_obj[0,1],coordenadas_obj[0,0]))
            # plt.plot([coordenadas_obj[:,1],coordenadas_pred[:,1]],[coordenadas_obj[:,0],coordenadas_pred[:,0]],'-g',linewidth=0.5)
            # ax.imshow(map_image, zorder=0, extent = BBox, aspect= 'equal')
            # plt.savefig(namedir_export_test+'/Pruebamap_tray'+str(step)+'_predGuadaAfuerasTrain.png',format='png', dpi=600)

        test_losses.append(sum(test_loss) / len(test_loss))
        total = np.array(total)
        error_medio = sum(total)/len(total)
        max_error = max(total)
        min_error = min(total)
        mse = sum(total**2)/len(total)
        Info = 'Cogiendo todas las secuencias que salen de las trayectorias tomadas para el conjunto de '+dataset+', se obtiene:'
        Info = Info + '\n\tEl error medio es de: '+str(error_medio)+' metros. \n\tEl error máximo es de: '+str(max_error)+' metros y el mínimo es de: '+str(min_error)+' metros.\n\tEl error cuadrático medio es de: '+str(mse)+' metros.'
        
        f = open(namedir+'/informacion_error_total_secuencias.txt','wt')
        f.write(Info)
        f.close()   
        return pred_coordenatestot, coordenatestot




# In[14]:  Pruebas de test: se debe pasar batch_size uno y no representa r en la misma todas las salidas de batches sino representar una a una las 
# salidas que se van obteniendo del modelo.

propiedades = "Para entrenar este modelo se ha usado una longitud de secuencia de entrenamiento de "+str(sequence_length_train)+" y una longitud de secuencia de validacion de "+str(sequence_length_valid)+"."
propiedades = propiedades+'\nSe emplea un factor de solape de '+str(factor_solape)+' de la longitud de secuencia, de manera que hay '+str(solape)+' posiciones solapadas, un tamaño de batch_size de entrenamiento de '+str(batch_size)+' y uno de test de '+str(tbatch_size)+'. El modelo consta de '+str(num_lay)+' capas LSTM y no fully connected y de '+str(hidd_d)+' neuronas ocultas.'

propiedades = propiedades+'\nSe usa un learning rate de '+str(lr)+' y se entrena durante '+str(num_epochs)+' empleando la MSELoss y el optimizador '+str(optimizer)
if optimizer == 1:
    propiedades = propiedades+'.'
elif propiedades == 2:
    propiedades = propiedades+' con momento de '+str(momentum)+'.'
propiedades = propiedades+'\nEl modelo es el siguiente: \n\t '+str(model)+'\nY se usa TeacherForcing de '+str(teacher_forcing_ratio)+'.'

# new dataset: 
if sche == True:
    if optimizador == 1:
        model.load_state_dict(torch.load('models/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)+'.pt'))
    elif optimizador == 2:
        model.load_state_dict(torch.load('models/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_momentum'+str(momentum)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)+'.pt'))
else:
    if optimizador == 1:
        model.load_state_dict(torch.load('models/no_scheduler/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'NoScheduler.pt'))
    elif optimizador == 2:
        model.load_state_dict(torch.load('models/no_scheduler/model_numTrayectTrain'+str(numTrayect)+'seqtrain'+str(sequence_length_train)+'_numTrayectValid'+str(numTrayects_valid)+'_lr'+str(lr)+'_momentum'+str(momentum)+'_TeacherForcing'+str(teacher_forcing_ratio)+'_numLayers'+str(num_lay)+'_hiddNeurons'+str(hidd_d)+'_bs'+str(batch_size)+'_solape'+str(factor_solape)+'_optim'+str(optimizador)+'NoScheduler.pt'))
minmax = [minmaxlat_train, minmaxlon_train]

train_state = torch.zeros(num_lay, batch_size, hidd_d, dtype=torch.float).to(device)
valid_state = torch.zeros(num_lay, tbatch_size, hidd_d, dtype=torch.float).to(device)

train_loader = DataLoader(train_set, batch_size=tbatch_size, shuffle = False, drop_last=True)  # Para test solo, se cambia el batch_size a batch_size = 1

if num_lay == 3:
    if sche == True: 
        if optimizador == 1:
            nameprop='lr'+str(lr)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)
        elif optimizador == 2:
            nameprop='lr'+str(lr)+'_momentum'+str(momentum)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)
    else:
        if optimizador == 1:
            nameprop='lr'+str(lr)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'NoScheduler'
        elif optimizador == 2:
            nameprop='lr'+str(lr)+'_momentum'+str(momentum)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'NoScheduler'
    
    # new dataset:  
    predictions, targets = check_accuracy_imgs_seq(train_loader, model, 'train','numtrayectstrain'+str(numTrayect)+'_numtrayectsvalid'+str(numTrayects_valid)+'_seqlen'+str(sequence_length_train)+'_TeacherForcing'+str(teacher_forcing_ratio), loss_func, train_state,sequence_length_train,in_size,batch_size, device, minmax, propiedades, nameprop, lista_dias_train)
    predictions, targets = check_accuracy_imgs_seq(valid_loader, model, 'valid','numtrayectstrain'+str(numTrayect)+'_numtrayectsvalid'+str(numTrayects_valid)+'_seqlen'+str(sequence_length_train)+'_TeacherForcing'+str(teacher_forcing_ratio), loss_func, valid_state,sequence_length_valid,in_size,tbatch_size, device, minmax, propiedades, nameprop, lista_dias_valid)
elif num_lay == 2:
    # new dataset:  
    if sche == True:
        if optimizador == 1:
            nameprop='lr'+str(lr)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)
        elif optimizador == 2:
            nameprop='lr'+str(lr)+'_momentum'+str(momentum)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'schedulerLRstepsize'+str(stepsize)+'gamma'+str(gamma)
    else:
        if optimizador == 1:
            nameprop='lr'+str(lr)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'NoScheduler'
        elif optimizador == 2:
            nameprop='lr'+str(lr)+'_momentum'+str(momentum)+'batchsize'+str(batch_size)+'hidd'+str(hidd_d)+'numlay'+str(num_lay)+'%solape'+str(factor_solape)+'lonsec'+str(sequence_length_train)+'_optim'+str(optimizador)+'NoScheduler'
    predictions, targets = check_accuracy_imgs_seq2lay(train_loader, model, 'train','numtrayectstrain'+str(numTrayect)+'_numtrayectsvalid'+str(numTrayects_valid)+'_seqlen'+str(sequence_length_train)+'_TeacherForcing'+str(teacher_forcing_ratio), loss_func, train_state,sequence_length_train,in_size,batch_size, device, minmax, propiedades, nameprop, lista_dias_train)
    predictions, targets = check_accuracy_imgs_seq2lay(valid_loader, model, 'valid','numtrayectstrain'+str(numTrayect)+'_numtrayectsvalid'+str(numTrayects_valid)+'_seqlen'+str(sequence_length_valid)+'_TeacherForcing'+str(teacher_forcing_ratio), loss_func, valid_state,sequence_length_valid,in_size,tbatch_size, device, minmax, propiedades, nameprop, lista_dias_valid)
else:
    print('Fallo en la comprobación de eficiencia')

print('Fin prueba')