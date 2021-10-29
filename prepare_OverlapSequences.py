#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:16:12 2021

@author: lauram
"""
import numpy as np


def completeseq_consolape(trayectoria_ensecuencia, targets_ensecuencia, sampler, trayect_data, seq_len, trayect_labels, solape):
    numsecuencias_tot = [] 
    if solape == 0:
        i = 0
        for index in sampler:
            posiciones_trayectoria = trayect_data[index:index+sampler.__len__(index)]
            targets_trayectoria = trayect_labels[index:index+sampler.__len__(index)]
            # modulo = posiciones_trayectoria.shape[0]%seq_len
            num_trayectorias_nosolapadas = np.floor_divide(posiciones_trayectoria.shape[0],seq_len)
          #  print('Numero de secuencias creadas previsto = '+str(num_trayectorias_nosolapadas))
            num_secuencias_creadas = num_trayectorias_nosolapadas
            numsecuencias_tot.append(num_secuencias_creadas)
            seq_posiciones_trayectoria = posiciones_trayectoria[:num_secuencias_creadas*seq_len,:]
            seq_targets_trayectoria = targets_trayectoria[:num_secuencias_creadas*seq_len,:]
            for n in range(0,num_secuencias_creadas): 
           #     print('Paso '+str(paso)+'para n ='+str(n))
           #     paso = paso +1
                # recordar que el indexado es de 0 a nºposiciones-1
                try: 
                    x = seq_posiciones_trayectoria[n*seq_len:n*seq_len+seq_len,:]
           #         print(x.shape)
                    y = seq_targets_trayectoria[n*seq_len:n*seq_len+seq_len,:]
           #         print(y.shape)
                except IndexError:
                    x = None
                    y = None
                trayectoria_ensecuencia[i,:,:] = x
                targets_ensecuencia[i,:,:] = y
                i = i + 1
    else:
        i = 0
        for index in sampler:
            offset = seq_len - solape
            posiciones_trayectoria = trayect_data[index:index+sampler.__len__(index)]
            targets_trayectoria = trayect_labels[index:index+sampler.__len__(index)]
            modulo = posiciones_trayectoria.shape[0]%seq_len
           # print('Numero de secuencias creadas previsto = '+str(posiciones_trayectoria.shape[0]-seq_len+1-modulo))
           # num_secuencias_creadas = posiciones_trayectoria.shape[0]-seq_len+1-modulo
            num_secuencias_creadas = np.floor_divide(posiciones_trayectoria.shape[0],seq_len) + np.floor_divide(posiciones_trayectoria.shape[0]-offset,seq_len)
            numsecuencias_tot.append(num_secuencias_creadas)
            seq_posiciones_trayectoria = posiciones_trayectoria[:num_secuencias_creadas*seq_len,:]
            seq_targets_trayectoria = targets_trayectoria[:num_secuencias_creadas*seq_len,:]
            pares = 0
            impares = 1
            for n in range(0,num_secuencias_creadas): 
                if n%2 == 0: # representa las secuencias enteras sin solape
                    try: 
                        x = seq_posiciones_trayectoria[pares*seq_len:(pares+1)*seq_len,:]
                        y = seq_targets_trayectoria[pares*seq_len:(pares+1)*seq_len,:]
                    except IndexError:
                        x = None
                        y = None
                    pares = pares +1
                else:
                    try: 
                        x = seq_posiciones_trayectoria[impares*seq_len-solape:(impares+1)*seq_len-solape,:]
                        y = seq_targets_trayectoria[impares*seq_len-solape:(impares+1)*seq_len-solape,:]
                    except IndexError:
                        x = None
                        y = None
                    impares = impares +1
                trayectoria_ensecuencia[i,:,:] = x
                targets_ensecuencia[i,:,:] = y
                i = i + 1
    return trayectoria_ensecuencia, targets_ensecuencia

def num_seq_tot(sampler, seq_len, trayect_data, solape):
    numsecuencias_tot = [] 
    if solape == 0:
        for index in sampler:
            posiciones_trayectoria = trayect_data[index:index+sampler.__len__(index)]
            # modulo = posiciones_trayectoria.shape[0]%seq_len
            num_trayectorias_nosolapadas = np.floor_divide(posiciones_trayectoria.shape[0],seq_len)
          #  print('Numero de secuencias creadas previsto = '+str(num_trayectorias_nosolapadas))
            num_secuencias_creadas = num_trayectorias_nosolapadas
            numsecuencias_tot.append(num_secuencias_creadas)
        numsecuencias_tot = np.asarray(numsecuencias_tot)
        numsecuencias_trayect = numsecuencias_tot
        numsecuencias_tot = sum(numsecuencias_tot)
    else:
        for index in sampler:
            offset = seq_len - solape
            posiciones_trayectoria = trayect_data[index:index+sampler.__len__(index)]
            modulo = posiciones_trayectoria.shape[0]%seq_len
           # print('Numero de secuencias creadas previsto = '+str(posiciones_trayectoria.shape[0]-seq_len+1-modulo))
           # num_secuencias_creadas = posiciones_trayectoria.shape[0]-seq_len+1-modulo
            num_secuencias_creadas = np.floor_divide(posiciones_trayectoria.shape[0],seq_len) + np.floor_divide(posiciones_trayectoria.shape[0]-offset,seq_len)
            numsecuencias_tot.append(num_secuencias_creadas)
        numsecuencias_tot = np.asarray(numsecuencias_tot)
        numsecuencias_trayect = numsecuencias_tot
        numsecuencias_tot = sum(numsecuencias_tot)
    return numsecuencias_tot, numsecuencias_trayect

def complete_sequences(trayectoria_ensecuencia, targets_ensecuencia, sampler, trayect_data, seq_len, trayect_labels, solape):
    i = 0
    for index in sampler:
       # print('i = '+str(i))
        posiciones_trayectoria = trayect_data[index:index+sampler.__len__(index)]
        num_sequences = posiciones_trayectoria.shape[0]//seq_len
        modulo = posiciones_trayectoria.shape[0]%seq_len
       # print('Numero de secuencias creadas previsto = '+str(posiciones_trayectoria.shape[0]-seq_len+1-modulo))
       # print('Posiciones de la trayectoria = '+str(posiciones_trayectoria.shape[0]))
       # print('Modulo = '+str(modulo))
        seq_posiciones_trayectoria = posiciones_trayectoria[:num_sequences*seq_len,:]
        seq_targets_trayectoria = trayect_labels[index:index+num_sequences*seq_len,:]
        if solape == 0:
            for n in range(0, num_sequences): 
           #     print('Paso '+str(paso)+'para n ='+str(n))
           #     paso = paso +1
                # recordar que el indexado es de 0 a nºposiciones-1
                try: 
                    x = seq_posiciones_trayectoria[n*seq_len:n*seq_len+seq_len,:]
           #         print(x.shape)
                    y = seq_targets_trayectoria[n*seq_len:n*seq_len+seq_len,:]
           #         print(y.shape)
                except IndexError:
                    x = None
                    y = None
                trayectoria_ensecuencia[i,:,:] = x
                targets_ensecuencia[i,:,:] = y
                i = i + 1
        else:  
            for n in range(0, num_sequences): 
                if n == 0:
                    # recordar que el indexado es de 0 a nºposiciones-1
                    try: 
                        x = seq_posiciones_trayectoria[n:n+seq_len,:]
               #         print(x.shape)
                        y = seq_targets_trayectoria[n:n+seq_len,:]
               #         print(y.shape)
                    except IndexError:
                        x = None
                        y = None
                else:
                    try: 
                        x = seq_posiciones_trayectoria[n*seq_len-solape:n+seq_len-solape,:]
               #         print(x.shape)
                        y = seq_targets_trayectoria[n*seq_len-solape:n+seq_len-solape,:]
               #         print(y.shape)
                    except IndexError:
                        x = None
                        y = None
                    
                trayectoria_ensecuencia[i,:,:] = x
                targets_ensecuencia[i,:,:] = y
                i = i + 1
    
    return trayectoria_ensecuencia, targets_ensecuencia

def num_seq_tot_nosolapadas(sampler, seq_len, trayect_data):
    numsecuencias_tot = [] 
    for index in sampler:
        posiciones_trayectoria = trayect_data[index:index+sampler.__len__(index)]
        modulo = posiciones_trayectoria.shape[0]%seq_len
        num_trayectorias_nosolapadas = np.floor_divide(posiciones_trayectoria.shape[0],seq_len)
      #  print('Numero de secuencias creadas previsto = '+str(num_trayectorias_nosolapadas))
        num_secuencias_creadas = num_trayectorias_nosolapadas
        numsecuencias_tot.append(num_secuencias_creadas)
    numsecuencias_tot = np.asarray(numsecuencias_tot)
    numsecuencias_trayect = numsecuencias_tot
    numsecuencias_tot = sum(numsecuencias_tot)
    return numsecuencias_tot, numsecuencias_trayect

def complete_sequences_nosolapadas(trayectoria_ensecuencia, targets_ensecuencia, sampler, trayect_data, seq_len, trayect_labels):
    i = 0
    for index in sampler:
       # print('i = '+str(i))
        posiciones_trayectoria = trayect_data[index:index+sampler.__len__(index)]
        num_sequences = posiciones_trayectoria.shape[0]//seq_len
        modulo = posiciones_trayectoria.shape[0]%seq_len
       # print('Numero de secuencias creadas previsto = '+str(posiciones_trayectoria.shape[0]-seq_len+1-modulo))
       # print('Posiciones de la trayectoria = '+str(posiciones_trayectoria.shape[0]))
       # print('Modulo = '+str(modulo))
        seq_posiciones_trayectoria = posiciones_trayectoria[:num_sequences*seq_len,:]
        seq_targets_trayectoria = trayect_labels[index:index+num_sequences*seq_len,:]
       # paso = 0
        for n in range(0, num_sequences): 
       #     print('Paso '+str(paso)+'para n ='+str(n))
       #     paso = paso +1
            # recordar que el indexado es de 0 a nºposiciones-1
            try: 
                x = seq_posiciones_trayectoria[n*seq_len:n*seq_len+seq_len,:]
       #         print(x.shape)
                y = seq_targets_trayectoria[n*seq_len:n*seq_len+seq_len,:]
       #         print(y.shape)
            except IndexError:
                x = None
                y = None
            trayectoria_ensecuencia[i,:,:] = x
            targets_ensecuencia[i,:,:] = y
            i = i + 1
    
    return trayectoria_ensecuencia, targets_ensecuencia