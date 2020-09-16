#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:52:10 2020

@author: emmanuelmoudoute-bell
"""

def gain(y_test, y_pred, montant_initial=1000, seuil_hausse=0.5, seuil_baisse=-0.5, effet_de_levier=1):
    import numpy as np

    
    taille_y = len(y_test)
 
    #### var y
    Tx_croissance_y = np.zeros(taille_y).reshape(taille_y, 1)
    y_test = np.reshape(y_test, (taille_y, 1))
 
    for i in range(0, taille_y - 1):
        Tx_croissance_y[i][0] = (y_test[i][0] - y_test[i - 1][0]) / y_test[i - 1][0] * 100
    #### var y_pred
    Tx_croissance_y_pred = np.zeros(taille_y).reshape(taille_y, 1)
    y_pred = np.reshape(y_pred, (taille_y, 1))
 
    for i in range(0, taille_y - 1):
        Tx_croissance_y_pred[i][0] = (y_pred[i][0] - y_pred[i - 1][0]) / y_pred[i - 1][0] * 100
 
    # count /gain
    gain_hausse = np.zeros(taille_y).reshape(taille_y, 1)
    gain_baisse = np.zeros(taille_y).reshape(taille_y, 1)
    gain_total = np.zeros(taille_y).reshape(taille_y, 1)
    count_hausse = 0
    count_baisse = 0
 
    for i in range(1,taille_y):
        if Tx_croissance_y_pred[i] > seuil_hausse:
            gain_hausse[i][0] = Tx_croissance_y[i][0]
            gain_total[i][0] = Tx_croissance_y[i][0]
            count_hausse += 1
        elif Tx_croissance_y_pred[i] < seuil_baisse:
            gain_baisse[i][0] = - Tx_croissance_y[i][0]
            gain_total[i][0] = - Tx_croissance_y[i][0]
            count_baisse += 1
 
    ### Evalutation du modele
    somme_gain_hausse = 0
    somme_gain_baisse = 0
 
    for i in range(taille_y):
        somme_gain_hausse += gain_hausse[i][0]
        somme_gain_baisse += gain_baisse[i][0]
 
    ## count juste
    count_trade_juste_hausse = 0
    count_trade_juste_baisse = 0
 
    for i in range(taille_y):
        if gain_hausse[i][0] > 0:
            count_trade_juste_hausse += 1
        elif gain_baisse[i][0] > 0:
            count_trade_juste_baisse += 1
 
    ### ratio fin
    ratio_juste = count_trade_juste_hausse + count_trade_juste_baisse
    count_trade = count_hausse + count_baisse
    ratio_gain_trade = (somme_gain_hausse + somme_gain_baisse) / count_trade
 
    ### Calcul des pertes et gains max
    gain_max_hausse = np.max(gain_hausse, axis=0)
    gain_min_hausse = np.min(gain_hausse, axis=0)
 
    gain_max_baisse = np.max(gain_baisse, axis=0)
    gain_min_baisse = np.min(gain_baisse, axis=0)
 
    # calcul des tx de var
    montant_final = montant_initial
    coef = 1
 
    montant = np.zeros(taille_y).reshape(taille_y, 1)
    for i in range(taille_y - 1):
        montant_final = montant_final * (1 + effet_de_levier * gain_total[i][0] / 100)
        coef = coef * (1 + effet_de_levier * gain_total[i][0] / 100)
        montant[i][0] = (coef - 1) * 100
 
    pourcent = round((coef - 1) * 100, 2)
    montant = np.delete(montant, (len(montant) - 1), axis=0)
    
    montant_final = round(montant_final, 2)
    import matplotlib.pyplot as plt
    
    
    plt.plot(montant, color='green', label="Montant disponible")
    plt.title("Evolution du budget")
    plt.legend()
    plt.show()
 
    ecart_type_gain = np.std(gain_total, axis=0)
    moyenne_gain = np.mean(gain_total, axis=0)
 
    print('seuil hausse' '=', seuil_hausse)
    print('seuil baisse''=', seuil_baisse)
    print('count hausse''=', count_hausse)
    print('count baisse''=', count_baisse)
    print('count trade pris''=', ratio_juste)
    print('count hausse juste''=', count_trade_juste_hausse)
    print('count baisse juste''=', count_trade_juste_baisse)
    print('Somme des gains a la hausse''=', somme_gain_hausse)
    print('Somme des gains a la baisse''=', somme_gain_baisse)
    print('Somme des gains''=', montant_final)
    print('count trade juste''=', count_trade)
    print('Ratio des gains par jour trade' '=',  ratio_gain_trade)
    print('gain max a la hausse''=', gain_max_hausse)
    print('gain min a la hausse''=', gain_min_hausse)
    print('gain max a la baisse''=', gain_max_baisse)
    print('gain min a la baisse''=', gain_min_baisse)
    print('Votre budget est de', montant_final, "€", 'soit un gain en pourcentage de', pourcent, "%")
    print("ecart type gain", ecart_type_gain)
    print("moyenne gain", moyenne_gain)
        
        