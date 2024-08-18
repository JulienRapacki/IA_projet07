# OpenClassrooms - P07IA -Réalisez une analyse de sentiments grâce au Deep Learning

Le but est de prédire le sentiment des tweets afin d'anticiper les bad buzz sur les réseaux sociaux. 
Le projet inclut un modèle de Deep Learning type LSTM ainsi qu' une démarche orientée MLOps pour assurer une gestion et un déploiement efficaces du modèle.<br>


## Fonctionnalités

Prédiction du sentiment des tweets : Modèle de machine learning pour détecter si un tweet exprime un sentiment négatif.<br>
API déployée sur le Cloud : Interface pour l'interaction avec le modèle via une API.<br>
Gestion des modèles avec MLFlow : Suivi et gestion des expérimentations.<br>
Déploiement continu : Pipeline CI/CD pour automatiser le déploiement du modèle.<br>
Suivi de la performance : Utilisation de Azure Application Insights pour le suivi des performances en production.<br>


## Structure du projet

|── JulienRapacki/IA_P07<br>
|      └── app.py                _# API Flask pour servir les prédictions_<br> 
|      └── model_lstm_glove.h5   _# modèle LSTM entraîné_<br> 
|      └── tokenizer_lstm.pickle _# tockenizer entraîné_<br> 
|── Streamlit<br>  
|      └── steamlit-app.py       _# Interface utilisateur délivrée par Streamlit_<br> 
|── Test<br>  
|      └── P07_app_test.py       _# Script pour tests unitaires_<br> 
|── Model notebook<br>
|      └── Rapacki_Julien_2_scripts_notebook_modélisation_082024.ipynb    _# Notebook conception du modèle_<br> 
|── README.md                    _# Documentation du projet_<br> 
