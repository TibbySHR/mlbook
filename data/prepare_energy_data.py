#!/usr/bin/env python3
"""
Script de préparation des données pour le projet Énergie.
Télécharge les données de consommation énergétique depuis l'API Hydro-Québec.

Usage:
    python prepare_energy_data.py

Crée:
    - energy_train.csv : données d'entraînement (2023-12 à 2024-02)
    - energy_test.csv : données de test sans cible (pour Kaggle)
    - sample_submission.csv : format de soumission
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime

API_URL = "https://donnees.hydroquebec.com/api/explore/v2.1/catalog/datasets/consommation-clients-evenements-pointe/records"

def fetch_all_data(limit_per_request=100, max_records=10000):
    """Télécharge les données depuis l'API Hydro-Québec.
    
    Note: L'API a une limite d'offset de 10000, donc on ne peut pas
    télécharger plus de 10000 enregistrements.
    """
    all_records = []
    offset = 0
    
    print("Téléchargement des données depuis Hydro-Québec...")
    print(f"(Limite API: {max_records} enregistrements)")
    
    while offset < max_records:
        params = {
            "limit": limit_per_request,
            "offset": offset
        }
        
        try:
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400 and offset >= 10000:
                print(f"  Limite d'offset API atteinte à {offset}")
                break
            raise e
        
        records = data.get("results", [])
        if not records:
            break
            
        all_records.extend(records)
        offset += limit_per_request
        
        print(f"  Téléchargé: {len(all_records)} enregistrements...")
        
        if len(records) < limit_per_request:
            break
    
    print(f"Total: {len(all_records)} enregistrements")
    return all_records


def prepare_dataframe(records):
    """Convertit les enregistrements en DataFrame et nettoie les données."""
    df = pd.DataFrame(records)
    
    # Convertir les colonnes de date
    df['horodatage_local'] = pd.to_datetime(df['horodatage_local'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Renommer les colonnes pour clarté (en français)
    column_mapping = {
        'energie_totale_consommee': 'energie_kwh',
        'temperature_exterieure_moyenne': 'temperature_ext',
        'humidite_relative_moyenne': 'humidite',
        'vitesse_vent_moyenne': 'vitesse_vent',
        'precipitations_neige_moyenne': 'neige',
        'irradiance_solaire_moyenne': 'irradiance_solaire',
        'temperature_interieure_moyenne': 'temperature_int',
        'temperature_consigne_moyenne': 'consigne',
        'indicateur_evenement': 'evenement_pointe',
        'indicateur_weekend': 'est_weekend',
        'indicateur_jour_ferie': 'est_ferie',
        'heure_locale': 'heure',
        'jour_semaine': 'jour_semaine'
    }
    df = df.rename(columns=column_mapping)
    
    # Convertir les booléens
    df['est_weekend'] = df['est_weekend'].map({'TRUE': 1, 'FALSE': 0})
    df['est_ferie'] = df['est_ferie'].map({'TRUE': 1, 'FALSE': 0})
    
    # Sélectionner les colonnes utiles
    colonnes_utiles = [
        'horodatage_local', 'poste', 'heure', 'jour', 'mois', 'jour_semaine',
        'temperature_ext', 'humidite', 'vitesse_vent', 'neige', 'irradiance_solaire',
        'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos', 
        'jour_semaine_sin', 'jour_semaine_cos',
        'est_weekend', 'est_ferie',
        'clients_connectes', 'tstats_intelligents_connectes',
        'evenement_pointe', 'energie_kwh'
    ]
    
    df = df[[c for c in colonnes_utiles if c in df.columns]]
    
    # Trier par date
    df = df.sort_values('horodatage_local').reset_index(drop=True)
    
    # Supprimer les valeurs manquantes dans la cible
    df = df.dropna(subset=['energie_kwh'])
    
    return df


def create_train_test_split(df, test_start_date='2024-02-01'):
    """Crée une division temporelle train/test."""
    # Convertir en timezone-aware pour correspondre aux données
    test_start = pd.to_datetime(test_start_date).tz_localize('UTC')
    
    train = df[df['horodatage_local'] < test_start].copy()
    test = df[df['horodatage_local'] >= test_start].copy()
    
    print(f"Ensemble d'entraînement: {len(train)} observations")
    print(f"Ensemble de test: {len(test)} observations")
    
    return train, test


def main():
    # Télécharger les données
    records = fetch_all_data()
    
    # Préparer le DataFrame
    df = prepare_dataframe(records)
    
    print(f"\nDonnées préparées: {len(df)} observations")
    print(f"Période: {df['horodatage_local'].min()} à {df['horodatage_local'].max()}")
    
    # Division train/test
    train, test = create_train_test_split(df)
    
    # Sauvegarder les fichiers
    train.to_csv('energy_train.csv', index=False)
    print(f"\nSauvegardé: energy_train.csv ({len(train)} lignes)")
    
    # Pour le test, on garde la cible mais on créera un fichier sans pour Kaggle
    test_sans_cible = test.drop(columns=['energie_kwh'])
    test_sans_cible.to_csv('energy_test.csv', index=False)
    print(f"Sauvegardé: energy_test.csv ({len(test_sans_cible)} lignes, sans cible)")
    
    # Créer le fichier de soumission exemple
    submission = pd.DataFrame({
        'id': range(len(test)),
        'energie_kwh': test['energie_kwh'].values  # Valeurs réelles pour vérification
    })
    submission.to_csv('sample_submission.csv', index=False)
    print(f"Sauvegardé: sample_submission.csv")
    
    # Sauvegarder aussi les vraies valeurs de test (pour le professeur)
    test.to_csv('energy_test_avec_cible.csv', index=False)
    print(f"Sauvegardé: energy_test_avec_cible.csv (pour évaluation)")
    
    # Afficher quelques statistiques
    print("\n--- Statistiques des données d'entraînement ---")
    print(f"Cible (energie_kwh): moyenne={train['energie_kwh'].mean():.2f}, std={train['energie_kwh'].std():.2f}")
    print(f"Température: moyenne={train['temperature_ext'].mean():.2f}°C")
    print(f"Événements de pointe: {train['evenement_pointe'].sum()} ({100*train['evenement_pointe'].mean():.1f}%)")


if __name__ == "__main__":
    main()
