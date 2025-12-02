# ================================================================
# SERVICE API - Prédiction de consommation énergétique des bâtiments de Seattle (2016)
# ================================================================
# Ce fichier expose un modèle Machine Learning via une API BentoML.
# L'utilisateur envoie les caractéristiques d'un bâtiment
# et l'API renvoie une estimation de sa consommation énergétique.

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field, validator
import numpy as np


# ================================================================
# 1. Chargement du modèle BentoML
# ================================================================
# Le modèle a été sauvegardé dans le notebook avec :
# bentoml.sklearn.save_model("modele_energie_seattle", best_model...)
#
# "latest" sélectionne automatiquement la dernière version.
model = bentoml.sklearn.load_model("modele_energie_seattle:latest")


# ================================================================
# 2. Définition des données d'entrée avec Pydantic (validation)
# ================================================================
# Chaque champ correspond EXACTEMENT à une feature utilisée
# durant l'entraînement du modèle ML.
class BuildingInput(BaseModel):
    """
    Modèle Pydantic pour valider les données d'entrée de l'API.
    Chaque champ est vérifié pour éviter des valeurs incohérentes
    avant qu'elles ne soient envoyées au modèle ML.
    """

    BuildingAge: float = Field(..., description="Âge du bâtiment en années (>= 0)")
    BuildingSize: float = Field(..., description="Surface du bâtiment en m² (> 0)")
    NbPropertyUses: int = Field(..., description="Nombre de types d'usages dans le bâtiment (>= 1)")
    HasElectricity: int = Field(..., description="Présence d'électricité (0 ou 1)")
    HasNaturalGas: int = Field(..., description="Présence de gaz naturel (0 ou 1)")
    HasSteam: int = Field(..., description="Présence de vapeur (0 ou 1)")
    Latitude: float = Field(..., description="Latitude du bâtiment (-90 à 90)")
    Longitude: float = Field(..., description="Longitude du bâtiment (-180 à 180)")

    # ----------------------------------------
    # VALIDATEURS
    # ----------------------------------------
    
    @validator("BuildingAge")
    def check_building_age(cls, value):
        """Vérifie que l'âge du bâtiment est >= 0"""
        if value < 0:
            raise ValueError("BuildingAge doit être supérieur ou égal à 0.")
        return value

    @validator("BuildingSize")
    def check_building_size(cls, value):
        """Vérifie que la surface est > 0"""
        if value <= 0:
            raise ValueError("BuildingSize doit être supérieur à 0.")
        return value

    @validator("NbPropertyUses")
    def check_nb_property_uses(cls, value):
        """Vérifie que le nombre d'usages est au moins 1"""
        if value < 1:
            raise ValueError("NbPropertyUses doit être au moins 1.")
        return value

    @validator("HasElectricity", "HasNaturalGas", "HasSteam")
    def check_binary_flags(cls, value, field):
        """Vérifie que les champs électriques/gaz/vapeur sont 0 ou 1"""
        if value not in (0, 1):
            raise ValueError(f"{field.name} doit être 0 ou 1.")
        return value

    @validator("Latitude")
    def check_latitude_range(cls, value):
        """Vérifie que la latitude est comprise entre -90 et 90"""
        if not (-90 <= value <= 90):
            raise ValueError("Latitude doit être comprise entre -90 et 90.")
        return value

    @validator("Longitude")
    def check_longitude_range(cls, value):
        """Vérifie que la longitude est comprise entre -180 et 180"""
        if not (-180 <= value <= 180):
            raise ValueError("Longitude doit être comprise entre -180 et 180.")
        return value

# ================================================================
# 3. Création du service BentoML
# ================================================================
# Le nom "energie_api" sera visible dans BentoML Dashboard et Swagger.
svc = bentoml.Service("energie_api", runners=[])


# ================================================================
# 4. Définition de l'endpoint /predict
# ================================================================
# - Input : JSON validé par le modèle BuildingInput
# - Output : JSON contenant la consommation énergétique prédite
# ================================================================
@svc.api(input=JSON(pydantic_model=BuildingInput), output=JSON())
def predict(payload: BuildingInput):
    """
    Endpoint de prédiction - /predict

    Étapes :
    1. Conversion des données utilisateur en tableau numpy (2D)
       -> Obligatoire car scikit-learn attend un tableau 2D.
    2. Appel du modèle ML (RandomForestRegressor)
    3. Retour de la prédiction sous forme JSON (lisible par Swagger)

    Retourne :
    {
        "prediction_kBtu": valeur prédite
    }
    """

    # 1. Préparation des features dans l'ordre exact attendu par le modèle
    X = np.array([[
        payload.BuildingAge,
        payload.BuildingSize,
        payload.NbPropertyUses,
        payload.HasElectricity,
        payload.HasNaturalGas,
        payload.HasSteam,
        payload.Latitude,
        payload.Longitude
    ]])

    # 2. Faire la prédiction
    prediction = model.predict(X)

    # 3. Retourner une réponse JSON propre
    return {"prediction_kBtu": float(prediction[0])}
