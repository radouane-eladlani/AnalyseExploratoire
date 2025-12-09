# ================================================================
# SERVICE API - Prédiction de consommation énergétique des bâtiments
# ================================================================

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field, field_validator
import numpy as np


# ================================================================
# 1. Chargement du modèle BentoML + création du runner
# ================================================================
# get() permet de récupérer la dernière version du modèle enregistré
model_ref = bentoml.sklearn.get("modele_energie_seattle:latest")

# Un runner est l'interface BentoML pour exécuter le modèle
model_runner = model_ref.to_runner()

# Le service expose l’API et utilise le runner
svc = bentoml.Service("energie_api", runners=[model_runner])


# ================================================================
# 2. Définition des données d'entrée avec Pydantic V2
# ================================================================
class BuildingInput(BaseModel):
    """
    Modèle Pydantic pour valider les données envoyées par l'utilisateur.
    Compatible Pydantic V2 (nouvelle syntaxe).
    """

    BuildingAge: float = Field(..., ge=0, description="Âge du bâtiment en années (>= 0)")
    BuildingSize: float = Field(..., gt=0, description="Surface du bâtiment en m² (> 0)")
    NbPropertyUses: int = Field(..., ge=1, description="Nombre de types d'usages (>= 1)")
    HasElectricity: int = Field(..., description="Présence d'électricité (0 ou 1)")
    HasNaturalGas: int = Field(..., description="Présence de gaz naturel (0 ou 1)")
    HasSteam: int = Field(..., description="Présence de vapeur (0 ou 1)")
    Latitude: float = Field(..., description="Latitude (-90 à 90)")
    Longitude: float = Field(..., description="Longitude (-180 à 180)")

    # -------------------------------
    # VALIDATEURS Pydantic V2
    # -------------------------------

    @field_validator("HasElectricity", "HasNaturalGas", "HasSteam")
    def validate_binary(cls, value):
        """Valide les variables binaires (0 ou 1)"""
        if value not in (0, 1):
            raise ValueError("Les champs doivent être 0 ou 1.")
        return value

    @field_validator("Latitude")
    def validate_latitude(cls, value):
        """Valide que la latitude est entre -90 et 90"""
        if not (-90 <= value <= 90):
            raise ValueError("Latitude doit être comprise entre -90 et 90.")
        return value

    @field_validator("Longitude")
    def validate_longitude(cls, value):
        """Valide que la longitude est entre -180 et 180"""
        if not (-180 <= value <= 180):
            raise ValueError("Longitude doit être comprise entre -180 et 180.")
        return value


# ================================================================
# 3. Endpoint /predict (asynchrone obligatoire)
# ================================================================
@svc.api(input=JSON(pydantic_model=BuildingInput), output=JSON())
async def predict(payload: BuildingInput):
    """
    Endpoint de prédiction - /predict
    BentoML exécute tous les runners en mode ASYNCHRONE
    → l'endpoint doit obligatoirement être async

    Étapes :
    1. Transformer les données en matrice 2D NumPy
    2. Appeler le modèle ML via son runner
    3. Retourner un JSON propre
    """

    # 1. Préparer les features dans l'ordre exact du modèle
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

    # 2. Prédiction ASYNCHRONE via BentoML
    prediction = await model_runner.predict.async_run(X)

    # 3. Retourner une réponse JSON claire
    return {
        "prediction_kBtu": float(prediction[0]),
        "period": "annual (based on Seattle 2016 dataset)"
    }
