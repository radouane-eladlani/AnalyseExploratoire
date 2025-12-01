# ===============================================
# API pour prédire la consommation énergétique des bâtiments
# Utilise BentoML et Pydantic pour la validation des données
# ===============================================

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field
import numpy as np

# -----------------------------------------------
# Charger le modèle sauvegardé avec BentoML
# Remplace "modele_energie_seattle" par le nom exact de ton modèle
# -----------------------------------------------
model = bentoml.sklearn.load_model("modele_energie_seattle:latest")

# -----------------------------------------------
# Définir le modèle de validation des entrées
# On utilise Pydantic pour vérifier que l'utilisateur envoie des données cohérentes
# -----------------------------------------------
class BuildingInput(BaseModel):
    BuildingAge: float = Field(..., description="Âge du bâtiment en années")
    BuildingSize: float = Field(..., description="Surface du bâtiment en m²")
    NbPropertyUses: int = Field(..., description="Nombre de types d'usages dans le bâtiment")
    HasElectricity: int = Field(..., ge=0, le=1, description="Présence d'électricité (0 ou 1)")
    HasNaturalGas: int = Field(..., ge=0, le=1, description="Présence de gaz naturel (0 ou 1)")
    HasSteam: int = Field(..., ge=0, le=1, description="Présence de vapeur (0 ou 1)")
    Latitude: float = Field(..., description="Latitude du bâtiment")
    Longitude: float = Field(..., description="Longitude du bâtiment")

# -----------------------------------------------
# Créer le service BentoML
# -----------------------------------------------
svc = bentoml.Service("energie_api", runners=[])

# -----------------------------------------------
# Définir l'endpoint de prédiction
# -----------------------------------------------
@svc.api(input=JSON(pydantic_model=BuildingInput), output=JSON())
def predict(data: BuildingInput):
    """
    Endpoint de prédiction.
    Reçoit les informations d'un bâtiment et renvoie la consommation énergétique estimée.
    """

    # 1. Convertir les données Pydantic en tableau 2D pour le modèle
    X = np.array([[
        data.BuildingAge,
        data.BuildingSize,
        data.NbPropertyUses,
        data.HasElectricity,
        data.HasNaturalGas,
        data.HasSteam,
        data.Latitude,
        data.Longitude
    ]])

    # 2. Prédiction
    prediction = model.predict(X)

    # 3. Retourner la prédiction sous forme JSON
    return {"prediction_kBtu": float(prediction[0])}
