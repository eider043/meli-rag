import pandas as pd
from src.config import DATA_PATH, SUBSET_N, RANDOM_SEED

def load_laptops() -> pd.DataFrame:
    """Lee el CSV y hace normalización mínima para el pipeline."""
    df = pd.read_csv(DATA_PATH)

    # Normaliza nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]

    # Asegura una columna de id
    if "laptop_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "laptop_id"})

    # Limpieza básica de NA
    df = df.fillna("")

    # Subset reproducible (si aplica)
    if SUBSET_N and len(df) > SUBSET_N:
        df = df.sample(SUBSET_N, random_state=RANDOM_SEED).reset_index(drop=True)

    # laptop_id como string
    df["laptop_id"] = df["laptop_id"].astype(str)

    return df
