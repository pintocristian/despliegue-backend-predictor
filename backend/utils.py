import pandas as pd
from sklearn.impute import SimpleImputer

def dict_a_df(obs, columnas, dtypes):
    obs_df = pd.DataFrame([obs])

    # Encuentra las columnas con valores faltantes
    columnas_con_nulos = obs_df.columns[obs_df.isnull().any()]

    # Imputar solo en las columnas con valores faltantes
    if not columnas_con_nulos.empty:
        imputer = SimpleImputer(strategy='most_frequent')
        obs_df[columnas_con_nulos] = imputer.fit_transform(obs_df[columnas_con_nulos])

    # Convertir tipos de datos
    for col, dtype in dtypes.items():
        if col in obs_df.columns:
            obs_df[col] = obs_df[col].astype(dtype)

    return obs_df