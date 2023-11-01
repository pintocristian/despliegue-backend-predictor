import json
import joblib
import os

dir_actual = os.path.dirname(os.path.abspath(__file__))
ruta_json = os.path.join(dir_actual, 'resources/columnas_df.json')
ruta_pipeline = os.path.join(dir_actual, 'resources/pipeline.pkl')
ruta_pipeline_dtypes = os.path.join(dir_actual, 'resources/dtypes_df.pkl')
ruta_df_pkl = os.path.join(dir_actual, 'resources/data.pkl')

with open(ruta_json) as fname:
    pipeline_columnas = json.load(fname)

pipeline = joblib.load(ruta_pipeline)
pipeline_dtypes = joblib.load(ruta_pipeline_dtypes)