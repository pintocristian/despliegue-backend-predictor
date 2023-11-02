import json
import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import dict_a_df
from .data import pipeline, pipeline_columnas, pipeline_dtypes, ruta_df_pkl
import shap

def guardar_en_pkl(nuevo_registro):
    df = joblib.load(ruta_df_pkl)
    df = pd.concat([df, nuevo_registro], ignore_index=True)
    joblib.dump(df, ruta_df_pkl)


@csrf_exempt
def predecir(request):
    if request.method == 'POST':
        observacion_dict = json.loads(request.body)
        print("\nObservacion recibida JSON. Valores: {}".format(observacion_dict))
        obs_df = dict_a_df(observacion_dict, pipeline_columnas, pipeline_dtypes)
        
        # Configurar pandas para mostrar todas las columnas
        pd.set_option('display.max_columns', None)

        # Luego imprime tu DataFrame
        print(obs_df)

        # Suponiendo que 'pipeline' es tu modelo XGBoost entrenado
        pipeline_predicciones = pipeline.named_steps['procesado_variables'].transform(obs_df)  # Aplicar el pipeline completo

        ""# Predicciones con el modelo XGBoost
        xgb_model = pipeline.named_steps['estimador']
        predicciones = xgb_model.predict(pipeline_predicciones)

        """ # Crear un explainer SHAP para el modelo
        explainer = shap.Explainer(xgb_model)

        # Calcular las contribuciones SHAP para todas las instancias en 'pipeline_predicciones'
        shap_values = explainer.shap_values(pipeline_predicciones)"""

        # 'shap_values' es una matriz que contiene las contribuciones SHAP para todas las instancias.

        # Puedes acceder a las contribuciones SHAP para la primera instancia de la siguiente manera:
        # Obtener todas las importancias SHAP para todas las instancias
        importancia_caracteristicas = []

        definicion = ""
        if predicciones[0] == 0:
            definicion = "Violencia física o psicologica"
        elif predicciones[0] == 1:
            definicion = "Abuso Sexual"
        elif predicciones[0] == 2:
            definicion = "Negligencia y abandono"
        
        prediccion_int = int(predicciones[0])

        # Agrega el nuevo registro al DataFrame existente
        nuevo_registro = obs_df.copy()  # Asegúrate de que obs_df incluye todas las características necesarias
        nuevo_registro['naturaleza'] = prediccion_int  # Agrega la predicción

        # Llama a la función para guardar en 'data.pkl'
        guardar_en_pkl(nuevo_registro)

        """# Recorre todas las instancias en shap_values
        for instancia_shap_values in shap_values:
            instancia_importancias = [{'nombre': nombre, 'importancia': float(importancia)}
                                    for nombre, importancia in zip(pipeline_columnas, instancia_shap_values[0])]
            importancia_caracteristicas.append(instancia_importancias)
"""
        return JsonResponse({'prediccion': prediccion_int, 'definicion': definicion,  'importancia_caracteristicas': importancia_caracteristicas})

    return JsonResponse({'error': 'Invalid request method'}) 



@csrf_exempt
def obtener_dataframe(request):
    if request.method == 'GET':
        try:
            df = joblib.load(ruta_df_pkl)
            df_json = df.to_json(orient='records')
            return JsonResponse({'dataframe': df_json})
        except FileNotFoundError:
            return JsonResponse({'error': 'El archivo DataFrame no se encontró.'})
    return JsonResponse({'error': 'Invalid request method'})