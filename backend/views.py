import json
import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import dict_a_df
from .data import pipeline, pipeline_columnas, pipeline_dtypes, ruta_df_pkl
import shap
from concurrent.futures import ThreadPoolExecutor
import threading

lock = threading.Lock()

def guardar_en_pkl(nuevo_registro):
    with lock:
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

        def procesar_prediccion():
            
            pipeline_predicciones = pipeline.named_steps['procesado_variables'].transform(obs_df)  # Aplicar el pipeline completo

            #Se obtienen los nombres de las variables que resultan del rpocesamiento
            procesador_pipeline=pipeline.named_steps['procesado_variables']
            nombres_caracteristicas_procesadas = []

            for nombre, transformador, columnas in procesador_pipeline.transformers_:
                
                if hasattr(transformador, 'get_feature_names_out'):
                    # Obtener los nombres de características generados por el transformador
                    nombres_generados = transformador.get_feature_names_out()
                    # Extender la lista de nombres de características procesadas
                    nombres_caracteristicas_procesadas.extend(nombres_generados)
                else:
                    nombres_caracteristicas_procesadas.append(nombre)

            print(nombres_caracteristicas_procesadas)


            # Predicciones con el modelo XGBoost
            xgb_model = pipeline.named_steps['estimador']
            predicciones = xgb_model.predict(pipeline_predicciones)

            # Crear un explainer SHAP para el modelo
            explainer = shap.Explainer(xgb_model)

            # Calcular las contribuciones SHAP para todas las instancias en 'pipeline_predicciones'
            shap_values = explainer.shap_values(pipeline_predicciones)

            # 'shap_values' es una matriz que contiene las contribuciones SHAP para todas las instancias.

           

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

            importancia_caracteristicas_completo = dict(zip(nombres_caracteristicas_procesadas, shap_values[prediccion_int][0]))
     

            nuevo_dict_caracteristicas_procesadas = {}
            sumatoria = 0
            promedio = 0
            num_caracteristicas = 0
            for columna in pipeline_columnas:
                
                if columna != "naturaleza":
                    for clave in importancia_caracteristicas_completo:
                        if columna in clave:
                            sumatoria = sumatoria + importancia_caracteristicas_completo[clave]
                            num_caracteristicas = num_caracteristicas + 1
                    
                    if num_caracteristicas > 1:
                        promedio = sumatoria / num_caracteristicas
                        nuevo_dict_caracteristicas_procesadas[columna] = promedio
                    else:
                        nuevo_dict_caracteristicas_procesadas[columna] = sumatoria
                    
                    sumatoria = 0
                    promedio = 0
                    num_caracteristicas = 0

        
            # Ordenar el diccionario por los valores en orden descendente
            sorted_dict = sorted(nuevo_dict_caracteristicas_procesadas.items(), key=lambda x: x[1], reverse=True)

            top_10_tuples = sorted_dict[:10]

            top_10_formatted = [{'nombre': k, 'importancia': v} for k, v in top_10_tuples]


            importancia_caracteristicas = [[diccionario] for diccionario in top_10_formatted]

            return {'prediccion': prediccion_int, 'definicion': definicion,  'importancia_caracteristicas': importancia_caracteristicas}

        with ThreadPoolExecutor() as executor:
            result = executor.submit(procesar_prediccion)
            return JsonResponse(result.result())

    return JsonResponse({'error': 'Invalid request method'})



lock = threading.Lock()

@csrf_exempt
def obtener_dataframe(request):
    if request.method == 'GET':
        def cargar_dataframe():
            try:
                with lock:
                    df = joblib.load(ruta_df_pkl)
                    df_json = df.to_json(orient='records')
                    return JsonResponse({'dataframe': df_json})
            except FileNotFoundError:
                return JsonResponse({'error': 'El archivo DataFrame no se encontró.'})
        
        with ThreadPoolExecutor() as executor:
            result = executor.submit(cargar_dataframe)
            return result.result()
    
    return JsonResponse({'error': 'Invalid request method'})