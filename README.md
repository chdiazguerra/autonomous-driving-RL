# Autonomous Driving RL

Implementación de un sistema basado en Deep Reinforcement Learning para la conducción autónoma de un vehículo en un entorno simulado.

## Pre-requisitos

El código fue realizado para ejecutarse utilizando el Simulador CARLA versión 0.9.14, siendo incompatible con otras versiones. Para instalarlo, seguir las instrucciones en el siguiente enlace: https://carla.readthedocs.io/en/0.9.14/start_quickstart/  
Se debe tener instalado el cliente de CARLA para Python, lo que se puede realizar con el siguiente comando:  
```
pip install carla
```

El resto de las librerías pueden ser instaladas utilizado el archivo requirements.txt, con el siguiente comando:  
```
pip install -r requirements.txt
```

## Configuración

El archivo [config.py](configuration/config.py) contiene las variables de configuración del sistema. En este archivo se pueden modificar los parámetros de entrenamiento, el modelo a utilizar, el modo de ejecución, entre otros. Es importante que esté bien configurado antes de ejecutar el sistema, para asegurar que los archivos se guarden y lean en la ubicación correcta.

Para ejecutar todos los scripts (excepto el de entrenamiento del autoencoder) se debe tener abierto el simulador.

Ante cualquier duda de los parámetros de los scripts, se puede utilizar el siguiente comando:  
```
python <script> --help
```

## Entrenamiento del Autoencoder

Para entrenar el autoencoder, lo primero que se debe realizar es la extracción de datos de CARLA. Para esto, se debe ejecutar el script [collect_data_autoencoder.py](collect_data_autoencoder.py), el cual se encarga de extraer un dataset de imágenes e información de conducción. Para esto, se puede utilizar el siguiente comando:
```
python collect_data_autoencoder.py --out_folder ..\autoencoderData\ClearNoon --weather ClearNoon
```
Notar que se ha especificado la ruta de guardado y el clima, los que por defecto son *./sensor_data* y *ClearNoon* respectivamente.

Una vez que se tengan los datos extraídos, se procede a formar el archivo con los datos totales. Para ello, se debe ejecutar:
```
python data/create_dataset.py --folder ..\autoencoderData\ClearNoon ..\autoencoderData\HardRainNoon ..\autoencoderData\WetNoon ..\autoencoderData\ClearSunset --out_file ..\autoencoderData\ClearNoon\dataset.csv
```
Notar que aquí se han especificado las rutas de los datos extraídos y el archivo de salida.

Finalmente, se procede a entrenar el autoencoder. Para esto, se debe ejecutar el script [train_autoencoder.py](train_autoencoder.py), el cual se encarga de entrenar el autoencoder con el dataset creado. Para esto, se puede utilizar el siguiente comando:
```
python train_autoencoder.py
```
El cual utiliza el archivo de configuración para cargar los parámetros de entrenamiento. Se debe tener en cuenta que, dado la forma en que está programada la carga de datos, esto podría tomar gran parte de la memoria RAM. Se recomienda tener al menos 16gb de memoria.

Se proveen los [datos utilizados](https://uchile-my.sharepoint.com/:u:/g/personal/christian_diaz_g_uchile_cl/ET3rNmM3CmZFvL8RfhFagvYBmGmwVx3Hhuuf0Ux3RNv-bg?e=NAO6ov) y el [autoencoder entrenado](https://uchile-my.sharepoint.com/:f:/g/personal/christian_diaz_g_uchile_cl/Epfuw5T6DKtLskBleu3-SP8Bt-6hY04gH2ig_EQHKrc1Iw?e=6qv3Wt).

## Extracción de datos expertos

El modelo del agente hace uso de Behavior Cloning, con lo que se debe tener un experto que entregue las acciones correctas para cada estado. Para esto, se debe ejecutar el script [collect_expert_data.py](collect_expert_data.py), el cual se encarga de extraer un dataset de observaciones e información de conducción. Para esto, se puede utilizar el siguiente comando:
```
python collect_expert_data.py --weather ClearNoon
```
En este caso, la carpeta de salida se encuentra en el archivo de configuración, al igual que la mayoría de los parámetros. Se debe especificar el clima a utilizar, el cual por defecto es *ClearNoon*.

Una vez que se extraigan los datos para todos los climas de interés, se procede a formar el archivo con los datos totales. Para ello, se deja el archivo [utils.ipynb](utils.ipynb), la cual permite su creación. Este archivo creado es el que debe ser utilizado en la variable *DDPG_EXPERT_DATA_FILE* del archivo de configuración.

Se proveen los [datos expertos](https://uchile-my.sharepoint.com/:u:/g/personal/christian_diaz_g_uchile_cl/EeI_lEEZvLBJuIf3ewVT9NEBlACXI0x0Tmc-0ipDYEZsYQ?e=KLWc6c)

## Entrenamiento del agente

Los parámetros de entrenamiento para DDPG y SAC están dados en el archivo de configuración, en sus respectivos apartados. 

Para entrenar el agente DDPG, se debe ejecutar el script [train_ddpg_agent.py](train_ddpg_agent.py), mientras que para SAC se ejecuta [train_sac_agent.py](train_sac_agent.py). En ambos casos, se debe tener el simulador abierto, se debe haber entrenado el autoencoder previamente y tener el dataset experto si es que se utiliza.

Se proveen los [agentes entrenados](https://uchile-my.sharepoint.com/:u:/g/personal/christian_diaz_g_uchile_cl/EdUgQUjGtV9Btp_vnsoR4UsBloY4gWQSB-SJGZUcpxYjKw?e=AOcJlI).

## Evaluación del agente

Para evaluar el agente, se debe ejecutar el script [test_agent.py](test_agent.py), el cual se encarga de ejecutar el agente entrenado en el simulador. Para esto, se puede utilizar el siguiente comando:
```
python test_agent.py --route_id <id> --agent_model <model> --type <type>
```
En este caso, *route_id* especifica cual de las 3 rutas (de 0 a 2) se desea utilizar, *agent_model* especifica el modelo del agente a utilizar (archivo entrenado) y *type* especifica si se desea evaluar en el entorno de entrenamiento o de prueba. También se puede utilizar el flag *-exo_vehicles* si se desea que el entorno tenga vehículos extra.

Se proveen [videos](https://uchile-my.sharepoint.com/:u:/g/personal/christian_diaz_g_uchile_cl/ER-a5GcwflJLhpWTzJ3UUxABAjAnRTExH4TD5zHTPaU_qA?e=DtIC9n) con la evaluación del agente.