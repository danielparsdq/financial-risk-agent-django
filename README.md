# financial-risk-agent-django

Sistema multiagente en Django para el análisis de riesgo crediticio con IA explicable (XAI). Incluye un chatbot financiero que consulta un modelo de ML entrenado desde el admin, explica cada predicción con SHAP y registra cada respuesta en un agente auditor basado en LLM.

Código asociado al artículo *"Multi-agent system in Django using explainable AI for the detection of credit risk in the financial sector of Ecuadorian companies"*.

## Contenido del repositorio

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── train_model.ipynb              # Notebook de entrenamiento (referencia del pipeline)
│   └── archivos/
│       └── credit_risk_dataset.csv    # Dataset público de Kaggle usado en los experimentos
└── sistema_xia/                       # Proyecto Django
    ├── manage.py
    ├── .env.example                   # Plantilla de variables de entorno
    ├── sistema_xia/                   # Settings, URLs, views y agentes (base_agent, chatbot_agent)
    ├── ml_models/                     # Entrenamiento de modelos desde el admin (utils.py)
    ├── audit/                         # Registro de auditoría y dashboard de cumplimiento
    ├── chats/                         # Historial de conversaciones por usuario
    ├── templates/                     # Chat, login, registro y plantillas del admin
    └── static/                        # JS y CSS del chat
```

## Requisitos

- Python 3.13
- Un servidor LLM compatible con la API de OpenAI (`/v1/chat/completions`). Por ejemplo, [Ollama](https://ollama.com) en local. El modelo del chatbot debe soportar *tool calling*, y el del auditor debe soportar `response_format={"type": "json_object"}`.

## Instalación y arranque

Desde la raíz del repositorio:

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp sistema_xia/.env.example sistema_xia/.env   # Luego editar SECRET_KEY y los valores LLM
cd sistema_xia
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Después abre:

- `http://127.0.0.1:8000/`: chat (puede usarse sin cuenta, pero sin historial).
- `http://127.0.0.1:8000/admin/`: administración con el usuario creado.

### Configurar el LLM

Las variables se leen de `sistema_xia/.env`. Las que ya existan en el entorno tienen prioridad.

| Variable | Uso |
|---|---|
| `LLM_HOST` | URL base del servidor compatible con OpenAI. Debe incluir `/v1`. |
| `LLM_API_KEY` | Clave del servidor. Con Ollama puede ser cualquier valor. |
| `LLM_MODEL` | Modelo del chatbot (con tool calling). |
| `AUDIT_LLM_MODEL` | Modelo del agente auditor (con JSON mode). |
| `LLM_TIMEOUT`, `LLM_TEMPERATURE` | Tiempo máximo en segundos y temperatura. |
| `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS` | Parámetros estándar de Django. |

Para usar Ollama en local, instala el servidor, descarga los modelos que pongas en `.env` (`ollama pull <modelo>`) y déjalo corriendo (`ollama serve`).

## Entrenar el modelo de riesgo

El chatbot no trae un modelo preentrenado: necesita al menos un entrenamiento completado.

1. Entra al admin como superusuario y abre el modelo `MLModelTraining` de la app `ml_models`.
2. Crea un registro subiendo `notebooks/archivos/credit_risk_dataset.csv` y ajusta `optuna_trials` y `optuna_timeout`.
3. Guarda. El entrenamiento corre en un hilo aparte y el estado pasa de `pending` a `training` y luego a `completed` o `failed` (el detalle del error queda en `log`).

El CSV debe tener las columnas originales del dataset de Kaggle (`person_age`, `person_income`, `loan_status`, etc.). El código las renombra internamente.

Al terminar, el chatbot usa automáticamente el último entrenamiento `completed`. Cada entrenamiento también se guarda en `models/training_<id>/` (formato MLflow), en la raíz del repositorio.

## Cómo funciona

### 1. Flujo del chat

1. El navegador envía los mensajes a `POST /api/chat/stream/`.
2. La vista recorta el historial a los últimos 20 mensajes y devuelve la respuesta en NDJSON (streaming).
3. `ChatbotAgent` ejecuta un bucle de *tool calling* con el LLM. Si el modelo decide evaluar a un cliente, llama a la herramienta `risk_ai`, que ejecuta `predecir_riesgo()` con el modelo entrenado. El resultado vuelve al LLM para redactar la respuesta.
4. Solo se piden siete datos no sensibles: ingresos, años de trabajo, tipo de vivienda, monto, motivo, historial de mora y años de historial crediticio. La edad, el género y la etnia no se usan.

### 2. Agentes

- `BaseAgent` (`sistema_xia/agents/base_agent.py`): cliente OpenAI, registro de herramientas, bucle de *tool calling*, auditoría y registro de transacciones.
- `ChatbotAgent` (`sistema_xia/agents/chatbot_agent.py`): analista financiero limitado a riesgo crediticio, con la herramienta `risk_ai`.
- Agente auditor: tras cada respuesta final, `BaseAgent.audit_compliance()` pide al modelo `AUDIT_LLM_MODEL` un JSON con el nivel de riesgo, la base legal y si la respuesta cumple las reglas del agente. En el chat por streaming el texto ya se envió, así que un fallo de auditoría solo queda registrado con estado `blocked`.

### 3. Pipeline de entrenamiento (`ml_models/utils.py`)

1. Renombra columnas y convierte el target: `riesgo_crediticio = 1 - loan_status`, así que **1 = Aprobado** y **0 = Riesgo**.
2. Excluye `edad`, `tasa_interes` y `grado_prestamo` para evitar fuga de información.
3. Divide los datos 60/20/20 de forma estratificada (semilla 42).
4. Preprocesamiento con `ColumnTransformer`: imputación de numéricas (media), limpieza y `OneHotEncoder` para categóricas.
5. Optimiza XGBoost, LightGBM y Random Forest con Optuna. **Nota:** en el código actual, Optuna puntúa cada prueba con F1 sobre el conjunto de prueba, no sobre el de validación.
6. Elige el mejor modelo por *accuracy* en el conjunto de validación.
7. Calcula los valores SHAP medios por variable (`TreeExplainer`) y los guarda en el registro.
8. Guarda el modelo en la base de datos (joblib) y en `models/` (MLflow).

### 4. Explicabilidad (XAI)

- **Global:** importancia media |SHAP| por variable, guardada al entrenar.
- **Local:** para cada predicción, `risk_motivation` calcula la contribución de cada variable y devuelve el top 10 en porcentaje. Se muestra en el chat como tabla.

### 5. Auditoría y cumplimiento

- Cada consulta crea un `AgentTransactionLog` con los payloads, la latencia, el modelo usado y el resultado de la auditoría.
- `/admin/audit/agenttransactionlog/dashboard/` muestra el dashboard de cumplimiento (solo staff).
- Los logs son de solo lectura en el admin.

### 6. Control de acceso

- Registro en `/accounts/register/`: el usuario entra al grupo `public_user`, que le da permisos sobre sus chats y mensajes.
- Usuarios anónimos pueden chatear, pero su conversación no se guarda.
- Staff: admin, entrenamiento y dashboards.

## Replicar el notebook

`notebooks/train_model.ipynb` contiene el mismo pipeline en formato de notebook. Ejecútalo desde la carpeta `notebooks/`, porque lee `archivos/credit_risk_dataset.csv`. Ejecuta los pasos en el orden del notebook.

Las salidas del notebook muestran la ruta de la máquina donde se ejecutó, reemplazada por `<ruta-del-proyecto>`.

## Limitaciones conocidas

- El dataset es público (Kaggle). Revisa su licencia antes de redistribuirlo.
- La selección del modelo usa el conjunto de validación, pero Optuna optimiza F1 sobre el conjunto de prueba (ver paso 5 del pipeline).
- El chatbot depende de un LLM externo o local. Sin él, el chat no responde.
- Los logos de la interfaz (`static/brand/`) no se incluyen. Las plantillas los referencian, así que esas imágenes no cargarán.
- El proyecto no está pensado para producción sin revisar `SECRET_KEY`, `DEBUG` y `ALLOWED_HOSTS`.
