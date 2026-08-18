# RAGAS Evaluation API

API REST modular para evaluar respuestas de sistemas RAG. Recibe una pregunta, la respuesta generada, los contextos recuperados y una respuesta de referencia opcional; devuelve metricas calculadas realmente por RAGAS y OpenAI.

El proyecto esta pensado para ejecutarse primero de forma local y luego copiarse a otro host. No configura n8n, no modifica otros contenedores y no despliega servicios automaticamente.

## Arquitectura

```text
n8n (futuro) -> HTTP POST /ragas/eval -> FastAPI -> RAGAS -> OpenAI
                                             |
                                             +-> JSON con metricas
```

La capa HTTP esta separada de la configuracion, los schemas y el servicio de evaluacion. Esta separacion permite incorporar mas adelante Django, PostgreSQL, autenticacion, usuarios, historial, dashboard y administracion sin reescribir la evaluacion RAGAS.

## Que es RAGAS

[RAGAS](https://docs.ragas.io/) es un framework de evaluacion para aplicaciones basadas en RAG y LLM. Este proyecto fija **RAGAS 0.4.3** y utiliza exclusivamente su API moderna:

- `ragas.metrics.collections`
- `ragas.llms.llm_factory`
- `ragas.embeddings.OpenAIEmbeddings`
- metodos asincronos `ascore()` y valores `MetricResult.value`

No usa `evaluate()`, `SingleTurnSample`, wrappers de LangChain ni imports legacy, porque en RAGAS 0.4 la API recomendada es la API de colecciones.

## Metricas implementadas

| Metrica | Que mide | Requiere `reference` |
|---|---|---|
| `faithfulness` | Consistencia factual de la respuesta con los contextos recuperados | No |
| `answer_relevancy` | Relevancia de la respuesta respecto de la pregunta | No |
| `context_precision` | Calidad y orden de los contextos recuperados respecto de la referencia | Si |
| `context_recall` | Cobertura de la informacion de referencia por los contextos | Si |

Cuando no se envia `reference`, `context_precision` y `context_recall` se devuelven como `null`; no se inventan scores. Cualquier `NaN`, infinito positivo o infinito negativo producido legitimamente por una metrica tambien se normaliza a `null`, para garantizar JSON valido.

## Estructura

```text
ragas-api/
|-- app/
|   |-- api/routes.py
|   |-- services/ragas_service.py
|   |-- config.py
|   `-- schemas.py
|-- tests/
|   |-- conftest.py
|   |-- test_api.py
|   |-- test_service.py
|   `-- test_integration.py
|-- .env.example
|-- .gitignore
|-- .dockerignore
|-- Dockerfile
|-- docker-compose.yml
|-- main.py
|-- requirements.txt
`-- README.md
```

## Requisitos

- Python 3.11
- Una API key de OpenAI para evaluaciones reales
- Docker y Docker Compose v2, solo si se usara la ejecucion en contenedor

## Instalacion local

Copie primero el archivo de configuracion:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows CMD:

```bat
copy .env.example .env
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configuracion `.env`

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAGAS_TIMEOUT=120
LOG_LEVEL=INFO
```

- `OPENAI_API_KEY`: obligatoria para `POST /ragas/eval`. Nunca aparece en `/config`, logs ni respuestas.
- `OPENAI_MODEL`: modelo evaluador utilizado por RAGAS.
- `OPENAI_EMBEDDING_MODEL`: embeddings usados por Answer Relevancy.
- `RAGAS_TIMEOUT`: limite total de una evaluacion y timeout del cliente OpenAI, en segundos.
- `LOG_LEVEL`: nivel estandar de logging, por ejemplo `INFO` o `DEBUG`.

No confirme `.env` al control de versiones; esta incluido en `.gitignore`.

## Ejecucion con Python

Con el entorno virtual activado:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8800
```

Comprobacion rapida:

```bash
curl http://localhost:8800/health
curl http://localhost:8800/config
```

`/config` solo informa si OpenAI esta configurado y los nombres de modelos; nunca devuelve la API key.

## Ejecucion con Docker

1. Copie `.env.example` como `.env` y configure `OPENAI_API_KEY`.
2. Valide y construya:

```bash
docker compose config
docker compose build
```

3. Inicie el servicio local:

```bash
docker compose up -d
docker compose logs -f ragas-api
```

El puerto interno es `8000` y Compose publica `8800` en el host. El contenedor no usa modo privilegiado y se ejecuta con un usuario sin privilegios.

Para detenerlo:

```bash
docker compose down
```

## Endpoints y ejemplos curl

### `GET /health`

```json
{"status":"ok","service":"ragas-api"}
```

### `GET /config`

```json
{
  "openai_configured": true,
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small"
}
```

### `POST /ragas/eval` con referencia

```bash
curl -X POST "http://localhost:8800/ragas/eval" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cual es la capital de Chile?",
    "answer": "La capital de Chile es Santiago.",
    "contexts": [
      "Santiago es la capital y principal ciudad de Chile."
    ],
    "reference": "Santiago"
  }'
```

Ejemplo de respuesta:

```json
{
  "status": "success",
  "question": "¿Cual es la capital de Chile?",
  "answer": "La capital de Chile es Santiago.",
  "context_items": 1,
  "reference_provided": true,
  "metrics": {
    "faithfulness": 0.98,
    "answer_relevancy": 0.95,
    "context_precision": 1.0,
    "context_recall": 1.0
  },
  "evaluation_time_ms": 2450
}
```

Los valores anteriores solo ilustran el formato; la API devuelve los scores que RAGAS calcule realmente en cada solicitud.

### Alias compatible `context` y referencia opcional

```bash
curl -X POST "http://localhost:8800/ragas/eval" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cual es la capital de Chile?",
    "answer": "La capital de Chile es Santiago.",
    "context": [
      "Santiago es la capital y principal ciudad de Chile."
    ]
  }'
```

Se acepta `context` o `contexts`, pero no ambos en la misma solicitud. Internamente siempre se normaliza a `contexts: list[str]`.

## Swagger y OpenAPI

Con la API iniciada:

- Swagger UI: <http://localhost:8800/docs>
- Esquema OpenAPI: <http://localhost:8800/openapi.json>

## Tests

Los tests unitarios mockean la capa RAGAS/OpenAI y no consumen tokens ni requieren una API key:

```bash
pytest
```

El test de integracion real esta desactivado por defecto. Para habilitarlo deliberadamente:

Windows PowerShell:

```powershell
$env:RUN_INTEGRATION_TESTS = "true"
$env:OPENAI_API_KEY = "su-api-key"
pytest tests/test_integration.py -v
```

Linux/macOS:

```bash
RUN_INTEGRATION_TESTS=true OPENAI_API_KEY="su-api-key" pytest tests/test_integration.py -v
```

Este test hace llamadas reales a OpenAI y puede generar costos.

## Integracion futura con n8n

No se configura n8n desde este proyecto. Cuando la API se encuentre en el host elegido, agregue un nodo **HTTP Request** con:

- Method: `POST`
- URL: `http://SERVIDOR:8800/ragas/eval`
- Send Body: activado
- Body Content Type: JSON

Body:

```json
{
  "question": "{{$json.question}}",
  "answer": "{{$json.answer}}",
  "contexts": {{$json.contexts}},
  "reference": "{{$json.reference}}"
}
```

Si el flujo no dispone de referencia, omita completamente la propiedad `reference` o envie `null`. Asegurese de que `contexts` sea un array JSON, no un string que contenga JSON serializado.

## Validacion y errores

- Payload invalido: HTTP `422` generado por Pydantic.
- `OPENAI_API_KEY` ausente: HTTP `503`, `configuration_error`.
- Error de OpenAI o indisponibilidad: HTTP `503`, `openai_error`.
- Timeout: HTTP `504`, `evaluation_timeout`.
- Error de RAGAS: HTTP `500`, `evaluation_failed`.
- Error interno inesperado: HTTP `500`, `internal_error`.

Formato de error de la aplicacion:

```json
{
  "status": "error",
  "error": "evaluation_failed",
  "detail": "Descripcion segura del error"
}
```

## Logging

Se registra el inicio de cada evaluacion, cantidad de contextos, presencia de referencia, metricas solicitadas, duracion y errores. La pregunta, respuesta, contextos y `OPENAI_API_KEY` no se registran, para reducir la exposicion de datos sensibles.

## Troubleshooting

### `/ragas/eval` devuelve 503 `configuration_error`

Verifique que `.env` exista, que `OPENAI_API_KEY` no este vacia y reinicie Uvicorn o el contenedor despues de cambiarla.

### OpenAI devuelve 401, 429 o errores de conexion

Revise la validez de la key, permisos del proyecto, cuota, limites de tasa y conectividad saliente HTTPS. La API traduce esos fallos a `openai_error` sin exponer la key.

### La evaluacion supera el tiempo disponible

Aumente `RAGAS_TIMEOUT`. Las cuatro metricas realizan varias llamadas al modelo; con referencia el trabajo es mayor. Revise tambien latencia y limites de tasa de OpenAI.

### Docker no puede conectarse al daemon

Inicie Docker Desktop o el servicio Docker y compruebe `docker info`. `docker compose config` solo valida YAML y puede funcionar aunque el daemon este detenido.

### El puerto 8800 esta ocupado

En ejecucion Python, elija otro puerto con `--port`. En Compose, cambie solo el lado izquierdo de `"8800:8000"`, por ejemplo `"8801:8000"`.

### PowerShell bloquea la activacion del entorno

Puede ejecutar directamente `.venv\Scripts\python.exe -m pytest` o ajustar la politica solo para la sesion actual mediante `Set-ExecutionPolicy -Scope Process RemoteSigned`.

## Despliegue posterior en otro servidor

El despliegue es manual y queda bajo control del operador:

1. Copie la carpeta del proyecto al servidor elegido, excluyendo `.venv`, `.env`, caches y artefactos locales.
2. Instale Docker/Compose o Python 3.11.
3. Cree un `.env` nuevo en el servidor y configure una API key mediante un canal seguro.
4. Ejecute `docker compose config` y `docker compose build`, o cree un entorno virtual e instale `requirements.txt`.
5. Inicie la API y compruebe `/health`, `/config`, `/docs` y una evaluacion controlada.
6. Restrinja red, firewall y acceso al puerto segun su entorno. Esta primera version no incorpora autenticacion; no exponga el endpoint directamente a Internet sin un control de acceso delante.
7. Configure el nodo HTTP Request de n8n solo despues de validar la API en ese host.

No copie el archivo `.env` de desarrollo ni incluya secretos dentro de la imagen Docker.
