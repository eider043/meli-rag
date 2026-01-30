# RAG + Agente Crítico para QA sobre Laptops

## Objetivo
Construir un sistema **RAG (Retrieval-Augmented Generation)** que responda preguntas sobre laptops a partir de un dataset técnico, recuperando pasajes relevantes y generando respuestas **con citas explícitas**, verificadas por un **agente crítico** que garantiza que cada afirmación esté respaldada por evidencia.

El sistema incluye:
- Indexación ligera (BM25)
- Generación con LLM
- Agente crítico de verificación (faithfulness)
- Evaluación cuantitativa del retrieval y la generación

---

## Dataset
- **Fuente:** Kaggle  
- **Archivo:** `Laptops_with_technical_specifications.csv`
- **Subconjunto:** ~300 laptops (para optimizar velocidad)
- Cada fila representa un modelo de laptop con especificaciones técnicas.

---

## Arquitectura de la Solución

**Flujo principal:**

1. **Ingesta y normalización**
   - Limpieza básica y estandarización de columnas
2. **Chunking**
   - Pasajes cortos (50–120 tokens aprox.)
   - Cada chunk conserva `laptop_id` y `campo`
3. **Indexación**
   - Índice BM25 sobre los chunks
4. **Retrieve**
   - Top-k = 5 chunks más relevantes
5. **Generate**
   - Respuesta generada con LLM (`llm_openai.py`)
   - Máx. 120 palabras
   - Citas obligatorias `[laptop_id:campo]`
6. **Agente Crítico**
   - Verifica que **cada oración** tenga soporte en los chunks
   - Reintenta o descarta la respuesta si no es fiel
7. **Logs y Evaluación**
   - Logs JSONL
   - Métricas de retrieval y generación

---

## Agente Crítico (Obligatorio)
El agente crítico:
- Divide la respuesta en oraciones
- Verifica:
  - Presencia de citas
  - Que las citas pertenezcan a chunks recuperados
  - Overlap semántico entre oración y evidencia
- Calcula **Faithfulness** como:
  > % de oraciones respaldadas por evidencia

Si detecta fallas:
- Reintenta la generación con reglas más estrictas
- O descarta la respuesta

---

## Evaluación

Se evaluaron **8 queries curadas** usando las siguientes métricas:

### Métricas de Retrieval
- **Precision@5**
- **Recall@5**

### Métricas de Generación
- **Faithfulness:** % de oraciones con soporte
- **Answer Coverage:** % de chunks citados en la respuesta
- **critic_ok:** aceptación final del agente crítico

### Resultados

| Métrica | Valor promedio |
|------|------|
| Precision@5 | **0.675** |
| Recall@5 | **0.675** |
| Faithfulness | **0.646** |
| Answer Coverage | **0.375** |
| critic_ok | **0.50** |

---

## Análisis de Resultados

- El **retrieval es sólido** (Precision y Recall ~0.67), lo que indica que el índice BM25 recupera correctamente laptops relevantes.
- El **faithfulness es moderado (~65%)**, mostrando que el LLM a veces introduce afirmaciones no completamente respaldadas.
- Solo el **50% de las respuestas fueron aceptadas** por el agente crítico, validando su rol como mecanismo de control de calidad.
- Las queries más complejas (precio/calidad, batería, GPU) presentan menor cobertura y mayor tasa de rechazo.

---

## Conclusiones

- El sistema cumple correctamente con el paradigma **RAG + Agente Crítico**.
- El agente crítico es clave para detectar alucinaciones y evitar respuestas sin soporte.
- Existe un trade-off natural entre **fluidez** y **fidelidad**, controlado eficazmente por el agente.
- El enfoque es extensible a otros dominios técnicos (e-commerce, catálogos, fichas técnicas).

---

## Posibles Mejoras
- Usar embeddings semánticos híbridos (BM25 + embeddings)
- Ajustar el prompt para reducir afirmaciones implícitas
- Enriquecer el dataset con campos normalizados (RAM, GPU, batería)
- Integrar MCP para exponer el índice como servicio

---

## Estructura del Proyecto

├── main.py
├── run_eval_batch.py
├── data/
│ ├── Laptops_with_technical_specifications.csv
│ └── eval_queries.json
├── src/
│ ├── data_loader.py
│ ├── chunking.py
│ ├── retriever_bm25.py
│ ├── generator.py
│ ├── llm_openai.py
│ ├── critic_agent.py
│ ├── pipeline.py
│ └── evaluation.py
└── outputs/
├── eval_runs.jsonl
├── metrics_eval.csv
├── critic_logs.jsonl


---

## Entregables Cumplidos
- RAG funcional con citas
- Agente crítico con logs
- Evaluación cuantitativa
- Ejemplos trazables query → chunks → respuesta

## Conclusión corta (por si la necesitas aparte)

El sistema demuestra que un RAG con agente crítico reduce significativamente respuestas no fundamentadas. Aunque el retrieval es robusto, el agente evidencia que la generación libre del LLM puede introducir afirmaciones no respaldadas, validando la necesidad de mecanismos automáticos de verificación en sistemas de QA técnicos.

## Diagrama de Arquitectura (lógico)

Componentes:

Usuario
  ↓
Query
  ↓
BM25 Retriever
  ↓
Top-K Chunks
  ↓
LLM Generator (OpenAI)
  ↓
Respuesta con citas
  ↓
Agente Crítico
  ├─ OK → Respuesta Final
  └─ FAIL → Reintento / Descarte
