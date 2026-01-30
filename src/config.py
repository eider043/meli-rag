# ==============================
# Configuración del proyecto RAG
# ==============================

# Ruta al CSV (ajusta según tu repo)
DATA_PATH = r"data\Laptops_with_technical_specifications.csv"

# Subconjunto para velocidad (recomendado 200-400)
SUBSET_N = 300
RANDOM_SEED = 42

# Chunks cortos (aprox. tokens por palabras)
CHUNK_MIN_TOKENS = 50
CHUNK_MAX_TOKENS = 120

# Retrieve / Generate
TOP_K = 5                 # 3-5 recomendado
MAX_ANSWER_WORDS = 120    # <= 120 palabras (requisito)

# Evaluación
EVAL_QUERIES_PATH = r"data\eval_queries.json"

# Outputs / logs
OUTPUT_DIR = "outputs"
RUNS_PATH = r"outputs\runs.jsonl"
CRITIC_LOGS_PATH = r"outputs\critic_logs.jsonl"
INDEX_PREVIEW_PATH = r"outputs\index_preview.csv"
