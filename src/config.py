MODEL_EXPLORER = "claude-haiku-4-5"
MODEL_SCOUT    = "claude-haiku-4-5"
MODEL_CRITIC   = "claude-haiku-4-5"
MODEL_WRITER   = "claude-haiku-4-5"

MAX_TOOL_CALLS   = 10
MAX_REVISIONS    = 2
MAX_TOKENS       = 1200
MAX_TOKENS_WRITER = 3500

MIN_MINUTES      = 900
SHRINKAGE_K      = 900
MIN_MARKET_VALUE = 100000

# Faz B-C'de dogrulanan value_residuals() varyanti (raw npxg_per90 ile,
# 4-lig birlesik modelde R-squared=0.4625 - bkz. reports/npxg_vs_goals_faz_b.md).
# Scout ajaninin run_value_residuals araci bunu SABIT kullanir - LLM'e perf_col
# secme sansi TOOLS semasindan CIKARILARAK verilmez (bkz. src/agents/scout.py),
# cunku farkli bir perf_col (orn. npxg_per90_pct) hic dogrulanmamis, sessizce
# farkli bir regresyon varyantina gecmek anlamina gelirdi.
DEFAULT_PERF_COL = "npxg_per90"
