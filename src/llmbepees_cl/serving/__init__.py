"""Serving (Fase 5): benchmark latensi + API FastAPI.

`benchmark` tak butuh FastAPI. App HTTP ada di `llmbepees_cl.serving.app` (extra
`serve`) — impor di sana supaya paket inti tetap bisa dipakai tanpa FastAPI:
`from llmbepees_cl.serving.app import create_app`.
"""

from llmbepees_cl.serving.benchmark import BenchResult, benchmark_ask, percentile

__all__ = ["BenchResult", "benchmark_ask", "percentile"]
