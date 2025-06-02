import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes - Configuración centralizada
workers = int(os.environ.get('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
max_workers = int(os.environ.get('GUNICORN_MAX_WORKERS', 6))
workers = min(workers, max_workers)

worker_class = 'sync'
worker_connections = 1000
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 300))
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', 60))

# Restart workers after this many requests
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', 1000))
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', 50))

# Preload app para compartir memoria entre workers
preload_app = True

# Logging - Configuración centralizada
accesslog = '-' if os.environ.get('FLASK_ENV') == 'production' else None
errorlog = '-'
loglevel = os.environ.get('LOG_LEVEL', 'info').lower()
access_log_format = os.environ.get('ACCESS_LOG_FORMAT',
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s')

# Process naming
proc_name = 'meter-processing-api'

# Server mechanics
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = 1000
group = 1000
tmp_upload_dir = None

# Configuración de memoria y límites
worker_tmp_dir = '/dev/shm'  # Usar memoria compartida para temp files

def when_ready(server):
    server.log.info("=== METER PROCESSING API ===")
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Timeout: {timeout}s")
    server.log.info(f"Max requests per worker: {max_requests}")
    server.log.info(f"Bind: {bind}")
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")