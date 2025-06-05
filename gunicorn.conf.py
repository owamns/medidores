import os
from dotenv import load_dotenv

load_dotenv()

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Device detection
device = os.environ.get('DEVICE', 'auto').lower()
is_gpu = device in ('gpu', 'cuda') or (device == 'auto' and 'CUDA_VISIBLE_DEVICES' in os.environ)

if is_gpu:
    # GPU-specific configuration
    workers = int(os.environ.get('GUNICORN_GPU_WORKERS', 1))
    worker_class = os.environ.get('GUNICORN_GPU_WORKER_CLASS', 'gthread')
    threads = int(os.environ.get('GUNICORN_GPU_THREADS', 4))
    timeout = int(os.environ.get('GUNICORN_GPU_TIMEOUT', 600))
    keepalive = int(os.environ.get('GUNICORN_GPU_KEEPALIVE', 120))
    max_requests = int(os.environ.get('GUNICORN_GPU_MAX_REQUESTS', 500))
    max_requests_jitter = int(os.environ.get('GUNICORN_GPU_MAX_REQUESTS_JITTER', 50))
    graceful_timeout = int(os.environ.get('GUNICORN_GPU_GRACEFUL_TIMEOUT', 120))
    preload_app = False
else:
    # CPU-specific configuration
    workers = int(os.environ.get('GUNICORN_CPU_WORKERS', 3))
    worker_class = os.environ.get('GUNICORN_CPU_WORKER_CLASS', 'sync')
    threads = int(os.environ.get('GUNICORN_CPU_THREADS', 2))
    timeout = int(os.environ.get('GUNICORN_CPU_TIMEOUT', 120))
    keepalive = int(os.environ.get('GUNICORN_CPU_KEEPALIVE', 60))
    max_requests = int(os.environ.get('GUNICORN_CPU_MAX_REQUESTS', 1000))
    max_requests_jitter = int(os.environ.get('GUNICORN_CPU_MAX_REQUESTS_JITTER', 100))
    graceful_timeout = int(os.environ.get('GUNICORN_CPU_GRACEFUL_TIMEOUT', 60))
    preload_app = True

# Shared settings
worker_tmp_dir = '/dev/shm'
accesslog = '-' if os.environ.get('FLASK_ENV') == 'production' else None
errorlog = '-'
loglevel = os.environ.get('LOG_LEVEL', 'info').lower()
access_log_format = os.environ.get('ACCESS_LOG_FORMAT',
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s')
proc_name = 'meter-processing-api'
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = 1000
group = 1000
tmp_upload_dir = None

# Gunicorn hooks with device-specific logging
def on_starting(server):
    server.log.info(f"=== METER PROCESSING API WITH {'GPU' if is_gpu else 'CPU'} ===")
    server.log.info(f"Initializing {'GPU' if is_gpu else 'CPU'} resources...")

def when_ready(server):
    server.log.info(f"=== METER PROCESSING API READY ({'GPU' if is_gpu else 'CPU'} MODE) ===")
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Worker class: {worker_class}")
    server.log.info(f"Threads: {threads}")
    server.log.info(f"Timeout: {timeout}s")
    server.log.info(f"Graceful timeout: {graceful_timeout}s")
    server.log.info(f"Max requests per worker: {max_requests}")
    server.log.info(f"Bind: {bind}")
    server.log.info(f"{'GPU' if is_gpu else 'CPU'} resources initialized. Server ready.")

def worker_int(worker):
    worker.log.info(f"Worker received INT or QUIT signal - cleaning {'GPU' if is_gpu else 'CPU'} resources")

def pre_fork(server, worker):
    server.log.info(f"Worker about to spawn (pid: {worker.pid}) in {'GPU' if is_gpu else 'CPU'} mode")

def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid}) in {'GPU' if is_gpu else 'CPU'} mode")

def post_worker_init(worker):
    worker.log.info(f"Worker initialized with {'GPU' if is_gpu else 'CPU'} access (pid: {worker.pid})")

def worker_abort(worker):
    worker.log.info(f"Worker received SIGABRT signal - force cleaning {'GPU' if is_gpu else 'CPU'} resources")

def on_exit(server):
    server.log.info(f"Server shutting down - cleaning up {'GPU' if is_gpu else 'CPU'} resources")

limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190