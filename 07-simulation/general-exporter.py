from prometheus_client import start_http_server, Gauge, REGISTRY
import time
import psutil
import threading

# Metrics
elapsed_time = Gauge('total_elapsed_time', 'Total elapsed time since port 11001 started listening')
finished_jobs = Gauge('total_finished_jobs', 'Number of finished jobs based on smallest open port > 11000')

# Track whether port 11001 has started listening
timer_started = False
start_time = None

def is_port_open(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
            return True
    return False

def get_open_ports_above_11000():
    ports = set()
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port > 11000:
            ports.add(conn.laddr.port)
    return ports

def update_metrics():
    global timer_started, start_time
    while True:
        # Start timer when port 11001 starts listening
        if not timer_started and is_port_open(11001):
            timer_started = True
            start_time = time.time()

        # Update elapsed_time if the timer has started
        if timer_started:
            seconds_elapsed = int(time.time() - start_time)
            elapsed_time.set(seconds_elapsed)

        # Update finished_jobs based on smallest open port > 11000
        open_ports = get_open_ports_above_11000()
        smallest = min(open_ports) if open_ports else None
        if smallest and smallest > 11001:
            finished_jobs.set(smallest - 11001)
        else:
            finished_jobs.set(0)

        time.sleep(1)

def deregister_unwanted_metrics():
    unwanted_metrics = [
        'python_gc_objects_uncollectable_total',
        'python_gc_collections_total',
        'python_info',
        'process_virtual_memory_bytes',
        'process_resident_memory_bytes',
        'process_start_time_seconds',
        'process_cpu_seconds_total',
        'process_open_fds',
        'process_max_fds'
    ]
    for metric_name in unwanted_metrics:
        if metric_name in REGISTRY._names_to_collectors:
            REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])

if __name__ == '__main__':
    # Expose metrics at http://localhost:65500/metrics
    deregister_unwanted_metrics()
    start_http_server(65500)
    threading.Thread(target=update_metrics).start()

