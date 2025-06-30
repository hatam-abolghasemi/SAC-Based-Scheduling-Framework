from prometheus_client import start_http_server, Gauge, REGISTRY
import time
import psutil
import threading

# Metrics
elapsed_time = Gauge('total_elapsed_time', 'Total elapsed time since port 11001 started listening')
finished_jobs = Gauge('total_finished_jobs', 'Number of finished jobs based on missing ports > 11000')

# Timer state
timer_started = False
start_time = None

# Set of all ports that were ever seen open
seen_ports = set()

def is_port_open(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
            return True
    return False

def get_open_ports_above_11000():
    return {conn.laddr.port for conn in psutil.net_connections(kind='inet')
            if conn.status == psutil.CONN_LISTEN and conn.laddr.port > 11000}

def update_metrics():
    global timer_started, start_time, seen_ports

    while True:
        # Start the timer once 11001 is open
        if not timer_started and is_port_open(11001):
            timer_started = True
            start_time = time.time()

        # Update elapsed time
        if timer_started:
            elapsed_time.set(int(time.time() - start_time))

        # Get current open ports >11000
        current_ports = get_open_ports_above_11000()

        # Update the set of all seen ports
        seen_ports.update(current_ports)

        # Calculate finished jobs = ports we've seen that are no longer open
        finished = seen_ports - current_ports
        finished_jobs.set(len(finished))

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
    deregister_unwanted_metrics()
    start_http_server(9809)
    threading.Thread(target=update_metrics).start()

