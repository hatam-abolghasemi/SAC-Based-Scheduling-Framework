import re

job_line_re = re.compile(
    r"\[JobStart\]\s+job_id=(\d+)\s+gen_id=(\d+)\s+port=(\d+)\s+node=([\w\-]+)"
    r"\s+gen_moment=(\d+)\s+sched_moment=(\d+)\s+epochs=(\d+)"
    r"\s+image=([\w\-]+)\s+container=([\w\-]+)"
)

def parse_log_file(log_path):
    jobs = []
    with open(log_path, 'r') as file:
        for line in file:
            match = job_line_re.search(line)
            if match:
                sched_moment = int(match.group(6))
                job = {
                    'sched_moment': sched_moment,
                    'job_id': int(match.group(1)),
                    'gen_id': int(match.group(2)),
                    'port': int(match.group(3)),
                    'node': match.group(4),
                    'epochs': int(match.group(7)),
                    'image': match.group(8),
                    'container': match.group(9)
                }
                jobs.append(job)
    return sorted(jobs, key=lambda j: j['sched_moment'])

def write_relative_time_pattern(jobs, output_path):
    if not jobs:
        print("No jobs found in the log.")
        return

    base_time = jobs[0]['sched_moment'] - 1

    with open(output_path, 'w') as f:
        for job in jobs:
            relative_time = job['sched_moment'] - base_time
            line = (
                f"{relative_time} "
                f"job_id={job['job_id']} gen_id={job['gen_id']} port={job['port']} "
                f"node={job['node']} epochs={job['epochs']} "
                f"image={job['image']} container={job['container']}"
            )
            f.write(line + "\n")

def main():
    log_path = 'job-exporter-manager-docker.log'  # change this if needed
    output_path = 'job_pattern.txt'

    jobs = parse_log_file(log_path)
    write_relative_time_pattern(jobs, output_path)

    print(f"Relative-time job pattern written to {output_path}")

if __name__ == '__main__':
    main()

