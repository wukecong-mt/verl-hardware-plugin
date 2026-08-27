#!/usr/bin/env bash

set -euo pipefail

ray_api_address="${1:-${RAY_API_SERVER_ADDRESS:-${RAY_ADDRESS:-http://127.0.0.1:8265}}}"

python3 - "${ray_api_address}" <<'PY'
import sys
import time

from ray.dashboard.modules.job.common import JobStatus
from ray.job_submission import JobSubmissionClient


address = sys.argv[1]
client = JobSubmissionClient(address)

running_job_ids = [
    job.submission_id
    for job in client.list_jobs()
    if job.status == JobStatus.RUNNING and job.submission_id is not None
]

if not running_job_ids:
    print(f"No RUNNING Ray jobs found at {address}.")
    raise SystemExit(0)

print(f"Stopping {len(running_job_ids)} RUNNING Ray job(s) at {address}:")
for job_id in running_job_ids:
    print(f"  {job_id}")
    client.stop_job(job_id)

deadline = time.monotonic() + 120
pending = set(running_job_ids)
terminal_statuses = {JobStatus.STOPPED, JobStatus.SUCCEEDED, JobStatus.FAILED}

while pending and time.monotonic() < deadline:
    for job_id in tuple(pending):
        status = client.get_job_status(job_id)
        if status in terminal_statuses:
            print(f"{job_id}: {status}")
            pending.remove(job_id)
    if pending:
        time.sleep(1)

if pending:
    print(
        "Timed out waiting for these jobs to stop: " + ", ".join(sorted(pending)),
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
