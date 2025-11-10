"""
Job Monitor Helper
Monitors and displays Databricks job progress.
"""

import time
from databricks.sdk import WorkspaceClient


def monitor_job_progress(run_id: int, update_interval: int = 30):
    """
    Monitor a Databricks job run and print progress updates.

    Args:
        run_id: The Databricks run ID to monitor
        update_interval: Seconds between status checks (default: 30)

    Returns:
        tuple: (success: bool, duration_seconds: int, result_state: str)
    """
    w = WorkspaceClient()

    print(f"⏳ Monitoring job progress...")
    print(f"   (Updates every {update_interval} seconds)\n")

    start_time = time.time()
    last_state = None
    last_task_status = {}

    while True:
        run_info = w.jobs.get_run(run_id=run_id)
        state = run_info.state
        life_cycle_state = state.life_cycle_state.value

        # Print state changes
        if life_cycle_state != last_state:
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Job status: {life_cycle_state}")
            last_state = life_cycle_state

        # Print task progress
        if run_info.tasks:
            for task in run_info.tasks:
                task_key = task.task_key
                task_state = task.state.life_cycle_state.value if task.state else "PENDING"

                if task_key not in last_task_status or last_task_status[task_key] != task_state:
                    elapsed = int(time.time() - start_time)
                    status_icon = "⏳" if task_state == "RUNNING" else "✅" if task_state == "TERMINATED" else "⏸️"
                    print(f"[{elapsed}s] {status_icon} {task_key}: {task_state}")
                    last_task_status[task_key] = task_state

        # Check if job is done
        if life_cycle_state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
            result_state = state.result_state.value if state.result_state else "UNKNOWN"
            elapsed = int(time.time() - start_time)

            print()
            if result_state == "SUCCESS":
                print(f"✅ Job completed successfully!")
                print(f"   Duration: {elapsed // 60}m {elapsed % 60}s")
                return True, elapsed, result_state
            else:
                print(f"❌ Job failed with state: {result_state}")
                print(f"   Duration: {elapsed // 60}m {elapsed % 60}s")
                if state.state_message:
                    print(f"   Error: {state.state_message}")
                return False, elapsed, result_state

        # Wait before next check
        time.sleep(update_interval)
