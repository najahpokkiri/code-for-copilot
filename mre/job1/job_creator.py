"""
Job Creator Helper
Creates and configures the Databricks job for the building enrichment pipeline.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, TaskDependency, SparkPythonTask, Library


def create_databricks_job(
    iso3: str,
    cluster_id: str,
    workspace_base: str,
    catalog: str,
    schema: str,
    volume_base: str,
    minimal_config_path: str,
    email: str = None
):
    """
    Create a Databricks job for the building enrichment pipeline.

    Args:
        iso3: Country ISO3 code
        cluster_id: Databricks cluster ID to use
        workspace_base: Workspace path where scripts are located
        catalog: Databricks catalog name
        schema: Databricks schema name
        volume_base: Volume root path
        minimal_config_path: Path to minimal config file in workspace
        email: Optional email for notifications

    Returns:
        tuple: (job_id, job_name)
    """
    w = WorkspaceClient()

    job_name = f"Building_Enrichment_{iso3}"
    requirements_path = f"{workspace_base}/requirements.txt"
    generated_config_path = f"{volume_base}/{iso3}/config.json"

    # Define all 8 tasks
    tasks = [
        # Task 0: Setup & config generation
        Task(
            task_key="task0_setup",
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task0_setup.py",
                parameters=["--minimal_config", minimal_config_path]
            ),
            libraries=[Library(requirements=requirements_path)]
        ),
        # Task 1: Proportions to Delta
        Task(
            task_key="task1_proportions_to_delta",
            depends_on=[TaskDependency(task_key="task0_setup")],
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task1_proportions_to_delta.py",
                parameters=["--config_path", generated_config_path]
            ),
            libraries=[Library(requirements=requirements_path)]
        ),
        # Task 2: Grid generation
        Task(
            task_key="task2_grid_generation",
            depends_on=[TaskDependency(task_key="task1_proportions_to_delta")],
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task2_grid_generation.py",
                parameters=["--config_path", generated_config_path]
            ),
            libraries=[Library(requirements=requirements_path)]
        ),
        # Task 3: Tile downloader
        Task(
            task_key="task3_tile_downloader",
            depends_on=[TaskDependency(task_key="task2_grid_generation")],
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task3_tile_downloader.py",
                parameters=["--config_path", generated_config_path]
            ),
            libraries=[Library(requirements=requirements_path)]
        ),
        # Task 4: Raster stats
        Task(
            task_key="task4_raster_stats",
            depends_on=[TaskDependency(task_key="task3_tile_downloader")],
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task4_raster_stats.py",
                parameters=["--config_path", generated_config_path]
            ),
            libraries=[Library(requirements=requirements_path)]
        ),
        # Task 5: Post processing
        Task(
            task_key="task5_post_processing",
            depends_on=[TaskDependency(task_key="task4_raster_stats")],
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task5_post_processing.py",
                parameters=["--config_path", generated_config_path]
            ),
            libraries=[Library(requirements=requirements_path)]
        ),
        # Task 6: Create views
        Task(
            task_key="task6_create_views",
            depends_on=[TaskDependency(task_key="task5_post_processing")],
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task6_create_views.py",
                parameters=["--config_path", generated_config_path]
            ),
            libraries=[Library(requirements=requirements_path)]
        ),
        # Task 7: Export
        Task(
            task_key="task7_export",
            depends_on=[TaskDependency(task_key="task6_create_views")],
            existing_cluster_id=cluster_id,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task7_export.py",
                parameters=["--config_path", generated_config_path, "--iso3", iso3]
            ),
            libraries=[Library(requirements=requirements_path)]
        )
    ]

    # Create job
    job = w.jobs.create(
        name=job_name,
        tasks=tasks,
        max_concurrent_runs=1,
        timeout_seconds=0,
        email_notifications={
            "on_success": [email],
            "on_failure": [email]
        } if email else None
    )

    return job.job_id, job_name
