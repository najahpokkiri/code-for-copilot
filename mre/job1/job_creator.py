"""
Job Creator Helper
Creates and configures the Databricks job for the building enrichment pipeline.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, TaskDependency, SparkPythonTask, JobEmailNotifications
from databricks.sdk.service.compute import Library, PythonPyPiLibrary
from datetime import datetime


def get_job_libraries():
    """
    Returns the list of PyPI libraries required for all job tasks.
    These packages are defined in requirements.txt and will be automatically
    installed when each task runs.
    """
    return [
        # Geospatial processing
        Library(pypi=PythonPyPiLibrary(package="geopandas==0.14.0")),
        Library(pypi=PythonPyPiLibrary(package="shapely==2.0.2")),
        Library(pypi=PythonPyPiLibrary(package="rasterio==1.3.9")),
        # Data manipulation
        Library(pypi=PythonPyPiLibrary(package="pandas==2.0.3")),
        Library(pypi=PythonPyPiLibrary(package="numpy==1.24.3")),
        Library(pypi=PythonPyPiLibrary(package="pyarrow==13.0.0")),
        # HTTP requests for tile downloads
        Library(pypi=PythonPyPiLibrary(package="requests==2.31.0")),
        # Configuration (Task 0 needs YAML)
        Library(pypi=PythonPyPiLibrary(package="pyyaml==6.0.1")),
    ]


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

    # Generate job name with date suffix matching table naming convention
    date_suffix = datetime.now().strftime("%y%m%d")
    job_name = f"inv_NoS_{iso3}_{date_suffix}"
    # Config is now saved to workspace instead of volume
    generated_config_path = f"{workspace_base}/config_{iso3}.json"

    # Get libraries list for all tasks
    job_libraries = get_job_libraries()

    # Define all 8 tasks
    tasks = [
        # Task 0: Setup & config generation
        Task(
            task_key="task0_setup",
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task0_setup.py",
                parameters=["--minimal_config", minimal_config_path]
            )
        ),
        # Task 1: Proportions to Delta
        Task(
            task_key="task1_proportions_to_delta",
            depends_on=[TaskDependency(task_key="task0_setup")],
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task1_proportions_to_delta.py",
                parameters=["--config_path", generated_config_path]
            )
        ),
        # Task 2: Grid generation
        Task(
            task_key="task2_grid_generation",
            depends_on=[TaskDependency(task_key="task1_proportions_to_delta")],
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task2_grid_generation.py",
                parameters=["--config_path", generated_config_path]
            )
        ),
        # Task 3: Tile downloader
        Task(
            task_key="task3_tile_downloader",
            depends_on=[TaskDependency(task_key="task2_grid_generation")],
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task3_tile_downloader.py",
                parameters=["--config_path", generated_config_path]
            )
        ),
        # Task 4: Raster stats
        Task(
            task_key="task4_raster_stats",
            depends_on=[TaskDependency(task_key="task3_tile_downloader")],
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task4_raster_stats.py",
                parameters=["--config_path", generated_config_path]
            )
        ),
        # Task 5: Post processing
        Task(
            task_key="task5_post_processing",
            depends_on=[TaskDependency(task_key="task4_raster_stats")],
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task5_post_processing.py",
                parameters=["--config_path", generated_config_path]
            )
        ),
        # Task 6: Create views
        Task(
            task_key="task6_create_views",
            depends_on=[TaskDependency(task_key="task5_post_processing")],
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task6_create_views.py",
                parameters=["--config_path", generated_config_path]
            )
        ),
        # Task 7: Export
        Task(
            task_key="task7_export",
            depends_on=[TaskDependency(task_key="task6_create_views")],
            existing_cluster_id=cluster_id,
            libraries=job_libraries,
            spark_python_task=SparkPythonTask(
                python_file=f"{workspace_base}/task7_export.py",
                parameters=["--config_path", generated_config_path, "--iso3", iso3]
            )
        )
    ]

    # Create job
    job = w.jobs.create(
        name=job_name,
        tasks=tasks,
        max_concurrent_runs=1,
        timeout_seconds=0,
        email_notifications=JobEmailNotifications(
            on_success=[email],
            on_failure=[email]
        ) if email else None
    )

    return job.job_id, job_name
