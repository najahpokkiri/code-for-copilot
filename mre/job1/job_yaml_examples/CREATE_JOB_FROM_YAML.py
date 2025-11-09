# Databricks notebook source
# MAGIC %md
# MAGIC # Create Databricks Job from YAML Configuration
# MAGIC
# MAGIC This notebook reads your job YAML file and creates a Databricks job programmatically using the Jobs API.
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Reads `building_enrichment.yml` from your Repo
# MAGIC 2. Parses the job configuration
# MAGIC 3. Creates the job using Databricks Jobs API
# MAGIC 4. Returns the job URL

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Libraries

# COMMAND ----------

%pip install pyyaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import yaml
import json
from pyspark.sql import SparkSession

# Paths
repo_path = "/Workspace/Repos/code-for-copilot/mre/job1"
job_yaml_path = f"{repo_path}/job_yaml_examples/databricks_bundle_example/resources/jobs/building_enrichment.yml"
cluster_yaml_path = f"{repo_path}/job_yaml_examples/databricks_bundle_example/resources/clusters/main_cluster.yml"

# Your workspace details
workspace_url = "https://adb-6685660099993059.19.azuredatabricks.net"
workspace_path = "/Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts"
email = "npokkiri@munichre.com"

print(f"📂 Job YAML: {job_yaml_path}")
print(f"📂 Cluster YAML: {cluster_yaml_path}")
print(f"🌐 Workspace: {workspace_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read YAML Configuration Files

# COMMAND ----------

# Read job YAML
with open(job_yaml_path.replace("/Workspace", "/dbfs"), 'r') as f:
    job_config = yaml.safe_load(f)

# Read cluster YAML
with open(cluster_yaml_path.replace("/Workspace", "/dbfs"), 'r') as f:
    cluster_config = yaml.safe_load(f)

print("✅ YAML files loaded")
print(f"📋 Job name: {list(job_config['resources']['jobs'].keys())[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform YAML to Jobs API Format

# COMMAND ----------

def transform_yaml_to_job_config(job_yaml, cluster_yaml, workspace_path, email):
    """
    Transform YAML configuration to Databricks Jobs API format
    """

    # Get job definition from YAML
    job_name = list(job_yaml['resources']['jobs'].keys())[0]
    job_def = job_yaml['resources']['jobs'][job_name]

    # Get cluster configuration
    cluster_name = list(cluster_yaml['resources']['job_clusters'].keys())[0]
    cluster_def = cluster_yaml['resources']['job_clusters'][cluster_name]

    # Build job configuration
    job_config = {
        "name": job_def['name'].replace('${bundle.target}', 'production'),
        "description": job_def.get('description', ''),
        "max_concurrent_runs": job_def.get('max_concurrent_runs', 1),
        "timeout_seconds": job_def.get('timeout_seconds', 0),
        "email_notifications": {
            "on_failure": [email],
            "on_success": [email]
        },
        "tasks": [],
        "job_clusters": [
            {
                "job_cluster_key": "main_cluster",
                "new_cluster": cluster_def['new_cluster']
            }
        ],
        "format": "MULTI_TASK"
    }

    # Add tasks
    for task in job_def['tasks']:
        task_config = {
            "task_key": task['task_key'],
            "description": task.get('description', ''),
            "job_cluster_key": "main_cluster"
        }

        # Add dependencies
        if 'depends_on' in task:
            task_config['depends_on'] = task['depends_on']

        # Add Python task
        if 'spark_python_task' in task:
            python_file = task['spark_python_task']['python_file']
            # Replace ${var.workspace_path} with actual path
            python_file = python_file.replace('${var.workspace_path}', workspace_path)

            task_config['spark_python_task'] = {
                'python_file': python_file,
                'parameters': task['spark_python_task'].get('parameters', [])
            }

            # Replace parameter variables
            if 'parameters' in task_config['spark_python_task']:
                params = []
                for param in task_config['spark_python_task']['parameters']:
                    param = param.replace('${var.workspace_path}', workspace_path)
                    params.append(param)
                task_config['spark_python_task']['parameters'] = params

        # Add libraries
        if 'libraries' in task:
            task_config['libraries'] = task['libraries']

        # Add timeout
        if 'timeout_seconds' in task:
            task_config['timeout_seconds'] = task['timeout_seconds']

        # Add retries
        if 'max_retries' in task:
            task_config['max_retries'] = task['max_retries']

        job_config['tasks'].append(task_config)

    return job_config

# Transform configuration
job_api_config = transform_yaml_to_job_config(
    job_config,
    cluster_config,
    workspace_path,
    email
)

print("✅ Configuration transformed to Jobs API format")
print(f"📊 Job has {len(job_api_config['tasks'])} tasks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Job Configuration

# COMMAND ----------

# Display the job configuration
print("="*60)
print("JOB CONFIGURATION PREVIEW")
print("="*60)
print(json.dumps(job_api_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Job Using Jobs API

# COMMAND ----------

# Get current workspace context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
api_url = ctx.apiUrl().get()
api_token = ctx.apiToken().get()

# Create job
import requests

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

# Create job endpoint
create_job_url = f"{api_url}/api/2.1/jobs/create"

print("🚀 Creating job...")
response = requests.post(
    create_job_url,
    headers=headers,
    json=job_api_config
)

if response.status_code == 200:
    result = response.json()
    job_id = result['job_id']

    print("="*60)
    print("✅ JOB CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"Job ID: {job_id}")
    print(f"Job URL: {workspace_url}/#job/{job_id}")
    print("="*60)

    # Store job ID for later use
    dbutils.widgets.text("job_id", str(job_id))

else:
    print("="*60)
    print("❌ ERROR CREATING JOB")
    print("="*60)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Job Creation

# COMMAND ----------

# Get job details
if response.status_code == 200:
    job_id = result['job_id']
    get_job_url = f"{api_url}/api/2.1/jobs/get"

    verify_response = requests.get(
        get_job_url,
        headers=headers,
        params={"job_id": job_id}
    )

    if verify_response.status_code == 200:
        job_details = verify_response.json()

        print("="*60)
        print("JOB DETAILS")
        print("="*60)
        print(f"Name: {job_details['settings']['name']}")
        print(f"Creator: {job_details['creator_user_name']}")
        print(f"Tasks: {len(job_details['settings']['tasks'])}")
        print(f"Clusters: {len(job_details['settings']['job_clusters'])}")
        print("="*60)

        # List all tasks
        print("\n📋 Tasks:")
        for task in job_details['settings']['tasks']:
            depends = ""
            if 'depends_on' in task:
                deps = [d['task_key'] for d in task['depends_on']]
                depends = f" (depends on: {', '.join(deps)})"
            print(f"  - {task['task_key']}{depends}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Run the Job

# COMMAND ----------

# Uncomment to run the job immediately after creation

# if response.status_code == 200:
#     run_job_url = f"{api_url}/api/2.1/jobs/run-now"
#
#     run_response = requests.post(
#         run_job_url,
#         headers=headers,
#         json={"job_id": job_id}
#     )
#
#     if run_response.status_code == 200:
#         run_result = run_response.json()
#         run_id = run_result['run_id']
#
#         print("="*60)
#         print("✅ JOB RUN STARTED")
#         print("="*60)
#         print(f"Run ID: {run_id}")
#         print(f"Run URL: {workspace_url}/#job/{job_id}/run/{run_id}")
#         print("="*60)
#     else:
#         print(f"❌ Error starting job: {run_response.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What was created:**
# MAGIC - Job with 8 tasks (Task 0 - Task 7)
# MAGIC - Task dependencies configured
# MAGIC - Cluster configuration attached
# MAGIC - Email notifications set up
# MAGIC - Libraries specified for tasks that need them
# MAGIC
# MAGIC **Next steps:**
# MAGIC 1. Go to the job URL above
# MAGIC 2. Review the configuration
# MAGIC 3. Click "Run Now" to test
# MAGIC 4. Monitor task execution
# MAGIC
# MAGIC **To update the job:**
# MAGIC - Modify the YAML files
# MAGIC - Re-run this notebook
# MAGIC - (You may need to delete old job first or use Jobs API update endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (Optional)

# COMMAND ----------

# Uncomment to delete the job if you need to recreate it

# if response.status_code == 200:
#     delete_job_url = f"{api_url}/api/2.1/jobs/delete"
#
#     delete_response = requests.post(
#         delete_job_url,
#         headers=headers,
#         json={"job_id": job_id}
#     )
#
#     if delete_response.status_code == 200:
#         print(f"✅ Job {job_id} deleted")
#     else:
#         print(f"❌ Error deleting job: {delete_response.text}")
