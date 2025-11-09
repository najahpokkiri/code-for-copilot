# Using Existing Cluster with Job Creation Notebook

## 🎯 Problem

The `CREATE_JOB_FROM_YAML.py` notebook creates a **new job cluster** from the YAML config.
But you already have a **personal compute cluster** you want to use!

---

## ✅ Solution 1: Use Your Existing Cluster (RECOMMENDED)

Modify the notebook to reference your existing cluster instead of creating a new one.

### Step 1: Find Your Cluster ID

In Databricks Web UI:
1. Go to **Compute** (left sidebar)
2. Find your personal cluster
3. Click on it
4. Look at the URL - you'll see something like:
   ```
   https://.../compute/clusters/1010-130900-1bz314p1
                                  ^^^^^^^^^^^^^^^^
                                  This is your cluster ID
   ```

Or get it programmatically in a notebook:
```python
# Get list of clusters
import requests

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
api_url = ctx.apiUrl().get()
api_token = ctx.apiToken().get()

headers = {"Authorization": f"Bearer {api_token}"}

response = requests.get(f"{api_url}/api/2.0/clusters/list", headers=headers)
clusters = response.json()

# Show all clusters
for cluster in clusters.get('clusters', []):
    print(f"Name: {cluster['cluster_name']}")
    print(f"ID: {cluster['cluster_id']}")
    print(f"State: {cluster['state']}")
    print("-" * 40)
```

### Step 2: Modify the Notebook

In `CREATE_JOB_FROM_YAML.py`, replace the cluster configuration section:

**OLD (creates new cluster):**
```python
api_config = {
    "name": "Building Data Enrichment - IND",
    "tasks": [],
    "job_clusters": [  # ← This creates NEW cluster
        {
            "job_cluster_key": "main_cluster",
            "new_cluster": cluster_def['new_cluster']
        }
    ],
    "format": "MULTI_TASK"
}

# And in tasks:
task_config['job_cluster_key'] = "main_cluster"  # ← References new cluster
```

**NEW (uses existing cluster):**
```python
# Your existing cluster ID
EXISTING_CLUSTER_ID = "1010-130900-1bz314p1"  # ← UPDATE THIS!

api_config = {
    "name": "Building Data Enrichment - IND",
    "tasks": [],
    # NO job_clusters section - using existing cluster instead
    "format": "MULTI_TASK"
}

# And in tasks:
task_config['existing_cluster_id'] = EXISTING_CLUSTER_ID  # ← Use existing cluster
# Remove: task_config['job_cluster_key'] = "main_cluster"
```

### Complete Modified Notebook Cell

Replace the configuration building section with:

```python
# COMMAND ----------

# YOUR EXISTING CLUSTER ID - UPDATE THIS!
EXISTING_CLUSTER_ID = "1010-130900-1bz314p1"  # Get from Compute UI

# Build job configuration
api_config = {
    "name": "Building Data Enrichment - IND",
    "max_concurrent_runs": 1,
    "email_notifications": {
        "on_failure": [your_email],
        "on_success": [your_email]
    },
    "tasks": [],
    # Don't create new cluster - use existing one
    "format": "MULTI_TASK"
}

# Add all tasks
for task in job_def['tasks']:
    task_config = {
        "task_key": task['task_key'],
        "description": task.get('description', ''),
        "existing_cluster_id": EXISTING_CLUSTER_ID  # ← Use existing cluster
    }

    # Add dependencies
    if 'depends_on' in task:
        task_config['depends_on'] = task['depends_on']

    # Add Python task
    if 'spark_python_task' in task:
        python_file = task['spark_python_task']['python_file']
        python_file = python_file.replace('${var.workspace_path}', workspace_path)

        params = task['spark_python_task'].get('parameters', [])
        params = [p.replace('${var.workspace_path}', workspace_path) for p in params]

        task_config['spark_python_task'] = {
            'python_file': python_file,
            'parameters': params
        }

    # Add libraries if specified
    if 'libraries' in task:
        task_config['libraries'] = task['libraries']

    api_config['tasks'].append(task_config)

print(f"✅ Configuration built with existing cluster: {EXISTING_CLUSTER_ID}")
```

---

## ✅ Solution 2: Let Job Create New Cluster (Current Approach)

Keep the current notebook as-is. It will create a **job cluster** that:
- Starts when the job runs
- Stops when the job completes
- Doesn't consume resources when idle

**Pros:**
- ✅ Isolated resources for the job
- ✅ Auto-scales based on job needs
- ✅ Cost-effective (only runs when job runs)

**Cons:**
- ⚠️ Cluster startup time (2-5 minutes)
- ⚠️ Need to ensure cluster config is correct

---

## ✅ Solution 3: Hybrid - Use Existing for Dev, New for Production

Create different configurations:

```python
# COMMAND ----------

# Configuration
USE_EXISTING_CLUSTER = True  # Set False for production

if USE_EXISTING_CLUSTER:
    # Development: Use your personal cluster
    EXISTING_CLUSTER_ID = "1010-130900-1bz314p1"
    print(f"📍 Using existing cluster: {EXISTING_CLUSTER_ID}")
else:
    # Production: Create new job cluster
    print("📍 Will create new job cluster")

# COMMAND ----------

# Build job configuration
api_config = {
    "name": "Building Data Enrichment - IND",
    "tasks": [],
    "format": "MULTI_TASK"
}

# Add job cluster only if NOT using existing
if not USE_EXISTING_CLUSTER:
    cluster_name = list(cluster_config['resources']['job_clusters'].keys())[0]
    cluster_def = cluster_config['resources']['job_clusters'][cluster_name]

    api_config['job_clusters'] = [
        {
            "job_cluster_key": "main_cluster",
            "new_cluster": cluster_def['new_cluster']
        }
    ]

# Add tasks
for task in job_def['tasks']:
    task_config = {
        "task_key": task['task_key'],
        "description": task.get('description', '')
    }

    # Set cluster reference
    if USE_EXISTING_CLUSTER:
        task_config['existing_cluster_id'] = EXISTING_CLUSTER_ID
    else:
        task_config['job_cluster_key'] = "main_cluster"

    # ... rest of task configuration ...
```

---

## 📊 Cluster Comparison

| Cluster Type | When to Use | Pros | Cons |
|--------------|-------------|------|------|
| **Existing (Personal)** | Development, Testing | Fast start, Always available | Uses resources even when idle |
| **Job Cluster (New)** | Production, Scheduled jobs | Cost-effective, Isolated | Startup time (2-5 min) |

---

## 🔍 How to Verify Cluster in Created Job

After creating the job:

1. Go to job URL in Databricks
2. Click on any task
3. Look at "Cluster" section
4. You'll see either:
   - **Existing cluster**: Shows cluster name/ID
   - **Job cluster**: Shows "main_cluster" with config

---

## 💡 Recommended Workflow

**For Development/Testing:**
```python
USE_EXISTING_CLUSTER = True
EXISTING_CLUSTER_ID = "your-cluster-id"
```
- Fast iteration
- No cluster startup wait
- Use your personal compute

**For Production/Scheduled:**
```python
USE_EXISTING_CLUSTER = False
# Creates job cluster from YAML config
```
- Automatic resource management
- Cost-effective
- Isolated from other workloads

---

## ⚠️ Important Notes

### Libraries on Existing Cluster

If using your existing cluster, make sure it has required libraries installed:
- geopandas==0.14.4
- shapely==2.0.4
- rasterio==1.3.9
- xlsxwriter==3.2.9

**To install on your cluster:**
1. Go to **Compute** → Your cluster
2. Click **Libraries** tab
3. Click **Install New**
4. Select **PyPI** and install each library

Or the job can install them per-task (slower but more flexible).

### Cluster Permissions

Your personal cluster must allow job tasks to run on it. If you get permission errors, you may need to:
- Use an all-purpose compute cluster
- Or create a job cluster instead

---

## 📝 Complete Example: Using Existing Cluster

```python
# Databricks notebook source

# MAGIC %md
# MAGIC # Create Job Using Existing Cluster

# COMMAND ----------

%pip install pyyaml

# COMMAND ----------

# Configuration
YOUR_CLUSTER_ID = "1010-130900-1bz314p1"  # ← UPDATE THIS!
workspace_path = "/Workspace/Repos/code-for-copilot/mre/job1"
your_email = "npokkiri@munichre.com"

# COMMAND ----------

import yaml
import json
import requests

# Read job YAML (but ignore cluster YAML since using existing)
job_yaml = f"{workspace_path}/job_yaml_examples/databricks_bundle_example/resources/jobs/building_enrichment.yml"

with open(job_yaml.replace("/Workspace", "/dbfs"), 'r') as f:
    job_config = yaml.safe_load(f)

job_name = list(job_config['resources']['jobs'].keys())[0]
job_def = job_config['resources']['jobs'][job_name]

# COMMAND ----------

# Build API configuration using existing cluster
api_config = {
    "name": "Building Data Enrichment - IND",
    "max_concurrent_runs": 1,
    "email_notifications": {
        "on_failure": [your_email],
        "on_success": [your_email]
    },
    "tasks": [],
    "format": "MULTI_TASK"
}

# Add tasks with existing cluster
for task in job_def['tasks']:
    task_config = {
        "task_key": task['task_key'],
        "description": task.get('description', ''),
        "existing_cluster_id": YOUR_CLUSTER_ID  # ← Use existing cluster
    }

    if 'depends_on' in task:
        task_config['depends_on'] = task['depends_on']

    if 'spark_python_task' in task:
        python_file = task['spark_python_task']['python_file']
        python_file = python_file.replace('${var.workspace_path}', workspace_path)

        params = task['spark_python_task'].get('parameters', [])
        params = [p.replace('${var.workspace_path}', workspace_path) for p in params]

        task_config['spark_python_task'] = {
            'python_file': python_file,
            'parameters': params
        }

    if 'libraries' in task:
        task_config['libraries'] = task['libraries']

    api_config['tasks'].append(task_config)

print(f"✅ Job configured with {len(api_config['tasks'])} tasks")
print(f"📍 Using existing cluster: {YOUR_CLUSTER_ID}")

# COMMAND ----------

# Create job via API
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
api_url = ctx.apiUrl().get()
api_token = ctx.apiToken().get()

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"{api_url}/api/2.1/jobs/create",
    headers=headers,
    json=api_config
)

if response.status_code == 200:
    job_id = response.json()['job_id']
    workspace_url = api_url.replace('/api', '')

    print("="*60)
    print("✅ JOB CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"Job ID: {job_id}")
    print(f"Job URL: {workspace_url}/#job/{job_id}")
    print(f"Using existing cluster: {YOUR_CLUSTER_ID}")
    print("="*60)
else:
    print("❌ ERROR:", response.text)
```

---

## 🎓 Summary

**To use your existing personal cluster:**

1. **Get your cluster ID** from Compute UI
2. **Modify notebook** to use `existing_cluster_id` instead of `job_cluster_key`
3. **Remove** the `job_clusters` section from API config
4. **Run** the notebook → Job created with your cluster!

**Advantages:**
- ✅ No cluster startup time
- ✅ Use your existing compute
- ✅ Libraries already installed (if you installed them)

**Make sure:**
- ⚠️ Cluster has required libraries
- ⚠️ Cluster allows job execution
- ⚠️ Cluster is running (or can auto-start)
