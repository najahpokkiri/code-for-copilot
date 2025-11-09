# Create Databricks Job from Notebook Using YAML

## 🎯 Perfect Solution: Use Notebook to Create Job!

Since you have notebook access but not CLI access, you can use a **notebook to programmatically create the Databricks job** from your YAML configuration files.

---

## ⚡ How It Works

```
Your YAML Files              Notebook (with API)           Databricks Job
├── building_enrichment.yml  ────────►  Reads YAML        ────────►  Job Created!
└── main_cluster.yml                    Calls Jobs API              (8 tasks, ready to run)
```

**Benefits:**
- ✅ No manual UI configuration
- ✅ No CLI access needed
- ✅ Uses your existing YAML files
- ✅ Reproducible (run notebook → job created)
- ✅ Easy to update (modify YAML → rerun notebook)

---

## 🚀 Quick Start

### Step 1: Connect Your Git Repo

1. **Workspace → Repos → Add Repo**
2. URL: `https://github.com/najahpokkiri/code-for-copilot`
3. Branch: `claude/job-yaml-config-structure-011CUthFZWYLsccLNE2WRN5c`

### Step 2: Create New Notebook

1. **Workspace → Create → Notebook**
2. Name: `Create_Job_From_YAML`
3. Language: Python

### Step 3: Copy the Notebook Code

Copy the content from `CREATE_JOB_FROM_YAML.py` into your notebook.

Or use this simplified version:

---

## 📝 Simplified Notebook Code

```python
# Databricks notebook source

# MAGIC %md
# MAGIC # Create Job from YAML Configuration

# COMMAND ----------

# Install PyYAML
%pip install pyyaml

# COMMAND ----------

import yaml
import json
import requests

# Configuration - UPDATE THESE!
repo_path = "/Workspace/Repos/code-for-copilot/mre/job1"
workspace_path = "/Workspace/Repos/code-for-copilot/mre/job1"  # Where your scripts are
your_email = "npokkiri@munichre.com"

# Paths to YAML files
job_yaml = f"{repo_path}/job_yaml_examples/databricks_bundle_example/resources/jobs/building_enrichment.yml"
cluster_yaml = f"{repo_path}/job_yaml_examples/databricks_bundle_example/resources/clusters/main_cluster.yml"

# COMMAND ----------

# Read YAML files
with open(job_yaml.replace("/Workspace", "/dbfs"), 'r') as f:
    job_config = yaml.safe_load(f)

with open(cluster_yaml.replace("/Workspace", "/dbfs"), 'r') as f:
    cluster_config = yaml.safe_load(f)

print("✅ YAML files loaded")

# COMMAND ----------

# Get job and cluster definitions
job_name = list(job_config['resources']['jobs'].keys())[0]
job_def = job_config['resources']['jobs'][job_name]

cluster_name = list(cluster_config['resources']['job_clusters'].keys())[0]
cluster_def = cluster_config['resources']['job_clusters'][cluster_name]

# Build job configuration for API
api_config = {
    "name": "Building Data Enrichment - IND",
    "max_concurrent_runs": 1,
    "email_notifications": {
        "on_failure": [your_email],
        "on_success": [your_email]
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

# Add all tasks
for task in job_def['tasks']:
    task_config = {
        "task_key": task['task_key'],
        "description": task.get('description', ''),
        "job_cluster_key": "main_cluster"
    }

    # Add dependencies
    if 'depends_on' in task:
        task_config['depends_on'] = task['depends_on']

    # Add Python task with script path
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

print(f"✅ Configuration built: {len(api_config['tasks'])} tasks")

# COMMAND ----------

# Get API credentials
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
api_url = ctx.apiUrl().get()
api_token = ctx.apiToken().get()

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

# Create the job
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
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Click the URL above to view your job")
    print("2. Review the task configuration")
    print("3. Click 'Run Now' to test")
else:
    print("❌ ERROR:", response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC Your job has been created with all 8 tasks:
# MAGIC - Task 0: Config Generation
# MAGIC - Task 1-7: All pipeline tasks
# MAGIC
# MAGIC Go to the job URL above to run it!
```

---

## 🔧 Customization

### Update Paths

Before running, update these variables in the notebook:

```python
# Your repo path (after connecting Repos)
repo_path = "/Workspace/Repos/code-for-copilot/mre/job1"

# Where your Python scripts are located
workspace_path = "/Workspace/Repos/code-for-copilot/mre/job1"

# Your email for notifications
your_email = "npokkiri@munichre.com"
```

### Using Different YAML Files

If you created custom YAML files:

```python
job_yaml = f"{repo_path}/path/to/your/custom_job.yml"
cluster_yaml = f"{repo_path}/path/to/your/custom_cluster.yml"
```

---

## 🎯 What Gets Created

When you run the notebook, it creates a Databricks Job with:

**Job Configuration:**
- Name: "Building Data Enrichment - IND"
- Max concurrent runs: 1
- Email notifications on success/failure

**8 Tasks:**
- Task 0: Config Generation (no dependencies)
- Task 1: Import Proportions (depends on Task 0)
- Task 2: Grid Generation (depends on Task 1, has geopandas/shapely libraries)
- Task 3: Tile Download (depends on Task 2)
- Task 4: Raster Stats (depends on Task 3, has rasterio library)
- Task 5: Post-Processing (depends on Task 4)
- Task 6: Create Views (depends on Task 5)
- Task 7: Export (depends on Task 6, has xlsxwriter library)

**Cluster:**
- DBR 13.3.x
- Standard_DS3_v2 nodes
- 4 workers
- Spark configurations optimized for Delta

---

## 🔄 Updating the Job

### Option 1: Delete and Recreate

```python
# In a new notebook cell

# Delete old job
job_id = 123456  # Your job ID from previous run

response = requests.post(
    f"{api_url}/api/2.1/jobs/delete",
    headers=headers,
    json={"job_id": job_id}
)

# Then rerun the creation cells
```

### Option 2: Update Existing Job

```python
# Update instead of create
response = requests.post(
    f"{api_url}/api/2.1/jobs/update",
    headers=headers,
    json={
        "job_id": job_id,
        "new_settings": api_config
    }
)
```

---

## 🎓 Advanced: Creating Multiple Country Jobs

You can modify the notebook to create jobs for multiple countries:

```python
# Loop through countries
countries = ["IND", "USA", "BRA"]

for country_iso3 in countries:
    # Modify job name and paths
    api_config["name"] = f"Building Data Enrichment - {country_iso3}"

    # Update config path to use country-specific config
    # (you'd have config_IND.yaml, config_USA.yaml, etc.)

    # Create job
    response = requests.post(...)
    print(f"✅ Job created for {country_iso3}")
```

---

## 💡 Benefits of This Approach

| Aspect | Manual UI | CLI Bundle | **Notebook + YAML** ⭐ |
|--------|-----------|------------|------------------------|
| **Access needed** | Web UI | CLI + token | **Notebook only** |
| **Configuration** | Click 50+ times | One command | **Run one notebook** |
| **Reproducible** | ❌ Hard | ✅ Yes | **✅ Yes** |
| **Version control** | ❌ No | ✅ Yes | **✅ Yes (YAML files)** |
| **Easy to update** | ❌ Tedious | ✅ Easy | **✅ Easy (edit YAML)** |
| **Team friendly** | ❌ No | ⚠️ Need CLI | **✅ Anyone with notebook** |

---

## ✅ Complete Workflow

```
1. Connect Git Repo (Workspace → Repos)
   ↓
2. Upload data files (Catalog → Volumes)
   ↓
3. Create notebook from CREATE_JOB_FROM_YAML.py
   ↓
4. Update configuration (paths, email)
   ↓
5. Run notebook → Job created! ✅
   ↓
6. Go to job URL → Run job → Done!
```

---

## 📖 Files You'll Use

**From your Git repository:**
- `building_enrichment.yml` - Job definition (8 tasks)
- `main_cluster.yml` - Cluster configuration
- All `task*.py` files - Your Python scripts

**Notebook creates:**
- Databricks Job (via Jobs API)
- All task configurations
- Task dependencies
- Library requirements

**Notebook does NOT upload:**
- Your Python scripts (already in Repos)
- Data files (upload separately to Volumes)

---

## ❓ FAQ

**Q: Do I need to manually configure anything in the UI?**
A: No! The notebook creates everything via API.

**Q: Can I schedule the created job?**
A: Yes! After creation, go to the job UI and add a schedule.

**Q: What if job creation fails?**
A: Check the error message. Common issues:
- YAML file path incorrect
- Workspace path doesn't exist
- Library name misspelled

**Q: Can I use this for other jobs?**
A: Yes! Just point to different YAML files or modify the YAML structure.

**Q: Do I need to delete the old job before creating a new one?**
A: If the name is the same, you'll get an error. Either:
- Delete the old job first
- Use a different name
- Use the update API instead of create

---

## 🎉 Summary

You've effectively created a **"Databricks Bundle Deploy" using notebooks**!

**Instead of:**
- ❌ `databricks bundle deploy` (no CLI access)
- ❌ Creating 8 tasks manually in UI (tedious)

**You can:**
- ✅ Run one notebook
- ✅ Job created from YAML
- ✅ All tasks configured
- ✅ Ready to run!

**All the YAML work we did is now being used** to programmatically create your job! 🚀

---

**Next**:
1. Copy `CREATE_JOB_FROM_YAML.py` to a new notebook
2. Update the configuration variables
3. Run it!
4. Get your job URL
5. Click "Run Now"
