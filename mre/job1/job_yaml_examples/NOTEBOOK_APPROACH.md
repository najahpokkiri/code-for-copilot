# Using Databricks Notebooks with Your Pipeline

## ✨ Great News: Notebooks Make This Easier!

You can run your entire pipeline from a **single master notebook** instead of creating a job in the UI!

---

## 🚀 Option 1: Master Notebook (RECOMMENDED)

Create one notebook that runs all tasks sequentially.

### Step 1: Connect Your Git Repo (Same as Before)

1. **Workspace → Repos → Add Repo**
2. URL: `https://github.com/najahpokkiri/code-for-copilot`
3. Branch: `claude/job-yaml-config-structure-011CUthFZWYLsccLNE2WRN5c`

### Step 2: Create Master Notebook

**Create new notebook**: `Building_Data_Enrichment_Master`

**Cell 1: Setup**
```python
# Master Pipeline Notebook - Building Data Enrichment

# Set paths
repo_path = "/Repos/code-for-copilot/mre/job1"
config_path = f"{repo_path}/config.json"

print("🚀 Starting Building Data Enrichment Pipeline")
print(f"📂 Repo path: {repo_path}")
```

**Cell 2: Task 0 - Config Generation**
```python
# Task 0: Generate config.json from config.yaml
print("\n" + "="*60)
print("TASK 0: Config Generation")
print("="*60)

%run $repo_path/config_builder.py config.yaml

print("✅ Task 0 Complete: config.json generated")
```

**Cell 3: Task 1 - Import Proportions**
```python
# Task 1: Import Proportions
print("\n" + "="*60)
print("TASK 1: Import Proportions")
print("="*60)

# Set arguments
import sys
sys.argv = ["task1_proportions_to_delta.py", "--config_path", config_path]

# Run task
%run $repo_path/task1_proportions_to_delta.py

print("✅ Task 1 Complete: Proportions imported")
```

**Cell 4: Task 2 - Grid Generation**
```python
# Task 2: Grid Generation
print("\n" + "="*60)
print("TASK 2: Grid Generation")
print("="*60)

# Need libraries
%pip install geopandas==0.14.4 shapely==2.0.4

# Set arguments
sys.argv = ["task2_grid_generation.py", "--config_path", config_path]

# Run task
%run $repo_path/task2_grid_generation.py

print("✅ Task 2 Complete: Grid generated")
```

**Cell 5: Task 3 - Tile Download**
```python
# Task 3: Tile Download
print("\n" + "="*60)
print("TASK 3: Tile Download")
print("="*60)

sys.argv = ["task3_tile_downloader.py", "--config_path", config_path]
%run $repo_path/task3_tile_downloader.py

print("✅ Task 3 Complete: Tiles downloaded")
```

**Cell 6: Task 4 - Raster Stats**
```python
# Task 4: Raster Statistics
print("\n" + "="*60)
print("TASK 4: Raster Statistics")
print("="*60)

# Need rasterio
%pip install rasterio==1.3.9

sys.argv = ["task4_raster_stats.py", "--config_path", config_path]
%run $repo_path/task4_raster_stats.py

print("✅ Task 4 Complete: Raster stats extracted")
```

**Cell 7: Task 5 - Post-Processing**
```python
# Task 5: Post-Processing
print("\n" + "="*60)
print("TASK 5: Post-Processing")
print("="*60)

sys.argv = ["task5_post_processing.py", "--config_path", config_path]
%run $repo_path/task5_post_processing.py

print("✅ Task 5 Complete: Post-processing done")
```

**Cell 8: Task 6 - Create Views**
```python
# Task 6: Create Views
print("\n" + "="*60)
print("TASK 6: Create Views")
print("="*60)

sys.argv = ["task6_create_views.py", "--config_path", config_path]
%run $repo_path/task6_create_views.py

print("✅ Task 6 Complete: Views created")
```

**Cell 9: Task 7 - Export**
```python
# Task 7: Export
print("\n" + "="*60)
print("TASK 7: Export")
print("="*60)

# Need xlsxwriter
%pip install xlsxwriter==3.2.9

sys.argv = ["task7_export.py", "--config_path", config_path]
%run $repo_path/task7_export.py

print("✅ Task 7 Complete: Export done")
print("\n" + "="*60)
print("🎉 PIPELINE COMPLETE!")
print("="*60)
```

### Step 3: Run the Notebook

**Just click "Run All"!**
- All tasks run sequentially
- Each cell's output shows task progress
- If one task fails, you can fix and re-run just that cell

---

## 🚀 Option 2: Individual Task Notebooks

Create separate notebooks for each task, then create a workflow of notebooks.

### Create 8 Notebooks:

**Notebook: Task_0_Config_Generation**
```python
# Task 0: Config Generation

repo_path = "/Repos/code-for-copilot/mre/job1"

%run $repo_path/config_builder.py config.yaml

print("✅ Config generated")
```

**Notebook: Task_1_Import_Proportions**
```python
# Task 1: Import Proportions

repo_path = "/Repos/code-for-copilot/mre/job1"
config_path = f"{repo_path}/config.json"

import sys
sys.argv = ["task1_proportions_to_delta.py", "--config_path", config_path]

%run $repo_path/task1_proportions_to_delta.py

print("✅ Proportions imported")
```

**Repeat for Tasks 2-7...**

### Create Job from Notebooks

1. **Workflows → Jobs → Create Job**
2. Add tasks:
   - Type: **Notebook**
   - Notebook path: `/Users/your_email/Task_0_Config_Generation`
   - Dependencies: Set appropriately

---

## 🚀 Option 3: Run Tasks Directly with dbutils

If your scripts support it, you can execute them directly:

```python
# In a notebook

# Method 1: Using dbutils.notebook.run
result = dbutils.notebook.run(
    "/Repos/code-for-copilot/mre/job1/task1_proportions_to_delta.py",
    timeout_seconds=3600,
    arguments={"config_path": "/Repos/code-for-copilot/mre/job1/config.json"}
)

# Method 2: Using %run magic
%run /Repos/code-for-copilot/mre/job1/task1_proportions_to_delta.py
```

---

## 📊 Comparison: Which Option?

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Master Notebook** ⭐ | One file, easy to run, see all output | Long notebook | **Development, Testing** |
| **Individual Notebooks** | Modular, can run in parallel | More files to manage | **Production, Scheduled** |
| **Direct .py via %run** | Uses existing scripts | Need to handle args | **Quick testing** |

**Recommendation**: Start with **Master Notebook** for development!

---

## ✅ Complete Example: Master Notebook

Here's a **complete, copy-paste ready** master notebook:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Building Data Enrichment Pipeline - Master Notebook
# MAGIC
# MAGIC This notebook runs all 8 tasks of the pipeline sequentially.
# MAGIC
# MAGIC **Tasks:**
# MAGIC - Task 0: Config Generation
# MAGIC - Task 1: Import Proportions
# MAGIC - Task 2: Grid Generation
# MAGIC - Task 3: Tile Download
# MAGIC - Task 4: Raster Statistics
# MAGIC - Task 5: Post-Processing
# MAGIC - Task 6: Create Views
# MAGIC - Task 7: Export

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Set paths
repo_path = "/Repos/code-for-copilot/mre/job1"
config_path = f"{repo_path}/config.json"

print(f"📂 Repo path: {repo_path}")
print(f"📄 Config path: {config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 0: Config Generation

# COMMAND ----------

print("="*60)
print("TASK 0: Config Generation")
print("="*60)

%run $repo_path/config_builder.py config.yaml

print("✅ Task 0 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Import Proportions

# COMMAND ----------

print("="*60)
print("TASK 1: Import Proportions")
print("="*60)

import sys
sys.argv = ["task1_proportions_to_delta.py", "--config_path", config_path]

%run $repo_path/task1_proportions_to_delta.py

print("✅ Task 1 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Grid Generation
# MAGIC
# MAGIC Requires: geopandas, shapely

# COMMAND ----------

# Install libraries
%pip install geopandas==0.14.4 shapely==2.0.4

# COMMAND ----------

print("="*60)
print("TASK 2: Grid Generation")
print("="*60)

sys.argv = ["task2_grid_generation.py", "--config_path", config_path]
%run $repo_path/task2_grid_generation.py

print("✅ Task 2 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Tile Download

# COMMAND ----------

print("="*60)
print("TASK 3: Tile Download")
print("="*60)

sys.argv = ["task3_tile_downloader.py", "--config_path", config_path]
%run $repo_path/task3_tile_downloader.py

print("✅ Task 3 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Raster Statistics
# MAGIC
# MAGIC Requires: rasterio

# COMMAND ----------

# Install rasterio
%pip install rasterio==1.3.9

# COMMAND ----------

print("="*60)
print("TASK 4: Raster Statistics")
print("="*60)

sys.argv = ["task4_raster_stats.py", "--config_path", config_path]
%run $repo_path/task4_raster_stats.py

print("✅ Task 4 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Post-Processing

# COMMAND ----------

print("="*60)
print("TASK 5: Post-Processing")
print("="*60)

sys.argv = ["task5_post_processing.py", "--config_path", config_path]
%run $repo_path/task5_post_processing.py

print("✅ Task 5 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Create Views

# COMMAND ----------

print("="*60)
print("TASK 6: Create Views")
print("="*60)

sys.argv = ["task6_create_views.py", "--config_path", config_path]
%run $repo_path/task6_create_views.py

print("✅ Task 6 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7: Export
# MAGIC
# MAGIC Requires: xlsxwriter

# COMMAND ----------

# Install xlsxwriter
%pip install xlsxwriter==3.2.9

# COMMAND ----------

print("="*60)
print("TASK 7: Export")
print("="*60)

sys.argv = ["task7_export.py", "--config_path", config_path]
%run $repo_path/task7_export.py

print("✅ Task 7 Complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Pipeline Complete!

# COMMAND ----------

print("="*60)
print("🎉 PIPELINE COMPLETE!")
print("="*60)
print("\n✅ All tasks finished successfully")
print("\n📊 Check your results:")
print("  - Tables: Catalog → prp_mr_bdap_projects.geospatialsolutions")
print("  - Exports: Volumes → jrc/data/outputs/")
```

---

## 🎯 Quick Start (Notebook Approach)

### 3 Steps:

1. **Connect Git Repo** (Workspace → Repos)
2. **Upload Data to Volumes** (Same as before)
3. **Create Master Notebook** (Copy code above)
4. **Run All** → Done! ✅

---

## 💡 Benefits of Notebook Approach

✅ **Easier than UI job creation** - No manual task configuration
✅ **See all output** - All task logs in one place
✅ **Easy debugging** - Re-run just failed cells
✅ **Interactive** - Can modify and test
✅ **Schedules** - Can still schedule notebook to run
✅ **Markdown docs** - Add explanations between tasks

---

## 🔄 Converting to Scheduled Job

Once your master notebook works:

1. **Workflows → Jobs → Create Job**
2. **Task type**: Notebook
3. **Notebook path**: `/Users/your_email/Building_Data_Enrichment_Master`
4. **Schedule**: Set cron schedule
5. **Done!**

Much simpler than creating 8 separate tasks!

---

## ❓ FAQ

**Q: Can I use %run with .py files from Repos?**
A: Yes! `%run /Repos/your-repo/path/to/script.py` works perfectly.

**Q: How do I pass arguments to scripts?**
A: Set `sys.argv` before running:
```python
import sys
sys.argv = ["script.py", "--config_path", "config.json"]
%run /Repos/your-repo/script.py
```

**Q: What about library dependencies?**
A: Use `%pip install` in cells before running tasks that need them.

**Q: Can I still use the job YAML files?**
A: They're great reference! They show exactly what each task needs.

---

## 🎓 Summary

**Instead of creating job in UI, you can:**

1. Create one **Master Notebook**
2. Use `%run` to execute your `.py` scripts from Repos
3. Click "Run All"
4. ✅ Done!

**This is actually EASIER than:**
- Creating 8 tasks in UI
- Configuring each task separately
- Setting dependencies manually

**Plus you get:**
- All output in one place
- Easy to debug
- Can run interactively
- Still schedulable

---

**Next Step**: Copy the master notebook code above, create a new notebook, paste it, and run it! 🚀
