# Quick Reference: Databricks Job YAML Configuration

## 🎯 Key Decision: Which Approach?

| If you have... | Use this approach | Location |
|----------------|-------------------|----------|
| **1-3 people, simple pipeline** | Monolithic YAML | See `APPROACH_COMPARISON.md` |
| **3-10 people, multiple environments** | **Asset Bundle** ⭐ | `databricks_bundle_example/` |
| **10+ people, many similar pipelines** | Template-based | See `APPROACH_COMPARISON.md` |

**Recommended**: **Asset Bundle** (Approach 2) for most teams

---

## 🚀 Quick Start: Asset Bundle

### Setup (One Time)

```bash
# 1. Install CLI
pip install databricks-cli

# 2. Authenticate
databricks configure --token

# 3. Copy bundle template
cp -r job_yaml_examples/databricks_bundle_example ./databricks_bundle
cd databricks_bundle

# 4. Customize databricks.yml
vim databricks.yml  # Update workspace_path and email

# 5. Deploy
databricks bundle deploy
```

### Daily Usage

```bash
# Make changes
vim config.yaml                           # Update pipeline config
vim resources/jobs/building_enrichment.yml  # Update tasks

# Deploy changes
databricks bundle deploy                  # Deploy to dev
databricks bundle deploy -t production    # Deploy to prod

# Run job
databricks bundle run building_enrichment_IND
```

---

## 📋 Task 0: Config Generation

### Why Task 0?

**Before** (Manual):
```bash
# Developer must remember:
python config_builder.py config.yaml  # Easy to forget!
```

**After** (Automated):
```yaml
# Task 0 in job YAML:
- task_key: task0_config_generation
  spark_python_task:
    python_file: config_builder.py
    parameters: [config.yaml]
```

**Benefits**:
- ✅ Always runs before other tasks
- ✅ Config.json always fresh
- ✅ Git tracks only config.yaml
- ✅ New team members don't need to know about config_builder.py

### Adding Task 0 to Existing Job

```yaml
tasks:
  # Add this as first task
  - task_key: task0_config_generation
    description: "Generate config.json from config.yaml"
    spark_python_task:
      python_file: /path/to/config_builder.py
      parameters:
        - config.yaml
    job_cluster_key: main_cluster

  # Make Task 1 depend on Task 0
  - task_key: task1_import_proportions
    depends_on:
      - task_key: task0_config_generation  # ← Add dependency
    spark_python_task:
      python_file: /path/to/task1_proportions_to_delta.py
```

---

## 🏗️ Directory Structure: Asset Bundle

```
your_project/
├── databricks.yml                    # Root config (environments, variables)
├── resources/
│   ├── jobs/
│   │   └── building_enrichment.yml   # Job definition (tasks)
│   └── clusters/
│       └── main_cluster.yml          # Cluster config
├── config.yaml                       # Pipeline config (source of truth)
├── .gitignore                        # Ignore config.json, *.pyc, etc.
└── README.md
```

**Git Tracking**:
- ✅ **Commit**: `databricks.yml`, `resources/*.yml`, `config.yaml`
- ❌ **Ignore**: `config.json`, `__pycache__`, logs

---

## 🔧 Common Tasks

### Switch Country (IND → USA)

```bash
# 1. Update config
vim databricks.yml
# Change: country_iso3: IND
# To:     country_iso3: USA

# 2. Update pipeline config
vim config.yaml
# Change: iso3: IND
# To:     iso3: USA

# 3. Redeploy
databricks bundle deploy

# Done! Task 0 will generate new config.json with USA settings
```

### Add New Task

```bash
# 1. Edit job YAML
vim resources/jobs/building_enrichment.yml

# 2. Add new task (example: Task 8 - Data Quality)
tasks:
  # ... existing tasks ...

  - task_key: task8_data_quality
    description: "Run data quality checks"
    depends_on:
      - task_key: task7_exports
    spark_python_task:
      python_file: ${var.workspace_path}/task8_data_quality.py
      parameters:
        - --config_path
        - ${var.workspace_path}/config.json
    job_cluster_key: main_cluster

# 3. Deploy
databricks bundle deploy
```

### Update Cluster Size

```bash
# 1. Edit cluster config
vim resources/clusters/main_cluster.yml

# 2. Change worker count
num_workers: 8  # Was: 4

# 3. Redeploy
databricks bundle deploy
```

### Add Environment Variable

```bash
# 1. Edit databricks.yml
vim databricks.yml

# 2. Add variable
targets:
  production:
    variables:
      my_new_var: "some_value"  # ← Add here

# 3. Use in job YAML
# ${var.my_new_var}

# 4. Deploy
databricks bundle deploy -t production
```

---

## 📊 Task Dependencies

### Linear Dependencies (Sequential)

```yaml
tasks:
  - task_key: task1
    # No depends_on

  - task_key: task2
    depends_on:
      - task_key: task1  # Runs after task1

  - task_key: task3
    depends_on:
      - task_key: task2  # Runs after task2
```

**Flow**: task1 → task2 → task3

### Parallel Dependencies

```yaml
tasks:
  - task_key: task1
    # No depends_on

  - task_key: task2a
    depends_on:
      - task_key: task1

  - task_key: task2b
    depends_on:
      - task_key: task1  # Both run in parallel after task1

  - task_key: task3
    depends_on:
      - task_key: task2a
      - task_key: task2b  # Waits for both to complete
```

**Flow**:
```
task1
  ├─→ task2a ─┐
  └─→ task2b ─┴─→ task3
```

### Your Pipeline Structure

```
Task 0: Config Generation
    ↓
Task 1: Import Proportions
    ↓
Task 2: Grid Generation
    ↓
Task 3: Tile Download  ─┐
                         ├─→ Task 5: Post-Processing
Task 4: Raster Stats   ─┘       ↓
                            Task 6: Create Views
                                ↓
                            Task 7: Export
```

---

## 🎨 YAML Variable Substitution

### Syntax

```yaml
# Define in databricks.yml
variables:
  my_var: "value"

# Use in resources/*.yml
some_field: ${var.my_var}          # ← Substituted at deploy time
another_field: ${bundle.name}      # ← Built-in variable
path: ${workspace.root_path}/file  # ← Workspace variable
```

### Common Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `${var.XXX}` | Custom variable | `${var.country_iso3}` |
| `${bundle.name}` | Bundle name | `building_enrichment` |
| `${bundle.target}` | Target environment | `development`, `production` |
| `${workspace.host}` | Workspace URL | `https://adb-xxx...` |
| `${workspace.root_path}` | Bundle root path | `/Users/.../.bundle/...` |

### Example

```yaml
# databricks.yml
variables:
  catalog: prp_mr_bdap_projects
  schema: geospatialsolutions
  country_iso3: IND

# resources/jobs/building_enrichment.yml
description: |
  Processing ${var.country_iso3} data
  Output: ${var.catalog}.${var.schema}.estimates_${var.country_iso3}

# Result after substitution:
description: |
  Processing IND data
  Output: prp_mr_bdap_projects.geospatialsolutions.estimates_IND
```

---

## 🐛 Troubleshooting

### Issue: "Bundle validation failed"

```bash
# Check YAML syntax
databricks bundle validate

# Common causes:
# - Incorrect indentation (use 2 spaces, not tabs)
# - Missing required fields
# - Invalid variable references
```

### Issue: "Task 0 fails with 'file not found'"

```bash
# Verify config.yaml exists in workspace
databricks workspace ls /Workspace/Users/.../scripts/

# Upload if missing
databricks workspace import config.yaml /Workspace/.../scripts/config.yaml
```

### Issue: "Variables not substituted (shows ${var.xxx})"

```bash
# Check variables are defined in databricks.yml
cat databricks.yml | grep -A 5 "variables:"

# Ensure you're deploying with correct target
databricks bundle deploy -t production  # Use -t flag
```

### Issue: "Permission denied"

```bash
# Check workspace permissions
databricks workspace get-status /Workspace/Users/.../

# Grant permissions if needed (ask admin)
```

---

## 📚 Cheat Sheet

### Deployment Commands

```bash
# Validate bundle
databricks bundle validate

# Deploy to dev (default)
databricks bundle deploy

# Deploy to production
databricks bundle deploy -t production

# View deployed resources
databricks bundle summary

# Destroy bundle (careful!)
databricks bundle destroy
```

### Job Management

```bash
# List jobs
databricks jobs list

# Run job
databricks bundle run building_enrichment_IND

# Run specific task
databricks jobs run-now --job-id <ID> --task-keys task0_config_generation

# Get run status
databricks runs get --run-id <RUN_ID>

# Get run output
databricks runs get-output --run-id <RUN_ID>
```

### Workspace Operations

```bash
# List workspace files
databricks workspace ls /Workspace/Users/.../

# Upload file
databricks workspace import local_file.py /Workspace/.../file.py

# Download file
databricks workspace export /Workspace/.../file.py local_file.py
```

---

## 🔗 Quick Links

### Documentation
- **This Directory**:
  - [Approach Comparison](./APPROACH_COMPARISON.md) - Compare 3 approaches
  - [Migration Guide](./MIGRATION_GUIDE.md) - Step-by-step migration
  - [Bundle Example](./databricks_bundle_example/) - Ready-to-use template

- **Official Docs**:
  - [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles)
  - [Job Configuration Reference](https://docs.databricks.com/workflows/jobs/jobs-api)
  - [CLI Reference](https://docs.databricks.com/dev-tools/cli)

### Examples
- [Bundle Examples Repo](https://github.com/databricks/bundle-examples)
- [MLOps Stacks](https://github.com/databricks/mlops-stacks)

---

## 💡 Pro Tips

### Tip 1: Use `--dry-run` for Testing

```bash
# See what would be deployed without actually deploying
databricks bundle deploy --dry-run
```

### Tip 2: Parameterize Everything

```yaml
# Instead of hardcoding:
email_notifications:
  on_failure: [npokkiri@munichre.com]

# Use variables:
email_notifications:
  on_failure: [${var.email_notifications}]
```

### Tip 3: Add Descriptions Everywhere

```yaml
- task_key: task1_import_proportions
  description: |
    Load building type proportions CSV to Delta table.
    Input: config.yaml -> proportions_csv_path
    Output: ${var.catalog}.${var.schema}.proportions_input
```

**Why?** Descriptions show up in Databricks UI, helping team understand pipeline.

### Tip 4: Use Tags for Cost Tracking

```yaml
tags:
  project: geospatial_solutions
  pipeline: building_enrichment
  cost_center: "1234"
  owner: npokkiri@munichre.com
```

### Tip 5: Version Your Bundles

```yaml
# databricks.yml
bundle:
  name: building_data_enrichment
  version: 2.0  # ← Track versions
```

---

## ✅ Checklist: Production Readiness

Before deploying to production:

- [ ] Task 0 tested and generates config.json successfully
- [ ] All tasks have clear descriptions
- [ ] Email notifications configured
- [ ] Retry logic added to critical tasks
- [ ] Timeout values set appropriately
- [ ] Cluster size validated for workload
- [ ] Environment variables defined for prod
- [ ] `.gitignore` prevents committing generated files
- [ ] Team trained on new workflow
- [ ] Rollback plan documented
- [ ] Monitoring/alerts set up

---

**Questions?** See [APPROACH_COMPARISON.md](./APPROACH_COMPARISON.md) for detailed guidance.

**Need Help?** Contact: npokkiri@munichre.com
