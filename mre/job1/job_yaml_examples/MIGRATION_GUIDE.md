# Migration Guide: Moving to Databricks Asset Bundles

This guide helps you migrate from a single monolithic job YAML file to a structured Databricks Asset Bundle with Task 0 config generation.

## 📋 Table of Contents

1. [Why Migrate?](#why-migrate)
2. [Migration Steps](#migration-steps)
3. [Detailed Walkthrough](#detailed-walkthrough)
4. [Verification](#verification)
5. [Rollback Plan](#rollback-plan)

---

## Why Migrate?

### Current State Issues

❌ **Single large YAML file** (200+ lines)
❌ **Manual config generation** (config_builder.py run locally)
❌ **No environment separation** (dev/prod mixed)
❌ **Git conflicts** on every change
❌ **Hard to onboard** new team members

### After Migration

✅ **Modular structure** (50 lines per file)
✅ **Automated config generation** (Task 0 in pipeline)
✅ **Environment support** (dev/staging/prod)
✅ **Clean git diffs** (only relevant files change)
✅ **Easy onboarding** (clear structure, self-documenting)

---

## Migration Steps

### Overview

```
Current State:
  job.yaml (200 lines)
  config.json (manual generation)

↓ Migration ↓

New State:
  databricks.yml
  resources/jobs/building_enrichment.yml
  resources/clusters/main_cluster.yml
  Task 0: Config generation automated
```

### Prerequisites

1. **Backup current configuration**:
   ```bash
   cp job.yaml job.yaml.backup
   cp config.json config.json.backup
   ```

2. **Install Databricks CLI**:
   ```bash
   pip install databricks-cli
   ```

3. **Authenticate**:
   ```bash
   databricks configure --token
   ```

---

## Detailed Walkthrough

### Step 1: Create Bundle Structure (5 minutes)

```bash
# Navigate to your project
cd /home/user/code-for-copilot/mre/job1

# Create directory structure
mkdir -p databricks_bundle/resources/{jobs,clusters}

# Copy template files
cp job_yaml_examples/databricks_bundle_example/databricks.yml databricks_bundle/
cp job_yaml_examples/databricks_bundle_example/resources/jobs/building_enrichment.yml databricks_bundle/resources/jobs/
cp job_yaml_examples/databricks_bundle_example/resources/clusters/main_cluster.yml databricks_bundle/resources/clusters/
cp job_yaml_examples/databricks_bundle_example/.gitignore databricks_bundle/
```

### Step 2: Customize Configuration (10 minutes)

#### 2.1 Edit `databricks.yml`

Update workspace settings:

```yaml
workspace:
  host: YOUR_DATABRICKS_HOST  # e.g., https://adb-xxx.azuredatabricks.net

targets:
  development:
    variables:
      workspace_path: /Workspace/Users/YOUR_EMAIL/YOUR_PROJECT/scripts
      email_notifications: YOUR_EMAIL

  production:
    variables:
      workspace_path: /Workspace/Users/YOUR_EMAIL/YOUR_PROJECT/scripts
      email_notifications: YOUR_EMAIL
```

#### 2.2 Verify Task Paths

In `resources/jobs/building_enrichment.yml`, ensure paths match your workspace:

```yaml
tasks:
  - task_key: task0_config_generation
    spark_python_task:
      python_file: ${var.workspace_path}/config_builder.py  # ← Verify this exists
      parameters:
        - ${var.workspace_path}/config.yaml                 # ← Verify this exists
```

#### 2.3 Adjust Cluster Settings

In `resources/clusters/main_cluster.yml`, adjust based on your workload:

```yaml
resources:
  job_clusters:
    main_cluster:
      new_cluster:
        node_type_id: Standard_DS3_v2  # Adjust for your needs
        num_workers: 4                  # Adjust based on data volume
```

### Step 3: Validate Bundle (2 minutes)

```bash
cd databricks_bundle

# Validate configuration
databricks bundle validate

# Expected output:
# ✅ Configuration is valid
```

**Common Validation Errors**:

| Error | Solution |
|-------|----------|
| `yaml: unmarshal errors` | Check YAML syntax (indentation, colons) |
| `variable not defined` | Ensure all `${var.xxx}` are defined in `databricks.yml` |
| `invalid workspace path` | Verify paths exist in Databricks workspace |

### Step 4: Test Deployment to Dev (5 minutes)

```bash
# Deploy to development environment
databricks bundle deploy

# View what was deployed
databricks bundle summary

# Expected output:
# Job: building_enrichment_IND [deployed]
# Cluster: main_cluster [configured]
```

### Step 5: Test Task 0 Execution (5 minutes)

**Option A: Run Entire Job**

```bash
# Trigger full job run
databricks bundle run building_enrichment_IND

# Monitor progress
databricks jobs runs list --job-id <JOB_ID>
```

**Option B: Run Only Task 0**

```bash
# Get job ID
JOB_ID=$(databricks jobs list | grep "Building Data Enrichment" | awk '{print $1}')

# Run task 0 only
databricks jobs run-now --job-id $JOB_ID --task-keys task0_config_generation

# Check output
databricks runs get-output --run-id <RUN_ID>
```

**Expected Task 0 Output**:

```
✅ Loaded config.yaml
✅ Generated 51 configuration values
✅ Written to config.json
✅ Validation: All paths generated successfully
```

### Step 6: Verify Generated Config (3 minutes)

After Task 0 runs:

```bash
# Download generated config.json
databricks workspace export ${WORKSPACE_PATH}/config.json config.json.generated

# Compare with backup
diff config.json.backup config.json.generated

# Should show only expected differences (if any)
```

### Step 7: Run Full Pipeline (30+ minutes)

```bash
# Run complete pipeline
databricks bundle run building_enrichment_IND

# Monitor in Databricks UI:
# 1. Go to Workflows
# 2. Find "Building Data Enrichment - IND [development]"
# 3. Click to view run details
# 4. Watch tasks progress: Task 0 → 1 → 2 → ... → 7
```

### Step 8: Deploy to Production (5 minutes)

Once dev testing passes:

```bash
# Deploy to production environment
databricks bundle deploy -t production

# Verify deployment
databricks bundle summary -t production

# Run production job (when ready)
databricks bundle run building_enrichment_IND -t production
```

---

## Verification

### Checklist

After migration, verify:

- [ ] Task 0 generates `config.json` successfully
- [ ] All 7 subsequent tasks use generated config
- [ ] Email notifications work (check inbox)
- [ ] Output tables created correctly:
  - [ ] `building_enrichment_proportions_input`
  - [ ] `building_enrichment_tsi_input`
  - [ ] `grid_centroids`
  - [ ] `download_status`
  - [ ] `grid_counts`
  - [ ] `building_enrichment_output`
  - [ ] Views: `tsi_proportions_*`
- [ ] Exports generated successfully

### Testing Task 0 Independently

Create a test script:

```bash
#!/bin/bash
# test_task0.sh

echo "🧪 Testing Task 0: Config Generation"

# 1. Backup existing config.json
cp config.json config.json.before

# 2. Delete config.json
rm config.json

# 3. Run Task 0
databricks jobs run-now --job-id <JOB_ID> --task-keys task0_config_generation

# 4. Wait for completion
sleep 10

# 5. Download generated config
databricks workspace export ${WORKSPACE_PATH}/config.json config.json.after

# 6. Verify
if [ -f "config.json.after" ]; then
    echo "✅ Task 0 generated config.json successfully"
    diff config.json.before config.json.after
else
    echo "❌ Task 0 failed to generate config.json"
    exit 1
fi
```

---

## Rollback Plan

If migration fails, rollback:

### Quick Rollback (2 minutes)

```bash
# 1. Restore backup files
cp job.yaml.backup job.yaml
cp config.json.backup config.json

# 2. Delete bundle deployment
databricks bundle destroy

# 3. Redeploy old job
databricks jobs create --json-file job.yaml
```

### Gradual Migration (Lower Risk)

Instead of full migration, do gradual rollout:

1. **Week 1**: Deploy bundle to dev only, keep production on old system
2. **Week 2**: Test extensively in dev, fix issues
3. **Week 3**: Deploy to staging (if available)
4. **Week 4**: Deploy to production after team approval

---

## Comparison: Before vs. After

### Before Migration

**Developer Workflow**:
```bash
# 1. Edit config.yaml locally
vim config.yaml

# 2. Run config builder locally
python config_builder.py config.yaml

# 3. Upload both files
databricks workspace import config.yaml ...
databricks workspace import config.json ...

# 4. Manually trigger job
databricks jobs run-now --job-id <ID>

# ❌ Problems:
# - Easy to forget step 2
# - config.json might be stale
# - Git conflicts on large job.yaml
```

**Git Repository**:
```
repo/
├── job.yaml (200+ lines, frequent conflicts)
├── config.json (generated, but committed?)
├── config.yaml (source of truth)
└── 20+ task scripts
```

### After Migration

**Developer Workflow**:
```bash
# 1. Edit config.yaml
vim config.yaml

# 2. Commit and push
git add config.yaml
git commit -m "Update country to USA"
git push

# 3. Deploy bundle
databricks bundle deploy

# 4. Run job (Task 0 generates config.json automatically)
databricks bundle run building_enrichment_IND

# ✅ Benefits:
# - Task 0 always generates fresh config
# - No manual steps
# - Clean git history
```

**Git Repository**:
```
repo/
├── databricks.yml (50 lines, environment config)
├── resources/
│   ├── jobs/building_enrichment.yml (150 lines, job only)
│   └── clusters/main_cluster.yml (30 lines, cluster only)
├── config.yaml (source of truth)
├── .gitignore (config.json not tracked)
└── src/ (task scripts)
```

---

## Best Practices After Migration

### 1. Git Workflow

```bash
# Only commit YAML files
git add databricks.yml
git add resources/**/*.yml
git add config.yaml

# Never commit generated files
# (.gitignore handles this)
git status  # Should not show config.json
```

### 2. Code Review

When reviewing PRs, focus on:
- Changes to `config.yaml` (pipeline parameters)
- Changes to `resources/jobs/*.yml` (task definitions)
- Changes to `resources/clusters/*.yml` (cluster config)

Ignore/skip:
- `config.json` (auto-generated)
- Bundle deployment artifacts

### 3. Deployment Process

```bash
# Development
databricks bundle deploy         # Default target

# Staging (if available)
databricks bundle deploy -t staging

# Production (after approval)
databricks bundle deploy -t production
```

### 4. Monitoring

Set up alerts:

```yaml
# In resources/jobs/building_enrichment.yml
email_notifications:
  on_start: [team@company.com]
  on_failure: [team@company.com, oncall@company.com]
  on_success: [team@company.com]
```

---

## Troubleshooting

### Issue: Task 0 Fails with "config.yaml not found"

**Solution**:
```bash
# Verify workspace path
databricks workspace ls ${WORKSPACE_PATH}

# Upload config.yaml if missing
databricks workspace import config.yaml ${WORKSPACE_PATH}/config.yaml
```

### Issue: Bundle Deploy Fails with "Job already exists"

**Solution**:
```bash
# Delete old job first
databricks jobs delete --job-id <OLD_JOB_ID>

# Redeploy bundle
databricks bundle deploy
```

### Issue: Variables Not Substituted (shows ${var.xxx})

**Solution**:

Check `databricks.yml` has variables defined:

```yaml
targets:
  development:
    variables:
      workspace_path: /Workspace/...  # ← Must be defined
```

---

## Success Metrics

After successful migration, you should see:

✅ **Reduced Git Conflicts**: 80% fewer merge conflicts
✅ **Faster Onboarding**: New devs productive in < 1 day
✅ **No Stale Configs**: config.json always fresh
✅ **Clear Audit Trail**: Git shows exact config changes
✅ **Environment Isolation**: Dev/prod fully separated

---

## Next Steps

After migration:

1. **Document**: Update team wiki with new workflow
2. **Train**: Hold team session on new bundle structure
3. **Monitor**: Watch first few production runs closely
4. **Iterate**: Gather feedback and refine

---

## Resources

- [Databricks Asset Bundles Docs](https://docs.databricks.com/dev-tools/bundles)
- [Example Bundle Repository](https://github.com/databricks/bundle-examples)
- [Internal Comparison Guide](./APPROACH_COMPARISON.md)
- [Bundle Example](./databricks_bundle_example/)

---

**Questions?** Contact: npokkiri@munichre.com

**Migration Support**: Post in #databricks-help Slack channel
