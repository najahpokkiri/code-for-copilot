# Databricks Job Definitions Guide

This folder contains **3 different approaches** for deploying the geospatial pipeline as a Databricks job.

Choose the approach that best fits your needs.

---

## 📁 Available Approaches

| File | Approach | Best For | Complexity |
|------|----------|----------|------------|
| `approach1_simple.yml` | Hardcoded paths | Quick testing, single environment | ⭐ Simple |
| `approach2_parameterized.yml` | Parameterized paths | Multiple users/environments | ⭐⭐ Moderate |
| `approach3_asset_bundles.yml` | Databricks Asset Bundles | Production, CI/CD, teams | ⭐⭐⭐ Advanced |

---

## 🎯 Decision Tree

```
Do you need CI/CD integration or multiple environments?
│
├─ YES → Use Approach 3 (Asset Bundles) ✅
│
└─ NO
   │
   └─ Will multiple people use this job?
      │
      ├─ YES → Use Approach 2 (Parameterized) ✅
      │
      └─ NO → Use Approach 1 (Simple) ✅
```

---

## Approach 1: Simple Hardcoded Paths

**File:** `approach1_simple.yml`

### Pros
- ✅ Easiest to understand
- ✅ Works immediately
- ✅ No setup required

### Cons
- ❌ Must edit paths for each user
- ❌ Not portable
- ❌ Code changes needed for different environments

### When to Use
- Quick testing
- Single user
- One-time runs
- Throwaway jobs

### Setup

1. **Edit the file:**
   ```yaml
   # Change this path to your location
   python_file: /Workspace/Users/YOUR_NAME/geospatial_pipeline/task1_...

   # Change cluster ID
   existing_cluster_id: YOUR_CLUSTER_ID
   ```

2. **Deploy:**
   ```bash
   # Via Databricks CLI
   databricks jobs create --json-file approach1_simple.yml

   # Or via UI: Copy/paste into Workflows > Create Job
   ```

3. **Run:**
   ```bash
   databricks jobs run-now --job-id YOUR_JOB_ID
   ```

---

## Approach 2: Parameterized Paths ⭐ RECOMMENDED

**File:** `approach2_parameterized.yml`

### Pros
- ✅ Single definition for all users
- ✅ Easy to customize per user
- ✅ No code changes needed
- ✅ Runtime parameter overrides

### Cons
- ⚠️ Slightly more setup
- ⚠️ Must set default parameters

### When to Use
- **Recommended for most cases**
- Multiple users
- Multiple environments
- Shareable job definitions
- Production deployments

### Setup

1. **Edit job parameters** (one time):
   ```yaml
   parameters:
     - name: workspace_path
       default: /Workspace/Users/YOUR_NAME/geospatial_pipeline

     - name: cluster_id
       default: YOUR_CLUSTER_ID
   ```

2. **Deploy:**
   ```bash
   databricks jobs create --json-file approach2_parameterized.yml
   ```

3. **Run with custom parameters:**
   ```bash
   # Override at runtime
   databricks jobs run-now --job-id YOUR_JOB_ID \
     --notebook-params '{"workspace_path": "/Workspace/Users/alice/pipeline"}'
   ```

### Git Clone Scenario

**When someone clones your repo:**

```bash
# 1. Clone
git clone https://github.com/your-org/geospatial-pipeline.git
cd geospatial-pipeline

# 2. Edit config.yaml for their country
vim config.yaml

# 3. Update job parameters in approach2_parameterized.yml
#    (just workspace_path and cluster_id)

# 4. Deploy
databricks jobs create --json-file databricks_jobs/approach2_parameterized.yml

# 5. Run!
databricks jobs run-now --job-id <returned-id>
```

**No code changes needed!**

---

## Approach 3: Asset Bundles (Modern Best Practice)

**File:** `approach3_asset_bundles.yml`

### Pros
- ✅ Modern Databricks standard
- ✅ Full CI/CD support
- ✅ Environment management (dev/staging/prod)
- ✅ Git integration
- ✅ Automatic code sync
- ✅ Permission management

### Cons
- ⚠️ Requires Databricks CLI >= 0.205.0
- ⚠️ Learning curve
- ⚠️ More complex setup

### When to Use
- **Best for production**
- CI/CD pipelines
- Multiple environments (dev/staging/prod)
- Team collaboration
- Enterprise deployments

### Setup

**Prerequisites:**
```bash
# Install Databricks CLI (latest)
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Authenticate
databricks auth login --host https://your-workspace.cloud.databricks.com
```

**Deployment:**

```bash
# 1. Validate
databricks bundle validate --target dev

# 2. Deploy to dev
databricks bundle deploy --target dev

# 3. Run job
databricks bundle run geospatial_pipeline --target dev

# 4. Deploy to prod (when ready)
databricks bundle deploy --target prod
```

### Environment Configuration

The bundle supports 3 environments:

**Dev:**
```bash
databricks bundle deploy --target dev
# Deploys to: /Workspace/Users/{you}/.bundle/geospatial_pipeline/dev
```

**Staging:**
```bash
databricks bundle deploy --target staging
# Deploys to: /Workspace/Users/{you}/.bundle/geospatial_pipeline/staging
```

**Prod:**
```bash
databricks bundle deploy --target prod
# Deploys to: /Shared/.bundle/geospatial_pipeline/prod
```

### Git Clone Scenario (Asset Bundles)

**When someone clones your repo:**

```bash
# 1. Clone
git clone https://github.com/your-org/geospatial-pipeline.git
cd geospatial-pipeline

# 2. Authenticate with Databricks
databricks auth login

# 3. Edit config.yaml for your country
vim config.yaml

# 4. Edit bundle variables (if needed)
vim databricks.yml  # Update cluster_id, iso3, etc.

# 5. Deploy to dev
databricks bundle deploy --target dev

# 6. Run
databricks bundle run geospatial_pipeline --target dev
```

**The bundle automatically:**
- Syncs code to Databricks Workspace
- Creates/updates job definitions
- Manages permissions
- Handles environment separation

---

## 🔄 CI/CD Integration

### GitHub Actions (Asset Bundles)

**`.github/workflows/deploy.yml`:**

```yaml
name: Deploy Geospatial Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Databricks CLI
        uses: databricks/setup-cli@main

      - name: Validate bundle
        env:
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
        run: |
          databricks bundle validate --target prod

      - name: Deploy to production
        if: github.event_name == 'push'
        env:
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
        run: |
          databricks bundle deploy --target prod

      - name: Run job
        if: github.event_name == 'push'
        env:
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
        run: |
          databricks bundle run geospatial_pipeline --target prod
```

### GitLab CI/CD (Asset Bundles)

**`.gitlab-ci.yml`:**

```yaml
stages:
  - validate
  - deploy
  - run

variables:
  DATABRICKS_HOST: $DATABRICKS_HOST
  DATABRICKS_TOKEN: $DATABRICKS_TOKEN

validate:
  stage: validate
  image: databricks/databricks-cli:latest
  script:
    - databricks bundle validate --target prod

deploy_prod:
  stage: deploy
  image: databricks/databricks-cli:latest
  script:
    - databricks bundle deploy --target prod
  only:
    - main

run_job:
  stage: run
  image: databricks/databricks-cli:latest
  script:
    - databricks bundle run geospatial_pipeline --target prod
  only:
    - main
  when: manual
```

---

## 📊 Comparison Table

| Feature | Simple | Parameterized | Asset Bundles |
|---------|--------|---------------|---------------|
| **Ease of setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Portability** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Multi-environment** | ❌ | ⚠️ Manual | ✅ Built-in |
| **CI/CD support** | ⚠️ Basic | ⚠️ Manual | ✅ Full |
| **Git integration** | Manual | Manual | ✅ Automatic |
| **Code sync** | Manual | Manual | ✅ Automatic |
| **Permission mgmt** | Manual | Manual | ✅ Automatic |
| **Recommended for** | Testing | Production | Enterprise |

---

## 🚀 Quick Start Guide

### For Individual Users (Testing)

1. Use **Approach 2 (Parameterized)**
2. Edit `approach2_parameterized.yml` parameters
3. Deploy and run

### For Teams (Production)

1. Use **Approach 3 (Asset Bundles)**
2. Setup CI/CD pipeline
3. Use environment targets (dev/staging/prod)

### For Quick Throwaway Jobs

1. Use **Approach 1 (Simple)**
2. Edit paths directly
3. Deploy via UI

---

## 📚 Additional Resources

**Databricks Documentation:**
- [Jobs API](https://docs.databricks.com/workflows/jobs/jobs-api.html)
- [Asset Bundles](https://docs.databricks.com/dev-tools/bundles/)
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html)

**Best Practices:**
- [Workflow best practices](https://docs.databricks.com/workflows/jobs/jobs-best-practices.html)
- [CI/CD on Databricks](https://docs.databricks.com/dev-tools/ci-cd/index.html)

---

## ❓ FAQ

**Q: Which approach should I use?**
A: For most cases, use **Approach 2 (Parameterized)**. For enterprise/CI/CD, use **Approach 3 (Asset Bundles)**.

**Q: Can I convert between approaches later?**
A: Yes! They all define the same job structure, just different deployment methods.

**Q: Do I need to change my Python scripts?**
A: No! All approaches use the same Python scripts with the same interface.

**Q: What if I don't have Databricks CLI?**
A: Use Approach 1 or 2 and deploy via Databricks UI (copy/paste YAML).

**Q: Can I use these with notebook-based workflows?**
A: Yes, replace `spark_python_task` with `notebook_task` and update paths to notebooks.

**Q: How do I handle secrets (like API keys)?**
A: Use Databricks Secrets. Reference in job params: `{{secrets/scope/key}}`

---

## 🎓 Tutorial: From Git Clone to Running Job

### Scenario: Alice clones the repo and wants to run for USA

**Using Approach 2 (Parameterized) - Recommended:**

```bash
# 1. Clone repo
git clone https://github.com/your-org/geospatial-pipeline.git
cd geospatial-pipeline/mre/job1

# 2. Edit config.yaml
vim config.yaml
# Change:
#   country.iso3: USA
#   inputs.proportions_csv: /path/to/USA_proportions.csv

# 3. Edit job parameters
vim databricks_jobs/approach2_parameterized.yml
# Change default parameters:
#   workspace_path: /Workspace/Users/alice@company.com/geospatial_pipeline
#   cluster_id: alice-cluster-id

# 4. Deploy job
databricks jobs create --json-file databricks_jobs/approach2_parameterized.yml
# Returns: Created job with ID: 12345

# 5. Run job
databricks jobs run-now --job-id 12345
```

**Done! Job runs with Task 0 auto-generating config.json from config.yaml**

---

**Using Approach 3 (Asset Bundles) - Best for teams:**

```bash
# 1. Clone repo
git clone https://github.com/your-org/geospatial-pipeline.git
cd geospatial-pipeline

# 2. Authenticate
databricks auth login

# 3. Edit config.yaml
vim mre/job1/config.yaml
# Change country to USA

# 4. Deploy to dev
databricks bundle deploy --target dev

# 5. Run
databricks bundle run geospatial_pipeline --target dev
```

**Even simpler! Bundle handles code sync, job creation, everything!**

---

## 📝 Summary

- **Testing?** → Use Approach 1 (Simple)
- **Production?** → Use Approach 2 (Parameterized)
- **Enterprise/CI/CD?** → Use Approach 3 (Asset Bundles)

All approaches work with **Task 0** to auto-generate config from YAML, making the pipeline fully self-contained and git-clone friendly!

---

**Questions?** Check the main pipeline documentation in `PIPELINE_GUIDE.md`
