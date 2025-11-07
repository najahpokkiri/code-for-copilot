# Databricks Asset Bundle - Building Data Enrichment

This directory contains a complete Databricks Asset Bundle configuration for the Building Data Enrichment pipeline.

## 📁 Directory Structure

```
databricks_bundle_example/
├── databricks.yml                    # Root bundle configuration
├── resources/
│   ├── jobs/
│   │   └── building_enrichment.yml   # Job definition with Task 0-7
│   └── clusters/
│       └── main_cluster.yml          # Cluster configuration
├── .gitignore                        # Git ignore patterns
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Install Databricks CLI**:
   ```bash
   pip install databricks-cli
   ```

2. **Authenticate**:
   ```bash
   databricks configure --token
   ```
   - Host: `https://adb-6685660099993059.19.azuredatabricks.net`
   - Token: Your personal access token

### Deployment

#### Option 1: Deploy to Development

```bash
# Validate bundle configuration
databricks bundle validate

# Deploy to development environment (default)
databricks bundle deploy

# View deployed resources
databricks bundle summary
```

#### Option 2: Deploy to Production

```bash
# Deploy to production environment
databricks bundle deploy -t production

# Run the job immediately
databricks bundle run building_enrichment_IND -t production
```

## 🏗️ Architecture

### Task Flow with Task 0

```
Task 0: Config Generation (config.yaml → config.json)
    ↓
Task 1: Import Proportions (CSV → Delta)
    ↓
Task 2: Grid Generation (Boundaries → Grid)
    ↓
Task 3: Tile Download (JRC → Volumes)
    ↓
Task 4: Raster Stats (Tiles → Counts)
    ↓
Task 5: Post-Processing (Counts → Estimates)
    ↓
Task 6: Create Views (Estimates → SQL Views)
    ↓
Task 7: Export (Views → CSV/Excel)
```

### Key Features

✅ **Task 0 Integration**: Config generation is now a formal task
✅ **Environment Support**: Dev, Staging, Production configs
✅ **Git-Friendly**: Only YAML files tracked, JSON generated
✅ **Parameterized**: Easy to switch countries (IND → USA)
✅ **Self-Documenting**: Descriptions in YAML
✅ **Retry Logic**: Automatic retries with backoff
✅ **Notifications**: Email alerts on start/success/failure

## 📋 Configuration

### Environment Variables

Each target (dev/staging/production) has its own variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `prp_mr_bdap_projects` |
| `schema` | Schema name | `geospatialsolutions` |
| `country_iso3` | Country code | `IND`, `USA`, `BRA` |
| `workspace_path` | Script location | `/Workspace/Users/.../scripts` |
| `email_notifications` | Alert email | `user@company.com` |

### Switching Countries

To process a different country:

```yaml
# In databricks.yml, under target:
variables:
  country_iso3: USA  # Change from IND to USA
```

Then redeploy:

```bash
databricks bundle deploy -t production
```

### Customizing Cluster

Edit `resources/clusters/main_cluster.yml`:

```yaml
resources:
  job_clusters:
    main_cluster:
      new_cluster:
        node_type_id: Standard_DS4_v2  # Bigger nodes
        num_workers: 8                  # More workers
```

## 🔄 Workflow

### For New Team Members (Git Clone Workflow)

```bash
# 1. Clone repository
git clone <repo-url>
cd building_data_enrichment

# 2. Install Databricks CLI
pip install databricks-cli

# 3. Authenticate
databricks configure --token

# 4. Validate configuration
databricks bundle validate

# 5. Deploy to dev environment
databricks bundle deploy

# 6. Run the job
databricks bundle run building_enrichment_IND

# ✅ Done! Task 0 will generate config.json automatically
```

**Key Point**: New team members don't need to manually create `config.json` - Task 0 does it!

### For Development

```bash
# Make changes to YAML files
vim resources/jobs/building_enrichment.yml

# Validate changes
databricks bundle validate

# Deploy to dev
databricks bundle deploy

# Test run
databricks bundle run building_enrichment_IND
```

### For Production Deployment

```bash
# Deploy to production
databricks bundle deploy -t production

# Monitor job
databricks jobs list --output JSON | jq '.jobs[] | select(.settings.name | contains("Building Data Enrichment"))'
```

## 📊 Benefits vs. Single YAML File

| Aspect | Single YAML | Asset Bundle |
|--------|------------|--------------|
| **File Size** | 200+ lines | ~50 lines per file |
| **Git Diffs** | Large, hard to review | Small, focused |
| **Merge Conflicts** | Frequent | Rare |
| **Reusability** | Low | High (cluster config shared) |
| **Environment Support** | Manual | Built-in |
| **Onboarding** | Confusing | Clear structure |

## 🎯 Task 0: Why It Matters

### Before (Manual Config)

```bash
# Developer workflow:
1. Edit config.yaml
2. Run: python config_builder.py config.yaml
3. Commit config.yaml (maybe forget config.json?)
4. Other dev pulls, gets stale config.json
5. Pipeline fails with "file not found"
```

### After (Task 0 in Pipeline)

```bash
# Developer workflow:
1. Edit config.yaml
2. Commit config.yaml
3. Push to git
4. Pipeline runs → Task 0 generates fresh config.json
5. ✅ Always consistent, always works
```

### Benefits

- ✅ **Reproducible**: Config always generated from YAML
- ✅ **No drift**: Can't have stale config.json
- ✅ **Self-documenting**: Pipeline shows config generation
- ✅ **Git-friendly**: Only YAML tracked
- ✅ **Onboarding**: New devs don't need to know about config_builder.py

## 🔍 Troubleshooting

### Validation Errors

```bash
# Check what's wrong
databricks bundle validate

# Common issues:
# - Invalid YAML syntax
# - Missing required fields
# - Incorrect variable references
```

### Deployment Failures

```bash
# Check deployment status
databricks bundle summary

# View logs
databricks jobs runs get-output --run-id <run-id>
```

### Task 0 Failures

If Task 0 fails:

1. Check `config.yaml` exists in workspace
2. Verify `config_builder.py` is executable
3. Check workspace path is correct in `databricks.yml`

## 📚 Resources

### Official Documentation
- [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles)
- [Bundle Configuration Reference](https://docs.databricks.com/dev-tools/bundles/settings)
- [CLI Reference](https://docs.databricks.com/dev-tools/cli)

### Example Bundles
- [Databricks Bundle Examples](https://github.com/databricks/bundle-examples)
- [MLOps Stacks](https://github.com/databricks/mlops-stacks)

### Community
- [Databricks Community](https://community.databricks.com/)
- [DAB Guide on Medium](https://medium.com/@aryan.sinanan/databricks-dab-databricks-assets-bundles-yaml-semi-definitive-guide-06a8859166d1)

## 🤝 Contributing

When making changes:

1. Edit YAML files (not generated JSON)
2. Validate: `databricks bundle validate`
3. Test in dev: `databricks bundle deploy`
4. Create PR with clear description
5. Deploy to production after approval

## 📧 Support

For questions or issues:
- Check Databricks logs in workspace
- Review task descriptions in job UI
- Contact: npokkiri@munichre.com

---

**Happy bundling! 🎉**
