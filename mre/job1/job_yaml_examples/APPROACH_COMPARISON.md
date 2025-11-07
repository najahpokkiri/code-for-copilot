# Databricks Job YAML Structure: Three Approaches

## Overview

This document compares three approaches to structuring Databricks job YAML files, considering:
- **Git-friendliness**: Easy to review diffs, merge conflicts
- **Maintainability**: Easy to update, extend, reuse
- **Onboarding**: New team members can understand quickly
- **CI/CD**: Works well with automated deployments

---

## Approach 1: Monolithic Single-File YAML ⚙️

**Pattern**: Everything in one `job.yaml` file

### Example Structure

```yaml
resources:
  jobs:
    building_data_enrichment:
      name: Building Data Enrichment Pipeline
      description: >-
        Processes GHSL data to generate building density estimates and TSI calculations
        at 5km grid resolution. Handles config generation, grid creation, tile downloads,
        raster statistics, post-processing, views, and exports.

      max_concurrent_runs: 1
      timeout_seconds: 0

      email_notifications:
        on_failure:
          - npokkiri@munichre.com
        on_success:
          - npokkiri@munichre.com

      schedule:
        quartz_cron_expression: "0 0 2 * * ?"  # 2 AM daily
        timezone_id: "America/New_York"
        pause_status: UNPAUSED

      job_clusters:
        - job_cluster_key: main_cluster
          new_cluster:
            spark_version: "13.3.x-scala2.12"
            node_type_id: "Standard_DS3_v2"
            num_workers: 4
            spark_conf:
              "spark.databricks.delta.optimizeWrite.enabled": "true"
              "spark.databricks.delta.autoCompact.enabled": "true"

      tasks:
        # Task 0: Generate Configuration
        - task_key: task0_config_generation
          description: "Generate config.json from config.yaml using config_builder.py"
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config_builder.py
            parameters:
              - config.yaml
          job_cluster_key: main_cluster

        # Task 1: Load Proportions
        - task_key: task1_import_proportions
          description: "Load multiplier CSVs to Delta tables"
          depends_on:
            - task_key: task0_config_generation
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/task1_proportions_to_delta.py
            parameters:
              - --config_path
              - /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config.json
          job_cluster_key: main_cluster

        # Task 2: Grid Generation
        - task_key: task2_grid_generation
          description: "Generate 5km grid centroids with stable cell IDs"
          depends_on:
            - task_key: task1_import_proportions
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/task2_grid_generation.py
            parameters:
              - --config_path
              - /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config.json
          job_cluster_key: main_cluster
          libraries:
            - pypi:
                package: geopandas==0.14.4
            - pypi:
                package: shapely==2.0.4

        # Task 3: Tile Download
        - task_key: task3_tile_download
          description: "Download GHSL tiles from public repositories"
          depends_on:
            - task_key: task2_grid_generation
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/task3_tile_downloader.py
            parameters:
              - --config_path
              - /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config.json
          job_cluster_key: main_cluster
          timeout_seconds: 7200  # 2 hours for downloads

        # Task 4: Raster Statistics
        - task_key: task4_raster_stats
          description: "Extract building counts from raster tiles"
          depends_on:
            - task_key: task3_tile_download
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/task4_raster_stats.py
            parameters:
              - --config_path
              - /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config.json
          job_cluster_key: main_cluster
          libraries:
            - pypi:
                package: rasterio==1.3.9

        # Task 5: Post-Processing
        - task_key: task5_proportions_multiplications
          description: "Calculate sector estimates with TSI proportions"
          depends_on:
            - task_key: task4_raster_stats
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/task5_post_processing.py
            parameters:
              - --config_path
              - /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config.json
          job_cluster_key: main_cluster

        # Task 6: Create Views
        - task_key: task6_create_views
          description: "Create TSI proportion SQL views"
          depends_on:
            - task_key: task5_proportions_multiplications
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/task6_create_views.py
            parameters:
              - --config_path
              - /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config.json
          job_cluster_key: main_cluster

        # Task 7: Export Results
        - task_key: task7_exports
          description: "Export final results to various formats"
          depends_on:
            - task_key: task6_create_views
          spark_python_task:
            python_file: /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/task7_export.py
            parameters:
              - --config_path
              - /Workspace/Users/npokkiri@munichre.com/inventory_nos_db/scripts/config.json
          job_cluster_key: main_cluster
          libraries:
            - pypi:
                package: xlsxwriter==3.2.9
```

### Pros ✅
- **Simple**: Everything in one place
- **Complete visibility**: See entire workflow at once
- **Easy to deploy**: Single file to manage

### Cons ❌
- **Large diffs**: Small changes create large git diffs
- **Merge conflicts**: Multiple people editing = conflicts
- **Hard to reuse**: Can't easily share task definitions across jobs
- **Cluttered**: 200+ lines for complex jobs

### Best For
- Small teams (1-3 people)
- Simple pipelines (< 10 tasks)
- Infrequent changes

---

## Approach 2: Modular Databricks Asset Bundle 📦

**Pattern**: Use Databricks Asset Bundles with separate files per resource type

### Directory Structure

```
mre/job1/
├── databricks.yml              # Main bundle config
├── config.yaml                 # Pipeline config (your existing)
├── resources/
│   ├── jobs/
│   │   └── building_enrichment.yml
│   └── clusters/
│       └── main_cluster.yml
├── src/
│   ├── task0_config_builder.py
│   ├── task1_proportions_to_delta.py
│   ├── task2_grid_generation.py
│   └── ...
└── .gitignore
```

### databricks.yml (Root Config)

```yaml
bundle:
  name: building_data_enrichment

workspace:
  host: https://adb-6685660099993059.19.azuredatabricks.net
  root_path: /Users/npokkiri@munichre.com/.bundle/${bundle.name}/${bundle.target}

include:
  - resources/**/*.yml

targets:
  development:
    mode: development
    default: true
    workspace:
      host: https://adb-6685660099993059.19.azuredatabricks.net
    variables:
      catalog: prp_mr_bdap_projects_dev
      schema: geospatialsolutions_dev

  production:
    mode: production
    workspace:
      host: https://adb-6685660099993059.19.azuredatabricks.net
    variables:
      catalog: prp_mr_bdap_projects
      schema: geospatialsolutions

variables:
  country_iso3:
    description: "ISO3 country code to process"
    default: IND
```

### resources/jobs/building_enrichment.yml

```yaml
resources:
  jobs:
    building_data_enrichment_${var.country_iso3}:
      name: Building Data Enrichment - ${var.country_iso3}
      description: |
        Processes GHSL data for ${var.country_iso3} to generate building estimates.

        Pipeline stages:
        0. Config generation from YAML
        1. Load multipliers to Delta
        2. Generate grid centroids
        3. Download GHSL tiles
        4. Extract raster statistics
        5. Post-process with TSI calculations
        6. Create SQL views
        7. Export to various formats

      max_concurrent_runs: 1
      job_cluster_key: ${resources.job_clusters.main_cluster.id}

      email_notifications:
        on_failure: [npokkiri@munichre.com]

      tasks:
        - task_key: task0_config_generation
          description: Generate config.json from config.yaml
          spark_python_task:
            python_file: ../src/config_builder.py
            parameters: ["config.yaml"]

        - task_key: task1_import_proportions
          depends_on: [{task_key: task0_config_generation}]
          spark_python_task:
            python_file: ../src/task1_proportions_to_delta.py
            parameters:
              - --config_path
              - config.json

        - task_key: task2_grid_generation
          depends_on: [{task_key: task1_import_proportions}]
          spark_python_task:
            python_file: ../src/task2_grid_generation.py
            parameters:
              - --config_path
              - config.json
          libraries:
            - pypi: {package: "geopandas==0.14.4"}
            - pypi: {package: "shapely==2.0.4"}

        # ... (remaining tasks follow same pattern)
```

### resources/clusters/main_cluster.yml

```yaml
resources:
  job_clusters:
    main_cluster:
      new_cluster:
        spark_version: "13.3.x-scala2.12"
        node_type_id: Standard_DS3_v2
        num_workers: 4
        spark_conf:
          spark.databricks.delta.optimizeWrite.enabled: "true"
          spark.databricks.delta.autoCompact.enabled: "true"
```

### Pros ✅
- **Modular**: Separate concerns (jobs, clusters, permissions)
- **Environment-aware**: Easy dev/staging/prod configs
- **Reusable**: Share cluster configs across jobs
- **Clean diffs**: Changes isolated to specific files
- **Variables**: Parameterize for flexibility
- **CI/CD native**: Built-in deployment commands

### Cons ❌
- **More files**: Need to navigate multiple files
- **Learning curve**: Need to learn DAB concepts
- **Tooling required**: Need Databricks CLI

### Best For
- Medium to large teams (3+ people)
- Multiple environments (dev/staging/prod)
- Reusable components across jobs
- CI/CD pipelines

---

## Approach 3: Template-Based with Jinja2 🎨

**Pattern**: Use Jinja2 templates to generate job YAML dynamically

### Directory Structure

```
mre/job1/
├── templates/
│   ├── job.yml.j2              # Job template
│   └── task.yml.j2             # Task template
├── config/
│   ├── pipeline_config.yaml    # Pipeline definition
│   └── environments.yaml       # Environment configs
├── scripts/
│   └── generate_job_yaml.py    # Template renderer
└── generated/
    └── job.yml                 # Generated (gitignored)
```

### templates/job.yml.j2

```yaml
resources:
  jobs:
    {{ job.name }}:
      name: {{ job.display_name }}
      description: {{ job.description }}

      max_concurrent_runs: {{ job.max_concurrent_runs | default(1) }}

      {% if job.schedule %}
      schedule:
        quartz_cron_expression: "{{ job.schedule.cron }}"
        timezone_id: "{{ job.schedule.timezone }}"
      {% endif %}

      job_cluster_key: main_cluster

      tasks:
      {% for task in job.tasks %}
        - task_key: {{ task.key }}
          description: "{{ task.description }}"
          {% if task.depends_on %}
          depends_on:
            {% for dep in task.depends_on %}
            - task_key: {{ dep }}
            {% endfor %}
          {% endif %}
          spark_python_task:
            python_file: {{ workspace_path }}/{{ task.script }}
            parameters:
              - --config_path
              - {{ workspace_path }}/config.json
          job_cluster_key: main_cluster
          {% if task.libraries %}
          libraries:
            {% for lib in task.libraries %}
            - pypi:
                package: {{ lib }}
            {% endfor %}
          {% endif %}
      {% endfor %}
```

### config/pipeline_config.yaml

```yaml
job:
  name: building_data_enrichment
  display_name: Building Data Enrichment Pipeline
  description: Processes GHSL data to generate building estimates
  max_concurrent_runs: 1

  tasks:
    - key: task0_config_generation
      description: Generate config.json from config.yaml
      script: config_builder.py
      depends_on: []
      libraries: []

    - key: task1_import_proportions
      description: Load multiplier CSVs to Delta
      script: task1_proportions_to_delta.py
      depends_on: [task0_config_generation]
      libraries: []

    - key: task2_grid_generation
      description: Generate 5km grid centroids
      script: task2_grid_generation.py
      depends_on: [task1_import_proportions]
      libraries:
        - geopandas==0.14.4
        - shapely==2.0.4

    - key: task3_tile_download
      description: Download GHSL tiles
      script: task3_tile_downloader.py
      depends_on: [task2_grid_generation]
      libraries: []

    - key: task4_raster_stats
      description: Extract building counts
      script: task4_raster_stats.py
      depends_on: [task3_tile_download]
      libraries:
        - rasterio==1.3.9

    - key: task5_proportions_multiplications
      description: Calculate sector estimates
      script: task5_post_processing.py
      depends_on: [task4_raster_stats]
      libraries: []

    - key: task6_create_views
      description: Create TSI views
      script: task6_create_views.py
      depends_on: [task5_proportions_multiplications]
      libraries: []

    - key: task7_exports
      description: Export final results
      script: task7_export.py
      depends_on: [task6_create_views]
      libraries:
        - xlsxwriter==3.2.9
```

### scripts/generate_job_yaml.py

```python
#!/usr/bin/env python3
"""
Generate Databricks job YAML from templates
"""
import yaml
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def generate_job_yaml(
    pipeline_config_path: str,
    environment: str = "production",
    output_path: str = "generated/job.yml"
):
    """Generate job YAML from template and config"""

    # Load configs
    with open(pipeline_config_path) as f:
        config = yaml.safe_load(f)

    with open(f"config/environments/{environment}.yaml") as f:
        env_config = yaml.safe_load(f)

    # Setup Jinja2
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("job.yml.j2")

    # Render
    output = template.render(
        job=config["job"],
        workspace_path=env_config["workspace_path"],
        cluster=env_config["cluster"]
    )

    # Write
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output)

    print(f"✅ Generated {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="production")
    args = parser.parse_args()

    generate_job_yaml(
        "config/pipeline_config.yaml",
        environment=args.env
    )
```

### Usage

```bash
# Generate for production
python scripts/generate_job_yaml.py --env production

# Generate for dev
python scripts/generate_job_yaml.py --env dev

# Deploy
databricks workspace import generated/job.yml ...
```

### Pros ✅
- **DRY**: No repetition across tasks
- **Flexible**: Easy to add new tasks
- **Multi-environment**: Generate for dev/staging/prod
- **Type-safe**: Validate config before generation
- **Testable**: Can unit test generation logic

### Cons ❌
- **Abstraction layer**: Need to understand templates
- **Debugging**: Errors in templates can be cryptic
- **Custom tooling**: Maintain generation scripts

### Best For
- Large teams with many similar pipelines
- Standardized task patterns across jobs
- Multiple environments with variations
- When you want to enforce conventions

---

## Comparison Matrix

| Aspect | Monolithic | Asset Bundle | Template-Based |
|--------|-----------|--------------|----------------|
| **Setup Complexity** | Low | Medium | High |
| **Git-friendliness** | Low | High | High |
| **Reusability** | Low | High | Very High |
| **Environment Support** | Manual | Built-in | Custom |
| **Learning Curve** | Low | Medium | High |
| **Maintenance** | High | Low | Low |
| **CI/CD Integration** | Manual | Native | Custom |
| **Team Size** | 1-3 | 3-10 | 10+ |

---

## Recommended Approach: **Asset Bundle (Approach 2)**

For your use case, I recommend **Approach 2: Databricks Asset Bundle** because:

1. ✅ **Git-friendly**: Changes to tasks don't affect cluster configs
2. ✅ **Industry standard**: Databricks official pattern
3. ✅ **Environment support**: Built-in dev/prod separation
4. ✅ **Easy onboarding**: New cloners see clear structure
5. ✅ **Future-proof**: Supports expansion to more jobs

### Migration Path

```bash
# 1. Install Databricks CLI
pip install databricks-cli

# 2. Create bundle structure
databricks bundle init

# 3. Move job config to resources/jobs/
# 4. Add Task 0 for config generation
# 5. Test deployment
databricks bundle validate
databricks bundle deploy -t dev

# 6. Deploy to production
databricks bundle deploy -t production
```

---

## Task 0: Config Generation Integration

Regardless of approach, here's how to add config generation as Task 0:

```yaml
- task_key: task0_config_generation
  description: |
    Generate config.json from config.yaml using config_builder.py.
    This ensures reproducible configs from version-controlled YAML.
  spark_python_task:
    python_file: /Workspace/.../config_builder.py
    parameters:
      - config.yaml
      - --output
      - config.json
  job_cluster_key: main_cluster
  timeout_seconds: 300  # 5 minutes max
```

**Key Points**:
- All subsequent tasks depend on this
- Fast execution (config generation is quick)
- Self-documenting: Anyone can see config is generated
- Reproducible: Clone repo → run job → works

---

## Resources

### Official Documentation
- [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles)
- [Job Configuration Reference](https://docs.databricks.com/workflows/jobs/jobs-api)
- [Bundle Settings](https://docs.databricks.com/dev-tools/bundles/settings)

### Community Resources
- [DAB Guide (Medium)](https://medium.com/@aryan.sinanan/databricks-dab-databricks-assets-bundles-yaml-semi-definitive-guide-06a8859166d1)
- [DAB Best Practices](https://community.databricks.com/t5/technical-blog/customizing-target-deployments-in-databricks-asset-bundles/ba-p/124772)

### Example Repositories
- [Databricks DAB Examples](https://github.com/databricks/bundle-examples)
- [MLOps Stack](https://github.com/databricks/mlops-stacks)
