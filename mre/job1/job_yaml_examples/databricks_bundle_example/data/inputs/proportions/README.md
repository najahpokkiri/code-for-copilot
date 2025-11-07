# Proportions CSV Files

Place your building type proportion CSV files here.

## 📋 File Format

**File naming convention**: `{COUNTRY_ISO3}_NOS_storey_mapping.csv`

**Examples**:
- `IND_NOS_storey_mapping.csv` - India
- `USA_NOS_storey_mapping.csv` - United States
- `BRA_NOS_storey_mapping.csv` - Brazil

## 📊 Expected CSV Structure

Your CSV should contain building type proportions by storey levels.

**Example columns**:
```csv
smod,built,RES_1,RES_2,RES_3,COM_1,COM_2,IND_1,IND_2,...
0,1,0.45,0.30,0.15,0.05,0.03,0.01,0.01,...
0,2,0.40,0.35,0.20,0.03,0.02,0.00,0.00,...
```

## 📥 How to Add Your File

```bash
# Copy your proportions CSV to this directory
cp /path/to/your/IND_NOS_storey_mapping.csv .

# Update config.yaml to reference it
# Edit: ../../config.yaml
# Set: proportions_csv path to match your filename
```

## 🔍 Verification

After adding your file:

```bash
# Check it's here
ls -lh *.csv

# Should see:
# IND_NOS_storey_mapping.csv (or your country)
```

## ⚙️ Configuration

In `config.yaml`:

```yaml
country:
  iso3: IND  # Your country code

inputs:
  proportions_csv: ${workspace.root_path}/files/data/inputs/proportions/IND_NOS_storey_mapping.csv
```

## 📏 File Size

- ✅ **Recommended**: < 10 MB
- ⚠️ **Warning**: 10-50 MB (works, but slower uploads)
- ❌ **Too large**: > 50 MB (use Volumes instead)

If your file is > 50 MB, keep it in Databricks Volumes and reference the Volume path in `config.yaml`.

## 💡 Tips

1. **Multiple countries**: You can place multiple CSV files here and switch between them by updating `config.yaml`

2. **Version control**: Use descriptive names:
   - `IND_NOS_storey_mapping_v1.csv`
   - `IND_NOS_storey_mapping_041125.csv` (with date)

3. **Testing**: Use a sample/test file for development:
   - `IND_NOS_storey_mapping_sample.csv`
