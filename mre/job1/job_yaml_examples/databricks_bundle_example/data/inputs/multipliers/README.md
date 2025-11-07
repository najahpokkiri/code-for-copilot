# TSI Multipliers

Place your Total Sum Insured (TSI) multiplier CSV here.

## 📋 File Format

**File name**: `tsi.csv`

## 📊 Expected CSV Structure

Your CSV should contain TSI multipliers for different building types and storey levels.

**Example structure**:
```csv
category,storey_level,multiplier,unit
RES,1,100,per_sqm
RES,2,120,per_sqm
RES,3,140,per_sqm
COM,1,200,per_sqm
COM,2,220,per_sqm
IND,1,150,per_sqm
...
```

## 📥 How to Add Your File

```bash
# Copy your TSI CSV to this directory
cp /path/to/your/tsi.csv .

# The file should be named exactly: tsi.csv
```

## 🔍 Verification

After adding your file:

```bash
# Check it's here
ls -lh tsi.csv

# Should see:
# tsi.csv
```

## ⚙️ Configuration

In `config.yaml`:

```yaml
inputs:
  tsi_csv: ${workspace.root_path}/files/data/inputs/multipliers/tsi.csv
```

## 📏 File Size

- ✅ **Recommended**: < 5 MB
- ⚠️ **Warning**: 5-20 MB (works, but slower)
- ❌ **Too large**: > 20 MB (use Volumes instead)

## 💡 Tips

1. **Version control**: If your TSI values change, use versioned names:
   - `tsi_v1.csv`
   - `tsi_2024.csv`

   Then update `config.yaml` to reference the correct version.

2. **Backup**: Keep previous versions for reproducibility:
   ```
   multipliers/
   ├── tsi.csv           # Current
   ├── tsi_v1_backup.csv # Previous version
   └── tsi_2023.csv      # Archived version
   ```

3. **Documentation**: Consider adding a `tsi_metadata.txt` describing:
   - Data source
   - Update date
   - Methodology
   - Contact person
