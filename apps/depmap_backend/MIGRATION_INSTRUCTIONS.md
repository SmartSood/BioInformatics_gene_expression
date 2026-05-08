# Database Migration Instructions

## Add geneDepmapResults Field to TrainingRun

To add the `geneDepmapResults` JSON field to store DepMap CSV file locations:

### 1. Run Prisma Migration

```bash
cd /Users/smarthsood/Desktop/Gene_startup/gene_web
npx prisma migrate dev --name add_gene_depmap_results
```

### 2. Generate Prisma Clients

```bash
# Generate TypeScript client
npx prisma generate

# Generate Python client
cd packages/db
python -m prisma generate
```

### 3. Verify Migration

The `TrainingRun` model should now have:
```prisma
geneDepmapResults Json?  // JSON mapping gene names to DepMap CSV file paths
```

### 4. Database Structure

The `geneDepmapResults` field will store JSON in this format:
```json
{
  "ERCC3": "/path/to/outputs/11/experiment-id/job-id_associations.csv",
  "TP53": "/path/to/outputs/11/experiment-id/job-id2_associations.csv"
}
```

Each gene name maps to its corresponding CSV file path.

