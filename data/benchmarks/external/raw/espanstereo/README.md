# EspanStereo Import

Place the released EspanStereo source file here as a `.csv`, `.json`, or `.jsonl`.

The current converter expects triplet-style rows with fields equivalent to:

- `context`
- `stereotypical`
- `anti_stereotypical`
- `irrelevant`

Optional metadata fields:

- `country`
- `category`

Then run:

```powershell
conda run -n RAG python scripts/prepare_external_benchmarks.py --datasets espanstereo
```

Or pass an explicit file path:

```powershell
conda run -n RAG python scripts/prepare_external_benchmarks.py --datasets espanstereo --espanstereo-path path\\to\\file.csv
```

The converter builds a project-side 3-way MCQ proxy:

- question: choose the most stereotypical continuation
- choices: stereotypical / anti-stereotypical / irrelevant, shuffled

This is a proxy conversion for the current pipeline, not the original paper's full evaluation protocol.
