# Dialogue Summarization

# Installation
```bash
pip install [-e] .
```

# Running benchmarks
In this release, dataset is not included in the zip file due to file size limit.

Download AMI, ICSI, DialogSum, SAMSum, MediaSum, Summscreen, and ADSC. Modify the data paths in scripts below accordingly.

Public release of this repository will contain data files as well.

## Meeting benchmarks
```bash
scripts/run_meetings.sh
```

## Other baselines
```bash
scripts/run_baselines.sh
```
