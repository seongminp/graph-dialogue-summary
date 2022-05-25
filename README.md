# Dialogue Summarization

Code for [**_Unsupervised Abstractive Dialogue Summarization with Word Graphs and POV Conversion_**](https://aclanthology.org/2022.wit-1.1/) (our submission to [WIT Workshop @ ACL2022](https://megagon.ai/2nd-workshop-on-deriving-insights-from-user-generated-text-wit/)).


# Installation
```bash
pip install [-e] .
```

# Running benchmarks

## Meeting benchmarks
```bash
scripts/run_meetings.sh
```

## Other baselines

Download DialogSum, SAMSum, MediaSum, Summscreen, and ADSC. Modify the data paths in scripts below accordingly.

```bash
scripts/run_baselines.sh
```
