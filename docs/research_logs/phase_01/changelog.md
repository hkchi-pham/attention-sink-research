# Phase 1 Redesign Changelog

Date: July 2026

## Motivation

The original Phase 1 notebooks successfully extracted attention tensors and produced qualitative visualisations. However, the analysis was limited in scope and did not provide a robust quantitative characterisation of attention sinks.

The notebooks were redesigned to improve reproducibility, modularity, and experimental rigor before proceeding to Phase 2.

---

## Notebook 1 – Attention Extraction

### Improvements

- Refactored the extraction pipeline into a more modular structure.
- Added configurable experiment settings.
- Improved metadata collection for every prompt.
- Added automatic saving of attention tensors and experiment metadata.
- Standardised output directory structure.
- Added sanity checks for attention tensor shapes.
- Improved reproducibility through deterministic configuration where possible.

### Why

The original notebook focused primarily on collecting attention tensors. The redesign ensures that future experiments can be reproduced consistently and that extracted data contain sufficient metadata for downstream quantitative analysis.

---

## Notebook 2 – Analysis

### Improvements

- Replaced purely qualitative analysis with a quantitative analysis pipeline.
- Introduced multiple sink-score definitions:
  - mean_all
  - mean_from_k
  - mean_second_half
  - last
- Added layer-wise sink progression analysis.
- Added layer × head sink matrices.
- Added per-head sink profiles.
- Added language comparison (English vs Vietnamese).
- Added prompt-category comparison.
- Added sequence-length analysis.
- Added correlation analysis between sink score and sequence length.
- Added qualitative visualisation across early, middle, and late layers.
- Standardised plotting and figure generation.

### Why

Attention sinks can be measured in multiple ways, each with different biases. The redesigned notebook compares several definitions instead of assuming a single metric, allowing a more reliable choice for subsequent experiments.

---

## Outcome

Phase 1 now provides:

- a reproducible extraction pipeline,
- a quantitative characterisation framework,
- validated visualisations,
- candidate sink metrics for future experiments,
- identification of important experimental confounds.

Phase 1 is considered complete and provides the foundation for Phase 2.
