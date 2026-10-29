# Prostate imaging research

Research notebooks from Badrinath Padmanabhan's prostate imaging work: paired MRI and histopathology exploration, whole-slide histology processing, and a later restart of the histology-to-diffusion-MRI direction.

## What is here

| Component                                                                                  | Source and scope                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Paired MRI and pathology](notebooks/reference/assessing_prostate_fused_mri_dataset.ipynb) | Original exploration of DICOM/MHA images, XML annotations, spreadsheet-based slice correspondence and stain normalisation. This uses the fused MRI/pathology collection, not PI-CAI.                                                                      |
| [Histology ROI experiments](notebooks/reference/automating_ROI_PC.ipynb)                   | Original PANDA slide/mask inspection, patch selection, MSER, Cellpose experiments and SVG export. Historical exploratory code, not a reliable Run All pipeline.                                                                                           |
| [PICAI Histology Restart](notebooks/PICAI_Histology_Restart.ipynb)                         | September 2026 Codex-assisted continuation: a calibrated PANDA patch, provider-specific region labels, MSER candidates and SVG export in physical units. The historical filename is retained, but the notebook processes PANDA histology, not PI-CAI MRI. |
| [Reusable histology functions](histology_pipeline.py)                                      | The existing restart implementation extracted into an importable module, with its [six synthetic unit tests](test_histology.py).                                                                                                                          |

PI-CAI MRI belongs to the broader research direction. This release does **not** contain a verified trained PI-CAI detector, a completed 3D tissue model, or a validated diffusion-MRI simulation. Keeping these datasets and milestones separate is important.

## Start with the restart notebook

Use Python 3.11 in an isolated environment. The public environment retains the restart's core dependency pins, with SciPy 1.14.1 for compatibility with the tested macOS environment. This is not a promise that every historical experiment supports these versions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_histology
python -m pip install jupyterlab
export PANDA_DATASET_ROOT=/absolute/path/to/authorised/panda
export PANDA_IMAGE_ID=your_authorised_slide_id
jupyter lab notebooks/PICAI_Histology_Restart.ipynb
```

Supply `train.csv`, `train_images/<image_id>.tiff` and `train_label_masks/<image_id>_mask.tiff`. Image and mask grids must agree. Pixel calibration must come from verified image metadata or an explicit verified override. The notebook fails rather than guessing a missing calibration.

The default MSER path runs on CPU. Cellpose is optional, disabled by default and not validated in this release. Enabling it installs a separate model dependency and may download weights. Review that branch before use.

Outputs go into a new timestamped directory under `outputs/`, or `PROSTATE_OUTPUT_ROOT` if set. **Generated images, masks, filenames, manifests and SVG metadata can contain study information. Keep the entire output directory private.** Do not commit executed notebooks.

## Interpretation

PANDA masks describe tissue regions, not individual cell membranes. The MSER objects are unvalidated nucleus-like candidates. Candidate counts, physical coordinates and successful SVG export do not establish biological accuracy. Patch selection is biased toward the requested tissue region and is not an unbiased evaluation sample.

Before any later simulation, establish expert segmentation references, justify a 2D-to-3D geometry model and independently check physical units and mesh validity. An outline of a nucleus cannot silently stand in for a whole-cell membrane.

## Provenance and validation

[WORKFLOW.md](WORKFLOW.md) records the project stages, historical-notebook setup and remaining limitations. [source_manifest.json](source_manifest.json) records the original filenames, privately retained source hashes and publication changes.

The original notebooks were retrieved from the existing research Drive. Publication cleanup removes outputs, execution metadata, private paths, embedded slide identifiers and private document links. The paired-MRI examples use `0000` placeholders; these are not real patient examples. No datasets, images, grades tied to individual slides, credentials or model weights are published.

The six existing synthetic tests check mask labels, provider logic, physical scaling, missing inputs, MSER overlap handling and SVG geometry. Notebook structure and code-cell syntax are checked separately. Real-data research results and the historical Cellpose/stain-normalisation branches have not been rerun for this release. The 2026 restart is identified separately from the earlier research work.

This is collaborative research source, not a claim of sole ownership of the datasets or wider project. No project-wide licence is added. Obtain data through authorised providers and observe their terms.
