# Project map

## 1. Paired MRI and histopathology exploration

`notebooks/reference/assessing_prostate_fused_mri_dataset.ipynb` preserves the original correspondence prototype. It reads image metadata and annotations, parses histology filenames, reads the correspondence workbook, associates slices, displays matched inputs and experiments with stain normalisation.

Set `PROSTATE_DATA_ROOT` to a private data directory. The expected layout is `paired_mri/` with the original collection's `Prostate-fused-MRI-pathology`, `PKG - Prostate Fused-MRI-Pathology`, `fused_prostate_matlab` and correspondence-workbook paths. The publication uses `manifest-example` and `aaa0000` placeholders. Select your authorised manifest and example files before running those cells.

These are filename/spreadsheet correspondence heuristics, not a validated geometric registration. The source includes unfinished scaffolding, first-match choices and experimental stain-normalisation alternatives. Verify slice identity, coordinate systems and image metadata independently. Historical optional imports include SimpleITK, pydicom, openpyxl, histomicstk, torchstain, torch and torchvision; they are not included in the restart environment.

## 2. Original PANDA ROI automation

`notebooks/reference/automating_ROI_PC.ipynb` contains the earlier slide/mask inspection and patch experiments. It explores provider masks, MSER regions, Cellpose and vector outlines. It has repeated definitions and alternative branches with notebook-state dependencies.

Set `PROSTATE_DATA_ROOT`, with the PANDA collection at `panda/`, and set `PANDA_IMAGE_ID` to an authorised image. Former hardcoded slide IDs now refer to that explicit choice. Any standalone experiment patch belongs under `automation/input/`. The notebook is a source archive: read and run selected cells, not Run All. Optional historical dependencies include cellpose, svgwrite, tifffile and imagecodecs. Model downloads or GPU use must be chosen deliberately.

## 3. September 2026 restart

`notebooks/PICAI_Histology_Restart.ipynb` and `histology_pipeline.py` implement the later Codex-assisted restart. The code checks paired image/mask inputs and provider labels, uses micrometre calibration, selects a bounded patch, compares MSER settings and exports contours in a consistent physical frame.

The module and notebook retain the same existing core algorithm; the module enables synthetic unit tests without opening any medical images. The notebook retains its embedded implementation so it can be read independently. Publication changes are configuration and privacy cleanup, not a new scientific result.

## 4. Not yet established

The reviewed source does not establish a trained PI-CAI MRI model, cross-dataset correspondence between PANDA and PI-CAI, validated cellular segmentation, a 3D substrate, or a diffusion simulation. The PI-CAI dataset folder is not a substitute for such source code. No MRI detection metrics or biological accuracy claims are made here.

## Validation boundary

`python -m unittest -v test_histology` uses only synthetic arrays and temporary files. The six tests exercise red-channel annotation preservation, provider-specific labels, anisotropic scaling and patch edges, missing-input rejection, nested MSER candidates and SVG holes/physical units. They do not test biological accuracy or reproduce original study outcomes.

All generated results, including apparently harmless tables and SVGs, must remain private until independently reviewed for data rights and identifying metadata.
