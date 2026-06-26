# radioscience

This repository packages the Paper2Code inputs and generated outputs for a RadioScience reproduction run. It includes the source paper inputs, prompt/data assets, generated planning and analysis artifacts, and the generated Python reproduction project for the COSMIC-2 radio occultation back-propagation geolocation workflow.

## What This Is

The generated project targets a reproduction of the paper:

**"Geolocation of the Ionospheric Irregularities in the Equatorial F Layer by Back Propagation of COSMIC-2 Radio Occultation Signals."**

The scientific goal is to locate equatorial F-layer ionospheric irregularities from COSMIC-2 high-rate GNSS radio occultation scintillation observations. The method uses back propagation of radio signals to infer where along the receiver-transmitter line of sight the scintillation-producing irregularity layer is most likely located.

The reproduction code generated under `outputs/RadioScience_repo/` is organized around two major workflows:

1. **Synthetic validation**
   - Generates phase-screen scenarios.
   - Runs forward/back propagation experiments.
   - Tests single-screen recovery, scan-angle sensitivity, two-screen cases, thermal noise, and ambiguous geolocation behavior.

2. **Real COSMIC-2 processing**
   - Loads high-rate COSMIC-2 phase and amplitude/SNR observations.
   - Segments observations into short analysis windows.
   - Computes scintillation metrics such as S4 and sigma_phi.
   - Builds line-of-sight and magnetic-field-aware back-propagation geometry.
   - Runs geolocation quality control.
   - Produces geolocation tables, maps, and comparison products.

No machine-learning model is trained here. This is a scientific signal-processing and geolocation reproduction project.

## Repository Contents

- `data/`
  - Prompt templates and Paper2Code dataset metadata/assets.

- `examples/`
  - Original and cleaned paper JSON examples.
  - Includes RadioScience inputs plus the original copied Transformer example files from the source workspace.

- `input_pdfs/`
  - The RadioScience paper PDF used as source material.

- `outputs/RadioScience/`
  - Paper2Code planning, analysis, coding artifacts, model-call logs, and intermediate outputs for the RadioScience run.

- `outputs/RadioScience_repo/`
  - The generated reproduction codebase.
  - This is the directory to inspect or run if you want to test the generated implementation.

- `outputs/Transformer/` and `outputs/Transformer_repo/`
  - Copied from the original workspace because the full `outputs/` folder was included.
  - These are not the main RadioScience deliverable.

## Generated Project Layout

The generated RadioScience implementation is located at:

```text
outputs/RadioScience_repo/
```

Important files:

```text
outputs/RadioScience_repo/README.md
outputs/RadioScience_repo/requirements.txt
outputs/RadioScience_repo/config/default.yaml
outputs/RadioScience_repo/main.py
outputs/RadioScience_repo/src/
```

Major source modules include:

```text
src/config.py
src/core/constants.py
src/core/types.py
src/data/
src/geometry/
src/geolocation/
src/propagation/
src/signal/
src/experiments/
src/visualization/
```

## Setup

From the root of this repository:

```powershell
cd outputs/RadioScience_repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you already have an active virtual environment, you can skip creating `.venv` and run:

```powershell
pip install -r outputs/RadioScience_repo/requirements.txt
```

## Basic Checks

From the repository root:

```powershell
python -m compileall -q outputs/RadioScience_repo
```

After dependencies are installed:

```powershell
python outputs/RadioScience_repo/main.py --help
```

For first testing, prefer the synthetic workflow before attempting full real COSMIC-2 processing. The real-data path may require additional external datasets, orbit products, and local paths configured in `outputs/RadioScience_repo/config/default.yaml`.

## Notes And Caveats

- This repo contains generated code and generated reasoning artifacts. Treat it as a reproduction scaffold, not as a fully validated scientific package.
- The generated code compiled successfully at creation time, but dependency installation and runtime validation should still be performed.
- Some paper details were ambiguous and are documented in the generated planning/config files, including phase detrending choices, exact multi-valued `D(L)=0` handling, and L1/L2 comparison definitions.
- The copied folders include some non-RadioScience example/output material because the full folders were copied as requested.

## GitHub Upload

This repository is prepared as a standalone Git repository. To push it to GitHub:

```powershell
cd C:\Users\brett\Desktop\Paper2Code\radioscience
git remote add origin https://github.com/brettnguyen/radioscience.git
git push -u origin main
```
