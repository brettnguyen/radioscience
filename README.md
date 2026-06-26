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

## How This Was Created

This repository was created from a Paper2Code workflow rather than written manually from scratch.

High-level creation process:

1. **Started with the RadioScience paper PDF**
   - The source paper was saved under `input_pdfs/radio_science.pdf`.

2. **Converted the paper into machine-readable JSON**
   - The paper content was converted into JSON form so the Paper2Code pipeline could process the title, abstract, sections, equations, and extracted text.
   - The resulting paper files are stored under `examples/`, including:

```text
examples/RadioScience.json
examples/RadioScience_cleaned.json
```

3. **Ran the Paper2Code planning stage**
   - Paper2Code read the cleaned JSON paper representation.
   - It generated:
     - an overall reproduction plan,
     - an architecture design,
     - a file-by-file implementation plan,
     - a configuration file.
   - These artifacts are stored in:

```text
outputs/RadioScience/planning_artifacts/
outputs/RadioScience/planning_response.json
outputs/RadioScience/planning_trajectories.json
outputs/RadioScience/planning_config.yaml
```

4. **Ran the Paper2Code analysis stage**
   - For each planned source file, Paper2Code generated a detailed logic analysis explaining what that file should do and how it should connect to the paper methodology.
   - These analysis files are stored in:

```text
outputs/RadioScience/analyzing_artifacts/
outputs/RadioScience/*_simple_analysis_response.json
outputs/RadioScience/*_simple_analysis_trajectories.json
```

5. **Ran the Paper2Code coding stage**
   - Paper2Code then generated the actual Python reproduction project one file at a time.
   - The generated code is stored in:

```text
outputs/RadioScience_repo/
```

6. **Used OpenAI GPT-5.5 for generation**
   - The Paper2Code scripts were configured to use:

```text
gpt-5.5
```

   - The model was used for planning, logic analysis, and code generation.

7. **Approximate API cost**
   - The full run cost roughly **$25 USD** in OpenAI API usage.
   - The exact dollar amount is approximate because the local Paper2Code cost table did not include `gpt-5.5` pricing at generation time.
   - Token usage and model-call logs are preserved in:

```text
outputs/RadioScience/cost_info.log
```

In short: this repo is the result of taking a scientific PDF, converting it into structured JSON, passing that through the Paper2Code planning/analyzing/coding pipeline, and using GPT-5.5 to generate a reproducible Python project scaffold.

## What The Code Does

In simple terms, the generated code tries to recreate the paper's full processing pipeline.

1. **Simulates radio-signal distortion**
   - Creates synthetic ionosphere disturbance screens.
   - Sends simulated radio waves through them.
   - Checks whether the back-propagation method can recover where the disturbance was.

2. **Loads COSMIC-2 radio occultation data**
   - Reads satellite observation files.
   - Extracts phase, signal strength/SNR, transmitter position, receiver position, time, and metadata.

3. **Estimates where the ionospheric disturbance happened**
   - Splits observations into short time windows.
   - Calculates scintillation metrics such as `S4` and `sigma_phi`.
   - Builds the receiver-to-transmitter line of sight.
   - Uses magnetic-field geometry to define the back-propagation plane.
   - Runs back propagation over candidate distances, roughly 100 km to 6000 km.
   - Finds the distance where the distorted signal best focuses.
   - Converts that distance into latitude, longitude, and height.

4. **Creates outputs**
   - Writes geolocation tables.
   - Tracks accepted and rejected events.
   - Produces plots and maps.
   - Supports L1/L2 signal comparison products.

The main generated code lives in:

```text
outputs/RadioScience_repo/
```

The main entry point is:

```text
outputs/RadioScience_repo/main.py
```

The most important generated source folders are:

```text
src/data/          reads data
src/signal/        computes signal and scintillation metrics
src/geometry/      handles satellite geometry and magnetic-field setup
src/propagation/   performs back propagation
src/geolocation/   decides final location and quality control
src/experiments/   runs synthetic and real-data experiments
src/visualization/ makes plots and maps
```

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
