# Paleoclimate resistance-change analysis

The analysis uses climate and paleogeography only.

## 1. Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Directory structure

The defaults use project-relative directories:

```text
project/
├── resistance_change.py
├── requirements.txt
├── data/
│   ├── chelsa/
│   └── ice6g/
└── results/
```



## 3. Run the complete analysis

From the directory containing `resistance_change.py`:

```bash
python resistance_change.py \
  --climate-dir data/chelsa \
  --ice-dir data/ice6g \
  --output-dir results/resistance_change \
  --download-missing
```

`--download-missing` downloads and caches the required CHELSA subsets and ICE-6G_C NetCDF files. Subsequent runs can omit the flag.

For a quick test:

```bash
python resistance_change.py \
  --climate-dir data/chelsa \
  --ice-dir data/ice6g \
  --output-dir results/test \
  --n-models 50 \
  --download-missing
```

The manuscript analysis used `--n-models 1000 --seed 8160`.

## 4. Input filenames

CHELSA layers are cached as cropped NumPy arrays:

```text
chelsa_bio01_21000bp.npy
chelsa_bio04_21000bp.npy
chelsa_bio12_21000bp.npy
...
```

ICE-6G_C files retain their official names:

```text
I6_C.VM5a_10min.21.nc
I6_C.VM5a_10min.18.nc
...
```

If different bounds or resolution are requested, delete previously cached CHELSA `.npy` files so they can be regenerated for the new grid.

## 5. Main options

```text
--times 21000,18000,16000,12000,8000,6000,3000
--bounds 20,35,180,82
--resolution 0.5
--n-models 1000
--seed 8160
--regions regions.example.json
```

Dates must be ordered from older to younger. A custom region JSON uses `[west, south, east, north]` boxes. Regions are optional analytical summaries and do not affect the pixel-level change maps.

## 6. Outputs

The output directory contains:

- `resistance_change_maps.png` and `.svg`
- `map_change_summary.csv`
- `regional_resistance_levels.csv`
- `regional_resistance_changes.csv`
- `analysis_metadata.json`
- `geotiffs/` containing median change, 10th/90th percentiles, and probabilities of decreasing or increasing resistance for every adjacent time interval

Negative change means lower resistance at the later date. The probability columns are the fraction of ensemble parameterizations supporting a direction of change; they are not frequentist P values.
Map coastlines are reconstructed from the ICE-6G_C land mask at the later date of each interval; they are not modern coastlines.

## 7. Model definition

The four predictors are cold stress, aridity, temperature seasonality, and terrain ruggedness. In each ensemble model, weights are sampled from `Dirichlet(1,1,1,1)` and the response exponent from `Uniform(0.75,1.50)`. Relative resistance is

```text
R = 1 + 9 * (weighted environmental score ** exponent)
```

Ocean and ICE-6G_C ice cells are unavailable. Modern rivers are not included because they are not a consistent reconstruction of paleohydrology over the full interval.

## 8. Data citations

- Karger, D. N. et al. (2023). CHELSA-TraCE21k—high-resolution (1 km) downscaled transient temperature and precipitation data since the Last Glacial Maximum. *Climate of the Past* 19:439–456. https://doi.org/10.5194/cp-19-439-2023
- Argus, D. F. et al. (2014). The Antarctica component of postglacial rebound model ICE-6G_C (VM5a). *Geophysical Journal International* 198:537–563. https://doi.org/10.1093/gji/ggu140
- Peltier, W. R. et al. (2015). Space geodesy constrains ice-age terminal deglaciation: The global ICE-6G_C (VM5a) model. *Journal of Geophysical Research: Solid Earth* 120:450–487. https://doi.org/10.1002/2014JB011176
