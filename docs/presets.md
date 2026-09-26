# Presets

Open **Presets → Manage presets…** to create, edit, delete, import, or export a
preset. New starts from the settings currently shown in the editor. Save writes
the selected preset; Use preset saves and activates it. Selecting a preset
name directly from the Presets menu activates the saved version.

A preset contains every public setting in `Config`, using the same YAML schema
as `configs/in_vitro.yaml`. The editor groups these into Geometry, Plate search,
Plants, Registration, Tracking, Files, and Workflow. HSV ranges use the same
lower/upper color editor as the main window. Custom crop boxes use normalized
center and size values, with rotation in degrees. Leaving a custom box disabled
uses the configured whole-image search or fixed plate margins and crop ratios.

Workflow defaults are optional in existing YAML files:

```yaml
gui:
  auto_apply: true
  load_crop_editing: true
  preprocess_crop_editing: false
```

These control automatic setting application and whether crop editing starts
open in each step. `crop.background_enabled` independently controls plate search
by color. None of these editor-opening choices changes the detection algorithm.

Activating a preset switches input/output/statistics paths and filename parsing
rules as well as processing settings, reloads the dataset, and returns to Load.
In vitro is selected on the first launch; subsequent launches restore the last activated preset. The Presets menu checks the active preset, and the manager selects and marks its row. An explicit
`--config` takes precedence over the saved preset.

Presets live in the platform's user application-configuration directory, under
`presets/`. On first use, Defaults and copies of the bundled YAML configurations
are created there. Editing/deleting them does not modify files in the repository,
and deleted initial presets are not recreated on every launch. Relative paths
from imported YAML are resolved using the existing CLI convention (relative to
the configuration file's parent's parent directory); saved/exported presets use
absolute paths so moving a preset file does not silently change its dataset.

Configuration validation runs before a preset is saved or applied. Saves replace
the individual YAML file atomically. Invalid edits do not overwrite a saved preset.
