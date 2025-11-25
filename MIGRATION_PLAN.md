# Migration Plan: napari-ndev → ndevio

**Date:** November 24, 2025  
**Goal:** Replace napari-ndev's bioio/nImage implementation with ndevio package

---

## Overview

Migrate napari-ndev from its internal nImage implementation to use the extracted `ndevio` package. This will:
- Eliminate duplicate code
- Remove bioio dependency pins from napari-ndev
- Centralize I/O functionality in ndevio
- Enable better plugin management and error handling

---

## Pre-Migration Checklist

- [x] Verify ndevio has all required functionality
- [x] Confirm ndevio's nImage API is compatible
- [x] Identify all files using nImage/bioio
- [x] Document settings access pattern changes
- [ ] Ensure ndevio is published/available as dependency

---

## Phase 1: Add ndevio Dependency

### Files to Modify:
- `pyproject.toml`

### Changes:
1. Add `ndevio` to dependencies list
2. Remove bioio dependencies (will come from ndevio):
   - `bioio>=1.1.0`
   - `bioio-base==1.0.4` ⚠️ (This pin is problematic!)
   - `bioio-imageio>=1`
   - `bioio-tifffile>=1`
   - `bioio-ome-tiff>=1`
   - `bioio-ome-zarr>=1`
   - `bioio-nd2>=1`

**Note:** Keep these for now, remove in Phase 5 after testing

---

## Phase 2: Update Core Imports

### Files Requiring Import Changes:

#### 1. `src/napari_ndev/__init__.py`
**Change:**
```python
# OLD
from napari_ndev.nimage import nImage

# NEW
from ndevio import nImage
```

#### 2. `src/napari_ndev/_napari_reader.py`
**Changes:**
```python
# OLD
from napari_ndev import get_settings, nImage
from napari_ndev.nimage import get_preferred_reader

# NEW
from napari_ndev import get_settings
from ndevio import nImage
from ndevio.nimage import determine_reader_plugin
```

**Function updates:**
- Replace `get_preferred_reader(path)` → `determine_reader_plugin(path)`
- Import `DELIMITER` from ndevio.widgets or define locally
- Update `nImageSceneWidget` import to use `from ndevio.widgets import nImageSceneWidget`

#### 3. `src/napari_ndev/helpers.py`
**Change:**
```python
# OLD (line 24, inside function)
from napari_ndev import nImage

# NEW
from ndevio import nImage
```

#### 4. `src/napari_ndev/widgets/_utilities_container.py`
**Change:**
```python
# OLD
from napari_ndev import get_settings, helpers, nImage

# NEW
from napari_ndev import get_settings, helpers
from ndevio import nImage
```

**Also change imports on lines 33:**
```python
# OLD
from bioio import BioImage

# NEW
from bioio import BioImage  # Keep if still needed, or remove
```

#### 5. `src/napari_ndev/widgets/_workflow_container.py`
**Change:**
```python
# OLD
from napari_ndev import helpers, nImage

# NEW
from napari_ndev import helpers
from ndevio import nImage
```

#### 6. `src/napari_ndev/widgets/_measure_container.py`
**Changes (3 locations):**
```python
# Line 400, 472 (inside functions)
# OLD
from napari_ndev import nImage

# NEW
from ndevio import nImage
```

#### 7. `src/napari_ndev/widgets/_apoc_container.py`
**Changes (3 locations):**
```python
# Lines 386, 511, 610 (inside functions)
# OLD
from napari_ndev import nImage

# NEW
from ndevio import nImage
```

---

## Phase 3: Update Settings Access Patterns

### Settings Mapping Table:

| napari-ndev (OLD) | ndevio (NEW) |
|-------------------|--------------|
| `settings.PREFERRED_READER` | `settings.ndevio_Reader.preferred_reader` |
| `settings.UNPACK_CHANNELS_AS_LAYERS` | `settings.ndevio_Reader.unpack_channels_as_layers` |
| `settings.SCENE_HANDLING` | `settings.ndevio_Reader.scene_handling` |
| `settings.CLEAR_LAYERS_ON_NEW_SCENE` | `settings.ndevio_Reader.clear_layers_on_new_scene` |

### Files to Update:

#### 1. `src/napari_ndev/_napari_reader.py`
- Line ~50: `settings.SCENE_HANDLING` → `settings.ndevio_Reader.scene_handling`
- Line ~53: `settings.SCENE_HANDLING` → `settings.ndevio_Reader.scene_handling`
- Line ~229: `settings.CLEAR_LAYERS_ON_NEW_SCENE` → `settings.ndevio_Reader.clear_layers_on_new_scene`

#### 2. `src/napari_ndev/nimage.py` ⚠️
**This file will be DELETED in Phase 5, but update for transition:**
- Line ~70: `settings.PREFERRED_READER` → `settings.ndevio_Reader.preferred_reader`
- Line ~139: `settings.UNPACK_CHANNELS_AS_LAYERS` → `settings.ndevio_Reader.unpack_channels_as_layers`
- Line ~154: `settings.UNPACK_CHANNELS_AS_LAYERS` → `settings.ndevio_Reader.unpack_channels_as_layers`

---

## Phase 4: Update Tests

### Test Files Requiring Import Changes:

1. `src/napari_ndev/_tests/test_nimage.py`
   - Line 8: `from napari_ndev import nImage` → `from ndevio import nImage`

2. `src/napari_ndev/_tests/test_helpers.py`
   - Line 9: `from napari_ndev import nImage` → `from ndevio import nImage`

3. `src/napari_ndev/_tests/test_napari_reader.py`
   - Update mock paths: `napari_ndev._napari_reader.get_preferred_reader` → adjust as needed

4. `src/napari_ndev/_tests/test_settings.py`
   - Update `PREFERRED_READER` test to use `settings.ndevio_Reader.preferred_reader`

5. `src/napari_ndev/_tests/widgets/test_ndev_container.py`
   - Line 6: `from napari_ndev import __version__, nImage` → `from ndevio import nImage`

6. `src/napari_ndev/_tests/widgets/test_measure_container.py`
   - Line 5: `from napari_ndev import nImage` → `from ndevio import nImage`

7. `src/napari_ndev/_tests/widgets/test_utilities_container.py`
   - Line 7: `from napari_ndev import nImage` → `from ndevio import nImage`
   - Line 70, 93: Keep or adjust `nImage` usage

8. `src/napari_ndev/_tests/widgets/test_workflow_container.py`
   - Line 5: `from napari_ndev import nImage` → `from ndevio import nImage`
   - Lines 154, 186, 222: Keep usage

---

## Phase 5: Cleanup and Removal

### Files to Delete:
1. ✅ `src/napari_ndev/nimage.py` - Redundant, functionality in ndevio
2. ⚠️ Consider: `src/napari_ndev/_napari_reader.py` - Could use ndevio's reader directly

### Settings File Updates:
1. `src/napari_ndev/ndev_settings.yaml` - Remove reader-related settings:
   ```yaml
   # DELETE these (now in ndevio):
   PREFERRED_READER: bioio-ome-tiff
   SCENE_HANDLING: Open Scene Widget
   UNPACK_CHANNELS_AS_LAYERS: true
   CLEAR_LAYERS_ON_NEW_SCENE: false
   ```

2. Keep canvas-related settings:
   ```yaml
   CANVAS_SCALE: 1.0
   CANVAS_SIZE: [1024, 1024]
   OVERRIDE_CANVAS_SIZE: false
   ```

### Dependency Cleanup in `pyproject.toml`:
Remove after confirming all tests pass:
```toml
"bioio>=1.1.0",
"bioio-base==1.0.4",
"bioio-imageio>=1",
"bioio-tifffile>=1",
"bioio-ome-tiff>=1",
"bioio-ome-zarr>=1",
"bioio-nd2>=1",
```

---

## Phase 6: Testing & Validation

### Test Strategy:
1. **Unit Tests:**
   ```bash
   pytest src/napari_ndev/_tests/test_nimage.py -v
   pytest src/napari_ndev/_tests/test_napari_reader.py -v
   ```

2. **Widget Tests:**
   ```bash
   pytest src/napari_ndev/_tests/widgets/ -v
   ```

3. **Integration Tests:**
   ```bash
   pytest
   ```

4. **Manual Testing:**
   - Open napari with `pixi run napari`
   - Test file opening with various formats
   - Test multi-scene files
   - Test channel unpacking
   - Test settings persistence

### Validation Checklist:
- [ ] All tests pass
- [ ] Can open single-scene images
- [ ] Can open multi-scene images
- [ ] Scene widget appears for multi-scene files
- [ ] Channel unpacking works correctly
- [ ] Settings are respected
- [ ] Plugin installer appears for unsupported formats
- [ ] No import errors or circular dependencies

---

## Phase 7: Documentation Updates

### Files to Update:
1. `README.md` - Update installation and dependencies
2. Add note about ndevio dependency
3. Update contributor documentation if needed

---

## Rollback Plan

If issues arise:
1. Revert `pyproject.toml` changes
2. Restore `src/napari_ndev/nimage.py`
3. Revert import changes via git
4. Keep ndevio dependency for future retry

---

## Success Criteria

- ✅ All tests pass
- ✅ napari launches without errors
- ✅ Image I/O works correctly
- ✅ No duplicate bioio dependencies
- ✅ Settings work with new structure
- ✅ Code is cleaner and more maintainable

---

## Notes & Considerations

1. **Settings Migration:** Users with saved settings will need to migrate. Consider a settings migration utility.

2. **Backward Compatibility:** If napari-ndev is imported by other packages, they'll need to update their imports.

3. **Version Pinning:** Ensure ndevio version is properly pinned to avoid breaking changes.

4. **ndevio Development:** If ndevio isn't published yet, use local editable install:
   ```bash
   pip install -e ../ndevio
   ```

5. **Scene Widget:** Both implementations exist - verify they're compatible or migrate to ndevio's version entirely.

---

## Timeline Estimate

- Phase 1 (Dependencies): 15 minutes
- Phase 2 (Imports): 30 minutes
- Phase 3 (Settings): 20 minutes
- Phase 4 (Tests): 30 minutes
- Phase 5 (Cleanup): 20 minutes
- Phase 6 (Testing): 60 minutes
- Phase 7 (Documentation): 30 minutes

**Total:** ~3.5 hours

---

## Post-Migration Tasks

1. Update CI/CD pipelines if needed
2. Create release notes documenting breaking changes
3. Update ndev-kit documentation
4. Consider deprecation warnings for old import paths
5. Monitor for issues in production use
