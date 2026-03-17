# Pure Motion -- Lip Syncer Integration

## What this is

Pure Motion is a hybrid lip-sync refinement feature added to the existing `lip_syncer` processor in FaceFusion 3.6.0. It replaces the standard "paint pixels onto the mouth" approach (wav2lip / edtalk) with a two-stage pipeline:

1. Run the normal lip-sync model on a **neutral face template** to produce an audio-driven mouth shape.
2. Extract that mouth shape as a LivePortrait **expression vector**, then re-render the real target face with that expression blended in.

The result is audio-driven mouth motion that preserves the target face's identity and texture rather than compositing a synthetic mouth patch.

The feature is gated by a single slider: **LIP SYNCER PURE MOTION**. When set to 0 (default), the processor behaves identically to stock FaceFusion. When set above 0, the LivePortrait refinement path activates. Values above 1.0 over-drive the expression for more pronounced mouth movement.

## Origin

This work was ported from `pure_motion_2.patch` (still in the repo root for reference). That patch targeted an older FaceFusion layout with a monolithic `facefusion/processors/modules/lip_syncer.py` file. The current repo uses a package-based layout (`lip_syncer/core.py`, `choices.py`, `types.py`, `locales.py`), so the patch was manually adapted to the current conventions rather than applied directly.

## Architecture

```
sync_lip(target_face, audio_frame, vision_frame)
  |
  |-- warp face crop to FFHQ 512x512
  |-- create occlusion masks
  |
  |-- if pure_motion > 0:
  |     |-- extract_template_expression(audio_frame)
  |     |     |-- load face_template.npy (LRU-cached)
  |     |     |-- run lip-sync model (wav2lip/edtalk) on template
  |     |     |-- run motion_extractor on result -> expression vector
  |     |     |-- scale expression by pure_motion factor
  |     |
  |     |-- prepare_refine_frame (resize to 256x256, normalize)
  |     |-- forward_extract_feature -> feature volume
  |     |-- forward_extract_motion -> pitch/yaw/roll/scale/translation/expression/motion_points
  |     |-- calculate target motion points (original expression)
  |     |-- blend template expression into target expression
  |     |-- calculate source motion points (blended expression)
  |     |-- forward_stitch_motion_points
  |     |-- forward_generate_frame -> new face crop
  |     |-- normalize_refine_frame (back to 512x512 uint8)
  |     |-- box mask
  |
  |-- else (standard path):
  |     |-- wav2lip or edtalk on the face crop directly
  |     |-- area mask (wav2lip) or box mask (edtalk)
  |
  |-- combine masks, paste_back into original frame
```

### Key design decisions

- The lip-sync model is still used even in pure-motion mode, but it runs on a static neutral template face rather than the real target. This converts it from a pixel painter into an expression driver.
- Expression blending uses hardcoded per-index weights (indices 6, 12, 14, 17, 19, 20 of the 21-point LivePortrait expression tensor). These control how much of the audio-driven mouth shape transfers vs how much of the target's original expression is preserved.
- The inference pool is shared: when pure_motion > 0, both the lip-syncer ONNX session and all six LivePortrait ONNX sessions live in the same pool. The cache key uses the lip-syncer model name, so `clear_inference_pool()` must be called whenever pure_motion is toggled.

## Files changed (vs FaceFusion 3.6.0 base)

### `facefusion/processors/modules/lip_syncer/core.py`
The main logic file. ~340 lines added. Contains:
- Extended `create_static_model_set()` with `live_portrait` (6 sub-models) and `face_template` entries
- `collect_model_downloads()` -- builds combined DownloadSet for lip-syncer + conditionally LivePortrait models
- Modified `get_inference_pool()` to use `collect_model_downloads()` instead of just the lip-syncer sources
- `get_static_face_template()` -- LRU-cached loader for `face_template.npy`, runs face detection to get bounding box
- `has_pure_motion()` -- checks `lip_syncer_pure_motion > 0`
- Refactored `sync_lip()` into branching logic with `process_standard_lip_sync()` and `process_live_portrait_motion()`
- `apply_lip_syncer()` -- shared helper that runs the lip-sync model on any crop + bounding box
- `extract_template_expression()` -- the core trick: lip-sync on template, motion_extract, scale
- `create_blended_expression()` / `blend_expression()` -- per-index expression blending
- `calculate_target_motion_points()` / `calculate_source_motion_points()` -- motion point math
- LivePortrait forwards: `forward_extract_feature()`, `forward_extract_motion()`, `forward_stitch_motion_points()`, `forward_generate_frame()`
- `prepare_refine_frame()` / `normalize_refine_frame()` -- 256x256 input prep and 512x512 output denorm
- `resize_bounding_box()` -- helper for wav2lip bounding box adjustment
- Updated `register_args()`, `apply_args()`, `pre_check()` for the new `--lip-syncer-pure-motion` option

### `facefusion/processors/modules/lip_syncer/choices.py`
Added: `lip_syncer_pure_motion_range : Sequence[float] = create_float_range(0.0, 1.5, 0.25)`

### `facefusion/processors/modules/lip_syncer/locales.py`
Added: `help.pure_motion` and `uis.pure_motion_slider` locale strings.

### `facefusion/uis/components/lip_syncer_options.py`
Added: `LIP_SYNCER_PURE_MOTION_SLIDER` global, rendered between model dropdown and weight slider. Change handler calls `clear_inference_pool()` then updates state. `remote_update()` now returns 3 components.

### `facefusion/uis/components/preview.py`
Added: `'lip_syncer_pure_motion_slider'` to the list of sliders that trigger preview re-render on release.

### `facefusion/uis/types.py`
Added: `'lip_syncer_pure_motion_slider'` to the `ComponentName` literal.

## External dependencies

### Face template
- Source: https://huggingface.co/bluefoxcreation/Templates/tree/main/face-template
- Files: `face_template.hash` (8 bytes), `face_template.npy` (787 KB)
- Downloaded to: `.assets/templates/face_template.hash` and `.assets/templates/face_template.npy`
- Downloaded automatically on first `pre_check()` via FaceFusion's `conditional_download_hashes` / `conditional_download_sources`

### LivePortrait models
Same ONNX models already used by `face_editor` and `expression_restorer` processors:
- `live_portrait_feature_extractor.onnx`
- `live_portrait_motion_extractor.onnx`
- `live_portrait_eye_retargeter.onnx`
- `live_portrait_lip_retargeter.onnx`
- `live_portrait_stitcher.onnx`
- `live_portrait_generator.onnx`

All stored in `.assets/models/`. If you've already used face_editor or expression_restorer, these are already downloaded.

## How to test

### Prerequisites
- Python environment with FaceFusion 3.6.0 dependencies installed
- A target video with a visible face
- A source audio file (MP3 or WAV)

### Gradio UI test
```bash
python facefusion.py run
```
1. In the **processors** checkbox, select `lip_syncer`
2. Set a source audio file
3. Set a target video
4. The **LIP SYNCER PURE MOTION** slider appears between the model dropdown and weight slider
5. Set it to a value like 0.75 or 1.0
6. The preview should update showing audio-driven mouth motion
7. Run full video processing to verify output

### CLI test
```bash
python facefusion.py headless-run \
  --processors lip_syncer \
  -s /path/to/audio.mp3 \
  -t /path/to/target.mp4 \
  -o /path/to/output.mp4 \
  --lip-syncer-pure-motion 1.0 \
  --trim-frame-end 30
```

### What to verify
- With `pure_motion = 0`: output should be identical to stock FaceFusion lip sync
- With `pure_motion > 0`: mouth should move with audio but face texture/identity should be better preserved
- With `pure_motion > 1.0`: expression should be exaggerated (over-driven)
- Toggling between 0 and non-zero should work without errors (inference pool is cleared and rebuilt)
- First run with pure_motion > 0 will trigger model downloads (~1.5 GB total for LivePortrait + template)

## Known risks and tuning areas

### Expression blending weights
The `create_blended_expression()` function uses hardcoded per-index blend factors. These were ported directly from the original patch and are empirical. The indices correspond to specific facial control points in LivePortrait's 21-point expression tensor:
- Index 6: general face shape (0.5, 0.5, 0.5)
- Index 12: mid-face (0.5, 0.5, 0.5)
- Index 14: jaw area (0.6, 0.7, 0.7)
- Index 17: lower face (0.5, 0.8, 0.7)
- Index 19: lip opening -- dynamically scaled by detected lip openness (0.5, 0.9-1.5, 0.85)
- Index 20: chin area (0.5, 0.6, 0.7)

These may need visual tuning for different face types or audio characteristics.

### Memory
When pure_motion is active, the inference pool holds 7 ONNX sessions (1 lip-syncer + 6 LivePortrait). This is significant GPU memory usage. The existing `video_memory_strategy` setting handles cleanup between frames if set to `strict` or `moderate`.

### Inference pool cache key
The cache key for the inference pool is built from `model_names` (just the lip-syncer model name), not from the full source set. When pure_motion is toggled, `clear_inference_pool()` must be called to force pool recreation with the correct set of models. The UI handler does this automatically. If you add new code paths that change pure_motion without clearing the pool, you will get stale/missing model sessions.

### Template face detection
`get_static_face_template()` loads `face_template.npy` and runs FaceFusion's face detector on it to find the bounding box. If the face detector fails on the template (wrong settings, score threshold too high), this will crash with an `AttributeError` on `None.bounding_box`. The function is LRU-cached so this only happens once per session.

## File inventory

```
Changed files:
  facefusion/processors/modules/lip_syncer/core.py      (main logic, ~589 lines)
  facefusion/processors/modules/lip_syncer/choices.py    (added pure_motion_range)
  facefusion/processors/modules/lip_syncer/locales.py    (added locale strings)
  facefusion/uis/components/lip_syncer_options.py         (added slider + handler)
  facefusion/uis/components/preview.py                    (added preview trigger)
  facefusion/uis/types.py                                 (added component name)

Unchanged but relevant:
  facefusion/processors/modules/lip_syncer/types.py       (LipSyncerInputs, LipSyncerModel, LipSyncerWeight)
  facefusion/processors/live_portrait.py                   (create_rotation, limit_expression, EXPRESSION_MIN/MAX)
  facefusion/processors/types.py                           (LivePortrait* type aliases)
  facefusion/processors/modules/face_editor/core.py        (reference implementation for LivePortrait forwards)
  facefusion/processors/modules/expression_restorer/core.py (another LivePortrait reference)
  facefusion/inference_manager.py                           (pool caching, context key logic)
  facefusion/download.py                                    (conditional_download_*, curl --create-dirs)

Reference:
  pure_motion_2.patch                                      (original patch this was ported from)
```
