# sky-gradient-flatten

**English** | [简体中文](README.zh-CN.md)

Even out an uneven night-sky background — light-pollution glow, stacking seams, edge bands, colour blotches — in a **sky + foreground** photo (city / campus / landscape astrophotography), **without touching the buildings**.

General gradient tools (Siril background extraction, GraXpert, …) are built for pure deep-sky frames. With a skyline in the frame they either refuse to sample near the horizon or leave a dark ring and halos around buildings. This script uses your foreground mask, models the background from sky pixels only, and writes foreground pixels back bit-for-bit.

![Input / flatten_sky.py / final](docs/example_before_after.jpg)

*CUHK campus, Hong Kong, Bortle ~8, 150 × 4 s stack. **1** Input: stacked + composited, uneven light-pollution glow. **2** `flatten_sky.py` with default settings — buildings pixel-identical to the input. **3** The author's final image, shown for reference: the sky was re-rendered from the linear stack (star-colour calibration, Milky Way–protected background) and then edited in Lightroom. Panel 3 is **not** the output of this script alone.*

## What it fixes

- Light-pollution dome and colour cast near the horizon (optionally keeps a soft natural glow)
- Dark / bright bands at frame edges and corners
- Sharp-edged seams and steps left by stacking (frames covering different areas)
- Large colour blotches in the sky

What it does **not** do: noise reduction, star enhancement, sky replacement. No stars are generated or removed.

## Install

Python 3.9+:

```bash
pip install -r requirements.txt
```

or with [uv](https://github.com/astral-sh/uv), no install:

```bash
uv run --with tifffile,numpy,scipy,opencv-python-headless python flatten_sky.py ...
```

## Usage

```bash
python flatten_sky.py photo.tif photo_flat.tif --mask foreground_mask.tif
```

- **Input**: 8- or 16-bit RGB TIFF in sRGB (e.g. exported from Lightroom / Photoshop / Siril). The ICC profile is copied to the output.
- **Mask**: grayscale TIFF, same size as the photo. **White = foreground (buildings, trees), black = sky.** Soft edges are fine. If your mask is the other way round, add `--invert-mask`.
- **Output**: a new TIFF, plus `_compare.jpg` (top: before, bottom: after), `_bgmodel.jpg` (the fitted background) and `_log.json` (parameters, before/after uniformity, checks). The input is never overwritten.

### Mask on a different canvas

If you made the mask on the uncropped image and then cropped in Lightroom, pass the uncropped image as `--ref`; the crop offset is found automatically (pure translation crops only). Or give it yourself with `--offset DY DX`.

```bash
python flatten_sky.py cropped.tif out.tif --mask mask_full.tif --ref full.tif
```

### Making the mask

Any tool works: Photoshop *Select Sky* → invert → fill white on black; Lightroom *Select Sky* mask exported as an image; GIMP; or your own segmentation. A slightly generous foreground (a few px beyond the roofline) is safer than a tight one.

## Key options

| Option | Default | Meaning |
|---|---|---|
| `--keep-glow` | 0.5 | How much smooth horizon glow to keep. `0` = completely flat (can look dead and expose noise texture in the old glow area). |
| `--glow-sigma` | 400 | Smoothness of the kept glow, px. |
| `--sigma` | 250 | Scale of the main background model, px. Smaller fixes smaller blotches, but **can remove real structure such as the Milky Way** — raise it (e.g. 600+) when the Milky Way is visible. |
| `--fine-sigma` | 80 | Small-scale residual pass, px (`0` = off). Fixes over/under-correction right above the skyline. |
| `--seam` | 100 | Edge-preserving median window for stacking seams, px (`0` = off). Also lower this or turn it off for Milky Way images. |
| `--strength` | 1.0 | Overall strength, `0`–`1`. |
| `--target-pct` | 50 | Target sky level = this percentile of the fitted background. |

Run `python flatten_sky.py -h` for everything.

## How it works

Light pollution is additive, so the correction is done in **linear light** (sRGB curve removed):

1. Stars removed by a median filter at 1/4 scale; only *compact* bright outliers (bright stars, planes) are rejected, so wide bands stay in the samples and get corrected.
2. Background modelled from sky pixels only with a **locally linear masked normalized convolution**. A plain masked blur gets pulled down by the darker sky further from the horizon and under-corrects the glow at the roofline; the linear term follows the gradient up to the mask edge.
3. Target field = constant sky colour + an optional fraction of a very smooth horizon glow.
4. Two refinement passes: a small-scale residual pass, and an **edge-preserving median** pass that removes sharp stacking seams (a median keeps a step as a step, so subtracting it cancels the step).
5. Blend with the mask, back to sRGB; pixels with mask = 1 are copied from the input unchanged.

## Tips & limits

- Works on already-edited exports, but best results come from a gently processed image: flatten first, then do contrast / colour in Lightroom.
- The mask matters most. Halos already baked into the image along the skyline (e.g. from compositing) are not removed.
- Full-resolution 24 MP takes ~20 s on an Apple Silicon laptop and a few GB of RAM.
- Check the `_compare.jpg` and zoom to 100% near the skyline before using the result.

## License

MIT
