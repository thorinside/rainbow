# Rainbow

Wavetable FIR convolution effect plugin for [Expert Sleepers distingNT](https://expert-sleepers.co.uk/distingNT.html).

## What it does

Rainbow uses wavetable waves as FIR filter kernels to "colour" audio. Select a wavetable, sweep through wave positions with the Index parameter, and each position creates a different filter character.

- Sine waves → smooth filtering
- Complex waves → resonant/comb-like effects
- Morphing between waves → evolving timbral shifts

## Parameters

### Colour Page

| Parameter | Description |
|-----------|-------------|
| Wavetable | Select from wavetables on SD card |
| Index | Position within wavetable (0-100%) |
| Spread | Per-channel wavetable offset for stereo/multichannel spread (0-100%) |
| Depth | Wet/dry mix (0-100%) |
| Resolution | FIR kernel size: 64, 128, 256, or 512 taps (default: 256) |

### Output Page

| Parameter | Description |
|-----------|-------------|
| Gain | Output gain (-24 to +24 dB) |
| Saturation | Soft saturation amount (0-100%) |

### Routing Page

Per-channel input/output routing with numbered parameters (Input 1, Output 1, etc.).

## Specifications

| Spec | Range | Default |
|------|-------|---------|
| Channels | 1-12 | 2 |

With Spread at 0%, all channels use the same kernel. With Spread > 0%, each channel gets a different wavetable position offset, creating stereo width or multichannel variation.

## Installation

1. Download `rainbow.o` from [Releases](https://github.com/thorinside/rainbow/releases)
2. Copy to your distingNT SD card
3. Load in distingNT

## Building

Requires ARM toolchain:

```bash
# macOS
brew install --cask gcc-arm-embedded

# Build for hardware
make hardware

# Build for nt_emu testing
make test
```

## License

MIT
