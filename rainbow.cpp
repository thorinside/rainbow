/*
 * Rainbow - Wavetable FIR Convolution Effect
 * 
 * Convolves audio with a wavetable wave used as FIR filter kernel.
 * Different wave positions create different filter characteristics.
 * 
 * Developer: Thorinside (Th)
 * Plugin ID: Rb
 * GUID: ThRb
 */

#include <math.h>
#include <string.h>
#include <algorithm>
#include <new>
#include <distingnt/api.h>
#include <distingnt/wav.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// FIR kernel sizes (using wavetable mipmap levels)
// Higher = more filter character but more CPU
static constexpr int kMaxKernelSize = 512;
static constexpr int kKernelSizes[] = { 64, 128, 256, 512 };
static constexpr int kNumKernelSizes = 4;

// Maximum channels supported
static constexpr int kMaxChannels = 12;

// Wavetable buffer size (same as wavetableDemo)
static constexpr int kWavetableBufferSize = 256 * 2048;

// ============================================================================
// SPECIFICATIONS
// ============================================================================

enum {
	kSpecChannels,
};

static const _NT_specification specifications[] = {
	{
		.name = "Channels",
		.min = 1,
		.max = kMaxChannels,
		.def = 2,
		.type = kNT_typeGeneric
	},
};

// ============================================================================
// PARAMETER DEFINITIONS
// ============================================================================

// Shared parameters (not per-channel)
enum {
	kParamWavetable,
	kParamIndex,
	kParamSpread,
	kParamDepth,
	kParamGain,
	kParamSaturation,
	kParamKernelSize,
	
	kNumSharedParams,
};

// Per-channel parameters (offset by channel * kParamsPerChannel)
enum {
	kParamInput = 0,
	kParamOutput,
	kParamOutputMode,
	
	kParamsPerChannel,
};

// Kernel size enum strings
static const char* const kernelSizeStrings[] = { "64", "128", "256", "512", NULL };

// Base parameters (shared)
static const _NT_parameter sharedParameters[] = {
	{ .name = "Wavetable", .min = 0, .max = 32767, .def = 0, .unit = kNT_unitNone, .scaling = 0, .enumStrings = NULL },
	{ .name = "Index", .min = 0, .max = 1000, .def = 500, .unit = kNT_unitPercent, .scaling = kNT_scaling10, .enumStrings = NULL },
	{ .name = "Spread", .min = 0, .max = 1000, .def = 0, .unit = kNT_unitPercent, .scaling = kNT_scaling10, .enumStrings = NULL },
	{ .name = "Depth", .min = 0, .max = 100, .def = 50, .unit = kNT_unitPercent, .scaling = 0, .enumStrings = NULL },
	{ .name = "Gain", .min = -240, .max = 240, .def = 0, .unit = kNT_unitDb, .scaling = kNT_scaling10, .enumStrings = NULL },
	{ .name = "Saturation", .min = 0, .max = 100, .def = 0, .unit = kNT_unitPercent, .scaling = 0, .enumStrings = NULL },
	{ .name = "Resolution", .min = 0, .max = kNumKernelSizes - 1, .def = 2, .unit = kNT_unitEnum, .scaling = 0, .enumStrings = kernelSizeStrings },
};

// Per-channel parameter template
static const _NT_parameter channelParameterTemplate[] = {
	NT_PARAMETER_AUDIO_INPUT("Input", 1, 1)
	NT_PARAMETER_AUDIO_OUTPUT_WITH_MODE("Output", 1, 13)
};

// ============================================================================
// PARAMETER PAGES
// ============================================================================

static const uint8_t pageMain[] = { kParamWavetable, kParamIndex, kParamSpread, kParamDepth, kParamKernelSize };
static const uint8_t pageOutput[] = { kParamGain, kParamSaturation };

static const _NT_parameterPage sharedPages[] = {
	{ .name = "Colour", .numParams = ARRAY_SIZE(pageMain), .params = pageMain },
	{ .name = "Output", .numParams = ARRAY_SIZE(pageOutput), .params = pageOutput },
};

// ============================================================================
// ALGORITHM STRUCTURES
// ============================================================================

// Per-channel state (in DTC for fast access)
// Delay line is 2x size for contiguous convolution reads (no per-tap masking)
struct ChannelState {
	float delayLine[kMaxKernelSize * 2];
	int writePos;
};

// DTC structure - performance critical data
struct _rainbow_DTC {
	ChannelState channels[kMaxChannels];
	
	// Per-channel kernels for spread effect, with crossfade support
	float kernels[kMaxChannels][kMaxKernelSize];
	float newKernels[kMaxChannels][kMaxKernelSize];
	float crossfadeMix;
	bool crossfading;
	
	// Cached parameter values
	float depth;
	float gain;
	float saturation;
	float spread;
	int kernelSize;
	int kernelMask;
};

// Main algorithm structure
struct _rainbowAlgorithm : public _NT_algorithm {
	_rainbowAlgorithm() {}
	~_rainbowAlgorithm() {}
	
	// Mutable copy of parameters (for dynamic wavetable max)
	_NT_parameter* params;
	int numParams;
	
	// Parameter pages (dynamically built)
	_NT_parameterPage* pages;
	int numPages;
	uint8_t* pageArrays;
	char* paramNames;
	_NT_parameterPages paramPages;
	
	// Memory pointers
	_rainbow_DTC* dtc;
	int16_t* wavetableBuffer;
	
	// Wavetable request
	_NT_wavetableRequest request;
	
	// State
	int numChannels;
	bool cardMounted;
	bool awaitingCallback;
	bool wavetableLoaded;
	
	// Current wavetable info
	int currentWaveIndex;
	float currentIndexParam;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Soft saturation using tanh approximation
static inline float softSaturate(float x, float amount) {
	if (amount < 0.001f) return x;
	float drive = 1.0f + amount * 4.0f;  // 1x to 5x drive
	return tanh(x * drive) / tanh(drive);
}

static void buildKernelAtIndex(_rainbowAlgorithm* pThis, float* dest, float indexParam) {
	_rainbow_DTC* dtc = pThis->dtc;
	
	indexParam = std::max(0.0f, std::min(1.0f, indexParam));
	float offset = indexParam * (pThis->request.numWaves - 1);
	offset = std::max(0.0f, std::min(offset, (float)(pThis->request.numWaves - 1) - 0.0001f));
	
	int wave0 = (int)offset;
	int wave1 = std::min(wave0 + 1, (int)pThis->request.numWaves - 1);
	float frac = offset - wave0;
	
	const int kernelSize = dtc->kernelSize;
	
	const int16_t* mip0 = pThis->wavetableBuffer + kernelSize * (pThis->request.numWaves + wave0);
	const int16_t* mip1 = pThis->wavetableBuffer + kernelSize * (pThis->request.numWaves + wave1);
	
	float sum = 0.0f;
	for (int i = 0; i < kernelSize; ++i) {
		float v0 = mip0[i] / 32768.0f;
		float v1 = mip1[i] / 32768.0f;
		dest[i] = v0 + frac * (v1 - v0);
		sum += fabsf(dest[i]);
	}
	
	if (sum > 0.001f) {
		float scale = 1.0f / sum;
		for (int i = 0; i < kernelSize; ++i) {
			dest[i] *= scale;
		}
	}
}

static void buildAllKernels(_rainbowAlgorithm* pThis, float kernels[][kMaxKernelSize]) {
	float indexParam = pThis->v[kParamIndex] * 0.001f;
	float spread = pThis->v[kParamSpread] * 0.001f;
	int numChannels = pThis->numChannels;
	int kernelSize = pThis->dtc->kernelSize;
	
	if (spread < 0.001f || numChannels == 1) {
		buildKernelAtIndex(pThis, kernels[0], indexParam);
		for (int ch = 1; ch < numChannels; ++ch) {
			memcpy(kernels[ch], kernels[0], kernelSize * sizeof(float));
		}
	} else {
		for (int ch = 0; ch < numChannels; ++ch) {
			float chOffset = spread * ((float)ch / (numChannels - 1) - 0.5f);
			buildKernelAtIndex(pThis, kernels[ch], indexParam + chOffset);
		}
	}
	
	pThis->currentIndexParam = indexParam;
}

static void updateKernel(_rainbowAlgorithm* pThis) {
	if (!pThis->wavetableLoaded || pThis->request.error)
		return;
	if (!pThis->request.usingMipMaps || pThis->request.numWaves == 0)
		return;
	
	buildAllKernels(pThis, pThis->dtc->kernels);
}

static void updateKernelWithCrossfade(_rainbowAlgorithm* pThis) {
	if (pThis->request.error)
		return;
	if (!pThis->request.usingMipMaps || pThis->request.numWaves == 0)
		return;
	
	_rainbow_DTC* dtc = pThis->dtc;
	buildAllKernels(pThis, dtc->newKernels);
	dtc->crossfadeMix = 0.0f;
	dtc->crossfading = true;
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

static void calculateRequirements(_NT_algorithmRequirements& req, const int32_t* specifications) {
	int numChannels = specifications ? specifications[kSpecChannels] : 2;
	int numParams = kNumSharedParams + numChannels * kParamsPerChannel;
	
	// Parameter storage
	size_t paramSize = numParams * sizeof(_NT_parameter);
	
	// 3 pages: Wavetable, Output, Routing
	int numPages = 3;
	size_t pageSize = numPages * sizeof(_NT_parameterPage);
	size_t pageArraySize = numChannels * kParamsPerChannel * sizeof(uint8_t);
	size_t paramNameSize = numChannels * kParamsPerChannel * 16;  // "Output 12 Mode\0" etc
	
	req.numParameters = numParams;
	req.sram = sizeof(_rainbowAlgorithm) + paramSize + pageSize + pageArraySize + paramNameSize;
	req.dram = kWavetableBufferSize * sizeof(int16_t);
	req.dtc = sizeof(_rainbow_DTC);
	req.itc = 0;
}

static void wavetableCallback(void* callbackData) {
	_rainbowAlgorithm* pThis = (_rainbowAlgorithm*)callbackData;
	pThis->awaitingCallback = false;
	
	if (!pThis->request.error) {
		if (pThis->wavetableLoaded) {
			updateKernelWithCrossfade(pThis);
		} else {
			pThis->wavetableLoaded = true;
			buildAllKernels(pThis, pThis->dtc->kernels);
		}
	}
}

static _NT_algorithm* construct(const _NT_algorithmMemoryPtrs& ptrs,
                                const _NT_algorithmRequirements& req,
                                const int32_t* specifications) {
	int numChannels = specifications ? specifications[kSpecChannels] : 2;
	int numParams = kNumSharedParams + numChannels * kParamsPerChannel;
	int numPages = 3;
	
	// Place algorithm struct
	_rainbowAlgorithm* alg = new (ptrs.sram) _rainbowAlgorithm();
	uint8_t* mem = (uint8_t*)ptrs.sram + sizeof(_rainbowAlgorithm);
	
	// Allocate parameters
	alg->params = (_NT_parameter*)mem;
	mem += numParams * sizeof(_NT_parameter);
	alg->numParams = numParams;
	
	// Allocate pages
	alg->pages = (_NT_parameterPage*)mem;
	mem += numPages * sizeof(_NT_parameterPage);
	alg->numPages = numPages;
	
	// Allocate page array for routing page
	alg->pageArrays = mem;
	mem += numChannels * kParamsPerChannel * sizeof(uint8_t);
	
	// Allocate parameter name strings
	alg->paramNames = (char*)mem;
	mem += numChannels * kParamsPerChannel * 16;
	
	// Copy shared parameters
	memcpy(alg->params, sharedParameters, sizeof(sharedParameters));
	
	// Generate per-channel parameters with numbered names
	for (int ch = 0; ch < numChannels; ++ch) {
		int baseParam = kNumSharedParams + ch * kParamsPerChannel;
		
		// Copy template
		memcpy(&alg->params[baseParam], channelParameterTemplate, sizeof(channelParameterTemplate));
		
		// Adjust default bus assignments
		alg->params[baseParam + kParamInput].def = 1 + ch;
		alg->params[baseParam + kParamOutput].def = 13 + ch;
		
		// Build parameter names: "Input 1", "Output 1", "Out 1 Mode"
		for (int p = 0; p < kParamsPerChannel; ++p) {
			char* name = alg->paramNames + (ch * kParamsPerChannel + p) * 16;
			int len = 0;
			
			if (p == kParamInput) {
				const char* s = "Input ";
				while (*s) name[len++] = *s++;
			} else if (p == kParamOutput) {
				const char* s = "Output ";
				while (*s) name[len++] = *s++;
			} else {
				const char* s = "Out ";
				while (*s) name[len++] = *s++;
			}
			len += NT_intToString(name + len, 1 + ch);
			if (p == kParamOutputMode) {
				const char* s = " Mode";
				while (*s) name[len++] = *s++;
			}
			name[len] = 0;
			
			alg->params[baseParam + p].name = name;
		}
	}
	
	// Build parameter pages
	alg->pages[0] = sharedPages[0];
	alg->pages[1] = sharedPages[1];
	
	// Single routing page with all channel parameters
	for (int i = 0; i < numChannels * kParamsPerChannel; ++i) {
		alg->pageArrays[i] = kNumSharedParams + i;
	}
	alg->pages[2].name = "Routing";
	alg->pages[2].numParams = numChannels * kParamsPerChannel;
	alg->pages[2].params = alg->pageArrays;
	
	// Set up parameter pages struct
	alg->paramPages.numPages = numPages;
	alg->paramPages.pages = alg->pages;
	
	alg->parameters = alg->params;
	alg->parameterPages = &alg->paramPages;
	
	// Set up DTC
	alg->dtc = (_rainbow_DTC*)ptrs.dtc;
	memset(alg->dtc, 0, sizeof(_rainbow_DTC));
	
	// Set up wavetable buffer
	alg->wavetableBuffer = (int16_t*)ptrs.dram;
	memset(alg->wavetableBuffer, 0, req.dram);
	
	// Initialize wavetable request
	alg->request.table = alg->wavetableBuffer;
	alg->request.tableSize = kWavetableBufferSize;
	alg->request.callback = wavetableCallback;
	alg->request.callbackData = alg;
	
	// Initialize state
	alg->numChannels = numChannels;
	alg->cardMounted = false;
	alg->awaitingCallback = false;
	alg->wavetableLoaded = false;
	alg->currentWaveIndex = -1;
	alg->currentIndexParam = -1.0f;
	
	// Initialize cached values
	alg->dtc->depth = 0.5f;
	alg->dtc->gain = 1.0f;
	alg->dtc->saturation = 0.0f;
	alg->dtc->kernelSize = kKernelSizes[2];  // Default: 256
	alg->dtc->kernelMask = alg->dtc->kernelSize - 1;
	
	return alg;
}

static int parameterUiPrefix(_NT_algorithm*, int, char*) {
	return 0;
}

static void parameterChanged(_NT_algorithm* self, int p) {
	_rainbowAlgorithm* pThis = (_rainbowAlgorithm*)self;
	_rainbow_DTC* dtc = pThis->dtc;
	
	switch (p) {
	case kParamWavetable:
		if (!pThis->awaitingCallback && pThis->cardMounted) {
			pThis->request.index = pThis->v[kParamWavetable];
			if (NT_readWavetable(pThis->request)) {
				pThis->awaitingCallback = true;
			}
		}
		break;
		
	case kParamIndex:
	case kParamSpread:
		updateKernel(pThis);
		break;
		
	case kParamDepth:
		dtc->depth = pThis->v[kParamDepth] / 100.0f;
		break;
		
	case kParamGain:
		{
			float dB = pThis->v[kParamGain] / 10.0f;
			dtc->gain = pow(10.0f, dB / 20.0f);
		}
		break;
		
	case kParamSaturation:
		dtc->saturation = pThis->v[kParamSaturation] / 100.0f;
		break;
		
	case kParamKernelSize:
		{
			int idx = std::max(0, std::min((int)pThis->v[kParamKernelSize], kNumKernelSizes - 1));
			dtc->kernelSize = kKernelSizes[idx];
			dtc->kernelMask = dtc->kernelSize - 1;
			updateKernel(pThis);
		}
		break;
	}
}

static void step(_NT_algorithm* self, float* busFrames, int numFramesBy4) {
	_rainbowAlgorithm* pThis = (_rainbowAlgorithm*)self;
	_rainbow_DTC* dtc = pThis->dtc;
	
	bool cardMounted = NT_isSdCardMounted();
	if (pThis->cardMounted != cardMounted) {
		pThis->cardMounted = cardMounted;
		if (cardMounted) {
			pThis->params[kParamWavetable].max = NT_getNumWavetables() - 1;
			NT_updateParameterDefinition(NT_algorithmIndex(self), kParamWavetable);
			parameterChanged(self, kParamWavetable);
		}
	}
	
	const int numFrames = numFramesBy4 << 2;
	const float depth = dtc->depth;
	const float dryMix = 1.0f - depth;
	const float gain = dtc->gain;
	const float saturation = dtc->saturation;
	const bool doConvolve = pThis->wavetableLoaded;
	const bool doSaturate = saturation > 0.001f;
	const int kernelSize = dtc->kernelSize;
	const int kernelMask = dtc->kernelMask;
	
	bool crossfading = dtc->crossfading;
	float crossfadeMix = dtc->crossfadeMix;
	constexpr float kCrossfadeRate = 1.0f / 2400.0f;  // ~50ms at 48kHz
	
	for (int ch = 0; ch < pThis->numChannels; ++ch) {
		const int baseParam = kNumSharedParams + ch * kParamsPerChannel;
		const float* __restrict in = busFrames + (pThis->v[baseParam + kParamInput] - 1) * numFrames;
		float* __restrict out = busFrames + (pThis->v[baseParam + kParamOutput] - 1) * numFrames;
		const bool replace = pThis->v[baseParam + kParamOutputMode];
		ChannelState* state = &dtc->channels[ch];
		float* __restrict delay = state->delayLine;
		int wp = state->writePos;
		
		const float* __restrict kernel = dtc->kernels[ch];
		const float* __restrict newKernel = dtc->newKernels[ch];
		float localMix = crossfadeMix;
		
		for (int i = 0; i < numFrames; ++i) {
			const float dry = in[i];
			delay[wp] = dry;
			delay[wp + kernelSize] = dry;
			
			float wet;
			if (doConvolve) {
				const float* __restrict x = &delay[wp + kernelSize];
				const float* __restrict h = kernel;
				float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
				
				for (int k = 0; k < kernelSize; k += 4) {
					a0 = fmaf(x[0], h[k], a0);
					a1 = fmaf(x[-1], h[k + 1], a1);
					a2 = fmaf(x[-2], h[k + 2], a2);
					a3 = fmaf(x[-3], h[k + 3], a3);
					x -= 4;
				}
				float wetOld = (a0 + a1) + (a2 + a3);
				
				if (crossfading) {
					x = &delay[wp + kernelSize];
					h = newKernel;
					a0 = 0.0f; a1 = 0.0f; a2 = 0.0f; a3 = 0.0f;
					for (int k = 0; k < kernelSize; k += 4) {
						a0 = fmaf(x[0], h[k], a0);
						a1 = fmaf(x[-1], h[k + 1], a1);
						a2 = fmaf(x[-2], h[k + 2], a2);
						a3 = fmaf(x[-3], h[k + 3], a3);
						x -= 4;
					}
					float wetNew = (a0 + a1) + (a2 + a3);
					wet = wetOld + (wetNew - wetOld) * localMix;
					if (ch == 0) localMix += kCrossfadeRate;
				} else {
					wet = wetOld;
				}
			} else {
				wet = dry;
			}
			
			wp = (wp + 1) & kernelMask;
			
			float mixed = fmaf(dry, dryMix, wet * depth);
			if (doSaturate) mixed = softSaturate(mixed, saturation);
			mixed *= gain;
			
			if (replace) out[i] = mixed;
			else out[i] += mixed;
		}
		state->writePos = wp;
		
		if (ch == 0 && crossfading) {
			crossfadeMix = localMix;
		}
	}
	
	if (crossfading) {
		if (crossfadeMix >= 1.0f) {
			for (int ch = 0; ch < pThis->numChannels; ++ch) {
				for (int i = 0; i < kernelSize; ++i) {
					dtc->kernels[ch][i] = dtc->newKernels[ch][i];
				}
			}
			dtc->crossfading = false;
			dtc->crossfadeMix = 0.0f;
		} else {
			dtc->crossfadeMix = crossfadeMix;
		}
	}
}

static bool draw(_NT_algorithm* self) {
	_rainbowAlgorithm* pThis = (_rainbowAlgorithm*)self;
	
	// Draw wavetable name
	int index = pThis->v[kParamWavetable];
	_NT_wavetableInfo info;
	NT_getWavetableInfo(index, info);
	
	if (info.name) {
		NT_drawText(10, 20, info.name, 15, kNT_textLeft, kNT_textNormal);
	} else {
		NT_drawText(10, 20, "No wavetable", 8, kNT_textLeft, kNT_textNormal);
	}
	
	// Draw status
	if (pThis->awaitingCallback) {
		NT_drawText(10, 35, "Loading...", 8);
	} else if (pThis->request.error) {
		NT_drawText(10, 35, "Error", 8);
	}
	
	// Draw waveform if wavetable loaded and using mipmaps
	if (pThis->wavetableLoaded && pThis->request.usingMipMaps && pThis->request.numWaves > 0) {
		// Get current wave position
		float indexParam = pThis->v[kParamIndex] * 0.001f;
		float offset = indexParam * (pThis->request.numWaves - 1);
		offset = std::max(0.0f, std::min(offset, (float)(pThis->request.numWaves - 1) - 0.0001f));
		
		int wave = (int)offset;
		float frac = offset - wave;
		
		constexpr int kDisplaySize = 64;
		const int16_t* mip0 = pThis->wavetableBuffer + kDisplaySize * (pThis->request.numWaves + wave);
		const int16_t* mip1 = pThis->wavetableBuffer + kDisplaySize * (pThis->request.numWaves + std::min(wave + 1, (int)pThis->request.numWaves - 1));
		
		float prevX = 0, prevY = 0;
		for (int i = 0; i < kDisplaySize; ++i) {
			float v0 = mip0[i];
			float v1 = mip1[i];
			float v = v0 + frac * (v1 - v0);
			
			float x = 192.0f + i;
			float y = 36.0f - v * (28.0f / 32768.0f);
			
			if (i > 0) {
				NT_drawShapeF(kNT_line, prevX, prevY, x, y, 12);
			}
			prevX = x;
			prevY = y;
		}
		
		// Draw frame around waveform
		NT_drawShapeI(kNT_box, 191, 7, 256, 65, 6);
	}
	
	// Draw channel/depth info
	char buf[32];
	int len = NT_intToString(buf, pThis->numChannels);
	buf[len++] = 'c';
	buf[len++] = 'h';
	buf[len] = 0;
	NT_drawText(10, 50, buf, 10);
	
	return false;  // Show standard parameter line
}

// ============================================================================
// FACTORY DEFINITION
// ============================================================================

static const _NT_factory factory = {
	.guid = NT_MULTICHAR('T', 'h', 'R', 'b'),
	.name = "Rainbow",
	.description = "Wavetable FIR convolution effect",
	.numSpecifications = ARRAY_SIZE(specifications),
	.specifications = specifications,
	.calculateStaticRequirements = NULL,
	.initialise = NULL,
	.calculateRequirements = calculateRequirements,
	.construct = construct,
	.parameterChanged = parameterChanged,
	.step = step,
	.draw = draw,
	.midiRealtime = NULL,
	.midiMessage = NULL,
	.tags = kNT_tagEffect | kNT_tagFilterEQ,
	.hasCustomUi = NULL,
	.customUi = NULL,
	.setupUi = NULL,
	.serialise = NULL,
	.deserialise = NULL,
	.midiSysEx = NULL,
	.parameterUiPrefix = parameterUiPrefix,
};

// ============================================================================
// PLUGIN ENTRY POINT
// ============================================================================

uintptr_t pluginEntry(_NT_selector selector, uint32_t data) {
	switch (selector) {
	case kNT_selector_version:
		return kNT_apiVersion10;  // Using v10 for parameterUiPrefix
		
	case kNT_selector_numFactories:
		return 1;
		
	case kNT_selector_factoryInfo:
		return (uintptr_t)((data == 0) ? &factory : NULL);
	}
	return 0;
}
