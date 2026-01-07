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

// FIR kernel size (using wavetable mipmap level)
// 64 taps is a good balance between quality and CPU usage
static constexpr int kKernelSize = 64;

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
	kParamDepth,
	kParamGain,
	kParamSaturation,
	
	kNumSharedParams,
};

// Per-channel parameters (offset by channel * kParamsPerChannel)
enum {
	kParamInput = 0,
	kParamOutput,
	kParamOutputMode,
	
	kParamsPerChannel,
};

// Base parameters (shared)
static const _NT_parameter sharedParameters[] = {
	{ .name = "Wavetable", .min = 0, .max = 32767, .def = 0, .unit = kNT_unitNone, .scaling = 0, .enumStrings = NULL },
	{ .name = "Index", .min = 0, .max = 1000, .def = 500, .unit = kNT_unitPercent, .scaling = kNT_scaling10, .enumStrings = NULL },
	{ .name = "Depth", .min = 0, .max = 100, .def = 50, .unit = kNT_unitPercent, .scaling = 0, .enumStrings = NULL },
	{ .name = "Gain", .min = -240, .max = 240, .def = 0, .unit = kNT_unitDb, .scaling = kNT_scaling10, .enumStrings = NULL },
	{ .name = "Saturation", .min = 0, .max = 100, .def = 0, .unit = kNT_unitPercent, .scaling = 0, .enumStrings = NULL },
};

// Per-channel parameter template
static const _NT_parameter channelParameterTemplate[] = {
	NT_PARAMETER_AUDIO_INPUT("Input", 1, 1)
	NT_PARAMETER_AUDIO_OUTPUT_WITH_MODE("Output", 1, 13)
};

// ============================================================================
// PARAMETER PAGES
// ============================================================================

static const uint8_t pageMain[] = { kParamWavetable, kParamIndex, kParamDepth };
static const uint8_t pageOutput[] = { kParamGain, kParamSaturation };

static const _NT_parameterPage sharedPages[] = {
	{ .name = "Wavetable", .numParams = ARRAY_SIZE(pageMain), .params = pageMain },
	{ .name = "Output", .numParams = ARRAY_SIZE(pageOutput), .params = pageOutput },
};

// ============================================================================
// ALGORITHM STRUCTURES
// ============================================================================

// Per-channel state (in DTC for fast access)
struct ChannelState {
	float delayLine[kKernelSize];
	int writePos;
};

// DTC structure - performance critical data
struct _rainbow_DTC {
	ChannelState channels[kMaxChannels];
	
	// Current kernel (interpolated from wavetable)
	float kernel[kKernelSize];
	
	// Cached parameter values
	float depth;
	float gain;
	float saturation;
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

// Update kernel from wavetable based on index
static void updateKernel(_rainbowAlgorithm* pThis) {
	if (!pThis->wavetableLoaded || pThis->request.error)
		return;
	
	if (!pThis->request.usingMipMaps || pThis->request.numWaves == 0)
		return;
	
	_rainbow_DTC* dtc = pThis->dtc;
	
	// Get index parameter (0-100%)
	float indexParam = pThis->v[kParamWavetable + kParamIndex] * 0.001f;
	
	// Map to wave position
	float offset = indexParam * (pThis->request.numWaves - 1);
	offset = std::max(0.0f, std::min(offset, (float)(pThis->request.numWaves - 1) - 0.0001f));
	
	int wave0 = (int)offset;
	int wave1 = std::min(wave0 + 1, (int)pThis->request.numWaves - 1);
	float frac = offset - wave0;
	
	// Access 64x64 mipmap level
	// Mipmap offset for size s is: s * numWaves
	const int16_t* mip0 = pThis->wavetableBuffer + kKernelSize * (pThis->request.numWaves + wave0);
	const int16_t* mip1 = pThis->wavetableBuffer + kKernelSize * (pThis->request.numWaves + wave1);
	
	// Interpolate and normalize to float kernel
	float sum = 0.0f;
	for (int i = 0; i < kKernelSize; ++i) {
		float v0 = mip0[i] / 32768.0f;
		float v1 = mip1[i] / 32768.0f;
		dtc->kernel[i] = v0 + frac * (v1 - v0);
		sum += fabsf(dtc->kernel[i]);
	}
	
	// Normalize kernel to prevent gain changes
	if (sum > 0.001f) {
		float scale = 1.0f / sum;
		for (int i = 0; i < kKernelSize; ++i) {
			dtc->kernel[i] *= scale;
		}
	}
	
	pThis->currentIndexParam = indexParam;
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

static void calculateRequirements(_NT_algorithmRequirements& req, const int32_t* specifications) {
	int numChannels = specifications ? specifications[kSpecChannels] : 2;
	int numParams = kNumSharedParams + numChannels * kParamsPerChannel;
	
	// Parameter storage
	size_t paramSize = numParams * sizeof(_NT_parameter);
	
	// Page arrays (2 shared pages + 1 routing page per channel)
	int numPages = 2 + numChannels;
	size_t pageSize = numPages * sizeof(_NT_parameterPage);
	size_t pageArraySize = numChannels * kParamsPerChannel * sizeof(uint8_t);
	
	req.numParameters = numParams;
	req.sram = sizeof(_rainbowAlgorithm) + paramSize + pageSize + pageArraySize;
	req.dram = kWavetableBufferSize * sizeof(int16_t);
	req.dtc = sizeof(_rainbow_DTC);
	req.itc = 0;
}

static void wavetableCallback(void* callbackData) {
	_rainbowAlgorithm* pThis = (_rainbowAlgorithm*)callbackData;
	pThis->awaitingCallback = false;
	pThis->wavetableLoaded = !pThis->request.error;
	
	if (pThis->wavetableLoaded) {
		updateKernel(pThis);
	}
}

static _NT_algorithm* construct(const _NT_algorithmMemoryPtrs& ptrs,
                                const _NT_algorithmRequirements& req,
                                const int32_t* specifications) {
	int numChannels = specifications ? specifications[kSpecChannels] : 2;
	int numParams = kNumSharedParams + numChannels * kParamsPerChannel;
	int numPages = 2 + numChannels;
	
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
	
	// Allocate page arrays for channel routing
	alg->pageArrays = mem;
	
	// Copy shared parameters
	memcpy(alg->params, sharedParameters, sizeof(sharedParameters));
	
	// Generate per-channel parameters
	for (int ch = 0; ch < numChannels; ++ch) {
		int baseParam = kNumSharedParams + ch * kParamsPerChannel;
		
		// Copy template
		memcpy(&alg->params[baseParam], channelParameterTemplate, sizeof(channelParameterTemplate));
		
		// Adjust default bus assignments
		alg->params[baseParam + kParamInput].def = 1 + ch;  // Input 1, 2, 3...
		alg->params[baseParam + kParamOutput].def = 13 + ch;  // Output 13, 14, 15...
	}
	
	// Build parameter pages
	// First two pages are shared
	alg->pages[0] = sharedPages[0];
	alg->pages[1] = sharedPages[1];
	
	// Per-channel routing pages
	for (int ch = 0; ch < numChannels; ++ch) {
		uint8_t* pageArray = alg->pageArrays + ch * kParamsPerChannel;
		int baseParam = kNumSharedParams + ch * kParamsPerChannel;
		
		for (int p = 0; p < kParamsPerChannel; ++p) {
			pageArray[p] = baseParam + p;
		}
		
		alg->pages[2 + ch].name = "Routing";
		alg->pages[2 + ch].numParams = kParamsPerChannel;
		alg->pages[2 + ch].params = pageArray;
	}
	
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
	
	return alg;
}

static int parameterUiPrefix(_NT_algorithm*, int p, char* buff) {
	if (p >= kNumSharedParams) {
		int ch = (p - kNumSharedParams) / kParamsPerChannel;
		int len = NT_intToString(buff, 1 + ch);
		buff[len++] = ':';
		buff[len] = 0;
		return len;
	}
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
				pThis->wavetableLoaded = false;
			}
		}
		break;
		
	case kParamIndex:
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
	const float* __restrict kernel = dtc->kernel;
	const bool doConvolve = pThis->wavetableLoaded;
	const bool doSaturate = saturation > 0.001f;
	constexpr int kMask = kKernelSize - 1;
	
	for (int ch = 0; ch < pThis->numChannels; ++ch) {
		const int baseParam = kNumSharedParams + ch * kParamsPerChannel;
		const float* __restrict in = busFrames + (pThis->v[baseParam + kParamInput] - 1) * numFrames;
		float* __restrict out = busFrames + (pThis->v[baseParam + kParamOutput] - 1) * numFrames;
		const bool replace = pThis->v[baseParam + kParamOutputMode];
		ChannelState* state = &dtc->channels[ch];
		float* __restrict delay = state->delayLine;
		int wp = state->writePos;
		
		for (int i = 0; i < numFrames; ++i) {
			const float dry = in[i];
			delay[wp] = dry;
			
			float wet;
			if (doConvolve) {
				wet = 0.0f;
				int rp = wp;
				for (int k = 0; k < kKernelSize; k += 4) {
					wet += delay[rp] * kernel[k];
					rp = (rp - 1) & kMask;
					wet += delay[rp] * kernel[k + 1];
					rp = (rp - 1) & kMask;
					wet += delay[rp] * kernel[k + 2];
					rp = (rp - 1) & kMask;
					wet += delay[rp] * kernel[k + 3];
					rp = (rp - 1) & kMask;
				}
			} else {
				wet = dry;
			}
			
			wp = (wp + 1) & kMask;
			
			float mixed = dry * dryMix + wet * depth;
			if (doSaturate) mixed = softSaturate(mixed, saturation);
			mixed *= gain;
			
			if (replace) out[i] = mixed;
			else out[i] += mixed;
		}
		state->writePos = wp;
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
		
		// Access 64x64 mipmap
		const int16_t* mip0 = pThis->wavetableBuffer + kKernelSize * (pThis->request.numWaves + wave);
		const int16_t* mip1 = pThis->wavetableBuffer + kKernelSize * (pThis->request.numWaves + std::min(wave + 1, (int)pThis->request.numWaves - 1));
		
		// Draw waveform
		float prevX = 0, prevY = 0;
		for (int i = 0; i < kKernelSize; ++i) {
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
