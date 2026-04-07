# **PyTorch to CoreML on Apple Silicon: An Exhaustive Field Guide for Speech Model Deployment**

April 2, 2026

> **Scope:** This repository trains **Gemma** multimodal models only (see `gemma_tuner/models/gemma/`). This guide covers **general** Core ML export patterns for speech / seq2seq stacks.


## **1\. Executive Summary**

The transition of speech models—particularly sequence-to-sequence transformer architectures like Conformer—from PyTorch research environments to deployment-ready CoreML on Apple Silicon is a process defined by hardware idiosyncrasies, undocumented compiler behaviors, and strict memory layout requirements. This report serves as a hands-on operating manual for engineering teams actively shipping and debugging speech-model export and deployment workflows.

The following technical realities dictate CoreML deployment for speech on Apple Silicon:

1. **Full conversion of autoregressive decoders is a performance trap.** While encoder-only conversions perform exceptionally well on the Apple Neural Engine (ANE), forcing sequence-to-sequence decoders into standard stateless CoreML leads to catastrophic memory thrashing and latency spikes due to Key-Value (KV) cache copying.1  
2. **Hybrid deployment is the operational standard.** The most performant and stable architectures utilize a CoreML-compiled encoder running on the ANE, paired with a native C++ or MLX-based decoder running on the GPU or CPU.3  
3. **RangeDim destroys ANE performance.** Unbounded dynamic shapes force the MLIR runtime to fall back to the CPU or GPU. EnumeratedShapes is mandatory for achieving ANE acceleration on variable-length audio inputs.5  
4. **FP16 Softmax overflow is the primary cause of gibberish output.** The ANE executes exclusively in FP16 precision. Attention heads with unscaled logits exceeding \+1e4 will overflow to inf, destroying the output matrix. Manual maximum-value subtraction prior to Softmax is a non-negotiable PyTorch surgery before export.7  
5. **torch.jit.trace remains superior to torch.export for production.** Despite the beta introduction of torch.export in coremltools 8.0, tracing remains the most battle-tested path with the highest operator translation coverage (approximately 70% vs 56% for export) for complex speech models.9  
6. **Stateful models require iOS 18 / macOS 15+ targets.** For teams attempting full CoreML deployment with an internal KV cache, the register\_buffer API must be used to define stateful inputs, preventing the runtime from copying massive cache tensors across the Swift boundary on every token generation.11  
7. **PyTorch's Scaled Dot-Product Attention (SDPA) breaks conversions.** CoreML conversion requires deterministic graph tracing. PyTorch's SDPA implementation dynamically dispatches to different backend kernels (FlashAttention, Math, etc.), causing irrecoverable graph breaks. It must be explicitly disabled prior to export.12  
8. **LayerNorm bias/scale inversion is a known ANE hardware quirk.** Certain ANE compiler passes misinterpret the order of scale and bias in custom LayerNorm implementations. Specific hooks must be injected into the PyTorch module to divide the bias by the weight before export.12  
9. **Compute-unit selection cannot be completely forced.** Specifying ComputeUnit.ALL allows the espresso backend to map operations. If an operation violates ANE constraints (e.g., \>13 kernel size), it will silently fall back to the GPU or CPU, causing severe context-switching latency.13  
10. **Parity testing requires Signal-to-Noise Ratio (SNR), not exact matching.** Due to FP16 casting and hardware-specific math approximations, outputs will never match PyTorch FP32 exactly. An SNR greater than 20dB indicates acceptable parity.14  
11. **MILCompilerForANE crashes are uncatchable.** When the MLIR pass manager fails on unsupported memory alignments, the resulting C++ exception cannot be caught by Swift's do/catch blocks, resulting in an immediate application crash.16  
12. **Bridging MLMultiArray to C++ requires explicit lifetime management.** When sharing CoreML output pointers with an external C++ decoder, withExtendedLifetime must be utilized to prevent the Swift Automatic Reference Counting (ARC) garbage collector from destroying the tensor during inference.18  
13. **MultiFunctionDescriptor enables parameter-efficient architectures.** CoreML 8.1 allows packing multiple speech heads (e.g., transcription and speaker diarization) into a single .mlpackage while deduplicating shared backbone weights via hash comparison.20  
14. **Memory Layouts dictate speed.** The ANE prefers NHWC (Channel-Last) layout. Forcing standard PyTorch NCHW layouts through the compiler often results in Transpose nodes being injected, creating significant memory bandwidth bottlenecks.21  
15. **Benchmarking requires dedicated CLI tooling.** Relying on Xcode Instruments alone is insufficient for CI/CD pipelines. Using coreml-cli allows for isolated benchmarking of compiled models against specific compute targets to detect latency regressions prior to deployment.23

## ---

**2\. Decision Framework: Deployment Architectures**

Before initiating the coremltools conversion pipeline, the target deployment architecture must be formalized. Attempting to force an incompatible model architecture into a monolithic CoreML package will result in wasted engineering cycles, thermal throttling, and unacceptable latency.

The Apple hardware stack limits memory bandwidth differently across its compute units. While the GPU boasts high FLOPs and bandwidth, the ANE is highly optimized for energy-efficient matrix multiplication with a strictly limited memory cache.24 Therefore, the decision of *what* to convert is more important than *how* to convert it.

### **Deployment Strategy Matrix**

| Architecture Option | Best Use Case | Primary Hardware | Pros | Cons |
| :---- | :---- | :---- | :---- | :---- |
| **Encoder-Only CoreML** | Feature extraction, Voice Activity Detection, Speaker ID, Hubert embeddings. | ANE | Clean conversion pipeline, deterministic latency, near-zero fallback, highly power-efficient. | Limited exclusively to non-generative tasks. |
| **Hybrid Deployment (CoreML Encoder \+ External Decoder)** | Whisper, Conformer seq2seq, complex multi-modal speech systems. | Encoder: ANE Decoder: GPU/CPU | Avoids CoreML stateful decoding memory thrashing. Achieves up to 18x encoder speedup while preserving decoder flexibility.4 | Requires complex Swift-to-C++ (or MLX) memory bridging for cross-attention tensors. Increases binary size. |
| **Full CoreML (Stateless)** | Small, fixed-length sequence-to-sequence models (e.g., specific wakeword generators). | CPU/GPU (ANE fails on dynamic sequence loops). | Produces a single .mlpackage artifact. Simple Swift integration. | Exponential latency degradation. KV cache must be copied across the Swift/Obj-C boundary on every pass. Extreme battery drain. |
| **Full CoreML (Stateful)** | Production autoregressive models targeting iOS 18+ and macOS 15+. | ANE / GPU | Resolves KV cache thrashing by maintaining memory on-device. Preserves Apple Intelligence-level efficiency.11 | Relies on immature coremltools stateful APIs. Hard locked to modern OS versions. Requires complex PyTorch buffer preparation.11 |

### **When Full Conversion is Worth It**

Full CoreML conversion is only justifiable if the target audience is strictly on iOS 18 or macOS 15 and newer, allowing the use of the StateType API. Stateful models allow the coremltools compiler to allocate the Key-Value cache directly on the accelerator (ANE or GPU) and mutate it in place.11 If the model can be successfully mapped to stateful tensors using PyTorch's register\_buffer, the resulting .mlpackage provides the lowest possible power consumption profile.

### **When Encoder-Only Conversion is Best**

If the speech model relies on complex, dynamically changing sparse attention masks during decoding, or utilizes custom beam-search heuristics, the encoder-only path is optimal. The encoder processes the fixed or chunked audio waveform into high-dimensional embeddings. Because this process is largely static and highly parallelizable, it maps perfectly to the ANE.4

### **When Hybrid Deployment is the Right Answer**

Hybrid deployment is the right answer when supporting legacy OS versions (iOS 17 and below) while deploying complex models like Whisper. In this architecture, the CoreML runtime handles the heavy, static lifting of the encoder. The resulting contextual embeddings are passed as raw memory pointers to an external decoder written in C++ (e.g., ggml or whisper.cpp) or Apple's MLX framework.4 This allows the decoder to manage its own KV cache, execute complex sampling algorithms, and avoid the massive overhead of crossing the CoreML API boundary for every single token generated.

### **When to Stop Fighting CoreML**

The MLIR compiler utilized by CoreML is exceptionally rigid. Engineering teams should immediately pivot to a pure C++ or MLX deployment if the model exhibits any of the following characteristics:

* Operations requiring FP64 precision, or operations highly sensitive to FP16 rounding beyond standard Softmax stabilization.27  
* Continuous dynamic sequence lengths that cannot be chunked into a maximum of 128 enumerated shapes.6  
* Dependence on 1D/2D Convolutions with a kernel size greater than 13, or strides greater than 2, which are unsupported by the ANE and force GPU/CPU fallback.13

## ---

**3\. Migration Playbook**

Converting a PyTorch speech model to an ANE-optimized CoreML package is a structured, iterative engineering process. Skipping steps directly leads to runtime crashes and silent precision loss.

### **Phase 1: Audit Checklist Before Conversion**

Before writing export scripts, the PyTorch model must be surgically altered. CoreML requires deterministic, static operations.

* \[ \] **Disable SDPA:** PyTorch's scaled\_dot\_product\_attention dynamically selects backend kernels (FlashAttention, math, etc.) based on hardware and tensor shape at runtime. This breaks the torch.jit.trace graph. Explicitly disable it to force the manual attention implementation.12  
* \[ \] **Implement Stable Softmax:** Locate all F.softmax calls in the attention heads. Replace them with a max-value subtraction implementation (logits \- max(logits)) to prevent FP16 overflow on the ANE.7  
* \[ \] **Isolate the Encoder:** Hard-sever the encoder from the decoder module. Strip out all beam search, greedy decoding algorithms, and text tokenizers. These logical loops belong in Swift or C++, not the MLIR graph.  
* \[ \] **Mock Dynamic Shapes:** Determine the chunk sizes required for audio streaming and create corresponding dummy tensors matching the target EnumeratedShapes.6  
* \[ \] **Address LayerNorm Quirks:** If using custom LayerNorm modules, inject pre-hooks to mathematically divide the bias by the weight scale to prevent ANE hardware inversion bugs.12

### **Phase 2: Ordered Conversion Steps**

1. **Trace the Graph:** Instantiate the modified PyTorch module and trace it using torch.jit.trace with the largest required input shape. Ensure the trace completes without warnings regarding data-dependent control flow or dropped operations.  
2. **Define Shapes:** Construct the coremltools.EnumeratedShapes object containing up to 128 supported audio sequence lengths.5  
3. **Define Pass Pipelines:** If applying compression, configure the OpLinearQuantizerConfig or DEFAULT\_PALETTIZATION passes to compress linear weights.29  
4. **Execute Conversion:** Invoke coremltools.convert() utilizing the traced model, defined inputs, and minimum\_deployment\_target=ct.target.macOS14 (or newer) to ensure modern Model Intermediate Language (MIL) operations are utilized.  
5. **Multi-Function Merge (Optional):** If deploying multiple heads (e.g., ASR and VAD) that share the same backbone, utilize save\_multifunction to merge the converted models into a single package, automatically deduplicating shared weights.20

### **Phase 3: Validation Checklist After Each Step**

1. \[ \] **Python Parity Check:** Execute inference using the original PyTorch model and the new CoreML model within Python. Calculate the Signal-to-Noise Ratio (SNR). Reject any model with an SNR below 20dB.15  
2. \[ \] **Xcode Performance Report:** Open the .mlpackage in Xcode, select a physical device, and generate a performance report. Verify that the primary transformer blocks are assigned to the Neural Engine.30  
3. \[ \] **CLI Benchmarking:** Utilize coreml-cli to benchmark the model sequentially on CPU, GPU, and ANE. Confirm that the ANE latency is strictly lower than the CPU latency.23

## ---

**4\. Failure Modes Catalog**

Deploying speech models on Apple Silicon surfaces esoteric bugs deep within the CoreML compiler stack. This catalog details the most destructive issues, their root causes, and robust fixes.

### **Failure Mode 1: ANE Compiler Service Crash (MILCompilerForANE)**

**Symptom:** During inference initialization (MLModel(contentsOf:)), the application hangs indefinitely or crashes before the first frame is processed. Console logs display MILCompilerForANE error: failed to compile ANE model or an uncatchable \_\_cxa\_throw C++ exception in libBNNS.dylib.16 **Cause:** The MLIR compiler infrastructure (espresso backend) encounters memory layouts, unsupported operations (e.g., massive group convolutions), or unbounded dimension sizes it cannot resolve during Ahead-Of-Time (AOT) specialization. The compiler panics, throwing a C++ exception that Swift's do/catch mechanisms cannot trap, terminating the entire application process.16 **Minimal Fix:** Instantiate the model with MLModelConfiguration().computeUnits \=.cpuAndGPU to bypass the ANE compiler entirely. **Robust Fix:** Re-export the model strictly using EnumeratedShapes instead of RangeDim. If the issue persists, execute a layer bisection using coremltools.models.utils.bisect\_model to identify the specific PyTorch operator triggering the compilation failure. Once identified, rewrite the operator mathematically using simpler primitives.9 **Verification:** Load the model on a physical iOS 17+ device. The initialization should complete in under 1 second without throwing watchdog termination events.

### **Failure Mode 2: The Softmax "Gibberish" Trap (FP16 Overflow)**

**Symptom:** The converted model outputs perfectly accurate transcriptions when executed on the Mac CPU, but outputs nonsensical, repetitive strings (e.g., "The The The The") or complete silence when deployed on the ANE.7 **Cause:** The Apple Neural Engine executes exclusively in 16-bit floating-point precision (FP16). The maximum representable value in FP16 is 65504\. In deep transformer architectures, unscaled logits within the multi-head attention blocks frequently reach magnitudes of \+1e4. When calculating Softmax, ![][image1] immediately overflows to infinity. The subsequent division by the sum of exponentials results in NaN, permanently destroying the attention matrix.27 **Minimal Fix:** Force the specific Softmax layers to execute on the CPU via CoreML MIL manipulation (incurs massive context-switching overhead and latency spikes). **Robust Fix:** Replace PyTorch's native Softmax with a numerically stable max-subtraction identity prior to tracing.8 Subtracting the maximum logit ensures the largest value exponentiated is ![][image2], strictly bounding the results without altering the final probability distribution.33 **Code Example:**

Python

import torch  
import torch.nn.functional as F

def stable\_softmax(logits, dim=-1):  
    \# Subtract max to prevent FP16 overflow on ANE  
    logits\_max \= torch.max(logits, dim=dim, keepdim=True).values  
    stable\_logits \= logits \- logits\_max  
    return F.softmax(stable\_logits, dim=dim)

\# Monkey-patch the model's attention mechanism prior to tracing  
my\_model.attention.softmax \= stable\_softmax

**Verification:** Execute the model on the ANE. Transcriptions should perfectly match the CPU output, and SNR comparisons between the PyTorch FP32 baseline and the CoreML ANE output should exceed 20dB.

### **Failure Mode 3: Custom LayerNorm Bias/Scale Inversion**

**Symptom:** The model compiles successfully, runs on the ANE without crashing, but the SNR is exceptionally low (0 to 5 dB), resulting in severely degraded audio transcription accuracy or hallucinations. **Cause:** Certain custom PyTorch LayerNorm implementations, when translated to CoreML and mapped to ANE hardware primitives, suffer from a well-documented inversion bug where the ANE compiler incorrectly interprets the scale and bias parameter order.12 **Minimal Fix:** Restrict the compute units to .cpuAndGPU to bypass the ANE LayerNorm kernel. **Robust Fix:** Implement a PyTorch pre-hook to mathematically divide the bias by the weight scale prior to state dictionary loading and model tracing. This offsets the hardware's inversion bug.12 **Code Example:**

Python

def correct\_for\_bias\_scale\_order\_inversion(state\_dict, prefix, local\_metadata, strict, missing\_keys, unexpected\_keys, error\_msgs):  
    \# Fixes ANE bias scaling hardware quirk  
    state\_dict\[prefix \+ 'bias'\] \= state\_dict\[prefix \+ 'bias'\] / state\_dict\[prefix \+ 'weight'\]  
    return state\_dict

class LayerNormANE(LayerNormBase):  
    def \_\_init\_\_(self, \*args, \*\*kwargs):  
        super().\_\_init\_\_(\*args, \*\*kwargs)  
        self.\_register\_load\_state\_dict\_pre\_hook(correct\_for\_bias\_scale\_order\_inversion)

**Verification:** Run the parity testing suite. The SNR should immediately jump from \<5dB to \>20dB on ANE hardware.

### **Failure Mode 4: Infinite Latency on Dynamic Sequence Lengths (RangeDim)**

**Symptom:** A model initialized with RangeDim(lower\_bound=1, upper\_bound=-1) converts successfully and runs flawlessly on the Xcode Simulator (CPU), but exhibits multi-second latency spikes per inference when deployed to an M2 Max or iPhone 15 Pro in production.6 **Cause:** The CoreML runtime cannot pre-allocate static hardware buffers for undefined shapes on the Apple Neural Engine. When the runtime detects an unbounded dimension, it immediately abandons ANE execution, dynamically recompiling the MLIR graph on the fly, and falls back to CPU execution, entirely bypassing the hardware accelerator.17 **Minimal Fix:** Hardcode the PyTorch model to accept a single, massive static sequence length and zero-pad all incoming audio arrays. **Robust Fix:** Purge RangeDim entirely. Utilize coremltools.EnumeratedShapes to specify a finite set of supported sequence lengths, allowing the ANE compiler to Ahead-Of-Time (AOT) compile optimized execution graphs for each shape.5 **Code Example:**

Python

import coremltools as ct

\# Define standard audio chunk lengths (e.g., 1s, 5s, 15s, 30s)  
shapes \=

mel\_input \= ct.TensorType(  
    name="mel",   
    shape=ct.EnumeratedShapes(shapes=shapes, default=shapes)  
)

**Verification:** Run the model through Xcode Instruments. The Neural Engine instrument should display heavy duty cycles, and latency should drop by an order of magnitude.

### **Failure Mode 5: Layout Thrashing (NCHW to NHWC)**

**Symptom:** Xcode performance reports show the model successfully utilizing the ANE, but the overall latency is significantly worse than expected. The Instruments timeline reveals excessive time spent in Transpose or Reformat operations. **Cause:** PyTorch natively represents 4D tensors in NCHW (Batch, Channel, Height, Width) format. However, the optimal memory access pattern for Apple Tensor Cores and the ANE is NHWC (Channel-Last).21 If a speech model frequently switches between temporal and channel-based convolutions, coremltools automatically injects layout conversion operators. The ANE has a much lower memory bandwidth ceiling than the GPU; forcing it to continually shuffle memory layouts chokes the hardware.22 **Minimal Fix:** Ignore the layout and accept the latency penalty. **Robust Fix:** Rewrite the PyTorch model to utilize a consistent memory layout throughout the encoder block. Perform all spatial transformations at the very beginning or end of the network, preventing interleaved transposition operations within the deep transformer layers.21 **Verification:** The Xcode Performance Report should show Transpose operations consuming less than 5% of the total inference timeline.

## ---

**5\. Best Practices**

* **Enforce Strict FP16 Emulation in PyTorch:** Do not wait until the final CoreML conversion step to discover precision degradation. Cast the PyTorch model to .half() and execute extensive validation during Python testing. If the PyTorch FP16 execution fails or produces NaNs, the CoreML ANE execution will inevitably fail as well.  
* **Decouple the Audio Pre-Processor:** Audio preprocessing (e.g., Mel-Spectrogram extraction, windowing, STFT) must remain in native Swift (leveraging the Accelerate framework) or optimized C++. Attempting to trace and export raw-waveform-to-spectrogram transforms directly inside the CoreML graph bloats the model size and frequently causes immediate CPU fallback due to unsupported complex mathematical operations.  
* **Utilize PassPipelines for Compression:** Memory bandwidth, not compute FLOPs, is the primary bottleneck for speech models on Apple Silicon.24 Utilize coremltools optimization configurations (e.g., OpLinearQuantizerConfig) for weight palettization or Int4 blockwise quantization. Palettization groups similar weights into clusters, drastically reducing the physical memory size transferred to the ANE during execution without destroying structural accuracy.29  
* **Leverage CLI Tooling over Xcode:** Xcode Instruments is heavy and difficult to integrate into automated pipelines. Use coreml-cli (xcrun coremlcompiler) to build automated CI/CD checks that assert ANE latency targets on every commit, preventing silent hardware fallback regressions from reaching production.23

## ---

**6\. Worst Practices / Anti-Patterns**

* **Exporting Beam Search:** Attempting to force decoding logic, autoregressive loops, and conditional branches (e.g., beam search, nucleus sampling) into the MIL graph is a fatal error. coremltools struggles fundamentally with dynamic control flow. This logic belongs strictly in the application runtime (Swift/C++).  
* **Using np.allclose for Parity Validation:** Utilizing standard boolean equality checks (np.allclose(pytorch\_out, coreml\_out, atol=1e-4)) for parity validation will fail immediately on speech models. The accumulated numerical deviations caused by hardware-specific FP16 mathematical approximations in deep transformer architectures guarantee non-identical float outputs. SNR must be used instead.15  
* **Masking Bugs with CPU\_ONLY:** When encountering an MLIR crash or gibberish output, developers frequently specify computeUnits \=.cpuOnly to "fix" the bug. This fundamentally destroys the battery life, thermal profile, and user experience of the mobile application. The underlying operator incompatibility must be mathematically resolved.  
* **Passing Strings across the CoreML Boundary:** CoreML is highly inefficient at string manipulation. Tokenization and detokenization algorithms must occur outside the ML model. The CoreML I/O boundary should accept strictly integer ID arrays or floating-point audio arrays.

## ---

**7\. Weird Patterns that Actually Work on Apple Silicon**

* **Un-squeezing Linear Layers to Conv2D:** Due to how the ANE optimizes spatial memory layouts, manually unsqueezing linear layer weights (converting nn.Linear ![][image3] nn.Conv2d(1x1)) before export can occasionally trigger more efficient ANE block allocations, bypassing certain internal compiler limitations and resulting in subtle latency improvements.12  
* **The 128-Shape Trick:** CoreML EnumeratedShapes enforces a strict physical limit of exactly 128 pre-compiled shapes.6 Compiling exactly 128 shapes representing granular audio increments (e.g., 1-second, 2-second... up to 128 seconds) provides the illusion of flawless dynamic audio streaming to the end user without triggering the catastrophic performance cliff associated with RangeDim.  
* **Deduplicating Constants via save\_multifunction:** Merging multiple task-specific adapter models with a shared base model via save\_multifunction literally executes a cryptographic hash check over the model weights to identify structural overlaps. It functions as an exceptionally effective parameter-efficient deployment tool, shrinking overall application bundle sizes drastically without requiring the engineering overhead of building dynamic weight-loading architectures.20

## ---

**8\. Limitations and Unsolved Problems**

Despite continuous improvements in the Apple Machine Learning stack and coremltools 8.0/8.1, several systemic issues remain unsolvable dead-ends:

* **Uncatchable C++ Compiler Panics:** As of macOS 15, the MLIR compiler service can still crash the parent application when compiling incompatible ANE hardware instructions. Because the exception is thrown asynchronously inside libBNNS.dylib, there is no software catch mechanism available to iOS or macOS developers to gracefully recover or fallback to a safe state.16  
* **Complete KV Cache Transparency:** While iOS 18 introduced StateType for stateful models, inspecting, slicing, or evicting elements from the KV cache dynamically *during* execution (e.g., for complex beam-search branching or context eviction) is completely unsupported. The stateful cache remains an opaque memory block managed exclusively by the CoreML runtime.2  
* **Lossy Export Paths:** The PyTorch torch.export path is officially recommended by PyTorch as the future of model conversion, but its integration within coremltools remains lossy and brittle. Fallbacks to the deprecated torch.jit.trace are required for approximately 30-40% of standard speech models, forcing teams to maintain legacy tracing pipelines indefinitely.10

## ---

**9\. Copy-Paste Reference Snippets**

The following code snippets represent production-grade implementations of the concepts detailed in this report.

### **Snippet A: Export Preparation & Encoder-Only Conversion**

This script demonstrates disabling SDPA, applying enumerated shapes, and executing the CoreML conversion for an audio encoder.

Python

import torch
import coremltools as ct
from transformers import AutoModelForCausalLM

\# 1\. Disable SDPA to allow deterministic tracing (use attn_implementation="eager")
\# For HF models: pass attn_implementation="eager" to from_pretrained
\# For custom modules: set module.use_sdpa = False before torch.jit.trace

\# Assume \`encoder\` is the isolated PyTorch Audio Encoder module  
encoder.eval()

\# 2\. Define Enumerated Shapes for variable-length audio   
shapes \=

mel\_input \= ct.TensorType(  
    name="mel\_spectrogram",   
    shape=ct.EnumeratedShapes(shapes=shapes, default=shapes)  
)

\# 3\. Trace the model using the maximum shape  
dummy\_input \= torch.randn(1, 80, 3000)  
traced\_encoder \= torch.jit.trace(encoder, dummy\_input)

\# 4\. Execute CoreML Conversion with ANE optimizations  
mlmodel \= ct.convert(  
    traced\_encoder,  
    inputs=\[mel\_input\],  
    compute\_units=ct.ComputeUnit.ALL,  
    minimum\_deployment\_target=ct.target.macOS14  
)

mlmodel.save("GemmaEncoder\_ANE.mlpackage")

### **Snippet B: Parity Testing via Signal-to-Noise Ratio (SNR)**

This script calculates the SNR between PyTorch and CoreML outputs. Do not use np.allclose.15

Python

import numpy as np  
import coremltools as ct  
import torch

def compute\_snr(reference: np.ndarray, target: np.ndarray) \-\> float:  
    """Calculates SNR. Values \> 20dB indicate excellent parity."""  
    ref\_flat \= reference.flatten()  
    tgt\_flat \= target.flatten()  
      
    noise \= ref\_flat \- tgt\_flat  
    noise\_variance \= np.sum(noise \*\* 2) / len(noise) \+ 1e-9  
    signal\_energy \= np.sum(ref\_flat \*\* 2) / len(ref\_flat) \+ 1e-9  
      
    if signal\_energy \< 1e-10:  
        return 100.0 \# Clip maximum SNR if identically zero  
          
    snr \= 10 \* np.log10(signal\_energy / noise\_variance)  
    return snr

\# Run baseline PyTorch inference  
pytorch\_out \= pytorch\_encoder(dummy\_audio).detach().numpy()

\# Run CoreML inference  
coreml\_model \= ct.models.MLModel("GemmaEncoder\_ANE.mlpackage")  
coreml\_out \= coreml\_model.predict({"mel\_spectrogram": dummy\_audio.numpy()})\["output"\]

\# Validate  
snr\_value \= compute\_snr(pytorch\_out, coreml\_out)  
print(f"Encoder SNR: {snr\_value:.2f} dB")  
assert snr\_value \> 20.0, "Catastrophic precision loss detected. Check FP16 Softmax."

### **Snippet C: PyTorch Stateful Decoder Preparation (register\_buffer)**

This demonstrates preparing a PyTorch autoregressive model for iOS 18+ stateful decoding, preventing KV cache memory thrashing.9

Python

import torch

class StatefulDecoder(torch.nn.Module):  
    def \_\_init\_\_(self, max\_seq\_len, hidden\_dim):  
        super().\_\_init\_\_()  
        \# Register the KV cache as a persistent buffer  
        self.register\_buffer(  
            "kv\_cache",   
            torch.zeros((1, max\_seq\_len, hidden\_dim), dtype=torch.float16)  
        )  
        self.register\_buffer("cache\_idx", torch.tensor(, dtype=torch.int32))

    def forward(self, input\_token, cross\_attention\_embeds):  
        idx \= self.cache\_idx.item()  
          
        \# Calculate new Keys and Values  
        new\_kv \= self.\_compute\_kv(input\_token)  
          
        \# In-place slice update \- CoreML maps this to hardware state mutation  
        self.kv\_cache\[:, idx:idx+1, :\] \= new\_kv  
        self.cache\_idx \+= 1  
          
        \# Calculate attention using the updated cache  
        current\_cache \= self.kv\_cache\[:, :idx+1, :\]  
        logits \= self.\_compute\_attention(current\_cache, cross\_attention\_embeds)  
          
        return logits

\# During CoreML conversion, map 'kv\_cache' and 'cache\_idx' using ct.StateType()

### **Snippet D: Swift-to-C++ Hybrid Deployment Data Bridge**

When utilizing a hybrid deployment (CoreML Encoder \+ C++ Decoder), the MLMultiArray holding the cross-attention tensor must be passed directly to C++ without deep-copying memory. Swift's ARC will destroy the pointer if withExtendedLifetime is omitted.18

Swift

import CoreML

func runHybridInference(audioBuffer: MLMultiArray) throws \-\> String {  
    // 1\. Run ANE-optimized CoreML Encoder  
    let encoderOutput \= try coremlEncoder.prediction(mel\_spectrogram: audioBuffer)  
    let crossAttentionArray \= try encoderOutput.featureValue(for: "output")\!.multiArrayValue\!

    // 2\. Extract raw data pointer  
    let rawPointer \= crossAttentionArray.dataPointer  
    let typedPointer \= rawPointer.bindMemory(to: Float.self, capacity: crossAttentionArray.count)

    var decodedText \= ""

    // 3\. CRITICAL: Prevent Swift ARC from destroying the MLMultiArray   
    // while the C++ decoder is actively reading the tensor data.  
    withExtendedLifetime(crossAttentionArray) {  
        // Pass pointer to C++ decoder (e.g., ggml or custom implementation)  
        decodedText \= CPPExternalDecoder.decodeAutoregressive(  
            encodedFeatures: typedPointer,  
            sequenceLength: Int32(crossAttentionArray.shape.intValue),  
            featureDim: Int32(crossAttentionArray.shape.intValue)  
        )  
    }  
      
    return decodedText  
}

### **Snippet E: Compute-Unit Debugging via coreml-cli**

Automate latency and hardware fallback checks using the terminal.23

Bash

\#\!/bin/bash  
\# Benchmark ANE performance vs CPU performance to verify acceleration

echo "Benchmarking ANE Latency..."  
xcrun coremlcompiler benchmark GemmaEncoder\_ANE.mlpackage \\  
    \--input dummy\_mel.npy \\  
    \--device ane \\  
    \--json \> ane\_results.json

echo "Benchmarking CPU Latency..."  
xcrun coremlcompiler benchmark GemmaEncoder\_ANE.mlpackage \\  
    \--input dummy\_mel.npy \\  
    \--device cpu \\  
    \--json \> cpu\_results.json

\# Parse JSON to ensure ANE latency is strictly lower than CPU

## ---

**10\. Final "Red Flags" Checklist Before Shipping**

Before deploying the compiled .mlpackage to the production repository, execute the following audit to prevent disastrous application behaviors in the wild:

1. \[ \] **Compute Unit Verification:** Open the model in Xcode. Does the Performance Report confirm that ![][image4] of operations execute on the Neural Engine? If the timeline shows massive Cast or Transpose blocks on the GPU, the layout optimization failed.21  
2. \[ \] **Dynamic Shape Audit:** Verify RangeDim has been completely purged from all axes, barring the batch dimension (if strictly locked to 1). Ensure EnumeratedShapes is utilized and properly documented for the downstream Swift engineers.5  
3. \[ \] **Softmax Safety Verification:** Search the PyTorch codebase for F.softmax. Has maximum-value subtraction been implemented to protect against ANE FP16 overflow? If not, the model will output gibberish upon encountering loud or complex audio data.7  
4. \[ \] **SDPA Removal Check:** Has torch.nn.functional.scaled\_dot\_product\_attention been explicitly deactivated, and manual mathematical attention mapped in its place?.12  
5. \[ \] **SNR Validation Protocol:** Does the coremltools Python inference match the PyTorch FP32 baseline with an SNR ![][image5]dB on real-world audio samples? (Do not test exclusively with zero-arrays or random noise).15  
6. \[ \] **Memory Lifecycle Audit:** In hybrid deployment scenarios, is withExtendedLifetime wrapping every single MLMultiArray passed to the C++ decoder? Failing this check guarantees random, irreproducible memory access violation crashes (EXC\_BAD\_ACCESS) in production.19  
7. \[ \] **Physical Device Testing:** Has the model been instantiated via MLModel(contentsOf:) on physical A17/M3 silicon to ensure the MLIR AOT compiler does not trigger a kernel panic or watchdog termination event?.16

By rigorously enforcing these engineering methodologies, teams can successfully navigate the complexities of the CoreML compiler stack, achieving the exceptional performance, low-latency, and power efficiency characteristic of native Apple Silicon deployments.

#### **Works cited**

1. KV Caching Explained: Optimizing Transformer Inference Efficiency \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/blog/not-lain/kv-caching](https://huggingface.co/blog/not-lain/kv-caching)  
2. KV Caching from Scratch — Pytorch | by Ali Shafique | Medium, accessed April 2, 2026, [https://medium.com/@alishafique3/kv-caching-from-scratch-pytorch-5743ddcdc176](https://medium.com/@alishafique3/kv-caching-from-scratch-pytorch-5743ddcdc176)  
3. Core ML | Apple Developer Documentation, accessed April 2, 2026, [https://developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml)  
4. altalt-org/Lightning-SimulWhisper: An MLX/CoreML implementation of SimulStreaming. \~15x increase in performance \- GitHub, accessed April 2, 2026, [https://github.com/altalt-org/Lightning-SimulWhisper](https://github.com/altalt-org/Lightning-SimulWhisper)  
5. Flexible Input Shapes \- Core ML Tools, accessed April 2, 2026, [https://coremltools.readme.io/v4.0/docs/flexible-inputs](https://coremltools.readme.io/v4.0/docs/flexible-inputs)  
6. Flexible Input Shapes — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)  
7. Keep Attention Softmax FP32 during FP16/ZeRO Training · Issue \#1485 · hpcaitech/ColossalAI \- GitHub, accessed April 2, 2026, [https://github.com/hpcaitech/ColossalAI/issues/1485](https://github.com/hpcaitech/ColossalAI/issues/1485)  
8. Softmax Uncovered: Balancing Precision with Numerical Stability in Deep Learning | by Harriet | Medium, accessed April 2, 2026, [https://medium.com/@harrietfiagbor/softmax-uncovered-balancing-precision-with-numerical-stability-in-deep-learning-b8876490d411](https://medium.com/@harrietfiagbor/softmax-uncovered-balancing-precision-with-numerical-stability-in-deep-learning-b8876490d411)  
9. Releases · apple/coremltools \- GitHub, accessed April 2, 2026, [https://github.com/apple/coremltools/releases](https://github.com/apple/coremltools/releases)  
10. PyTorch Conversion Workflow — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)  
11. Stateful Models — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/stateful-models.html](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)  
12. whisper.cpp/models/convert-whisper-to-coreml.py at master · ggml ..., accessed April 2, 2026, [https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-whisper-to-coreml.py](https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-whisper-to-coreml.py)  
13. unsupported-layers.md \- hollance/neural-engine \- GitHub, accessed April 2, 2026, [https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md)  
14. Signal-to-Noise Ratio (SNR) — PyTorch-Metrics 1.9.0 documentation \- Lightning AI, accessed April 2, 2026, [https://lightning.ai/docs/torchmetrics/stable/audio/signal\_noise\_ratio.html](https://lightning.ai/docs/torchmetrics/stable/audio/signal_noise_ratio.html)  
15. Typed Execution Workflow Example — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/typed-execution-example.html](https://apple.github.io/coremltools/docs-guides/source/typed-execution-example.html)  
16. The Uncatchable CoreML Crash: How a single MLIR compiler failure on the iPhone SE2 cost me a week : r/swift \- Reddit, accessed April 2, 2026, [https://www.reddit.com/r/swift/comments/1s777t4/the\_uncatchable\_coreml\_crash\_how\_a\_single\_mlir/](https://www.reddit.com/r/swift/comments/1s777t4/the_uncatchable_coreml_crash_how_a_single_mlir/)  
17. CoreML MLE5ProgramLibrary AOT recompilation hangs/crashes on iOS 26.4 — C++ exception in espresso IR compiler bypasses Swift error handling \- Apple Developer, accessed April 2, 2026, [https://developer.apple.com/forums/thread/821073](https://developer.apple.com/forums/thread/821073)  
18. How to Integrate CoreML Models Into C/C++ Codebase \- Krisp, accessed April 2, 2026, [https://krisp.ai/blog/how-to-integrate-coreml-models-into-c-c-codebase/](https://krisp.ai/blog/how-to-integrate-coreml-models-into-c-c-codebase/)  
19. CoreMLHelpers/CoreMLHelpers/MLMultiArray+Helpers.swift at master · hollance/CoreMLHelpers \- GitHub, accessed April 2, 2026, [https://github.com/hollance/CoreMLHelpers/blob/master/CoreMLHelpers/MLMultiArray%2BHelpers.swift](https://github.com/hollance/CoreMLHelpers/blob/master/CoreMLHelpers/MLMultiArray%2BHelpers.swift)  
20. Multifunction Models — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html)  
21. Deploying Attention-Based Vision Transformers to Apple Neural Engine, accessed April 2, 2026, [https://machinelearning.apple.com/research/vision-transformers](https://machinelearning.apple.com/research/vision-transformers)  
22. NHWC vs NCHW : A memory access perspective on GPUs | by Deepika \- Medium, accessed April 2, 2026, [https://medium.com/@deepika\_writes/nhwc-vs-nchw-a-memory-access-perspective-on-gpus-4e79bd3b1b54](https://medium.com/@deepika_writes/nhwc-vs-nchw-a-memory-access-perspective-on-gpus-4e79bd3b1b54)  
23. Inspect, Benchmark, and Run Core ML Models from the Command Line | Schappi.com, accessed April 2, 2026, [https://schappi.com/blog/inspect-benchmark-and-run-core-ml-models-from-the-command-line](https://schappi.com/blog/inspect-benchmark-and-run-core-ml-models-from-the-command-line)  
24. Whisper.cpp has a coreml option which gives 3x speed up over cpu only according, accessed April 2, 2026, [https://news.ycombinator.com/item?id=43880345](https://news.ycombinator.com/item?id=43880345)  
25. WhisperKit: On-device Real-time ASR with Billion-Scale Transformers \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2507.10860v1](https://arxiv.org/html/2507.10860v1)  
26. ANE inference fails on M4 \+ macOS 26.4 beta with CoreML encoder \#3702 \- GitHub, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp/issues/3702](https://github.com/ggml-org/whisper.cpp/issues/3702)  
27. Accuracy Considerations — NVIDIA TensorRT, accessed April 2, 2026, [https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)  
28. Performance — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)  
29. Conversion — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/opt-conversion.html](https://apple.github.io/coremltools/docs-guides/source/opt-conversion.html)  
30. Core ML \- Machine Learning \- Apple Developer, accessed April 2, 2026, [https://developer.apple.com/machine-learning/core-ml/](https://developer.apple.com/machine-learning/core-ml/)  
31. schappim/coreml-cli: A native command-line interface for working with Apple Core ML models on macOS \- GitHub, accessed April 2, 2026, [https://github.com/schappim/coreml-cli](https://github.com/schappim/coreml-cli)  
32. Fp16 overflow when computing matmul in autocast context \- PyTorch Forums, accessed April 2, 2026, [https://discuss.pytorch.org/t/fp16-overflow-when-computing-matmul-in-autocast-context/183373](https://discuss.pytorch.org/t/fp16-overflow-when-computing-matmul-in-autocast-context/183373)  
33. python \- Numerically stable softmax \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/42599498/numerically-stable-softmax](https://stackoverflow.com/questions/42599498/numerically-stable-softmax)  
34. Numerically Stable Softmax \- Brian Lester, accessed April 2, 2026, [https://blester125.com/blog/softmax.html](https://blester125.com/blog/softmax.html)  
35. Flexible Shapes not working with ONNX to MLModel conversion using coremltools 4, accessed April 2, 2026, [https://stackoverflow.com/questions/64708252/flexible-shapes-not-working-with-onnx-to-mlmodel-conversion-using-coremltools-4](https://stackoverflow.com/questions/64708252/flexible-shapes-not-working-with-onnx-to-mlmodel-conversion-using-coremltools-4)  
36. Flexible Input Shapes on Neural Engine · Issue \#2370 · apple/coremltools \- GitHub, accessed April 2, 2026, [https://github.com/apple/coremltools/issues/2370](https://github.com/apple/coremltools/issues/2370)  
37. Optimization Guidelines for the Apple Neural Engine (ANE) \- GitHub Gist, accessed April 2, 2026, [https://gist.github.com/antmikinka/715499ae63630575065b22e5cb6ad8dd](https://gist.github.com/antmikinka/715499ae63630575065b22e5cb6ad8dd)  
38. Optimizing Data Layout for Training Deep Neural Networks | Request PDF \- ResearchGate, accessed April 2, 2026, [https://www.researchgate.net/publication/362730913\_Optimizing\_Data\_Layout\_for\_Training\_Deep\_Neural\_Networks](https://www.researchgate.net/publication/362730913_Optimizing_Data_Layout_for_Training_Deep_Neural_Networks)  
39. On-Device AI Models and Core ML Tools: Insights From WWDC 2024 | HackerNoon, accessed April 2, 2026, [https://hackernoon.com/on-device-ai-models-and-core-ml-tools-insights-from-wwdc-2024](https://hackernoon.com/on-device-ai-models-and-core-ml-tools-insights-from-wwdc-2024)  
40. This is really interesting, thank you. What would be the downside to padding all... | Hacker News, accessed April 2, 2026, [https://news.ycombinator.com/item?id=38908463](https://news.ycombinator.com/item?id=38908463)  
41. Examples \- Core ML Tools, accessed April 2, 2026, [https://coremltools.readme.io/v6.3/page/examples](https://coremltools.readme.io/v6.3/page/examples)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAYCAYAAAB6OOplAAAFfklEQVR4XsVYZ6geRRS9Y15iRbH7TIioIDbsLaJo7BpsWLBhAQV7Q5CgRlFR/GHDiiXBBkZFRBC7xgqiYkRFRUUD+kPsih3zPGfu7Lezd2bna+/lHTj77Z6502fu3PlECOefBipmk2LEBsY4m7dgk7XPwvVjPI6YlEoJMxn9jlo2Y+8YLJcU56lNL6OQq07qd3Sk2NAUPZc/DalrWbELpluhhtZVrJHINi/NlSq9Y5i8w2WOEIZjCh5Pg7s0ErvjPuQ9xoq9Y7x6YZAvNq9WKKcOgPwKulH8oCXYCDwbvAacY9KIUfB7lLObTfAYpvFtedv0GsuBp0jbbmvLn+qpUkZX+4PB38BRYzkbXAKeB+4FvoSiFqaluYvx+BKcalOWMU4DHwV/AMfAHauEtM2Tg1fAuaY1U/D5NX4virQ1RCfkpErQLG4aHp+Bpzf1ZYBmReeI7rrrxQz0OGDoLm0t2ih7CB4S9O0aVTh5Hc8XwnuMeeD7DcVigKb2laU25qIZc9mB7hj5l3VEO7pNnBIA/+P2xu/2rk7bXNSXxlgV3AdcJXyvCR4luipjXAL+bDTiJtGB3tDoT4J/o+oRox8HLgXXNnoX2O61Qy2jXrdjrpRWNPJvINqRRaJ+cQF4R2QyA3wTvBZ8FnwQuR4QXU0fiU4OsaJoBPE8uBg8F7wLvAD8ADwg2BH3gO9kWv+QaGN52MWgD6S+rtF3CvrhRp8M+IHOr2gd5G/AV8GpodsrgN/WJt6XVitsX/Edc4wGDtJ3P+AE/eocjN0JQb8q6MQt4NudL4cDTuThzncNTmRuQBcGfbOG6vxOoX5WQ0/BHcY62ZccF4EvBxvyRXBbZuwDxRX9FPi7aMdGMNBb4vcx0ZVNcCLOD+8EO8TCNhbNcyG4ekjjiucBdSv4h3+vFyx3CLe4uhDno4V4Iio8I1r+ekZn1EF9E6MTP4KXW1GR7JieMFiuzkBzl8VwoyhxqfOr1z0Bge4AJ6ezhnXFTh4RH3oVm0I3YQ+u18C/pA7FfhWdJFsU3JJvbBSLegNOPnXr64kvwKvrz2LbumCYvJ2B3tkmzAoJV/qv7nXQ4jvwfpsQgVEEV+6lkUbf/Sf4RqQtcbrym/U6uU5yLkLkOdGdZ1vJm+U/kBnLxguihr7zycOcrq9AF7/zIA+w1eaqaB/oTUPCGTYBWF9/3HSUwm2/hTDkUvtTNck/OTA8LCscIWqze6QdCf4H+1l1n+VdUTdhsado/v2N/gn4uNEIRj2wd4x0SlhJNNLhAuiVdI+tyExqNdD27wT/P8PH4qMMF2c4UfQwYDp937/gVjCoAvIDg90O4A3hvQIPPZ68h4Xv1cDPJbWji6BuwTrpehjmVaBf5o7IDeZ+om1qhpnpApREbHS5f2TycqzYd7bJI7ZhzPwpSB99t3jf6q+2lS/dVfTAvF3Up14BvgfOFx2slYNdhQ/BxU7DxQWo6S38Hm9s2IKTRSdwJNPimeBXonWwTk7IsbFBhDPBX4TxdVpOAbFxmjFViuDOZBt5L0Bb/C2WN9Z5thx+c0U0YtfIiNfimZEC3+WqSCMGLzxjMLssfHPAuGVTOO+a4FuxKzrFNprFiwnPEF52eGi34V7wZiu2or2ciYSLKo7f9TNFVoxxtOg23sMmeKTZ7xS93AwK3kxxW6xDvrSKVElh+q5K9n1y0GwBXRC2kKsGOheGWcwAfwIPHbAzjERui4UBy2nB8KVlSshI/YH+nWEfY3He+JJYvIG6Ov4NygN5+Y6SINs2Hrb+Nquf6apMEdt0NR4e3asoNDrW22wMejBjSDjbil3AvwDsP389odyecmqKAQakFW3jnhWHxESUmYVW1Ht1EzII/WZusx9stnu3LCBbSFaccPwPrWDjPB54aNEAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAABU0lEQVR4XnVUO05DQQxcSxTU6UCUcIpcB9FQIQ5AzTEoUCradBE1PedBzNperz+7E9k7Hn/W7ylJa43aHiNH/MnqRFAk6F6MSsdyFM0O0xdzE3T45gK39bLbwJcHQZ0tkldZJ1TrzuubmqKEbVejFRRzdhe3yfWmlEfbj42YZcrChl4SUf1LKfIIaw/Pj1Ou6/iAPUxF62aY4RRPuYeOoOfao+gJf88euje1Z7CnqAo9wN1mw1Dh1HnXNMZJon3CvrTetniHncBPfFLDSRY3jsVITbUf1P1hzTceZihP5oXINbqHXWD9/bmvkzL2g6dZkjN+Df+L6M5VKcpWC7jpYI84XkfgVhkY14amwhTfUK6WY+y3Vn9HhtRwY4I0ulQIV9+ypNGs3810irfofYWHi5W6G0t57mYsRSC/r/EeOzVtkDyGZvs8hY1EyHcXps5x4x/jHySRFb8H/URUAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAXCAYAAABefIz9AAAEb0lEQVR4Xq1XW4hWVRReW8YmyposyNQuRBmZNjLhZNOdinqIyl7sAkmURKASFcygUaEV00swBTEN3S9UAzVFaYUFRaOGSoq+VPTgkw9BlycrpJi+b699zr6cvf/zD/TBN/tfa317nX3fe0TEiAd/m9gVGwly+lkgrefSRfYs4OVdVywL44jvZLFGMdA9ohRJvv8vve2LH+ruE3evTOFq9qE8MY40wPhxqTODlsYYH4+VJX8KO0pZBKlDbAD3oM4kyicl34n54LfgxWmgA/Jfa4OfX1kMroP1GMqBQOJhZA7+3oIfz6B8GOXCRIEc5k/oTnf2NHgAXA//RSiXgg+BP4PPOk0ZjS4ljkY8B52GlRDvQ7ke1hDK98CPI53OxAfgdvBqVByG9oiovsK94MHAvt9xldgBkRHwLvB7sLcS6TbyOEmCYBGuTledFDkMnpP49qHuPYH9IPgLeELg40xyNnqc/Tq+OB189AZwc20RRqYQvtT9zmIF+BX4BHhaSZXzqq+K1IozwRnwisrhMAa+GNj7wU8Cm2AHWPdKZ3PZUVdl52zd7GLEWsyWLs1cAxU2wj+3gV+LNuSsUNEd6i9wX/0qOhOXOR9n6UexS9HiFOjZkTedXeES0Q4+7pb6reBv4PEuPgHOc78Xgbukm9WXgKP4OfgaeGEUaYySdyQhnnZs6L/g8+AXEGwMVEtcfIKuoO4y5x8PjtRXwTcgGkb5SuWEzdm3S5PKRtMsyhGClaeEB4GRQSvMamNnkHJctLEk99p1tUAPEu1IAKOnIv2TQV6uiGvB28XvzfvA0eDTjG8B+2tPjVyjY98ymG/Dud34JRYPTpLD6GHCvXMTuBv2DHgMkWuchEuXHXmpqqM5TNXBd2t/E9zjWJpGl6Yx7Oxe8HqjZwlXRyfkemz35EeiJ2FvQVNhENE/UFZ3F2fgEfBv8CdX9XzRjrzsNOJGbLnzv+D9DWwTXkMeh8TepRas/2gQa8UFontgB75/YxosgI1mHdvmYCi2ijb+VOFBYeQfsUsxwuWimpF4DGsDDwd5uraN4GFgD6uBQJPm9Ahy8uXxvuishRdvNxhDIu1gDF5HfyHW4770DfhdshbuFO1gf2aRnA3ulPipxuuE+qWB/LNM3RpXgZ9iXWPPYbprYbNGBw/32e/geYnmOYkPlTvAo6LHvYN5S7TjzowKvnjSJx/foOxgtWS5L3lqN8DDYIfYi9icmwYbyHU8/rlWtPEfgpvgYW6sCGNfLYH0KfAH0ffkO+Bu8GQfJqz6AdGrJzeyPPyqFcPn22of0ruGJ9oouKBZVzIJG44GrMJIH/LzTr1bdPP7WIwzwDXgEGJz06DoJf+l+CsiRZ/ovc0Z5t1d0jmkLUjtEJ1inRDWa8nhBmtOq1BPa2no0pd3hdjrrbw6BVVlZTlSAYrCP4etcCNSQCnAD/qiibxX0SkWojTUik4xi5wg52tDWsfbYUR/Z2PJzKT5ZoP2uk6RFWad7YjaX/9Oklmz9IG8P+9tokrN8j9v4pPq6hOmkAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAADA0lEQVR4Xp1WW4hOURTeW26JaUq5lQeFZhqXqBFCSBRJXlDe5U1eeDFKQskDSkgUI1GYoknJLYWkpJAHb0J5JW80vrXX2vusfTnn/OOr7+y9vnU5a1/O/GNMx7CpwKiRI3QSY0phuaLQ5BSfjaOaMjJQcFtC7OeMthxCFKOMutw63aFUrJRQaam3sNSkaJpBqFQ9y+G9ukoep5TUaVPBwxZiExtYDR4RbiYhjYE9Bs+tmB4D90OYqXw5gugnxSgPdkYhVeOnwEHMd2M8DfUvxkfgxCrYjAdvg8PgGvAA+A1c4byq8BRwQqKNDm6rQ/Za8B3YpY7uDDgCnvQCsBf8AU4KinU7+Rkcq5tZbHh1hyFOrWRCTcs1MsMeMtzMBSXSDpH2VWlvwXvKJmwwHLcqlnmp2/B4CgNHYmbHfr1D1TkWG7W2H8/X4HalTrP84i9id4t9VcUQlhrWB/LSolhexQPwCtijInK4vvNSupiMWwy/+Czbdp7YF6tFukkfhhHwvFe1Mx2WgXctX+T+8na1Q7JeYPITY68I9CFQg2kj5Cf9ltJaX9yHmEGMw5bvkqA1T2D34PELXG/DbtvlJr+nhF7SEXUjKPWvUR7r7uQQ+AZz98UHd30Bwjrwu3EnIKEcP9dwg5ckzmOB6HwVWooT5oOXsfCHGDemziag9EIM7zHpodfIq/aJezL4x8RHSUErDTd4UGkxBUvAm7CHrP/DKSh+Ew6Rg3b8iYwe4wx/3R7PwFc0UZm7DDe4qJJi0M/TfZDuHG23Sbc57S+1oXTj8RH8AD429KfLmucYP4F3VMZO8Dc4K0jWXDPcuLcDNoF0jOfAOZXchLw1AX4NLO2CYrCPc0jIPYo5NY6jt9ehvgS7vNODvqgT4PTUUYva3irEv36NmAHuQDBdJboGct1Ccn2V4j9TTmK94K2QOWOhsvQsS2rGKMMVJFMV4I/LC/+zwA6R5kV2MNIoQkkTtegqiozcRUqulqQcaZCyZZpGOGSiDpZ+4pgko65AKgVZ+9PiiS3zf23+YpUzlcObAAAAAElFTkSuQmCC>