# Known Issues

## General Issues

- distil-whisper IS NOT UP TO DATE, this includes pseudo_label.py
- Cache not utilized properly? Every time finetuning starts, it does pre-processing again
- Blacklist script overwrites evaluate metadata. Run blacklist first then evaluate!

## MPS (Apple Silicon) Specific Issues

### Known Limitations
- **Flash Attention 2**: Not supported on MPS. Use `sdpa` attention instead (already configured)
- **Mixed Precision**: Limited autocast support. Use bfloat16 or float32
- **Multi-GPU**: MPS doesn't support multi-GPU training (single GPU only)
- **Some Operations**: Certain PyTorch ops may fall back to CPU with PYTORCH_ENABLE_MPS_FALLBACK=1

### Common MPS Errors and Solutions

1. **"MPS backend out of memory"**
   - Solution: Reduce batch size or set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`
   - The unified memory architecture can cause swapping instead of OOM errors

2. **"Operation X not implemented for MPS"**
   - Solution: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (performance impact)
   - Report specific operations for future fixes

3. **Slower than expected performance**
   - Check if operations are falling back to CPU
   - Disable `PYTORCH_ENABLE_MPS_FALLBACK` after testing
   - Use appropriate batch sizes for your chip

4. **Numerical differences from CUDA**
   - MPS uses different floating-point optimizations
   - Increase tolerance in accuracy checks if needed
   - Results should be functionally equivalent

### Performance Considerations
- **Memory**: Apple Silicon uses unified memory - monitor total system RAM usage
- **Batch Sizes**: Start conservative (8-16) and increase gradually
- **Data Loading**: May need to adjust num_workers for optimal performance
