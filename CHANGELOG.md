# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Full Apple Silicon (M1/M2/M3) support via Metal Performance Shaders (MPS)
- Device-agnostic code that automatically selects MPS/CUDA/CPU
- `utils/device.py` module for unified device management
- MPS-specific memory management and optimization
- Comprehensive README.md with setup instructions
- `setup_mps.sh` automated setup script for Apple Silicon
- `test_mps.py` for verifying MPS functionality
- MPS-specific documentation in KNOWN_ISSUES.md
- Requirements.txt for easy dependency installation

### Changed
- Replaced Flash Attention 2 with SDPA for MPS compatibility
- Updated all CUDA-specific calls to be device-agnostic
- Reduced default batch sizes for Apple Silicon optimization
- Updated system_check.py to detect and report MPS capabilities
- Improved error handling for device-specific operations


### Fixed
- Hard-coded CUDA device references in evaluation scripts
- Memory management calls that were CUDA-specific
- Device placement issues in distil-whisper implementation
- README_MANAGE.md duplicate content issue

### Deprecated
- Flash Attention 2 usage (replaced with SDPA)


### Known Issues
- distil-whisper implementation needs updates
- Cache not utilized properly during finetuning preprocessing
- Blacklist script overwrites evaluate metadata
