# Multimodal Soccer Highlight Detection

A deep learning system for detecting highlight moments in soccer videos using multimodal data (vision, audio, and text).

## Overview

This project combines three modalities to identify highlight events in soccer matches:
- **Vision**: Extracted video frames and spectrograms for visual features
- **Audio**: Mel-spectrograms from match audio commentary
- **Text**: Commentary text extracted from match broadcasts

## Dataset

The dataset consists of 4,552 clips from soccer matches, split into:
- **Train**: 3,111 clips
- **Validation**: 664 clips
- **Test**: 778 clips

Each clip contains:
- Video frames (8-second clips from matches)
- Audio waveforms and mel-spectrograms
- Textual commentary
- Binary highlight/non-highlight labels

## Requirements

- Python 3.8+
- PyTorch
- TorchAudio
- TorchVision
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

[Paper Link](tinyurl.com/multimodal-hd-paper)