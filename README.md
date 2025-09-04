# DeepLung

Here, we present DeepLung, a deep learning framework that extracts three complementary spectral representations of lung sounds short-time Fourier transform, wavelet transform, and mel spectrogram-via a tri-branch VGG16 architecture, which are then fused into a unified latent space for classification.

Furthermore, we introduce an asymmetric multimodal knowledge distillation strategy, transferring rich representations from a multimodal teacher (CT images, audio, vital signs, and clinical reports) to the audio-based student model (DeepLung)


