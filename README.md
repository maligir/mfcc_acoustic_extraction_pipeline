# mfcc_acoustic_extraction_pipeline

This is an implementation of the Mel-Frequency Cepstral Coefficient (MFCC) acoustic feature extraction pipeline.

## File Descriptions:

1. signal.wav : A speech recording to test the pipeline.
2. mel_filters.npy : A set of pre-computed Mel filter banks.

## MFCC Pipeline Implementation Steps:

1. Load an audio waveform.
2. DC subtraction (i.e. remove the mean of the signal)
3. Pre-emphasis filtering
4. Transforming the audio into a sequence of frames 5. For each frame:

(a) Multiplying the frame with a window function (b) Computing the Fourier transform

(c) Computing the magnitude spectrum from the Fourier transform (d) Computing the power spectrum from the magnitude spectrum (e) Binning the power spectrum with Mel filterbanks

(f) Taking the logarithm of the Mel-filterbank power spectrum

(g) Computing Discrete Cosine Transform (DCT) of the log-Mel-power spectrum (h) Truncating the DCT to keep only the first 𝐶 elements

For an input audio waveform that occupies 𝐹 frames, the resulting MFCC representation of the audio will be a 𝐹 by 𝐶 matrix.

## Loading the File

Using scipy.io.wavfile.read(), we load the signal.wav file from disk. 

## Mean Subtraction

Let 𝑥1[𝑛] be the raw samples of the audio file. The first step is to perform mean subtraction:

𝑥2[𝑛] = 𝑥1[𝑛] − 𝑥0

where 𝑥0 is the average value of all the samples in the waveform.

## Pre-emphasis

The periodic excitation from the glottis in voiced speech is not an ideal delta train, but is “smoothed.” To be more concrete, the harmonics of the excitation fall off in magnitude as a function of frequency at a rate of 1/𝑓 2. The radiation characteristics at the lips partially counteract this by a factor of 𝑓 , but in order to flatten the spectrum completely we need to apply another filter whose magnitude response is proportional to 𝑓 . This is achieved with a first order finite impulse response (FIR) filter:

𝑥3[𝑛] = 𝑥2[𝑛] − 𝑏𝑥2[𝑛 − 1] 

where 𝑏 = 0.97 is conventionally used.

## Computing Frames

Thenextstepistocomputeaseriesof𝑁𝑓 frames,whereeachframe captures a small, fixed-length piece of the signal. The two parameters that control the nature of the frames we will extract are the frame 𝑙𝑒𝑛𝑔𝑡h (sometimes also called the width) and frame 𝑠h𝑖𝑓 𝑡 (sometimes also called the “hop”). Assuming a window length 𝐿 and shift 𝑆 (both specified in samples), The 𝑘𝑡h frame will be:

𝑥𝑘[𝑛] = 𝑥3[𝑘𝑆 +𝑛],𝑛 = 0,1,2,...,𝐿−1

Unless (𝑙𝑒𝑛𝑔𝑡h(𝑥3) − 𝐿) is exactly divisible by 𝑆, the very last frame won’t have enough samples to fill an entire window length 𝐿. In this case, we can just pad the final window with zeros up to length 𝐿.

## Applying the Window Function

The next step is to element-wise multiply each frame 𝑥𝑘 [𝑛] with a window function 𝑤[𝑛]:

𝑤 𝑥𝑘 [𝑛]=𝑥𝑘[𝑛]⋅𝑤[𝑛]

The Hamming window is most commonly used in ASR, and can be created simply by making an appropriate call to the scipy.signal.hamming() function (scipy.signal also has factory functions for a large number of other window types).

## Computing the Fourier Transform

Next, we will compute the Fourier transform of each windowed frame. The “direct” implementation of the DFT is:

𝑁 𝑋[𝑚] = ∑𝑥[𝑛]𝑒 𝑛=0 𝑚 −2𝜋𝑗 𝑛 𝑁
