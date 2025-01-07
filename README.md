# Literature_Review
## 1. Seeing With Sound:Long-range Acoustic Beamforming for Multimodal Scene Understanding

### Summary:

By combining microphone beamforming information with camera images, target localization under specific conditions (e.g., vehicles) is achieved. Additionally, the author proposes a neural network-based method to optimize the beamforming resolution of the microphone array, enabling higher recognition resolution for more
accurate multi-sensor fusion-based target localization.

### Take Away Information:

1. Sound can help cameras overcome limitations in low-light and partially occluded environments.

2. A neural network can enhance the beamforming resolution of a microphone array, effectively simulating a high-aperture system.

## 2. Direction of Arrival Estimation for Multiple Sound Sources Using Convolutional Recurrent Neural Network

### Summary:

The authors designed a network for sound source localization (direction), combining CNN and RNN methods to predict the direction of the sound source with a better understanding of temporal information. Based on the results shown, the authors successfully achieved accurate prediction for single-source scenarios, but the prediction performance was average when multiple sound sources were present.

### Take Away Information:

1. A new kind of neural network that could be used for sound source localization, this structure could be used in siren detection.

2. Dataset for sound localization training.

3. Combining CNN and RNN structure could extract the direction information from the spectrogram.

## 3. A LEARNING-BASED APPROACH TO DIRECTION OF ARRIVAL ESTIMATION IN NOISY AND REVERBERANT ENVIRONMENTS

### Summary:

The author designed an learning based approach directly learns the nonlinear relationship between the received signals and the DOA from a large amount of trainiing data synthesised for many noisy and reverberant environments. To investigate how to use a big set of ttraining data to achieve reliably DOA estimation in adverse environments. The idea is to learn a mapping from featuress extracted from the microphone array inputs to the DOA using a big set of training data.

### Take Away Information:

1. The DOA estimation is formulated as a 360-class pattern classification problem (0 degrees to 359 degrees).

2. Use GCC feature for DOA estimation: TDOA of microphone pairs are estimated from their GCC vectors but TDOA estimation is often enreliable in low SNR and high reverberation conditions. Also, compared with the raw data from the microphone which only shows the signal from one microphone, GCC always give an information about the relationship between two microphone. 

3. A simple neural network structure with only one hidden layer is used 

4. To improve the robustness of the estimation in noisy environment, we proposed to use a weighted sum of the GCC patterns from all the frames of the test.

5. The performance is tested in small, medium and big  meeting rooms 

## 4. Deep Neural Networks for Multiple Speaker Detection and Localization

### Summary: 

Propose to use neural networks for simultaneous detection and localization of multiple sound sources in human-robot interaction. In contrast to conventional signal processing techniques, neural network-based sound source localization methods require fewer strong assumptions about the environment. Three NN architectures for multiple SSL(Sound Source Localization) are propoesed and NN adopt a likelihood-based output encoding that can handle an arbitrary number of sources.

### Take Away Information:

1. Use generalizaed cross-correlation with phase transform for the input instead a single estimation of the TDOA

2. The speech signals in the time-frequency domain is sparsity and the rrandomly distributed noise which might be stronger than the signal in some TF bins. So the author proposed to use GCC-PHAT on mel-scale filter bank. 

3. The output is encoded into a vector of 360 values, each of which is associated with an individual azimuth direction. The possibility could be zero when there is no sound source or contains N peaks when there are N sources.

4. Three neural network is proposed: 

    (a) Multilayer perceptron with GCC-PHAT

    (b) Convolutional neural network with GCCFB

    (c) Convolutional neural network with GCCFB: first extract the direction information from the delay message within similar frequency region, then extract the possibility of DoA from all-frequency information (The model is shown in the network structure part)

## 5. SOUND EVENT LOCALIZATION AND DETECTION USING ACTIVITY-COUPLED CARTESIAN DOA VECTOR AND RD3NET

### Summary:

A two stage system is proposed to solve the sound event localization and sound event detection task simultaneously using an activity-coupled Cartesian DOA vector representation. During the preprocessing, two data augmentation techniques are applied to input signals prior to the feature extraction while one data augmentation techinique exploiting multichannel information in the feature domain is performed after the feature extraction.

### Take Away Information:

1. Multichannel amplitude spectrograms and inter-channel phase differences are used as frame-wise features

2. Data Augmentation: 

    (a) EMDA: Mix the sound with random amplitudes to simulate a noisy environment

    (b) Rotate: Rotate the sound source (they used a circular microphone layout in the research)

    (c) Multichannel SpecAugment: Extend a channel dimension to the feature map

3. The RD3Net model is adopt from D3Net architecture which is previously used for sound separation, the changes of the model would be compared in the nerual network part. 

4. A binary cross entropy is used for the event detection head and masked MSE is used for DoA detection had.

## 6. D3Net*

### Summary:

The paper ultilize a CNN based nerual network to solve the music sound separation problem. In this work, the author combine the advantages of DenseNet and dilated convolution to propose a novel network architecture.

### Take Away Information: 

1. Propose a nested architecture of dilated dense block to effectively repeat dilation factors multiple times with dense connections that ensure the sufficient depth required for modeling resolution

<div style="text-align: center;">
    <img src="images/multi_dilated_convolution.png" alt="Example Image" width="400" height="auto">
</div>

2. Indicate that applying a dilated convolution to skip connections from early layers without handling the aliasing problem makes it difﬁcult to extract information.

## 7. Environmental Sound Classification with CNN

## 8. Deep CNN for Environmental Sound Classification and Data Augmentation

### Summary:

Deep convolutional neural networks (CNNs) are well-suited for environmental sound classification due to their ability to learn discriminative spectro-temporal features. However, the scarcity of labeled data limits their application. This study proposes a deep CNN architecture and audio data augmentation techniques to address this issue. Experiments show that the combination of data augmentation and the proposed CNN achieves state-of-the-art performance, outperforming both the CNN without augmentation and shallow dictionary learning models. Additionally, different augmentation methods affect the classification accuracy of each class differently, suggesting that class-conditional data augmentation could further enhance performance.

### Take Away Information:

1. A deep CNN arhitecture with three convolutional layers interleaved with two pooling operations, followed by two fully connected layers is contrusted for sound classification.

2. Four different kinds of audio data augmentation methods are used:

    (1) Time stretching: slow down or speed up the audio sample 

    (2) Pitch shifting (Greatest Positive Impact): raise or lower the pitch of the audio sample 

    (3) Dynamic range compression: compress the dynamic range of the sample using four parameterizations (What is dynamic range: the sound from a boardingcast and a high-quality speakers is different)

    (4) Background noise: mix the sample with another recording containing background sounds from different types of acoustic scenes

3. The result shows that CNN with augmentation method perform much better than the traditional methods, also it indicates that the superiorperformance of the proposed SB-CNN is not only due to theaugmented training set, but rather thanks to the combinationof an augmented training set with the increased capacity andrepresentational power of the deep learning model.

##  9. Long-range Acoustic Beamforming for Multimodal Scene Understanding

### Summary:

The author introduce long-range acoustic beamforming of sound produced by road users in-the-wild as a complementary sensing modality to traditional electromagnetic radiation-based sensors. With the help of a neural aperture expansion method for beamforming, it shows its effectiveness for multimodal automotive object detection when coupled with RGB images in challenging automotive scenarios.

### Take Away Information:

1. Multimodal signals, combining beamforming and RGB data, are utilized for object detection to enhance detection accuracy.

2. To improve beamforming performance, the author trained a neural network to synthetically expand the microphone array's aperture, enabling higher-fidelity beamforming maps with reduced PSF distortion.

3. Acoustic sensors complement photon-based sensors by detecting incoming objects that are not directly visible to RGB or lidar systems.

## 10. Hearing what you cannot see: Acoustic Vehicle Detection Around Corners

### Summary:

This work proposes to use passive acoustic percep-tion as an additional sensing modality for intelligent vehicles. We demonstrate that approaching vehicles behind blind corners can be detected by sound before such vehicles enter in line-of-sight. We have equipped a research vehicle with a roof-mounted microphone array, and show on data collected with this sensor setup that wall reﬂections provide information on the presence and direction of occluded approaching vehicles.

### Take Away Information:

1. Sound -> STFT -> Bandpass and Segmentation -> Directional of Arrival -> Classifier

2. An approaching vehicle could be detected with the same accuracy as visual baseline already more than one second ahead. 

3. It has difficulties to perform reliably in unseen test environments. 


# Network Structure

## 1. Seld Net (DoA (Multi) Sound Source Estimation)

<div style="text-align: center;">
    <img src="images/1734362522334.png" alt="Example Image">
</div>

### Input 

Spectrogram with 40 ms Hamming window and 50% overlap, each containing 1024 magnitude and phase value: L\*1024\*2C (L=100 -> 2s)

### Spatial Pseudo-spectrum Phase

<table align="center">
  <tr>
    <td>Layer</td>
    <td>Dimension</td>
  </tr>
  <tr>
    <td>Raw Data</td>
    <td>100*1024*2C</td>
  </tr>
  <tr>
    <td>4 Convolution Layers</td>
    <td>100*2*64</td>
  </tr>
  <tr>
    <td>Concatenate</td>
    <td>100*128</td>
  </tr>
  <tr>
    <td>2 Bidirectional GRU Layers</td>
    <td>100*128</td>
  </tr>
  <tr>
    <td>1 Linear Layer</td>
    <td>100*614</td>
  </tr>
</table>


### Direction of Arrival Phase

<table align="center">
  <tr>
    <td>Layer</td>
    <td>Dimension</td>
  </tr>
  <tr>
    <td>SPS</td>
    <td>100*614</td>
  </tr>
  <tr>
    <td>Reshape</td>
    <td>100*614*1</td>
  </tr>
  <tr>
    <td>2 Convolution Layers</td>
    <td>100*307*16</td>
  </tr>
  <tr>
    <td>Concatenate and 1 Linear Layer</td>
    <td>100*32</td>
  </tr>
  <tr>
    <td>2 Bidirectional GRU Layers</td>
    <td>100*32</td>
  </tr>
  <tr>
    <td>1 Linear Layer</td>
    <td>100*432</td>
  </tr>
</table>

## 2. MLP based DoA estimator

<div style="text-align: center;">
    <img src="images/MLP_DoA_estimator.png" alt="Example Image" width="400" height="auto">
</div>

### Input 

A 28×21 GCC graph, where 28 represents 28 microphone pairs and 21 indicates the 21 time delay samples selected for calculation for each pair of microphones. In addition, some weights are added to the input to make the model robust to the noise.

<div style="text-align: center;">
    <img src="images/MLP_DoA_Estimator_Noise_Filter.png" alt="Example Image" width="300" height="auto">
</div>

### Output

The posterior probability of a DOA angle is p(θt|ot) for θt = 0, ..., 359 degrees.

## 3. Convolutional neural network with GCCFB

<div style="text-align: center;">
    <img src="images/CNN_GCCFB.png" alt="Example Image" width="200" height="auto">
</div>

### Preprocessing

**GCCFB**: it works by decomposing the signal into multiple mel-scale frequency bands (which are well-suited to the spectral characteristics of speech), and then calculating GCC-PHAT for each frequency band separately. This helps to preserve delay information for each frequency band.

### Input 

51 * 40 * 6 GCCFB graph

### Output

The same output as the MLP based DoA neural network

## 4. Two-stage neural network with GCCFB

<div style="text-align: center;">
    <img src="images/TSNN-GCCFB.png" alt="Example Image" width="500" height="auto">
</div>

### Subnet I

Time delay feature space to  degree feature: extract latent DoA features in each fiter bank  on individual frequency regions.

### Subnet II

The second stage aggregates information across all frequencies in a neighbor DoA area and outputs the likelihood of a sound being in each DoA area.

## 5. RD3Net

<div style="text-align: center;">
    <img src="images/R3DNet.png" alt="Example Image" width="250" height="auto">
</div>

### Input

Multichannel amplitude spectrograms and inter-channel phase differencees are used as frame-wised features. The STFT was applied with a configuration of 20 ms frame length and 10 ms frame hop.

### Output

The name of the sound event and source angle.

## 6. SB-CNN (Sound Classification Neural Network)

<table align="center">
  <tr>
    <td>Layer</td>
    <td>Size</td>
  </tr>
  <tr>
    <td>Conv1</td>
    <td>24*5*5</td>
  </tr>
  <tr>
    <td>Conv2</td>
    <td>48*5*5</td>
  </tr>
  <tr>
    <td>Conv3</td>
    <td>48*5*5</td>
  </tr>
  <tr>
    <td>Linear Layer</td>
    <td>64</td>
  </tr>
  <tr>
    <td>Linear Layer</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Softmax</td>
    <td>10</td>
  </tr>
</table>

### Input 

Time-frequency patches taken from the log-scaled mel-spectrogram representation of the audio signal. The spectrogram is extracted with 128 components covering the audible frequency range (0-22050 Hz), using a window size of 23 ms.

### Output

Possibility of the related class 

# Other Knowledge

## GCC: Generalized Cross Correlation

GCC is used to estimate the time delay between two microphone, where \( 𝜏 \) is the delay in the discrete domain, \( * \) denotes the complex conjugation, and \( R \) denotes the real part of a complex number. The peak in GCC-PHAT is used to estimate the TDOA.


<div style="text-align: center;">
    <img src="images/gcc_formulation.png" alt="Example Image" width="300" height="auto">
</div>

For example, if the correlation coefficient reaches its highest value when 𝜏 = 0 it indicates that there is no time delay between the two signals.




