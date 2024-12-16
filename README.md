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

<div style="display: flex; justify-content: center;">
  <table>
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
</div>



