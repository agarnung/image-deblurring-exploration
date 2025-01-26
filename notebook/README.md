# notebook

This document is a self-contained tutorial/introduction to the basics of image deblurring, through a _hands-on_ approach.   

## Index
* [Introduction](#Introduction)
    * .
* [Testing](#Testing)
    * [Non-blind deconvolution](#non-blind-deconvolution)
        * [Wiener Filter](#wiener)
        * [Lucy-Richardson](#Lucy-Richardson)
        * [Modified unsupervised Wiener](#uns-wiener)
        * [Lucy-Richardson with TV prior](#Lucy-Richardson-TV)
    * [Blind deconvolution](#bind-deconvolution)
    * [Semi-blind deconvolution](#semi-bind-deconvolution)
* [References](#References)
    * [C++](#C++)
    * [Python](#Python)
    * [MATLAB](#MATLAB)
    * [Links](#Links)
    * [Books](#Books)


## Introduction <a class="anchor" id="Introduction"></a>

Capturing an image using any type of analog or digital camera, and even through the human visual system, as it is represented in the real and physical world, is impossible due to processes like electronic noise, non-linear lens distortion caused by inherent imperfections, focusing issues, quantization, discretization, noise in the acquisition system, uncontrolled environmental factors, mechanical vibrations, acquisition time, movement of the captured objects, etc. Various degradation phenomena (Noise, scatter, glare, and blur) are discussed in https://temchromatinlab.wordpress.com/deconvolution/. All these sources of error have in common that they cause, in the output image, either the incident light on the sensor to scatter across neighboring pixels, or information from neighboring pixels to combine with each other, disturbing the actual information.

All these factors make images a (pure) approximation of reality, since the problem is ill-posed and information is lost. However, the PSNR is usually so high that "we don't care". Still, there are occasions (see next image) where it is so low that one can call it "degraded." Imagine photographing an infinitely distant light point using an infinitely perfect camera; the resulting image will be a single bright pixel. However, if we photograph a star with our camera, the result is far from being a single bright pixel; instead, we will see a region scattered in a more or less circular shape that decreases in brightness with the radius. The effect that a real image acquisition system has on a "perfect" light source is called the **Point Spread Function** (PSF). If the convolution of a perfect image with this PSF produces the real image we would acquire, then by performing the deconvolution of our degraded image with the PSF, we obtain the restored, "ideal" image.

Furthermore, after this effect, noise is added to the image due to processes like those mentioned earlier. Sometimes (e.g., space telescopes), it is a poor assumption that the PSF is the same across all pixels in our real image. A degraded image like this is often denoted **blurred**. So, in a basic model, the ```Imaging Process = Convolve(ideal image, PSF) + Noise```. All these factors make PSF estimation theoretically possible, but practically impossible.


### Disambiguation

- **Smoothing**: We try to suppress superfluous features and false discontinuities.
- **Enhancing**: We try to create discontinuities where they should appear.
- **Deblurring**: The process of removing (any kind of) blur from an image. Sometimes deblurring is included as an example of **enhancing**.
- **Deconvolution**: The process of applying the inverse operation of a convolution, often used in restoration and deblurring problems.
- **Sharpening**: The process of improving the perceived acuity and visual sharpness of an image, especially for viewing it on a screen or in print. Sharpening can be used to approximate deconvolution and often provides reasonable results.
- **Super-resolution**: The process of reconstructing missing detail in an image.
- **Upsampling**: The process of increasing the image resolution, with no relation to blur, sharpness, or detail – but aiming to at least not reduce it. Sadly, ML literature calls upsampling “single frame super-resolution”.

### Deconvolution

Deconvolution can be one of the most effective methods for performing deblurring. There are other simple image restoration processes via the "Imaging Process", but they might not be very robust in real cases.

Essentially, deconvolution restores high frequencies, but since the captured image contains noise that the "perfect" image does not, the direct deconvolution process can amplify the noise, which also contains high frequencies.

Due to the ill-posedness of the image restoration problem (in general), it is common to concatenate deconvolution and denoising stages to leverage the performance and benefits of each efficient method.

Some widely used approaches are:
1. Solve in the frequency domain (e.g., Inverse Filter [i.e., direct deconvolution])
2. Solve in the frequency domain and use regularization to minimize noise (e.g., Wiener Filter)
3. Iterative approaches (e.g., Richardson-Lucy)
4. Iterative approaches with regularization (e.g., Richardson-Lucy with Total Variation Regularization)

### Types of Deconvolution

Numerous methods can be used to improve the quality of a blurred image, i.e., **deblurring**, and objectively and subjectively assess the quality of the deblurring process. This process can involve several approaches:
1. **Blind deconvolution**: Trying to estimate the PSF "blindly" from (only) the degraded image.
2. **Non-blind deconvolution**: Assuming a certain predefined PSF based on heuristic assumptions or formal knowledge, and "hoping for the best" with deblurring algorithms.
3. **Semi-blind deconvolution**: BLABLABLA.

Moreover, one can categorize the types of deblurring based on how the PSF is assumed:
- **Linear Model**.
  - Time / Spatial Invariant Model.
  - Time / Spatial Variant Model.
- **Non-Linear Model**.
  - Time / Spatial Invariant Model.
  - Time / Spatial Variant Model.

We can consider many different kinds of blurring models, and their complexity depends on whether one assumes the image has noise, the PDF is space-dependent, etc. For example, characteristics like the kernel used to model the process (emulating hand shake, out-of-focus camera, etc.). For a complete discussion, see https://disp.ee.ntu.edu.tw/class/tutorial/Deblurring_tutorial.pdf.

One can use very simple kernels (e.g., Gaussian) in developing efficient deconvolution processes to "get by" in many practical cases.


## Testing

Now that we have some background, we are prepared to implement some algorithm to try improving the quality of my mobile phone's camera.

Here is my original image:

<p align="center">
  <img src="../assets/7.jpg" alt="Original image" title="Original image" style="display: inline-block; width: 300px" />,
</p>

We will test various methods to perform deblurring, mainly through deconvolution, and analyze the results.

### 1) Non-blind deconvolution <a class="anchor" id="non-blind-deconvolution"></a>

We assume that we (perfectly) know the (blur kernel of the) PSF.

For instance, we assume the Imaging Process process is modeled by the following kernel, which is a Gaussian kernel with $size = 33$ and $\sigma = 3$:

<p align="center">
  <img src="../assets/blur_kernel.png" alt="blur_kernel" title="blur_kernel" style="display: inline-block; width: 300px" />,
</p>

Assuming a known kernel we can test various non-blind deconvolution methods:

#### 1.1) Wiener Filter <a class="anchor" id="wiener"></a>
BREVE TEORÍA Y ECUACION WIENER

FOTO RESULTADO

Of course, the result is horrible. We do not know what PSF would have my camera. This is trial and error and the time complexity depends directly on the patience and expertise of the user.

#### 1.2) Modified unsupervised Wiener <a class="anchor" id="uns-wiener"></a>
This algorithm has a self-tuned regularization parameters based on data learning. Based on an iterative Gibbs sampler that draw alternatively samples of posterior conditional law of the image, the noise power and the image frequency power.

See [scki-kit doc](https://scikit-image.org/docs/stable/auto_examples/filters/plot_restoration.html) and [paper](https://hal.archives-ouvertes.fr/hal-00674508).

FOTO RESULTADO

#### 1.3) Lucy-Richardson <a class="anchor" id="Lucy-Richardson"></a>
Partimos de cierta conjetura de nuestra imagen ideal desconocida; aplicamos un esquema iterativo actualizando la estimación hasta su convergencia.

FOTO RESULTADO

#### 1.4) Lucy-Richardson with TV prior <a class="anchor" id="Lucy-Richardson-TV"></a>

Añadiendo un regularizador anisotrópico como el operador TV se puede incorporar disparidad de los gradientes a la solución requerida, lo que, junto con la maximización del MAP que busca fidelidad, puede ser beneficioso para recuperar una imagen más natural y no tan rodeada de altas frecuencias artificiales.

FOTO RESULTADO

### 2) Blind deconvolution <a class="anchor" id="blind-deconvolution"></a>

We intend to find the PSF directly from the degraded image.

#### 2.1) ...

UN MÉTODO TRADICIONAL...

#### 2.2) ...

UN MODELO DL DE GITHUB...

### 3) Semi-blind deconvolution <a class="anchor" id="semi-blind-deconvolution"></a>

~~We use an initially aproximated PSF and then refine the result.~~

¿¿QUITAR?? <= igual solo mejor mencioanr que existe en la introducción, como caso especial, y no poner ejemplos o nombrar solo teóricos

## References

### C++
* https://github.com/y3nr1ng/DeconvLR
* https://github.com/tianyishan/Blind_Deconvolution
* https://github.com/chrrrisw/RL_deconv
* https://github.com/DoubleBiao/fast_deblurring

### Python
* https://scikit-image.org/docs/0.24.x/auto_examples/filters/plot_deconvolution.html
* https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/18a_deconvolution/introduction_deconvolution.html
* https://github.com/sylvainprigent/sdeconv/tree/main
* https://github.com/sovit-123/image-deblurring-using-deep-learning
* https://github.com/nkanven/gan-deblurring
* https://github.com/dongjxjx/dwdn
* https://github.com/axium/Blind-Image-Deconvolution-using-Deep-Generative-Priors
* https://github.com/Tmodrzyk/richardson-lucy-python

### MATLAB
* https://es.mathworks.com/help/images/deblurring-images-using-a-wiener-filter.html
* https://es.mathworks.com/help/images/deblurring-images-using-a-regularized-filter.html
* https://es.mathworks.com/help/images/deblurring-images-using-the-lucy-richardson-algorithm.html
* https://es.mathworks.com/help/images/deblurring-images-using-the-blind-deconvolution-algorithm.html
* https://es.mathworks.com/help/images/deblur-with-the-blind-deconvolution-algorithm.html

### Links
* https://biapol.github.io/PoL-BioImage-Analysis-TS-GPU-Accelerated-Image-Analysis/30_Deconvolution/0_intro_to_decon.html
* https://disp.ee.ntu.edu.tw/class/tutorial/Deblurring_tutorial.pdf
* https://bartwronski.com/2022/05/26/removing-blur-from-images-deconvolution-and-using-optimized-simple-filters/
* https://temchromatinlab.wordpress.com/deconvolution/
* https://en.wikipedia.org/wiki/Deconvolution

### Books
* Algorithms for Image Processing and Computer Vision 2nd ed. J.R. Parker (p. 251)
* Mathematical Problems in Image Processing (p. 128)
