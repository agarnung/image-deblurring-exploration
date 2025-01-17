# notebook

This document is a self-contained tutorial/introduction to the basics of image deblurring, through a _hands-on_ approach.   

## Introduction

La captura de una imagen mediante cualquier tipo de cámara analógica o digital, e incluso bajo el sistema visual humano, tal como está representada en el mundo real y físico, es imposible, por culpa de procesos como el ruido electrónico, la distorsión no lineal de las lentes por inherentes imperfecciones, problemas de enfoque, la cuantización y cuantificacíon, el ruido contenido al sistema de adquisición y factores no controlados del entorno o ambiente, las vibraciones mecánicas, el tiempo de adquisición, el movimiento de los objetos capturados, etc. Varios tipos de fenómenos de degradación (Noise, scatter, glare y blur) se reseñan en https://temchromatinlab.wordpress.com/deconvolution/. Todas estas fuentes de error tienen en común que causan, en la imagen de salida, o bien que la luz incidente en el sensor se disperse por los píxeles vecinos, o bien que información de píxeles vecinos se combinen entre sí, perturbando la información real.

Todos estos factores hacen de las imagenes una (pura) aproximación de la realidad, pues el problema es ill-posed y hay pérdida de información. Pero el PSNR usualmetne es tan alto que "no nos importa". Sin embargo, hay ocasiones (cer siguiente imagne) en las que, es tan bajo, que uno la puede calificar como "degradada". Imaginémonos que fotografiamos un punto luminoso _infinitamente_ lejano empleando una cámara _infinitamente perfecta_; el resultado de la imagen será un único píxel brillante. Ahora bien, si fotografiamos con nuestra cámara una estrella, el resultado está lejos de ser un único píxel brillante,por el contrario, veremos una región dispersa más o menos circular que decrece en brillo con el radio. El efecto que un sistema de adquisición de imágenes real provoca en una fuente de luz "perfecta" es lo que se llama _point spread function_ (PSF). Si la convolución de una imagen perfecto con esta PSF provoca la imagen real que adquiriríamos, entonces haciendo la deconvolucón de nuestra imagen degrada con la PSF conseguimos la imagen restaurada, "ideal". 

Más aún, tras este efecto, existe ruido que se agrega a la imagen debido a procesos como los mencionados al principio. Incluso a veces (e.g. telescopios espaciales) es una mala asunción que la PSF sea la misma en todos los píxeles de nuestra imagen real. A degraded image like this is often denoted **blurred**. So, in a basic model, the ```Imaging Process = Convolve(ideal image, PSF) + Noise```. Todos estos factores hcaen que la estimación de la PSF sea posible en la teoría, pero prácticamente imposible en práctica.

## Dissambiguation

Smoothing: Tratamos de suprimir caracteŕisticas superfluas y discontibuidades falsas

Enhancing: tratamos de crear discontibuidades en sitios donde deberían aparecer

Deblurring: process of removing (any kind of) blur from an image. A veces se incluye el deblurring como un ejemplo de **enhancing**.

Deconvolution: Process of applying the inverse operation of a convolution, often applied to restorarion and deblurring problems.

Sharpening: process of improving the perceived acuity and visual sharpness of an image, especially for viewing it on a screen or print. Sharpening can be used to approximate deconvolution and often provides reasonable results

Super-resolution: process of reconstructing missing detail in an image.

Upsampling: process of increasing the image resolution, with no relation to blur, sharpness, or detail – but aiming to at least not reduce it. Sadly ML literature calls upsampling “single frame super-resolution”.

## Deconvolution

Deconvolution puede ser uno de los ḿetodos más efectivos para realizar el deblurring. Existen otros procesos simples de restauración de imágenes degradas medinte el "Imaging Process", pero pueden no ser muy robustos ante casos reales. 

En esencia, deconvolution restores high frequencies, pero como la imagen capturada contiene ruido que la "perfecta" no, el proceso directo de deconvolución puede amplifica el ruido, el cual también contiene altas frecuencias.

Debido a la ill-posedness del problema de la restauración (en general) de iḿagenes, suele ser frecuente concatenar etapas de deconvolucíon y denoising para aprovechar el desempeño y los beneficios de métodos eficientes de cada uno.

Some widely used approaches are:
1. Solve in frequency domain (e.g. Inverse Filter [i.e. direct deconvolutoin])
2. Solve in frequency domain and use regularization to minimize noise (e.g. Wiener Filter)
3. Iterative approaches (e.g. Richardson Lucy)
4. Iterative approaches with regularization (e.g. Richardson Lucy with Total Variation Regularization)

## Types of deconvolution

Numerosos métodos se pueden llevar a cabo para mejorar la calidad de una imagen blurred, i.e. **deblurring**, andd objectivlyl and subjectivily assess the quality of the deblurring process. Este proceso puede conllevar varios enfoques:
1) blind deconvolution: tratar de estimar la PSF "a ciegas" a partir (únicamente) de la imagen degradada
2) non-blind deconvolution: asumir cierta PSF predefinida, basada en las suposiciones heurísticas o conocimiento formal que sea y "probar suerte" con los algoritmos de deblurring. 
3) semi-blind deconvolution: BLABLABLA

Moreover, uno puede categorizar los tipos de deblurring según cómo se asume que es la PSF
*Linear Model.
**Time / Spatial Invariant Model.
**Time / Spatial Variant Model.
*Non Linear Model.
**Time / Spatial Invariant Model.
**Time / Spatial Variant Model.

We can consider a lot of different kind of blurring models, and its complexity depends of wheter one assumes thee image have noise, the PDF is space-dependent, etc. E.g. characteristics like the kernel used to model the process (emulating the shake of a hand, the out-of-focus of a camera, etc.). For a complete discussion, see https://disp.ee.ntu.edu.tw/class/tutorial/Deblurring_tutorial.pdf.

Uno puede usar kernels muy sencillos (e.g. Gaussianos) en el desarrollo de procesos de deconvolutión eficiones para "salir del paso" en muchos casos prácticos.

## References

### C++
* https://github.com/y3nr1ng/DeconvLR
- https://github.com/tianyishan/Blind_Deconvolution
- https://github.com/chrrrisw/RL_deconv
- https://github.com/DoubleBiao/fast_deblurring

### Python
* https://scikit-image.org/docs/0.24.x/auto_examples/filters/plot_deconvolution.html
* https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/18a_deconvolution/introduction_deconvolution.html
* https://github.com/sylvainprigent/sdeconv/tree/main
* https://github.com/sovit-123/image-deblurring-using-deep-learning
* https://github.com/nkanven/gan-deblurring
* https://github.com/dongjxjx/dwdn
* https://github.com/axium/Blind-Image-Deconvolution-using-Deep-Generative-Priors
- https://github.com/Tmodrzyk/richardson-lucy-python

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