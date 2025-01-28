#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "deconv_hyper_lap.h"

void muestraImagenOpenCV(const cv::Mat img, std::string title, bool destroyAfter = true)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::resizeWindow(title, 800, 600);
    cv::imshow(title, img);
    cv::waitKey(0);

    if (destroyAfter)
        cv::destroyWindow(title);
}

void MyTimeOutput(const std::string& str, const std::chrono::high_resolution_clock::time_point& start_time, const std::chrono::high_resolution_clock::time_point& end_time)
{
    std::cout << str << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0 << " ms" << std::endl;
    return;
}

cv::Mat deconvolutionByTV(const cv::Mat& inputImage, const cv::Mat& kernel, int iterations = 100, double lambda = 0.1, double epsilon = 0.004)
{
    cv::Mat fTV;
    inputImage.copyTo(fTV);

    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat gradientX = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat gradientY = cv::Mat::zeros(height, width, CV_64F);

    int padSize = kernel.rows / 2;
    cv::Mat kernelPadded;
    cv::copyMakeBorder(kernel, kernelPadded, padSize, padSize, padSize, padSize, cv::BORDER_CONSTANT, cv::Scalar(0));

    for (int niter = 0; niter < iterations; niter++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double curValue = fTV.at<double>(y, x);
                if (y < height - 1)
                {
                    gradientY.at<double>(y, x) = fTV.at<double>(y + 1, x) - curValue; // Gradiente en Y
                }
                if (x < width - 1)
                {
                    gradientX.at<double>(y, x) = fTV.at<double>(y, x + 1) - curValue; // Gradiente en X
                }
            }
        }

        cv::Mat normGrad;
        cv::magnitude(gradientX, gradientY, normGrad);
        normGrad += epsilon;

        gradientX = gradientX / normGrad;
        gradientY = gradientY / normGrad;

        cv::Mat divergence = cv::Mat::zeros(height, width, CV_64F);
        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                divergence.at<double>(y, x) = gradientX.at<double>(y, x) - gradientX.at<double>(y - 1, x) +
                                              gradientY.at<double>(y, x) - gradientY.at<double>(y, x - 1);
            }
        }

        cv::Mat previous_fTV = fTV.clone();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double kValue = (1.0 / (1.0 + lambda * normGrad.at<double>(y, x)));

                double fidelityTerm = fTV.at<double>(y, x) - inputImage.at<double>(y, x);

                fTV.at<double>(y, x) -= (0.05 * kValue * (divergence.at<double>(y, x) + lambda * fidelityTerm));

                fTV.at<double>(y, x) = std::max(0.0, std::min(1.0, fTV.at<double>(y, x)));
            }
        }

        // cv::imshow("fTV", fTV);
        // cv::waitKey(5);
    }

    return fTV;
}

cv::Mat fftshift2D(const cv::Mat& originalMat)
{
    cv::Mat mat = originalMat.clone();

    // Recortar el espectro si tiene un número impar de filas o columnas
    cv::Rect rect(0, 0, mat.cols & -2, mat.rows & -2);
    mat = mat(rect);

    // Reorganizar los cuadrantes de la imagen de Fourier para que el origen esté en el centro de la imagen
    int cx = mat.cols / 2;
    int cy = mat.rows / 2;
    cv::Mat tmp;

    cv::Rect rect0(0, 0, cx, cy);
    cv::Rect rect1(cx, 0, cx, cy);
    cv::Rect rect2(0, cy, cx, cy);
    cv::Rect rect3(cx, cy, cx, cy);

    cv::Mat q0 = mat(rect0);
    cv::Mat q1 = mat(rect1);
    cv::Mat q2 = mat(rect2);
    cv::Mat q3 = mat(rect3);

    // Intercambiar los cuadrantes 1 y 4
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // Intercambiar los cuadrantes 2 y 3
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    return mat;
}

cv::Mat calculateConjugate(const cv::Mat& FH)
{
    cv::Mat planes[2];
    cv::split(FH, planes);

    cv::Mat conj_imaginary;
    cv::multiply(planes[1], -1, conj_imaginary);

    cv::Mat conjugate;
    std::vector<cv::Mat> channels;
    channels.push_back(planes[0]);
    channels.push_back(conj_imaginary);
    cv::merge(channels, conjugate);

    return conjugate;
}

void replaceNaNAndInfWithValues(cv::Mat& mat, double nanValue = 1, double infValue = 1, double negInfValue = 1,
                                double largeValue = 1, double smallValue = 1, bool enableLogs = false)
{
    const double THRESHOLD = 1e308;  // Umbral cercano al límite de double

    // Si la matriz tiene más de un canal, la dividimos y llamamos recursivamente a cada uno
    if (mat.channels() > 1)
    {
        std::vector<cv::Mat> channels;
        cv::split(mat, channels);  // Dividimos la matriz en sus canales individuales

        // Llamamos a la función recursivamente en cada canal
        for (auto& channel : channels)
        {
            replaceNaNAndInfWithValues(channel, nanValue, infValue, negInfValue, largeValue, smallValue, enableLogs);  // Llamada recursiva para cada canal
        }

        // Recomponemos la matriz con los canales procesados
        cv::merge(channels, mat);  // Unimos los canales nuevamente en una sola matriz
    }
    else
    {
        // Si la matriz tiene solo un canal, procesamos los elementos de la matriz
        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                double value = mat.at<double>(i, j);  // Obtenemos el valor del pixel en un canal

                // Check for NaN (Not a Number)
                if (std::isnan(value))
                {
                    if (enableLogs)
                        std::cout << "Entered NaN condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = nanValue;  // Reemplazar NaN por el valor definido
                }
                // Check for positive infinity (+Inf)
                else if (std::isinf(value) && value > 0)
                {
                    if (enableLogs)
                        std::cout << "Entered positive infinity condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = infValue;  // Reemplazar +Inf por el valor definido
                }
                // Check for negative infinity (-Inf)
                else if (std::isinf(value) && value < 0)
                {
                    if (enableLogs)
                        std::cout << "Entered negative infinity condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = negInfValue;  // Reemplazar -Inf por el valor definido
                }
                // Check for excessively large or small values
                else if (value > THRESHOLD)  // Cualquier valor por encima de un umbral muy alto se considera infinito.
                {
                    if (enableLogs)
                        std::cout << "Entered excessively large value condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = largeValue;  // Reemplazar valores grandes por el valor definido
                }
                else if (value < -THRESHOLD)
                {
                    if (enableLogs)
                        std::cout << "Entered excessively small value condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = smallValue;  // Reemplazar valores pequeños por el valor definido
                }
                // Check for values close to infinity using nextafter (for values approaching Inf)
                else if (std::nextafter(value, std::numeric_limits<double>::infinity()) == std::numeric_limits<double>::infinity())
                {
                    if (enableLogs)
                        std::cout << "Entered value close to positive infinity condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = infValue;  // Reemplazar valores cercanos a +Inf
                }
                else if (std::nextafter(value, -std::numeric_limits<double>::infinity()) == -std::numeric_limits<double>::infinity())
                {
                    if (enableLogs)
                        std::cout << "Entered value close to negative infinity condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = negInfValue;  // Reemplazar valores cercanos a -Inf
                }
                // Check for non-finite values (other than NaN and Inf)
                else if (!std::isfinite(value))
                {
                    if (enableLogs)
                        std::cout << "Entered non-finite value condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = (value > 0) ? infValue : smallValue;  // Reemplazar valores no finitos según el signo
                }
                // Check for very large positive values (greater than 10,000)
                else if (value > 10000)
                {
                    if (enableLogs)
                        std::cout << "Entered value greater than 10000 condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = largeValue;  // Reemplazar valores grandes por el valor definido
                }
                // Check for very large negative values (less than -10,000)
                else if (value < -10000)
                {
                    if (enableLogs)
                        std::cout << "Entered value less than -10000 condition at (" << i << ", " << j << ")" << std::endl;
                    mat.at<double>(i, j) = smallValue;  // Reemplazar valores pequeños por el valor definido
                }
            }
        }
    }
}

/*
 * cvWiener2 -- A Wiener 2D Filter implementation for OpenCV
 *  Author: Ray Juang  / rayver {_at_} hkn {/_dot_/} berkeley (_dot_) edu
 *  Date: 12.1.2006
 *
 * Modified 1.5.2007 (bug fix --
 *   Forgot to subtract off local mean from local variance estimate.
 *   (Credits to Kamal Ranaweera for the find)
 *
 * Modified 1.21.2007 (bug fix --
 *   OpenCV's documentation claims that the default anchor for cvFilter2D is center of kernel.
 *   This seems to be a lie -- the center has to be explicitly stated
 */
//
// cvWiener2  - Applies Wiener filtering on a 2D array of data
//   Args:
//      srcArr     -  source array to filter
//      dstArr     -  destination array to write filtered result to
//      szWindowX  -  [OPTIONAL] length of window in x dimension (default: 3)
//      szWindowY  -  [OPTIONAL] length of window in y dimension (default: 3)
//
void cvWiener2(const cv::Mat& srcMat, cv::Mat& dstMat, int szWindowX = 3, int szWindowY = 3)
{
    CV_Assert(!srcMat.empty());

    int nRows = szWindowY;
    int nCols = szWindowX;

    cv::Mat kernel = cv::Mat::ones(nRows, nCols, CV_32F) / static_cast<double>(nRows * nCols);

    cv::Mat tmpMat1, tmpMat2, tmpMat3, tmpMat4;

    // Local mean of input
    cv::filter2D(srcMat, tmpMat1, CV_64F, kernel);

    // Local variance of input
    cv::multiply(srcMat, srcMat, tmpMat2);  // in^2
    cv::filter2D(tmpMat2, tmpMat3, CV_64F, kernel);

    // Subtract local_mean^2 from local variance
    cv::multiply(tmpMat1, tmpMat1, tmpMat4); // localMean^2
    cv::subtract(tmpMat3, tmpMat4, tmpMat3); // localVariance

    // Estimate noise power
    double noisePower = cv::mean(tmpMat3)[0];

    // result = local_mean + (max(0, localVar - noise) / max(localVar, noise)) * (in - local_mean)
    cv::subtract(srcMat, tmpMat1, dstMat);  // in - local_mean

    cv::Mat maxLocalVarNoise;
    cv::max(tmpMat3, noisePower, maxLocalVarNoise);  // max(localVar, noise)

    cv::Mat localVarMinusNoise = tmpMat3 - noisePower;
    cv::max(localVarMinusNoise, 0.0, localVarMinusNoise);  // max(0, localVar - noise)

    cv::divide(localVarMinusNoise, maxLocalVarNoise, localVarMinusNoise);  // (max(0, localVar - noise) / max(localVar, noise))

    cv::multiply(localVarMinusNoise, dstMat, dstMat);  // Apply scaling factor to (in - local_mean)
    cv::add(dstMat, tmpMat1, dstMat);  // Add the local mean back

    // dstMat contains the filtered result
}

void applyGeneralWienerDeconvolution(const cv::Mat& G, cv::Mat& F_restored, int kSize = 5, double sigma = 5.0 / 8.5, bool constantK = false, double kValue = 1e-2, double C = 0.0251, double rho = 2.0, double a = 0.5)
{
    // double sigma{kSize / 8.5};
    cv::Mat H = cv::getGaussianKernel(kSize, sigma, CV_64F);
    H /= cv::sum(H)[0]; // Asegurarse de que esté normalizado
    H = H * H.t();
    std::cout << H << std::endl;

    muestraImagenOpenCV(G, "Imagen degradada G", false);
    cv::Mat H_vis;
    cv::normalize(H, H_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    muestraImagenOpenCV(H_vis, "Blur kernel", false);

    /// Definir SNR2inv(r) = C * r^rho
    int N = G.rows;
    int M = G.cols;
    int n = H.rows;
    int m = H.cols;
    int N2 = 2 * (N + n - 1);
    int M2 = 2 * (M + m - 1);
    int n2 = N2 / 2 + 1;
    int m2 = M2 / 2 + 1;
    cv::Mat C_SNR2inv = cv::Mat::zeros(N2, M2, CV_64F);
    if (constantK)
        C_SNR2inv.setTo(kValue);
    else
    {
        for (int i = 0; i < N2; ++i)
        {
            double* row_ptr = C_SNR2inv.ptr<double>(i);
            for (int j = 0; j < M2; ++j)
                row_ptr[j] = C * std::pow(std::sqrt((i - n2) * (i - n2) + (j - m2) * (j - m2)), rho);
        }
    }

    // DFT de H y G => H(r), G(r)
    cv::Mat G_ = cv::Mat::zeros(N2, M2, CV_64F);
    cv::Mat H_ = cv::Mat::zeros(N2, M2, CV_64F);
    G.copyTo(G_(cv::Rect(m, n, M, N)));
    H.copyTo(H_(cv::Rect(0, 0, m, n)));
    cv::Mat H_complex, G_complex;
    {
        cv::Mat plane0;
        H_.copyTo(plane0);
        cv::Mat planes[] = {plane0, cv::Mat::zeros(H_.size(), CV_64F)};
        cv::merge(planes, 2, H_complex);
    }
    {
        cv::Mat plane0;
        G_.copyTo(plane0);
        cv::Mat planes[] = {plane0, cv::Mat::zeros(G_.size(), CV_64F)};
        cv::merge(planes, 2, G_complex);
    }
    cv::dft(H_complex, H_);
    cv::dft(G_complex, G_);
    cv::Mat H_mag;
    {
        cv::Mat phase, planes[2];
        cv::split(H_, planes);
        cv::magnitude(planes[0], planes[1], H_mag);
        cv::add(H_mag, cv::Mat::ones(H_mag.rows, H_mag.cols, H_mag.type()), H_mag);
        cv::log(H_mag, H_mag);
        cv::phase(planes[0], planes[1], phase, false);
        // muestraImagenOpenCV(fftshift2D(H_mag), "H_ mag", false);
        // muestraImagenOpenCV(phase, "H_ phase", false);
    }
    {
        cv::Mat mag, phase, planes[2];
        cv::split(G_, planes);
        cv::magnitude(planes[0], planes[1], mag);
        cv::add(mag, cv::Mat::ones(mag.rows, mag.cols, mag.type()), mag);
        cv::log(mag, mag);
        cv::phase(planes[0], planes[1], phase, false);
        muestraImagenOpenCV(fftshift2D(mag), "G_ mag", false);
        // muestraImagenOpenCV(phase, "G_ phase", false);
    }

    // |H(r)|, |H(r)|^2
    cv::Mat H_mag_squared, planes[2], H_mag_squared_complex;
    cv::pow(H_mag, 2, H_mag_squared);
    // muestraImagenOpenCV(fftshift2D(H_mag_squared), "H_mag_squared", false);
    {
        cv::Mat plane0;
        H_mag_squared.copyTo(plane0);
        cv::Mat planes[] = {plane0, cv::Mat::zeros(H_mag_squared.size(), CV_64F)};
        cv::merge(planes, 2, H_mag_squared_complex);
    }

    // H*(r)
    cv::Mat H_conj = calculateConjugate(H_);

    // Calcular W(r)
    cv::Mat W1, W2, W;
    cv::divide(H_conj, H_mag_squared_complex + (cv::Mat::ones(H_mag_squared_complex.size(), H_mag_squared_complex.type()) * 1e-2), W1); // Regularizar un poco
    cv::pow(W1, a, W1);
    replaceNaNAndInfWithValues(W1);
    cv::Mat D2, D2_complex;
    cv::add(H_mag_squared, C_SNR2inv, D2);
    {
        cv::Mat plane0;
        D2.copyTo(plane0);
        cv::Mat planes[] = {plane0, cv::Mat::zeros(D2.size(), CV_64F)};
        cv::merge(planes, 2, D2_complex);
    }
    cv::divide(H_conj, D2_complex + (cv::Mat::ones(H_mag_squared_complex.size(), H_mag_squared_complex.type()) * 1e-2), W2);
    replaceNaNAndInfWithValues(W2);
    cv::pow(W2, 1 - a, W2);
    cv::multiply(W1, W2, W);
    replaceNaNAndInfWithValues(W);

    cv::Mat F_restored_ = G_.mul(W);
    {
        cv::Mat mag, phase, planes[2];
        cv::split(W, planes);
        cv::magnitude(planes[0], planes[1], mag);
        cv::add(mag, cv::Mat::ones(mag.rows, mag.cols, mag.type()), mag);
        cv::log(mag, mag);
        cv::phase(planes[0], planes[1], phase, false);
        muestraImagenOpenCV(fftshift2D(mag), "W mag", false);
        // muestraImagenOpenCV(phase, "W phase", false);
    }
    {
        cv::Mat mag, phase, planes[2];
        cv::split(F_restored_, planes);
        cv::magnitude(planes[0], planes[1], mag);
        cv::add(mag, cv::Mat::ones(mag.rows, mag.cols, mag.type()), mag);
        cv::log(mag, mag);
        cv::phase(planes[0], planes[1], phase, false);
        muestraImagenOpenCV(fftshift2D(mag), "F_restored_ mag", false);
        // muestraImagenOpenCV(phase, "F_restored_ phase", false);
    }

    cv::dft(F_restored_, F_restored, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    int Ns = N + n - 1;
    int Ms = M + m - 1;
    int k = 1;
    F_restored = F_restored(cv::Rect(k, k, Ms, Ns));
}

int main()
{
    std::string imagePath = "/media/sf_shared_folder/blurred-vision-678x446_compressed.jpg";
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_UNCHANGED);

    if (input.empty())
    {
        std::cerr << "Error: No se pudo cargar la imagen en el path especificado: " << imagePath << std::endl;
        return -1;
    }

    /// Wiener simple
    // {
    //     cv::Mat G;
    //     input.channels() == 3 ? cv::cvtColor(input, G, cv::COLOR_BGR2GRAY) : input.copyTo(G);
    //     G.convertTo(G, CV_64F, 1.0 / 255.0);

    //     cv::Mat F_restored;
    //     cvWiener2(G, F_restored, 5, 5);

    //     muestraImagenOpenCV(G, "Before", false);
    //     muestraImagenOpenCV(F_restored, "Imagen restaurada F^", false);
    // }

    /// Filtro de Wiener general
    {
        cv::Mat G;
        input.channels() == 3 ? cv::cvtColor(input, G, cv::COLOR_BGR2GRAY) : input.copyTo(G);
        G.convertTo(G, CV_64F, 1.0 / 255.0);
        // cv::resize(G, G, cv::Size(128, 128));

        /// Parámetros de la restauración:
        int kSize = 11;
        double sigma = 2.0;
        double C = 0.251;
        double rho = 2.0;
        double a = 0.5;
        bool constantK = false;
        double kValue = 100;

        cv::Mat F_restored;
        applyGeneralWienerDeconvolution(G, F_restored, kSize, sigma, constantK, kValue, C, rho, a);

        F_restored.convertTo(F_restored, CV_8UC1, 255.0);
        cv::normalize(F_restored, F_restored, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray());
        muestraImagenOpenCV(F_restored, "Imagen restaurada F^", false);
    }

    /// Deconvolution with TV prior
    {
        // if (input.channels() == 3)
        //     cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
        // cv::resize(input, input, cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
        // input.convertTo(input, CV_64F, 1.0 / 255.0);

        // /// PSF usada (gaussiana, motion, leer de archivo según datasheet de cámara...)
        // int kernel_size = 3;
        // // cv::Mat psf = cv::imread("psf.jpg", cv::IMREAD_GRAYSCALE);
        // cv::Mat psf = cv::getGaussianKernel(kernel_size, 1.0, CV_64F) * cv::getGaussianKernel(kernel_size, 1.0, CV_64F).t();

        // cv::Mat result;
        // muestraImagenOpenCV(input, "input", false);
        // start_time = std::chrono::high_resolution_clock::now();
        // cv::Mat tiko_deconv = deconvolutionByTV(input, psf, 100, 20.0, 0.0004);
        // end_time = std::chrono::high_resolution_clock::now();
        // MyTimeOutput("deconvolutionWithTVPrior: ", start_time, end_time);
        // muestraImagenOpenCV(tiko_deconv, "deconvolutionWithTVPrior", false);
    }

    /// Deconvolution with hyper-Laplacian prior
    {
        // cv::resize(input, input, cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
        // input.convertTo(input, CV_32F, 1.0 / 255.0);

        // cv::Mat out[3], src[3], imout;

        // cv::Mat kernel = (cv::Mat_<float>(11, 11) <<
        //                   2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        //                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        //                   5, 0, 2, 0, 1, 1, 4, 0, 3, 0, 3,
        //                   1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        //                   4, 1, 5, 0, 1, 3, 4, 0, 3, 0, 2,
        //                   0, 0, 0, 0, 1, 6, 2, 0, 0, 0, 0,
        //                   1, 0, 4, 5, 23, 37, 27, 2, 1, 0, 0,
        //                   0, 0, 0, 0, 17, 35, 23, 0, 0, 0, 0,
        //                   0, 0, 0, 0, 4, 9, 5, 0, 0, 0, 0,
        //                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        //                   1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0);
        // kernel /= 255.0;

        // start_time = std::chrono::high_resolution_clock::now();

        // cv::split(input, src);
        // std::cout << "Número de canales: " << input.channels() << std::endl;

        // for (int i = 0; i < input.channels(); i++) {
        //     fast_deblurring(src[i], kernel, out[i]);
        // }

        // cv::merge(out, input.channels(), imout);

        // end_time = std::chrono::high_resolution_clock::now();
        // MyTimeOutput("fast_deblurring: ", start_time, end_time);

        // imout *= 255.0;
        // imout.convertTo(imout, CV_8U);

        // cv::imshow("Input", input);
        // cv::imshow("Deblurred", imout);

        // cv::waitKey(0);
    }
    return 0;
}
