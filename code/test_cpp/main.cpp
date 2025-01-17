#include <iostream>

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

int main()
{
    std::string imagePath = "/opt/proyectos/image-deblurrer/assets/7.jpg";
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_UNCHANGED);

    if (input.empty())
    {
        std::cerr << "Error: No se pudo cargar la imagen en el path especificado: " << imagePath << std::endl;
        return -1;
    }

    std::chrono::high_resolution_clock::time_point start_time, end_time;

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
        cv::resize(input, input, cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
        input.convertTo(input, CV_32F, 1.0 / 255.0);

        cv::Mat out[3], src[3], imout;

        cv::Mat kernel = (cv::Mat_<float>(11, 11) <<
                          2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          5, 0, 2, 0, 1, 1, 4, 0, 3, 0, 3,
                          1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          4, 1, 5, 0, 1, 3, 4, 0, 3, 0, 2,
                          0, 0, 0, 0, 1, 6, 2, 0, 0, 0, 0,
                          1, 0, 4, 5, 23, 37, 27, 2, 1, 0, 0,
                          0, 0, 0, 0, 17, 35, 23, 0, 0, 0, 0,
                          0, 0, 0, 0, 4, 9, 5, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0);
        kernel /= 255.0;

        start_time = std::chrono::high_resolution_clock::now();

        cv::split(input, src);
        std::cout << "Número de canales: " << input.channels() << std::endl;

        for (int i = 0; i < input.channels(); i++) {
            fast_deblurring(src[i], kernel, out[i]);
        }

        cv::merge(out, input.channels(), imout);

        end_time = std::chrono::high_resolution_clock::now();
        MyTimeOutput("fast_deblurring: ", start_time, end_time);

        imout *= 255.0;
        imout.convertTo(imout, CV_8U);

        cv::imshow("Input", input);
        cv::imshow("Deblurred", imout);

        cv::waitKey(0);
    }
    return 0;
}
