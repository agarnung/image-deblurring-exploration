/// @brief Based on D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian Priors", Proceedings of NIPS 2009
/// Instalar libfftw3-dev

#pragma once

#include <fftw3.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv2/opencv.hpp>

const int Iteration_time = 5;
extern const float BO5UND[];
extern const float LEFT_LINE_P0[];
extern const float LEFT_LINE_P1[];
extern const float RIGHT_LINE_P0[];
extern const float RIGHT_LINE_P1[];

void circular_conv(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel, cv::Point anchor);
void divispectrum(const cv::Mat& X1, const cv::Mat& X2, cv::Mat& X_out);
void Solve_w(const cv::Mat& v, cv::Mat& w, int beta_chose);
void dft_fftw(const cv::Mat& src, cv::Mat& dst);
void idft_fftw(const cv::Mat& src, cv::Mat& dst);
void fast_deblurring(const cv::Mat& src_im, const cv::Mat& kernel, cv::Mat& yout);
