
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>


struct ProcCUDA {
    cv::cuda::Stream st;
    cv::cuda::GpuMat d_bgr, d_hsv, d_gray, d_blur, d_tmp, d_thresh, d_work;
    cv::Ptr<cv::cuda::Filter> gauss;
    cv::Ptr<cv::cuda::Filter> dil;
    cv::Ptr<cv::cuda::Filter> closef;

    ProcCUDA() {
        gauss = cv::cuda::createGaussianFilter(CV_8U, CV_8U, cv::Size(21,21), 0); // Blocksize 21
        cv::Mat se = cv::Mat::ones(2,2,CV_8U);
        dil    = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8U, se, cv::Point(-1,-1), 5);
        closef = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE,  CV_8U, se, cv::Point(-1,-1), 2);
    }

    // Runs: BGR->HSV, BGR->Gray, adaptive-thresh (Gaussian), dilate, close
    void run(const cv::Mat& frame, cv::Mat& hsv, cv::Mat& gray, cv::Mat& closed) {
        d_bgr.upload(frame, st);

        cv::cuda::cvtColor(d_bgr, d_hsv,  cv::COLOR_BGR2HSV,  0, st); // Convert to HSV
        cv::cuda::cvtColor(d_bgr, d_gray, cv::COLOR_BGR2GRAY, 0, st); // Convert to Grayscale

        cv::cuda::GpuMat d_meanminusC;
        gauss->apply(d_gray, d_blur, st); // For Adaptive Thresholding
        cv::cuda::subtract(d_blur, cv::Scalar(4), d_tmp, cv::noArray(), -1, st);  // C is 4
        cv::cuda::compare(d_gray, d_tmp, d_thresh, cv::CMP_LT, st);

        dil->apply(d_thresh, d_work, st); // Dilation to thicken edges
        closef->apply(d_work, d_thresh, st); // Close small gaps inside sticker areas

        d_hsv.download(hsv, st);
        d_gray.download(gray, st);
        d_thresh.download(closed, st);
        st.waitForCompletion();
    }
};
