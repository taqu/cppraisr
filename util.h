#ifndef INC_CPPRAISR_UTIL_H_
#define INC_CPPRAISR_UTIL_H_
// clang-format off
/**
# License
This software is distributed under two licenses, choose whichever you like.

## MIT License
Copyright (c) 2023 Takuro Sakai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Public Domain
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
*/
// clang-format on
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <tuple>
#include <utility>
#include <eigen/Eigen/Eigen>

namespace cppraisr
{

/**
 * @tparam  T ... type
 * @brief Swap values
 */
template<class T>
void swap(T& x0, T& x1)
{
    T t = x0;
    x0 = x1;
    x1 = t;
}

/**
 * @tparam T ... type
 * @brief Simple image class
 */
template<class T>
class Image
{
public:
    Image(std::function<void(void*)> deleter=::free);
    Image(int32_t width, int32_t height, int32_t channels, std::function<void(void*)> deleter=::free);
    ~Image();

    void reset(int32_t width, int32_t height, int32_t channels, T* pixels, std::function<void(void*)> deleter=::free);

    int32_t w() const
    {
        return width_;
    }
    int32_t h() const
    {
        return height_;
    }

    int32_t c() const
    {
        return channels_;
    }

    const T& operator()(int32_t x, int32_t y, int32_t c) const;
    T& operator()(int32_t x, int32_t y, int32_t c);

    void swap(Image& other);
    operator bool() const
    {
        return nullptr != pixels_;
    }
private:
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    int32_t width_;
    int32_t height_;
    int32_t channels_;
    T* pixels_ = nullptr;
    std::function<void(void*)> deleter_;
};

template<class T>
Image<T>::Image(std::function<void(void*)> deleter)
    : width_(0)
    , height_(0)
    , channels_(0)
    , pixels_(nullptr)
    , deleter_(deleter)
{
}

template<class T>
Image<T>::Image(int32_t width, int32_t height, int32_t channels, std::function<void(void*)> deleter)
    : width_(width)
    , height_(height)
    , channels_(channels)
    , pixels_(nullptr)
    , deleter_(deleter)
{
    pixels_ = static_cast<T*>(::malloc(sizeof(T) * width_ * height_ * channels_));
}

template<class T>
Image<T>::~Image()
{
    deleter_(pixels_);
    deleter_ = nullptr;
    pixels_ = nullptr;
    channels_ = 0;
    height_ = 0;
    width_ = 0;
}

template<class T>
void Image<T>::reset(int32_t width, int32_t height, int32_t channels, T* pixels, std::function<void(void*)> deleter)
{
    deleter_(pixels_);
    width_ = width;
    height_ = height;
    channels_ = channels;
    pixels_ = pixels;
    deleter_ = deleter;
}

template<class T>
const T& Image<T>::operator()(int32_t x, int32_t y, int32_t c) const
{
    return pixels_[(y * width_ + x) * channels_ + c];
}

template<class T>
T& Image<T>::operator()(int32_t x, int32_t y, int32_t c)
{
    return pixels_[(y * width_ + x) * channels_ + c];
}

template<class T>
void Image<T>::swap(Image<T>& other)
{
    cppraisr::swap(width_, other.width_);
    cppraisr::swap(height_, other.height_);
    cppraisr::swap(channels_, other.channels_);
    cppraisr::swap(pixels_, other.pixels_);
    deleter_.swap(other.deleter_);
}

/**
 * @brief Convert [0 255] integer to [0 1] float
 */
double to_double(uint8_t x);
inline double to_double(float x)
{
    return x;
}

/**
 * @brief Convert [0 1] float to [0 255] integer
 */
uint8_t to_uint8(double x);

template<class T>
T to_grey(T r, T g, T b, int32_t base)
{
    float x = 0.183f * r + 0.614f * g + 0.062f * b + 16.0f;
    int32_t i = static_cast<int32_t>(x);
    if(base<=i){
        return static_cast<T>(base-1);
    }
    if(i<0){
        return 0;
    }
    return static_cast<T>(i);
}

inline float to_grey(float r, float g, float b)
{
    return 0.183f * r + 0.614f * g + 0.062f * b + 16.0f/255.0f;
}

/**
 * @brief Parse a directory recursively and gather file paths
 * @return gathered paths
 * @param path ... directory to parse
 * @param predicate ... a function of returning desirable condition
 */
std::vector<std::filesystem::path> parse_directory(const char* path, std::function<bool(const std::filesystem::directory_entry&)> predicate, bool shuffle);

/**
 * @brief Generate a Gaussian filter
 */
template<class T>
void gaussian2d(int32_t size, T* w, T sigma)
{
    int32_t half = size >> 1;
    T total = 0.0;
    for(int32_t i = 0; i < size; ++i) {
        T dy = i - half;
        for(int32_t j = 0; j < size; ++j) {
            T dx = j - half;
            w[size * i + j] = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            total += w[size * i + j];
        }
    }
    total = 1.0 / total;
    for(int32_t i = 0; i < (size * size); ++i) {
        w[i] *= total;
    }
}

void conv2d(int32_t width, int32_t height, float* dst, const float* src, int32_t size, const float* weights);

/**
 * @brief Generate a Box filter
 */
void box2d(int32_t size, double* w);

/**
 * @brief Solve 2x2 equations and get eigen values, eigen vectors
 */
void solv2x2(double evalues[2], double evectors[4], const double m[4]);

/**
 * @brief Solve 2x2 equations and get eigen values, eigen vectors
 */
void solv2x2d(double evalues[2], double evectors[4], const double m[4]);

/**
 * @brief Hash function for RAISR
 */
std::tuple<int32_t, int32_t, int32_t> hashkey(const double* gradient_patch, int32_t angles);

/**
 * @brief Calc SSIM of a patch image
 */
template<class T>
double ssim(int32_t w, int32_t,int32_t c, int32_t patch_size, int32_t x, int32_t y, const T* x0, const T* x1)
{
    double avg0 = 0.0;
    double avg1 = 0.0;
    for(int32_t i = 0; i < patch_size; ++i) {
        for(int32_t j = 0; j < patch_size; ++j) {
            int32_t index = ((y + i) * w + x + j) * c;
            double l0 = 0;
            double l1 = 0;
            for(int32_t k = 0; k < c; ++k) {
                l0 += static_cast<double>(x0[index + k]) * x0[index + k];
                l1 += static_cast<double>(x1[index + k]) * x1[index + k];
            }
            l0 = std::sqrt(l0);
            l1 = std::sqrt(l1);
            avg0 += l0;
            avg1 += l1;
        }
    }

    double count = patch_size * patch_size;
    double inv_count = 1.0 / count;
    avg0 *= inv_count;
    avg1 *= inv_count;

    double d0 = 0;
    double d1 = 0;
    double cov = 0;
    for(int32_t i = 0; i < patch_size; ++i) {
        for(int32_t j = 0; j < patch_size; ++j) {
            int32_t index = ((y + i) * w + x + j) * c;
            double l0 = 0;
            double l1 = 0;
            for(int32_t k = 0; k < c; ++k) {
                l0 += static_cast<double>(x0[index + k]) * x0[index + k];
                l1 += static_cast<double>(x1[index + k]) * x1[index + k];
            }
            l0 = std::sqrt(l0);
            l1 = std::sqrt(l1);
            d0 += (l0 - avg0) * (l0 - avg0);
            d1 += (l1 - avg1) * (l1 - avg1);
            cov += (l0 - avg0) * (l1 - avg1);
        }
    }
    double sigma0 = std::sqrt(d0 * inv_count);
    double sigma1 = std::sqrt(d1 * inv_count);
    cov *= inv_count;
    static constexpr double C1 = 0.01*255*0.1*255;
    static constexpr double C2 = 0.03*255*0.3*255;

    double S0 = (2.0*avg0*avg1+C1)*(2.0*cov+C2);
    double S1 = (avg0*avg0+avg1*avg1+C1)*(sigma0*sigma0+sigma1*sigma1+C2);
    return S0/S1;
}

/**
 * @brief Calc MSSIM of a image
 */
template<class T>
double ssim(int32_t w, int32_t h, int32_t c, int32_t patch_size, const T* x0, const T* x1)
{
    double total = 0;
    int32_t count = 0;
    for(int32_t i=0; i<(h-patch_size); i+=patch_size){
        for(int32_t j=0; j<(w-patch_size); j+=patch_size){
            total += ssim<T>(w, h, c, patch_size, j, i, x0, x1);
            ++count;
        }
    }
    return total/count;
}

} // namespace cppraisr
#endif // INC_CPPRAISR_UTIL_H_

