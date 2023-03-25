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
#include <intrin.h>
#include <iostream>
#include <memory>
#include <numbers>
#include <tuple>
#include <utility>
#include <stb/stb_image.h>

namespace cppraisr
{

    template<class T>
    void swap(T& x0, T& x1)
{
        T t = x0;
        x0 = x1;
        x1 = t;
}

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
    int32_t width_ = 0;
    int32_t height_ = 0;
    int32_t channels_ = 0;
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
}

template<class T>
void Image<T>::reset(int32_t width, int32_t height, int32_t channels, T* pixels, std::function<void(void*)> deleter)
{
    ::free(pixels_);
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

template<class T, int32_t W, int32_t H>
class ImageStatic
{
public:
    ImageStatic();
    ~ImageStatic();

    constexpr int32_t count() const
    {
        return W*H;
    }

    constexpr int32_t size() const
    {
        return sizeof(T)*W*H;
    }

    void clear();

    constexpr int32_t w() const
    {
        return W;
    }
    constexpr int32_t h() const
    {
        return H;
    }

    const T& operator()(int32_t x, int32_t y) const;
    T& operator()(int32_t x, int32_t y);

private:
    ImageStatic(const ImageStatic&) = delete;
    ImageStatic& operator=(const ImageStatic&) = delete;
    T pixels_[W * H];
};

template<class T, int32_t W, int32_t H>
ImageStatic<T, W, H>::ImageStatic()
{
}

template<class T, int32_t W, int32_t H>
ImageStatic<T, W, H>::~ImageStatic()
{
}

template<class T, int32_t W, int32_t H>
void ImageStatic<T, W, H>::clear()
{
    ::memset(pixels_, 0, sizeof(T)*W*H);
}

template<class T, int32_t W, int32_t H>
const T& ImageStatic<T, W, H>::operator()(int32_t x, int32_t y) const
{
    return pixels_[y * W + x];
}

template<class T, int32_t W, int32_t H>
T& ImageStatic<T, W, H>::operator()(int32_t x, int32_t y)
{
    return pixels_[y * W + x];
}

template<class T, int32_t N0, int32_t N1, int32_t N2, int32_t N3, int32_t N4>
class Array5d
{
public:
    Array5d();
    ~Array5d();

    constexpr int32_t count() const
    {
        return N0 * N1 * N2 * N3 * N4;
    }

    constexpr size_t size() const
    {
        return sizeof(T) * count();
    }

    void clear();
    void write(std::ostream& os);
    void read(std::istream& is);

    const T& operator()(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4) const
    {
        int32_t index = i0 * N1 * N2 * N3 * N4 + i1 * N2 * N3 * N4 + i2 * N3 * N4 + i3 * N4 + i4;
        return items_[index];
    }

    T& operator()(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4)
    {
        int32_t index = i0 * N1 * N2 * N3 * N4 + i1 * N2 * N3 * N4 + i2 * N3 * N4 + i3 * N4 + i4;
        return items_[index];
    }

    const T* operator()(int32_t i0, int32_t i1, int32_t i2, int32_t i3) const
    {
        int32_t index = i0 * N1 * N2 * N3 * N4 + i1 * N2 * N3 * N4 + i2 * N3 * N4 + i3 * N4;
        return &items_[index];
    }

    T* operator()(int32_t i0, int32_t i1, int32_t i2, int32_t i3)
    {
        int32_t index = i0 * N1 * N2 * N3 * N4 + i1 * N2 * N3 * N4 + i2 * N3 * N4 + i3 * N4;
        return &items_[index];
    }

private:
    Array5d(const Array5d&) = delete;
    Array5d& operator=(const Array5d&) = delete;
    T* items_;
};

template<class T, int32_t N0, int32_t N1, int32_t N2, int32_t N3, int32_t N4>
Array5d<T, N0, N1, N2, N3, N4>::Array5d()
    :items_(nullptr)
{
    items_ = static_cast<T*>(::malloc(size()));
    ::memset(items_, 0, size());
}

template<class T, int32_t N0, int32_t N1, int32_t N2, int32_t N3, int32_t N4>
Array5d<T, N0, N1, N2, N3, N4>::~Array5d()
{
    ::free(items_);
    items_ = nullptr;
}

template<class T, int32_t N0, int32_t N1, int32_t N2, int32_t N3, int32_t N4>
void Array5d<T, N0, N1, N2, N3, N4>::clear()
{
    ::memset(items_, 0, size());
}

template<class T, int32_t N0, int32_t N1, int32_t N2, int32_t N3, int32_t N4>
void Array5d<T, N0, N1, N2, N3, N4>::write(std::ostream& os)
{
    os.write((const char*)items_, size());
}

template<class T, int32_t N0, int32_t N1, int32_t N2, int32_t N3, int32_t N4>
void Array5d<T, N0, N1, N2, N3, N4>::read(std::istream& is)
{
    is.read((char*)items_, size());
}

double to_double(uint8_t x);
uint8_t to_uint8(double x);

std::vector<std::filesystem::path> parse_directory(const char* path, std::function<bool(const std::filesystem::directory_entry&)> predicate);
void gaussian2d(int32_t size, double* w, double sigma);
void solv2x2(double evalues[2], double evectors[4], const double m[4]);
std::tuple<int32_t, int32_t, int32_t> hashkey(int32_t gradient_size, const double* gradient_patch, const double* weights, int32_t angles);

void transpose(int32_t size, double* dst, const double* src);
void power_m(int32_t size, double* m, const double* m0, int32_t p);
void mul_mm(int32_t size, double* m, const double* m0, const double* m1);
void mul_mm(int32_t rows, int32_t cols, double* r, const double* m0, const double* m1);
void mul_mv(int32_t size, double* r, const double* m, const double* v);
void mul_v(int32_t size, double* r, const double* v, const double a);
void add(int32_t size, double* m0, const double* m1);
void square_m(int32_t size, double* r, const double* v);
double dot(int32_t size, const double* x0, const double* x1);
double sum(int32_t size, const double* x);

class CGSolver
{
public:
    CGSolver(int32_t size);
    ~CGSolver();
    void solve(double* x, const double* A, const double* b, const int32_t max_iteration = 1000, const double epsilon = 1.0e-16);

private:
    CGSolver(const CGSolver&) = delete;
    CGSolver& operator=(const CGSolver&) = delete;
    int32_t size_;
    double* Ax_;
    double* r_;
    double* p_;
    double* Ap_;
};

class BiCGStabSolver
{
public:
    BiCGStabSolver(int32_t size);
    ~BiCGStabSolver();
    void solve(double* x, const double* A, const double* b, const int32_t max_iteration = 1000, const double epsilon = 1.0e-16);

private:
    BiCGStabSolver(const BiCGStabSolver&) = delete;
    BiCGStabSolver& operator=(const BiCGStabSolver&) = delete;
    int32_t size_;
    double* Ax_;
    double* r_;
    double* rr_;
    double* p_;
    double* Ap_;
    double* s_;
    double* As_;
};

class CGLSSolver
{
public:
    static void solve(int32_t size, double* x, double* A, const double* b, const int32_t max_iteration = 1000, const double epsilon = 1.0e-16, const double criteria=1.0);

};

double determinant(int32_t size, const double* m);
void LU(int32_t size, double* q, double* L, double* U, const double* m);
void invert(int32_t size, double* im, const double* m);

} // namespace cppraisr
#endif // INC_CPPRAISR_UTIL_H_
