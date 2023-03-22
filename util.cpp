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
#include "util.h"
#include <cassert>

namespace cppraisr
{
namespace
{
    bool is_equal(double x0, double x1, double torelance = 1.0e-10)
    {
        return std::abs(x0 - x1) < torelance;
    }

    bool is_zero(double x, double torelance = 1.0e-10)
    {
        return std::abs(x) < torelance;
    }

    void gradiant3(double gx[9], double gy[9], const double m[9], const double w[9])
    {
        // grad x
        gx[0] = (m[1] - m[0]) * w[0];
        gx[1] = (m[2] - m[0]) * w[1] * 0.5;
        gx[2] = (m[3] - m[2]) * w[2];

        gx[3] = (m[5] - m[3]) * w[3];
        gx[4] = (m[6] - m[3]) * w[4] * 0.5;
        gx[5] = (m[7] - m[6]) * w[5];

        gx[6] = (m[7] - m[6]) * w[6];
        gx[7] = (m[8] - m[6]) * w[7] * 0.5;
        gx[8] = (m[8] - m[7]) * w[8];

        // grad y
        gy[0] = (m[3] - m[0]) * w[0];
        gy[1] = (m[6] - m[0]) * w[1] * 0.5;
        gy[2] = (m[6] - m[3]) * w[2];

        gy[3] = (m[4] - m[1]) * w[3];
        gy[4] = (m[7] - m[1]) * w[4] * 0.5;
        gy[5] = (m[7] - m[4]) * w[5];

        gy[6] = (m[5] - m[2]) * w[6];
        gy[7] = (m[8] - m[2]) * w[7] * 0.5;
        gy[8] = (m[8] - m[5]) * w[8];
    }

    void gradiant4(double gx[16], double gy[16], const double m[16], const double w[16])
    {
        // grad x
        gx[0] = (m[1] - m[0]) * w[0];
        gx[1] = (m[2] - m[0]) * w[1] * 0.5;
        gx[2] = (m[3] - m[1]) * w[2] * 0.5;
        gx[3] = (m[3] - m[2]) * w[3];

        gx[4] = (m[5] - m[4]) * w[4];
        gx[5] = (m[6] - m[4]) * w[5] * 0.5;
        gx[6] = (m[7] - m[5]) * w[6] * 0.5;
        gx[7] = (m[7] - m[6]) * w[7];

        gx[8] = (m[9] - m[8]) * w[8];
        gx[9] = (m[10] - m[8]) * w[9] * 0.5;
        gx[10] = (m[11] - m[9]) * w[10] * 0.5;
        gx[11] = (m[11] - m[10]) * w[11];

        gx[12] = (m[13] - m[12]) * w[12];
        gx[13] = (m[14] - m[12]) * w[13] * 0.5;
        gx[14] = (m[15] - m[13]) * w[14] * 0.5;
        gx[15] = (m[15] - m[14]) * w[15];

        // grad y
        gy[0] = (m[4] - m[0]) * w[0];
        gy[1] = (m[8] - m[0]) * w[1] * 0.5;
        gy[2] = (m[8] - m[4]) * w[2] * 0.5;
        gy[3] = (m[12] - m[8]) * w[3];

        gy[4] = (m[5] - m[1]) * w[4];
        gy[5] = (m[9] - m[1]) * w[5] * 0.5;
        gy[6] = (m[9] - m[5]) * w[6] * 0.5;
        gy[7] = (m[13] - m[9]) * w[7];

        gy[8] = (m[6] - m[2]) * w[8];
        gy[9] = (m[10] - m[2]) * w[9] * 0.5;
        gy[10] = (m[10] - m[6]) * w[10] * 0.5;
        gy[11] = (m[14] - m[10]) * w[11];

        gy[12] = (m[7] - m[3]) * w[12];
        gy[13] = (m[11] - m[3]) * w[13] * 0.5;
        gy[14] = (m[11] - m[7]) * w[14] * 0.5;
        gy[15] = (m[15] - m[11]) * w[15];
    }

    void gradiant(int32_t size, double gx[], double gy[], const double m[], const double w[])
    {
        // grad x
        for(int32_t i = 0; i < size; ++i) {
            for(int32_t j = 0; j < size; ++j) {
                double weight = 0.5;
                int32_t prev = j - 1;
                int32_t next = j + 1;
                if(prev < 0) {
                    weight = 1.0;
                    prev = 0;
                }
                if(size <= next) {
                    weight = 1.0f;
                    next = size - 1;
                }
                int32_t index = i * size + j;
                gx[index] = (m[next] - m[prev]) * w[index] * weight;
            }
        }

        // grad y
        for(int32_t i = 0; i < size; ++i) {
            double weight = 0.5;
            int32_t prev = i - 1;
            int32_t next = i + 1;
            if(prev < 0) {
                weight = 1.0;
                prev = 0;
            }
            if(size <= next) {
                weight = 1.0f;
                next = size - 1;
            }

            for(int32_t j = 0; j < size; ++j) {
                int32_t index = i * size + j;
                gy[index] = (m[next] - m[prev]) * w[index] * weight;
            }
        }
    }

    void conv2(double g[4], const double gx[16], const double gy[16])
    {
        double gxgx = 0.0;
        double gxgy = 0.0;
        double gygy = 0.0;
        for(uint32_t i = 0; i < 16; ++i) {
            gxgx += gx[i] * gx[i];
            gxgy += gx[i] * gy[i];
            gygy += gy[i] * gy[i];
        }
        g[0] = gxgx;
        g[1] = gxgy;
        g[2] = gxgy;
        g[3] = gygy;
    }
} // namespace

double to_double(uint8_t x)
{
    return x / 255.0;
}

uint8_t to_uint8(double x)
{
    x *= 256.0;
    int32_t t = static_cast<int32_t>(x);
    if(t<0){
        return 0;
    }
    if(256<=t){
        return 255;
    }
    return static_cast<uint8_t>(t);
}

std::vector<std::filesystem::path> parse_directory(const char* path, std::function<bool(const std::filesystem::directory_entry&)> predicate)
{
    assert(nullptr != path);
    std::vector<std::filesystem::path> files;
    for(const std::filesystem::directory_entry& entry: std::filesystem::recursive_directory_iterator(path)) {
        if(!entry.is_regular_file()) {
            continue;
        }
        if(predicate(entry)) {
            files.push_back(entry.path());
        }
    }
    return files;
}

void gaussian2d(int32_t size, double* w, double sigma)
{
    int32_t half = size >> 1;
    double total = 0.0;
    for(int32_t i = 0; i < size; ++i) {
        double dy = i - half;
        for(int32_t j = 0; j < size; ++j) {
            double dx = j - half;
            w[size * i + j] = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            total += w[size * i + j];
        }
    }
    total = 1.0 / total;
    for(int32_t i = 0; i < (size * size); ++i) {
        w[i] *= total;
    }
}

void solv2x2(double evalues[2], double evectors[4], const double m[4])
{
    // assert(is_equal(m[1], m[2]));
    double a = m[0] + m[3];
    double descr = a * a - 4.0 * (m[0] * m[3] - m[1] * m[2]);
    descr = descr < 0.0 ? 0.0 : sqrt(descr);
    double b = m[0] + m[3];
    evalues[0] = (b + descr) / 2.0;
    evalues[1] = (b - descr) / 2.0;
    evectors[0] = m[1];
    evectors[1] = m[3] - evalues[1];
    evectors[2] = m[0] - evalues[0];
    evectors[3] = m[2];
}

std::tuple<int32_t, int32_t, int32_t> hashkey(int32_t gradient_size, const double* gradient_patch, const double* weights, int32_t angles)
{
    assert(0 < gradient_size && gradient_size <= 7);
    assert(nullptr != gradient_patch);
    assert(nullptr != weights);
    assert(0 < angles);
    double gx[7 * 7];
    double gy[7 * 7];
    switch(gradient_size) {
    case 3:
        gradiant3(gx, gy, gradient_patch, weights);
        break;
    case 4:
        gradiant4(gx, gy, gradient_patch, weights);
        break;
    case 5:
    case 7:
        gradiant(gradient_size, gx, gy, gradient_patch, weights);
        break;
    default:
        assert(false);
        return std::make_tuple(0, 0, 0);
    }
    double g[4];
    conv2(g, gx, gy);
    double evalues[2];
    double evectors[4];
    solv2x2(evalues, evectors, g);
    double theta = atan2(evectors[1], evectors[0]);
    while(theta < 0.0) {
        theta += std::numbers::pi_v<double>;
    }
    double lamda0 = sqrt(evalues[0]);
    double lamda1 = sqrt(evalues[1]);
    double u;
    if(is_zero(lamda0) && is_zero(lamda1)) {
        u = 0.0;
    } else {
        u = (lamda0 - lamda1) / (lamda0 + lamda1);
    }
    double lamda = evalues[0];
    int32_t strength;
    if(lamda < 0.0001) {
        strength = 0;
    } else if(0.001 < lamda) {
        strength = 2;
    } else {
        strength = 1;
    }

    int32_t coherence;
    if(u < 0.25) {
        coherence = 0;
    } else if(0.5 < u) {
        coherence = 2;
    } else {
        coherence = 1;
    }
    int32_t angle = static_cast<int32_t>(floor(theta / std::numbers::pi * angles));
    if(angles <= angle) {
        angle = angles - 1;
    }
    return std::make_tuple(angle, strength, coherence);
}

void transpose(int32_t size, double* dst, const double* src)
{
    for(int32_t i = 0; i < size; ++i) {
        for(int32_t j = 0; j < size; ++j) {
            dst[size * i + j] = src[size * j + i];
        }
    }
}

void mul_mm(int32_t rows, int32_t cols0, int32_t cols1, double* r, const double* m0, const double* m1)
{
    for(int32_t i = 0; i < rows; ++i) {
        for(int32_t j = 0; j < cols1; ++j) {
            double t = 0.0;
            for(int32_t k = 0; k < cols0; ++k) {
                t += m0[cols0 * i + k] * m1[cols1 * k + j];
            }
            r[cols1 * i + j] = t;
        }
    }
}

void mul_mv(int32_t rows, int32_t cols, double* r, const double* m, const double* v)
{
    for(int32_t i = 0; i < rows; ++i) {
        r[i] = 0.0;
        for(int32_t j = 0; j < cols; ++j) {
            r[i] += m[i * cols + j] * v[j];
        }
    }
}

void mul_v(int32_t size, double* r, const double* v, const double a)
{
    for(int32_t i = 0; i < size; ++i) {
        r[i] = v[i]*a;
    }
}

void add(int32_t size, double* m0, const double* m1)
{
    for(int32_t i = 0; i < size; ++i) {
        m0[i] += m1[i];
    }
}

void square(int32_t size, double* r, const double* v)
{
    for(int32_t i = 0; i < size; ++i) {
        for(int32_t j = 0; j < size; ++j) {
            r[i * size + j] = v[i] * v[j];
        }
    }
}

void dot(int32_t size, double* r, const double* m, const double* v)
{
    for(int32_t i = 0; i < size; ++i) {
        r[i] = 0.0;
        for(int32_t j = 0; j < size; ++j) {
            r[i] += m[i * size + j] * v[j];
        }
    }
}

double dot(int32_t size, const double* x0, const double* x1)
{
    double t = 0.0;
    for(int32_t i = 0; i < size; ++i) {
        t += x0[i] * x1[i];
    }
    return t;
}

CGSolver::CGSolver(int32_t size)
    : size_(size)
{
    assert(0 < size_);
    Ax_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    r_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    p_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    Ap_ = static_cast<double*>(::malloc(sizeof(double) * size_));
}

CGSolver::~CGSolver()
{
    ::free(Ap_);
    ::free(p_);
    ::free(r_);
    ::free(Ax_);
}

void CGSolver::solve(double* x, const double* A, const double* b, const int32_t max_iteration, const double epsilon)
{
    mul_mv(size_, size_, Ax_, A, x);
    for(int32_t i = 0; i < size_; ++i) {
        r_[i] = b[i] - Ax_[i];
        p_[i] = r_[i];
    }
    double r0 = ::sqrt(dot(size_, r_, r_));
    r0 = r0 * r0 * epsilon * epsilon;
    for(int32_t i = 0; i < max_iteration; ++i) {
        mul_mv(size_, size_, Ap_, A, p_);
        double d0 = dot(size_, r_, r_);
        double alpha = d0 / dot(size_, p_, Ap_);
        for(int32_t j = 0; j < size_; ++j) {
            x[j] += alpha * p_[j];
            r_[j] -= alpha * Ap_[j];
        }
        double d1 = dot(size_, r_, r_);
        if(d1 <= r0) {
            break;
        }
        double beta = d1 / d0;
        for(int32_t j = 0; j < size_; ++j) {
            p_[j] = r_[j] + beta * p_[j];
        }
    }
}

BiCGStabSolver::BiCGStabSolver(int32_t size)
    : size_(size)
{
    assert(0 < size_);
    Ax_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    r_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    rr_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    p_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    Ap_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    s_ = static_cast<double*>(::malloc(sizeof(double) * size_));
    As_ = static_cast<double*>(::malloc(sizeof(double) * size_));
}

BiCGStabSolver::~BiCGStabSolver()
{
    ::free(As_);
    ::free(s_);
    ::free(Ap_);
    ::free(p_);
    ::free(rr_);
    ::free(r_);
    ::free(Ax_);
}

void BiCGStabSolver::solve(double* x, const double* A, const double* b, const int32_t max_iteration, const double epsilon)
{
    mul_mv(size_, size_, Ax_, A, x);
    for(int32_t i = 0; i < size_; ++i) {
        r_[i] = b[i] - Ax_[i];
        p_[i] = r_[i];
        rr_[i] = r_[i];
    }
    double r0 = ::sqrt(dot(size_, r_, r_));
    r0 = r0 * r0 * epsilon * epsilon;
    for(int32_t count = 0; count < max_iteration; ++count) {
        mul_mv(size_, size_, Ap_, A, p_);
        double d0 = dot(size_, r_, rr_);
        double alpha = d0 / dot(size_, rr_, Ap_);
        for(int32_t i = 0; i < size_; ++i) {
            s_[i] = r_[i] - alpha * Ap_[i];
        }
        mul_mv(size_, size_, As_, A, s_);
        double w = dot(size_, As_, s_) / dot(size_, As_, As_);
        for(int32_t i = 0; i < size_; ++i) {
            x[i] += alpha * p_[i] + w * s_[i];
            r_[i] = s_[i] - w * As_[i];
        }
        double d1 = dot(size_, r_, r_);
        if(d1 <= r0) {
            break;
        }
        d1 = ::sqrt(d1);
        double beta = dot(size_, r_, rr_) / d1 * alpha * w;
        for(int32_t i = 0; i < size_; ++i) {
            p_[i] = r_[i] + beta * (p_[i] - w * Ap_[i]);
        }
    }
}

} // namespace cppraisr
