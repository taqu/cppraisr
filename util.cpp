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
#include <numbers>

namespace cppraisr
{
namespace
{
    /**
     * @brief Check zero
     */
    bool is_zero(double x, double torelance = 1.0e-10)
    {
        return std::abs(x) < torelance;
    }

#if 0
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
        gy[1] = (m[6] - m[0]) * w[3] * 0.5;
        gy[2] = (m[6] - m[3]) * w[6];

        gy[3] = (m[4] - m[1]) * w[1];
        gy[4] = (m[7] - m[1]) * w[4] * 0.5;
        gy[5] = (m[7] - m[4]) * w[7];

        gy[6] = (m[5] - m[2]) * w[2];
        gy[7] = (m[8] - m[2]) * w[5] * 0.5;
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
        gy[1] = (m[8] - m[0]) * w[4] * 0.5;
        gy[2] = (m[8] - m[4]) * w[8] * 0.5;
        gy[3] = (m[12] - m[8]) * w[12];

        gy[4] = (m[5] - m[1]) * w[1];
        gy[5] = (m[9] - m[1]) * w[5] * 0.5;
        gy[6] = (m[9] - m[5]) * w[9] * 0.5;
        gy[7] = (m[13] - m[9]) * w[13];

        gy[8] = (m[6] - m[2]) * w[2];
        gy[9] = (m[10] - m[2]) * w[6] * 0.5;
        gy[10] = (m[10] - m[6]) * w[10] * 0.5;
        gy[11] = (m[14] - m[10]) * w[14];

        gy[12] = (m[7] - m[3]) * w[3];
        gy[13] = (m[11] - m[3]) * w[7] * 0.5;
        gy[14] = (m[11] - m[7]) * w[11] * 0.5;
        gy[15] = (m[15] - m[11]) * w[15];
    }
#endif

    /**
     * @brief Calc convolutions of gradients
     */
    std::tuple<double, double, double> gradient(int32_t size, const double m[], const double w[])
    {
        double gx = 0.0;
        double gy = 0.0;
        double gxy = 0.0;
        for(int32_t i = 0; i < size; ++i) {
            int32_t py = i - 1;
            int32_t ny = i + 1;
            double wy = 1.0;
            if(py < 0) {
                py = 0;
                wy = 1.0;
            }
            if(size <= ny) {
                ny = size - 1;
                wy = 1.0;
            }

            for(int32_t j = 0; j < size; ++j) {
                int32_t px = j - 1;
                int32_t nx = j + 1;
                double wx = 1.0;
                if(px < 0) {
                    px = 0;
                    wx = 1.0;
                }
                if(size <= nx) {
                    nx = size - 1;
                    wx = 1.0;
                }
                int32_t ic = i * size + j;
                double x = (m[i * size + nx] - m[i * size + px]);
                double y = (m[ny * size + j] - m[py * size + j]);
                gx += x * x * wx*w[ic];
                gy += y * y * wx*w[ic];
                gxy += x * y * wx*wy*w[ic];
            }
        }
        return std::make_tuple(gx, gy, gxy);
    }
#if 0
    void conv2(double g[4], int32_t size, const double gx[], const double gy[])
    {
        double gxgx = 0.0;
        double gxgy = 0.0;
        double gygy = 0.0;
        for(int32_t i = 0; i < size; ++i) {
            gxgx += gx[i] * gx[i];
            gxgy += gx[i] * gy[i];
            gygy += gy[i] * gy[i];
        }
        g[0] = gxgx;
        g[1] = gxgy;
        g[2] = gxgy;
        g[3] = gygy;
    }
#endif
} // namespace

double to_double(uint8_t x)
{
    return x / 255.0;
}

uint8_t to_uint8(double x)
{
    x *= 256.0;
    int32_t t = static_cast<int32_t>(x);
    if(t < 0) {
        return 0;
    }
    if(256 <= t) {
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
            if(files.capacity() <= files.size()) {
                files.reserve(files.size() + 2048);
            }
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
    double b = m[0] + m[3];
    double c = m[0] * m[3] - m[1] * m[2];

    double descr = b * b - 4.0 * c;
    descr = descr < 0.0 ? 0.0 : sqrt(descr);
    evalues[0] = (b + descr) * 0.5;
    evalues[1] = (b - descr) * 0.5;

    if(1.0e-16 < abs(m[1])) {
        evectors[0] = m[1];
        evectors[1] = evalues[0] - m[0];
        evectors[2] = m[1];
        evectors[3] = evalues[1] - m[0];

    } else if(1.0e-16 < abs(m[2])) {
        evectors[0] = evalues[0] - m[3];
        evectors[1] = m[2];
        evectors[2] = evalues[1] - m[3];
        evectors[3] = m[2];

    } else {
        evectors[0] = 1;
        evectors[1] = 0;
        evectors[2] = 0;
        evectors[3] = 1;
    }
}

std::tuple<int32_t, int32_t, int32_t> hashkey(int32_t gradient_size, const double* gradient_patch, const double* weights, int32_t angles)
{
    assert(0 < gradient_size && gradient_size <= 7);
    assert(nullptr != gradient_patch);
    assert(nullptr != weights);
    assert(0 < angles);

    // Calc eigen values and eigen vectors of gradients
    auto [gx, gy, gxy] = gradient(gradient_size, gradient_patch, weights);

    double g[4] = {gx, gxy, gxy, gy};
    double evalues[2];
    double evectors[4];
    solv2x2(evalues, evectors, g);

    // Calc angle, strength, coherence
    double theta = atan2(evectors[1], evectors[0]);
    while(theta < 0.0) {
        theta += std::numbers::pi_v<double>;
    }
    const double lambda0 = sqrt(evalues[0]);
    const double lambda1 = sqrt(evalues[1]);
    double u;
    if(is_zero(lambda0) && is_zero(lambda1)) {
        u = 0.0;
    } else {
        u = (lambda0 - lambda1) / (lambda0 + lambda1);
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
    int32_t angle = static_cast<int32_t>(floor(theta / (2.0 * std::numbers::pi) * angles));
    if(angles <= angle) {
        angle = angles - 1;
    }
    return std::make_tuple(angle, strength, coherence);
}
} // namespace cppraisr
