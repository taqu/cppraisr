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
#include <algorithm>
#include <random>

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

     float dx(int32_t x, int32_t y, const double m[])
    {
         return m[y*3+x];
    }

 float dy(int32_t x, int32_t y, const double m[])
    {
     return m[y*3+x];
    }

    /**
     * @brief Calc convolutions of gradients
     */
    std::tuple<double, double> gradient(int32_t size, const double m[], const double w[])
    {
        double gx = 0;
        gx += (dx(0, 0, m) - dx(2, 0, m)) * 47.0;
        gx += (dx(0, 1, m) - dx(2, 1, m)) * 162.0;
        gx += (dx(0, 2, m) - dx(2, 2, m)) * 47.0;

        double gy = 0;
        gy += (dx(0, 0, m) - dx(0, 2, m)) * 47.0;
        gy += (dx(1, 0, m) - dx(1, 2, m)) * 162.0;
        gy += (dx(2, 0, m) - dx(2, 2, m)) * 47.0;
        return std::make_tuple(gx, gy);
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
    if(t < 0) {
        return 0;
    }
    if(256 <= t) {
        return 255;
    }
    return static_cast<uint8_t>(t);
}

std::vector<std::filesystem::path> parse_directory(const char* path, std::function<bool(const std::filesystem::directory_entry&)> predicate, bool shuffle)
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
    if(shuffle){
        std::random_device seed_gen;
        std::mt19937 engine(seed_gen());
        std::shuffle(files.begin(), files.end(), engine);
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

void box2d(int32_t size, double* w)
{
    for(int32_t i = 0; i < size; ++i) {
        for(int32_t j = 0; j < size; ++j) {
            w[size * i + j]  = 1.0;
        }
    }
}

void solv2x2(double evalues[2], double evectors[4], const double m[4])
{
    double b = m[0] + m[3];
    double c = m[0] * m[3] - m[1] * m[2];

    double descr = b * b - 4.0 * c;
    descr = descr < 0.0 ? 0.0 : sqrt(descr);
    evalues[0] = (b + descr) * 0.5;
    evalues[1] = (b - descr) * 0.5;

    if(1.0e-32 < abs(m[1])) {
        evectors[0] = m[1];
        evectors[1] = evalues[0] - m[0];
        //evectors[2] = m[1];
        //evectors[3] = evalues[1] - m[0];

    //} else if(1.0e-16 < abs(m[2])) {
    //    evectors[0] = evalues[0] - m[3];
    //    evectors[1] = m[2];
        //evectors[2] = evalues[1] - m[3];
        //evectors[3] = m[2];

    } else {
        evectors[0] = 1;
        evectors[1] = 0;
        //evectors[2] = 0;
        //evectors[3] = 1;
    }
}

void solv2x2d(double evalues[2], double evectors[4], const double m[4])
{
    // assert(is_equal(m[1], m[2]));
    double a = m[0];
    double b = m[1];
    double d = m[3];
    double det = std::sqrt((a-d)*(a-d) + 4.0*b*b);
    double L1 = (a+d+det)/2.0;
    double L2 = (a+d-det)/2.0;
    evalues[0] = L1;
    evalues[1] = L2;

    if(1.0e-32 < abs(m[1])) {
        evectors[0] = L1-m[3];
        evectors[1] = m[1];

    } else if(1.0e-32 < abs(m[3])) {
        evectors[0] = m[1];
        evectors[1] = L1 - m[0];

    } else {
        evectors[0] = 1;
        evectors[1] = 0;
    }
}

std::tuple<int32_t, int32_t, int32_t> hashkey(int32_t gradient_size, const double* gradient_patch, const double* weights, int32_t angles)
{
    assert(0 < gradient_size && gradient_size <= 7);
    assert(nullptr != gradient_patch);
    assert(nullptr != weights);
    assert(0 < angles);

    // Calc eigen values and eigen vectors of gradients
    auto [gx, gy] = gradient(gradient_size, gradient_patch, weights);

    // Calc angle, strength, coherence
    double theta = atan2(gy, gx) + std::numbers::pi_v<double>;
    const double lambda0 = abs(gx);
    const double lambda1 = abs(gy);
    double u;
    if(is_zero(lambda0) && is_zero(lambda1)) {
        u = 0.0;
    } else {
        u = abs(lambda0 - lambda1) / (lambda0 + lambda1);
        u /= 256.0;
    }
    double lamda = sqrt(gx*gx + gy*gy)/256.0;
    int32_t strength;
    if(lamda < 0.125) {
        strength = 0;
    } else if(0.25 < lamda) {
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
    coherence = 0;
    int32_t angle = static_cast<int32_t>(floor(theta / (2.0 * std::numbers::pi) * angles));
    if(angles <= angle) {
        angle = angles - 1;
    }
    return std::make_tuple(angle, strength, coherence);
}
} // namespace cppraisr
