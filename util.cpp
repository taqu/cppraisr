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
    float dx(int32_t x, int32_t y, const double m[])
    {
         return static_cast<float>(m[y*3+x]);
    }


    /**
     * @brief Calc convolutions of gradients
     */
    std::tuple<float, float> gradient(const double m[])
    {
        float gx = 0;
        gx += (dx(0, 0, m) - dx(2, 0, m)) * 47.0f;
        gx += (dx(0, 1, m) - dx(2, 1, m)) * 162.0f;
        gx += (dx(0, 2, m) - dx(2, 2, m)) * 47.0f;

        float gy = 0;
        gy += (dx(0, 0, m) - dx(0, 2, m)) * 47.0f;
        gy += (dx(1, 0, m) - dx(1, 2, m)) * 162.0f;
        gy += (dx(2, 0, m) - dx(2, 2, m)) * 47.0f;
        return std::make_tuple(static_cast<float>(gx), static_cast<float>(gy));
    }

#define PI (3.14159265359f)
#define PI2 (1.5707963268f)
    float atan2_fast(float y, float x)
    {
        if(x == 0.0f) {
            if(0.0f < y) {
                return PI2;
            }
            if(y == 0.0f) {
                return 0.0f;
            }
            return -PI2;
        }
        float atan;
        float z = y / x;
        if(abs(z) < 1.0f) {
            atan = z / (1.0f + 0.28f * z * z);
            if(x < 0.0f) {
                if(y < 0.0f) {
                    return atan - PI;
                }
                return atan + PI;
            }
        } else {
            atan = PI2 - z / (z * z + 0.28f);
            if(y < 0.0f) {
                return atan - PI;
            }
        }
        return atan;
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

namespace
{
    int32_t mirror(int32_t size, int32_t offset, int32_t x)
    {
        x += offset;
        if(x<0){
            x = std::min(-x, size-1);
        }else if(size<=x){
            x = std::max(size - (x-size)-1, 0);
        }
        return x;
    }

    float conv(int32_t width, int32_t height, int32_t x, int32_t y, const float* src, int32_t size, const float* weights)
    {
        int32_t half = size / 2;
        float total = 0.0f;
        for(int32_t i = -half; i <= half; ++i) {
            int32_t ty = mirror(height, i, y);
            for(int32_t j = -half; j <= half; ++j) {
                int32_t tx = mirror(width, j, x);
                float x = src[ty * width + tx];
                total += x * weights[(i + half) * size + (j + half)];
            }
        }
        return total;
    }
} // namespace

void conv2d(int32_t width, int32_t height, float* dst, const float* src, int32_t size, const float* weights)
{
    for(int32_t i = 0; i < height; ++i) {
        for(int32_t j = 0; j < width; ++j) {
            int32_t index = i * width + j;
            dst[index] = conv(width, height, j, i, src, size, weights);
        }
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

std::tuple<int32_t, int32_t, int32_t> hashkey(const double* gradient_patch, int32_t angles)
{
    assert(nullptr != gradient_patch);
    assert(0 < angles);

    // Calc eigen values and eigen vectors of gradients
    auto [gx, gy] = gradient(gradient_patch);

    // Calc angle, strength, coherence
    float theta = std::max(atan2_fast(gy, gx) + PI,0.0f);
    //const float lambda0 = abs(gx);
    //const float lambda1 = abs(gy);
    //float u;
    //if(is_zero(lambda0) && is_zero(lambda1)) {
    //    u = 0.0;
    //} else {
    //    u = abs(lambda0 - lambda1) / (lambda0 + lambda1);
    //    u *= (100.0f/256.0f);
    //}
    float lamda = sqrt(gx*gx + gy*gy)*(1.0f/256.0f);
    int32_t strength;
    if(lamda < 0.125f) {
        strength = 0;
    } else if(0.25f < lamda) {
        strength = 2;
    } else {
        strength = 1;
    }

    int32_t coherence;
    //if(u < 0.0003f) {
    //    coherence = 0;
    //} else if(0.0006f < u) {
    //    coherence = 2;
    //} else {
    //    coherence = 1;
    //}
    coherence = 0;
    int32_t angle = static_cast<int32_t>(floor(theta / (2.0f * PI) * angles));
    if(angles <= angle) {
        angle = angles - 1;
    }
    return std::make_tuple(angle, strength, coherence);
}

} // namespace cppraisr
