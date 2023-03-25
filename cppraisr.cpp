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
#include "cppraisr.h"
#include "util.h"
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>

namespace cppraisr
{
RAISRTrainer::RAISRTrainer()
{
}

RAISRTrainer::~RAISRTrainer()
{
}

void RAISRTrainer::train(const std::vector<std::filesystem::path>& images, int32_t max_images)
{
    count_ = 0;
    Q_.clear();
    V_.clear();
    H_.clear();
    gaussian2d(RAISRParam::GradientSize, &weights_(0, 0), 2.0);

    std::filesystem::path model_directory = std::filesystem::current_path();
    model_directory.append("model");
    if(!std::filesystem::exists(model_directory)) {
        std::filesystem::create_directory(model_directory);
    }
    int32_t image_count = 0;
    for(size_t i = 0; i < images.size(); ++i) {
        if(max_images <= image_count) {
            break;
        }
        Image<stbi_uc> original;
        std::u8string path = images[i].u8string();
        std::cout << "[" << std::setfill('0') << std::right << std::setw(4) << (image_count + 1) << '/' << images.size() << "] " << (char*)path.c_str() << std::endl;
        {
            int32_t w = 0, h = 0, c = 0;
            stbi_uc* pixels = stbi_load(reinterpret_cast<const char*>(path.c_str()), &w, &h, &c, STBI_default);
            if(nullptr == pixels) {
                continue;
            }
            original.reset(w, h, c, pixels, stbi_image_free);
        }

        if(1 < original.c()) {
            Image<stbi_uc> grey(original.w(), original.h(), 1);
            if(!grey) {
                continue;
            }
            for(int j = 0; j < original.h(); ++j) {
                for(int k = 0; k < original.w(); ++k) {
                    stbi_uc r = original(k, j, 0);
                    stbi_uc g = original(k, j, 1);
                    stbi_uc b = original(k, j, 2);
                    grey(k, j, 0) = static_cast<stbi_uc>(0.183f * r + 0.614f * g + 0.062f * b + 16);
                }
            }
            original.swap(grey);
        }
        Image<stbi_uc> upscaledLR(original.w(), original.h(), 1);
        if(!upscaledLR) {
            continue;
        }
        {
            Image<stbi_uc> tmp(original.w() >> 1, original.h() >> 1, 1);
            if(!tmp) {
                continue;
            }
            int r = stbir_resize_uint8_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(stbi_uc), &tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * sizeof(stbi_uc), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_BOX, STBIR_COLORSPACE_LINEAR, nullptr);
            if(!r) {
                continue;
            }

            r = stbir_resize_uint8_generic(&tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * tmp.c() * sizeof(stbi_uc), &upscaledLR(0, 0, 0), upscaledLR.w(), upscaledLR.h(), upscaledLR.w() * sizeof(stbi_uc), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_TRIANGLE, STBIR_COLORSPACE_LINEAR, nullptr);
            if(!r) {
                continue;
            }
        }
        train_image(upscaledLR, original);
        ++image_count;
    } // for(size_t i

    copy_examples();
    solve(model_directory);
}

void RAISRTrainer::train_image(const Image<stbi_uc>& upscaledLR, const Image<stbi_uc>& original)
{
    patch_image_.clear();
    gradient_patch_.clear();
    ATA_.clear();
    ATb_.clear();
    int32_t patch_begin;
    int32_t patch_end;
    int32_t gradient_begin;
    int32_t gradient_end;
    int32_t margin;
    int32_t patch_size;
    {
        patch_size = std::max(RAISRParam::PatchSize, RAISRParam::GradientSize);
        if constexpr((RAISRParam::PatchSize & 0x01U) == 0) {
            patch_begin = -(RAISRParam::PatchSize >> 1) + 1;
            patch_end = RAISRParam::PatchSize >> 1;
        } else {
            patch_begin = -(RAISRParam::PatchSize >> 1);
            patch_end = RAISRParam::PatchSize >> 1;
        }
        if constexpr((RAISRParam::GradientSize & 0x01U) == 0) {
            gradient_begin = -(RAISRParam::GradientSize >> 1) + 1;
            gradient_end = RAISRParam::GradientSize >> 1;
        } else {
            gradient_begin = -(RAISRParam::GradientSize >> 1);
            gradient_end = RAISRParam::GradientSize >> 1;
        }
        margin = (patch_size >> 1);
    }

    int32_t hend = upscaledLR.h() < margin ? 0 : upscaledLR.h() - margin;
    int32_t wend = upscaledLR.w() < margin ? 0 : upscaledLR.w() - margin;
    for(int32_t i = margin; i < hend; ++i) {
        for(int32_t j = margin; j < wend; ++j) {
            for(int32_t y = patch_begin; y <= patch_end; ++y) {
                for(int32_t x = patch_begin; x <= patch_end; ++x) {
                    patch_image_(x - patch_begin, y - patch_begin) = to_double(upscaledLR(j + x, i + y, 0));
                }
            }
            for(int32_t y = gradient_begin; y <= gradient_end; ++y) {
                for(int32_t x = gradient_begin; x <= gradient_end; ++x) {
                    gradient_patch_(x - gradient_begin, y - gradient_begin) = to_double(upscaledLR(j + x, i + y, 0));
                }
            }
            auto [angle, strength, coherence] = hashkey(RAISRParam::GradientSize, &gradient_patch_(0, 0), &weights_(0, 0), RAISRParam::Qangle);
            double pixelHR = to_double(original(j, i, 0));
            int32_t pixeltype = ((i - margin) % RAISRParam::R) * RAISRParam::R + ((j - margin) % RAISRParam::R);
            mul_mm(RAISRParam::PatchSize2, 1, &ATA_(0, 0), &patch_image_(0, 0), &patch_image_(0, 0));
            mul_v(RAISRParam::PatchSize2, &ATb_(0, 0), &patch_image_(0, 0), pixelHR);

            add(RAISRParam::PatchSize4, Q_(angle, strength, coherence, pixeltype), &ATA_(0, 0));
            add(RAISRParam::PatchSize2, V_(angle, strength, coherence, pixeltype), &ATb_(0, 0));
            ++count_;
        } // int32_t j = margin
    }     // int32_t i = margin
}

void RAISRTrainer::copy_examples()
{
    double* P = static_cast<double*>(::malloc(sizeof(double) * 7 * RAISRParam::PatchSize4));
    double* rotate = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize4));
    double* flip = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize4));
    double* trotate = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize4));
    double* tflip = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize4));

    ::memset(P, 0, 7 * sizeof(double) * RAISRParam::PatchSize4);
    ::memset(rotate, 0, sizeof(double) * RAISRParam::PatchSize4);
    ::memset(flip, 0, sizeof(double) * RAISRParam::PatchSize4);
    ::memset(trotate, 0, sizeof(double) * RAISRParam::PatchSize4);
    ::memset(tflip, 0, sizeof(double) * RAISRParam::PatchSize4);

    for(int32_t i = 0; i < (RAISRParam::PatchSize2); ++i) {
        int32_t i0 = i % RAISRParam::PatchSize;
        int32_t i1 = static_cast<int32_t>(floor(i / RAISRParam::PatchSize));
        int32_t j = RAISRParam::PatchSize2 - RAISRParam::PatchSize + i1 - RAISRParam::PatchSize * i0;
        rotate[j * RAISRParam::PatchSize2 + i] = 1;
        int32_t k = RAISRParam::PatchSize * (i1 + 1) - i0 - 1;
        flip[k * RAISRParam::PatchSize2 + i] = 1;
    }
    for(int32_t i = 1; i < 8; ++i) {
        int32_t i0 = i % 4;
        int32_t i1 = i >> 2;

        power_m(RAISRParam::PatchSize2, tflip, flip, i1);
        power_m(RAISRParam::PatchSize2, trotate, rotate, i0);
        double* p = P + (i - 1) * RAISRParam::PatchSize4;
        mul_mm(RAISRParam::PatchSize2, p, tflip, trotate);
    }
    ConjugateSet QExt;
    FilterSet VExt;
    QExt.clear();
    VExt.clear();

    double* Pt = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize4));
    double* newQ0 = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize4));
    double* newQ1 = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize4));
    double* newV = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize2));

    #if 0
    for(int32_t pixeltype = 0; pixeltype < (RAISRParam::R * RAISRParam::R); ++pixeltype) {
        for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
            for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
                for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
                    for(int32_t m = 1; m < 8; ++m) {
                        int32_t m0 = m % 4;
                        int32_t m1 = m >> 2;
                        int32_t newangle = angle;
                        if(m1 == 1) {
                            newangle = RAISRParam::Qangle - angle - 1;
                        }
                        newangle = newangle - RAISRParam::Qangle / 2 * m0;
                        while(newangle < 0) {
                            newangle += RAISRParam::Qangle;
                        }
                        const double* p = P + (m - 1) * RAISRParam::PatchSize4;
                        transpose(RAISRParam::PatchSize2, Pt, p);
                        const double* Q = Q_(angle, strength, coherence, pixeltype);
                        const double* V = V_(angle, strength, coherence, pixeltype);

                        double* q = QExt(angle, strength, coherence, pixeltype);
                        double* v = VExt(angle, strength, coherence, pixeltype);
                        mul_mm(RAISRParam::PatchSize2, newQ0, Pt, Q);
                        mul_mm(RAISRParam::PatchSize2, newQ1, newQ0, p);
                        mul_mm(RAISRParam::PatchSize2, newV, Pt, V);
                        add(RAISRParam::PatchSize4, q, newQ1);
                        add(RAISRParam::PatchSize2, v, newV);
                    }
                }
            }
        }
    }
    #endif
    add(Q_.count(), &Q_(0, 0, 0, 0, 0), &QExt(0, 0, 0, 0, 0));
    add(V_.count(), &V_(0, 0, 0, 0, 0), &VExt(0, 0, 0, 0, 0));

    ::free(newV);
    ::free(newQ1);
    ::free(newQ0);
    ::free(Pt);

    ::free(tflip);
    ::free(trotate);
    ::free(flip);
    ::free(rotate);
    ::free(P);
}

void RAISRTrainer::solve(const std::filesystem::path& model_directory)
{
    for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
        for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
            for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
                for(int32_t pixeltype = 0; pixeltype < RAISRParam::R2; ++pixeltype) {
                    CGLSSolver::solve(RAISRParam::PatchSize2, H_(angle, strength, coherence, pixeltype), Q_(angle, strength, coherence, pixeltype), V_(angle, strength, coherence, pixeltype));
                }
            }
        }
    }

    {
        time_t t = std::time(nullptr);
#ifdef _MSC_VER
        struct tm now;
        localtime_s(&now, &t);
#else
        struct tm now = *std::localtime(&t);
#endif
        now.tm_year += 1900;
        std::filesystem::path filepath = model_directory;
        filepath.append(std::format("filter_{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}.bin",
                                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min));
        std::ofstream file(filepath.c_str(), std::ios::binary);
        if(file.is_open()) {
            H_.write(file);
        }
    }
}
} // namespace cppraisr
