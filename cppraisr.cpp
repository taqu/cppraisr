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
#include <format>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>

namespace cppraisr
{
RAISRTrainer::RAISRTrainer()
    :solver_(RAISRParam::PatchSize*RAISRParam::PatchSize)
{
}

RAISRTrainer::~RAISRTrainer()
{
}

void RAISRTrainer::train(const std::vector<std::filesystem::path>& images)
{
    gaussian2d(RAISRParam::GradientSize, &weights_(0,0), 1.0);

    std::filesystem::path model_directory = std::filesystem::current_path();
    model_directory.append("model");
    if(!std::filesystem::exists(model_directory)){
        std::filesystem::create_directory(model_directory);
    }
    int32_t image_count = 0;
    int32_t operation_count = 0;
    for(size_t i = 0; i < images.size(); ++i) {
        Image<stbi_uc> original;
        std::u8string path = images[i].u8string();
        std::cout << "[" << std::setfill('0') << std::right << std::setw(4) << (image_count+1) << '/' << images.size() << "] " << (char*)path.c_str() << std::endl;
        {
            int32_t w=0,h=0,c=0;
            stbi_uc* pixels = stbi_load(reinterpret_cast<const char*>(path.c_str()), &w, &h, &c, STBI_default);
            if(nullptr == pixels){
                continue;
            }
            original.reset(w, h, c, pixels, stbi_image_free);
        }

        if(1 < original.c()) {
            Image<stbi_uc> grey(original.w(), original.h(), 1);
            if(!grey){
                continue;
            }
            for(int j = 0; j < original.h(); ++j) {
                for(int k = 0; k < original.w(); ++k) {
                    stbi_uc r = original(k, j, 0);
                    stbi_uc g = original(k, j, 1);
                    stbi_uc b = original(k, j, 2);
                    grey(k,j,0) = static_cast<stbi_uc>(0.183f * r + 0.614f * g + 0.062f * b + 16);
                }
            }
            original.swap(grey);
        }
        Image<stbi_uc> upscaledLR(original.w(), original.h(), 1);
        if(!upscaledLR){
            continue;
        }
        {
            Image<stbi_uc> tmp(original.w()>>1, original.h()>>1, 1);
            if(!tmp){
                continue;
            }
            int r = stbir_resize_uint8_generic(&original(0,0,0), original.w(), original.h(), original.w() * original.c() * sizeof(stbi_uc), &tmp(0,0,0), tmp.w(), tmp.h(), tmp.w() * sizeof(stbi_uc), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_BOX, STBIR_COLORSPACE_LINEAR, nullptr);
            if(!r) {
                continue;
            }

            r = stbir_resize_uint8_generic(&tmp(0,0,0), tmp.w(), tmp.h(), tmp.w() * tmp.c() * sizeof(stbi_uc), &upscaledLR(0,0,0), upscaledLR.w(), upscaledLR.h(), upscaledLR.w() * sizeof(stbi_uc), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_TRIANGLE, STBIR_COLORSPACE_LINEAR, nullptr);
            if(!r) {
                continue;
            }
        }
        operation_count += train_image(upscaledLR, original);
        {
            std::filesystem::path filepath = model_directory;
            filepath.append(std::format("filter_{0:06d}.bin", image_count));
            std::ofstream file(filepath.c_str(), std::ios::binary);
            if(file.is_open()) {
                H_.write(file);
            }
        }
        ++image_count;
    } // for(size_t i
}

int32_t RAISRTrainer::train_image(const Image<stbi_uc>& upscaledLR, const Image<stbi_uc>& original)
{
    int32_t patch_begin;
    int32_t patch_end;
    int32_t gradient_begin;
    int32_t gradient_end;
    int32_t margin;
    int32_t patch_size;
    {
        patch_size = std::max(RAISRParam::PatchSize, RAISRParam::GradientSize);
        if constexpr ((RAISRParam::PatchSize & 0x01U) == 0) {
            patch_begin = -(RAISRParam::PatchSize >> 1) + 1;
            patch_end = RAISRParam::PatchSize>>1;
        } else {
            patch_begin = -(RAISRParam::PatchSize >> 1);
            patch_end = RAISRParam::PatchSize>>1;
        }
        if constexpr ((RAISRParam::GradientSize & 0x01U) == 0) {
            gradient_begin = -(RAISRParam::GradientSize >> 1) + 1;
            gradient_end = RAISRParam::GradientSize>>1;
        } else {
            gradient_begin = -(RAISRParam::GradientSize >> 1);
            gradient_end = RAISRParam::GradientSize>>1;
        }
        margin = (patch_size >> 1);
    }

    int32_t half_patch_size = patch_size>>1;
    int32_t hend = upscaledLR.h() < margin ? 0 : upscaledLR.h() - margin;
    int32_t wend = upscaledLR.w() < margin ? 0 : upscaledLR.w() - margin;
    int32_t operation_count = 0;
    for(int32_t i = margin; i < hend; i+=half_patch_size) {
        for(int32_t j = margin; j < wend; j+=half_patch_size) {
            for(int32_t y = patch_begin; y <= patch_end; ++y) {
                for(int32_t x = patch_begin; x <= patch_end; ++x) {
                    patch_image_(x-patch_begin, y - patch_begin) = to_double(upscaledLR(j+x,i+y,0));
                }
            }
            for(int32_t y = gradient_begin; y <= gradient_end; ++y) {
                for(int32_t x = gradient_begin; x <= gradient_end; ++x) {
                    gradient_patch_(x-gradient_begin, y-gradient_begin) = to_double(upscaledLR(j+x,i+y,0));
                }
            }
            auto [angle, strength, coherence] = hashkey(RAISRParam::GradientSize, &gradient_patch_(0,0), &weights_(0,0), RAISRParam::Qangle);
            double pixelHR = to_double(original(j,i,0));
            int32_t pixeltype = ((i-margin) % RAISRParam::R) * RAISRParam::R + ((j-margin) % RAISRParam::R);
            mul_mm(RAISRParam::PatchSize*RAISRParam::PatchSize, 1, RAISRParam::PatchSize*RAISRParam::PatchSize, &ATA_(0,0), &patch_image_(0,0), &patch_image_(0,0));
            mul_v(RAISRParam::PatchSize*RAISRParam::PatchSize, &ATb_(0,0), &patch_image_(0,0), pixelHR);

            double* h = H_(angle, strength, coherence, pixeltype);
            solver_.solve(h, &ATA_(0,0), &ATb_(0,0));
            ++operation_count;
        } // int32_t j = margin
    }     // int32_t i = margin
    return operation_count;
}

} // namespace cppraisr
