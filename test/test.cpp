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
#include <filesystem>
#include <flags/include/flags.h>
#include <format>
#include <fstream>
#include <stb/stb_image.h>
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>

namespace
{
stbi_uc clamp(float x)
{
    if(x < 0.0f) {
        return 0;
    }
    if(256.0f <= x) {
        return 255;
    }
    return static_cast<stbi_uc>(x);
}

void rgb2ycbcr(cppraisr::Image<stbi_uc>& image)
{
    using namespace cppraisr;
    for(int32_t j = 0; j < image.h(); ++j) {
        for(int32_t k = 0; k < image.w(); ++k) {
            stbi_uc r = image(k, j, 0);
            stbi_uc g = image(k, j, 1);
            stbi_uc b = image(k, j, 2);
            stbi_uc y = clamp(0.183f * r + 0.614f * g + 0.062f * b + 16);
            stbi_uc cb = clamp(-0.101f * r - 0.339f * g + 0.439f * b + 128);
            stbi_uc cr = clamp(0.439f * r - 0.399f * g - 0.040f * b + 128);

            image(k, j, 0) = y;
            image(k, j, 1) = cb;
            image(k, j, 2) = cr;
        }
    }
}
void ycbcr2rgb(cppraisr::Image<stbi_uc>& image)
{
    using namespace cppraisr;
    for(int32_t j = 0; j < image.h(); ++j) {
        for(int32_t k = 0; k < image.w(); ++k) {
            int32_t y = image(k, j, 0);
            int32_t cb = image(k, j, 1);
            int32_t cr = image(k, j, 2);
            stbi_uc r = clamp(1.164f * (y - 16) + 1.793f * (cr - 128));
            stbi_uc g = clamp(1.164f * (y - 16) - 0.213f * (cb - 128) - 0.533f * (cr - 128));
            stbi_uc b = clamp(1.164f * (y - 16) + 2.112f * (cb - 128));

            image(k, j, 0) = r;
            image(k, j, 1) = g;
            image(k, j, 2) = b;
        }
    }
}

} // namespace

int32_t reflect(int32_t x, int32_t size)
{
    if(x < 0) {
        return -x;
    }
    if(size <= x) {
        return size - (x - size + 1);
    }
    return x;
}

void test(const std::vector<std::filesystem::path>& images, const cppraisr::RAISRTrainer::FilterSet& filters, int32_t max_images, bool measure_quality)
{
    using namespace cppraisr;
    ImageStatic<double, RAISRParam::PatchSize, RAISRParam::PatchSize> patch_image;
    ImageStatic<double, RAISRParam::GradientSize, RAISRParam::GradientSize> gradient_patch;
    ImageStatic<double, RAISRParam::GradientSize, RAISRParam::GradientSize> weights;

    gaussian2d(RAISRParam::GradientSize, &weights(0, 0), 2.0);
    std::filesystem::path result_directory = std::filesystem::current_path();
    result_directory.append("result");
    if(!std::filesystem::exists(result_directory)) {
        std::filesystem::create_directory(result_directory);
    }

    int32_t image_count = 0;
    uint8_t margin = RAISRParam::PatchSize >> 1;
    for(size_t count = 0; count < images.size(); ++count) {
        if(max_images <= count) {
            break;
        }
        Image<stbi_uc> original;
        std::u8string path = images[count].u8string();
        std::cout << "[" << std::setfill('0') << std::right << std::setw(4) << (image_count + 1) << '/' << std::setfill('0') << std::right << std::setw(4) << images.size() << "] " << (char*)path.c_str() << std::endl;
        {
            int32_t w = 0, h = 0, c = 0;
            stbi_uc* pixels = stbi_load(reinterpret_cast<const char*>(path.c_str()), &w, &h, &c, STBI_default);
            if(nullptr == pixels) {
                continue;
            }
            original.reset(w, h, c, pixels, stbi_image_free);
        }
        rgb2ycbcr(original);

        int32_t dw, dh;
        if(measure_quality) {
            dw = original.w();
            dh = original.h();
        } else {
            dw = original.w() >> 1;
            dh = original.h() >> 1;
        }
        Image<stbi_uc> upscaled(dw, dh, original.c());
        {
            if(measure_quality) {
                Image<stbi_uc> tmp(original.w() >> 1, original.h() >> 1, original.c());
                int r = stbir_resize_uint8_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(stbi_uc), &tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * sizeof(stbi_uc) * tmp.c(), tmp.c(), 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_BOX, STBIR_COLORSPACE_LINEAR, nullptr);
                if(!r) {
                    continue;
                }

                r = stbir_resize_uint8_generic(&tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * tmp.c() * sizeof(stbi_uc), &upscaled(0, 0, 0), upscaled.w(), upscaled.h(), upscaled.w() * sizeof(stbi_uc) * upscaled.c(), upscaled.c(), 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_TRIANGLE, STBIR_COLORSPACE_LINEAR, nullptr);
                if(!r) {
                    continue;
                }

            } else {
                int r = stbir_resize_uint8_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(stbi_uc), &upscaled(0, 0, 0), upscaled.w(), upscaled.h(), upscaled.w() * sizeof(stbi_uc) * upscaled.c(), upscaled.c(), 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_TRIANGLE, STBIR_COLORSPACE_LINEAR, nullptr);
                if(!r) {
                    continue;
                }
            }
        }

        int32_t half_patch_size = RAISRParam::PatchSize >> 1;
        int32_t half_gradient_size = RAISRParam::GradientSize >> 1;
        for(int32_t i = 0; i < upscaled.h(); ++i) {
            for(int32_t j = 0; j < upscaled.w(); ++j) {
                for(int32_t y = -half_patch_size; y <= half_patch_size; ++y) {
                    for(int32_t x = -half_patch_size; x <= half_patch_size; ++x) {
                        int32_t tx = reflect(x + j, upscaled.w());
                        int32_t ty = reflect(y + i, upscaled.h());
                        patch_image(x + half_patch_size, y + half_patch_size) = to_double(upscaled(tx, ty, 0));
                    }
                }
                for(int32_t y = -half_gradient_size; y <= half_gradient_size; ++y) {
                    for(int32_t x = -half_gradient_size; x <= half_gradient_size; ++x) {
                        int32_t tx = reflect(x + j, upscaled.w());
                        int32_t ty = reflect(y + i, upscaled.h());
                        gradient_patch(x + half_gradient_size, y + half_gradient_size) = to_double(upscaled(tx, ty, 0));
                    }
                }
                auto [angle, strength, coherence] = hashkey(RAISRParam::GradientSize, &gradient_patch(0, 0), &weights(0, 0), RAISRParam::Qangle);
                int32_t pixeltype = (i % RAISRParam::R) * RAISRParam::R + (j % RAISRParam::R);
                const RAISRTrainer::VectorParamSize2& h = *filters(angle, strength, coherence, pixeltype);
                double pixelHR = 0.0;
                for(int32_t y = 0; y < RAISRParam::PatchSize; ++y) {
                    for(int32_t x = 0; x < RAISRParam::PatchSize; ++x) {
                        pixelHR += patch_image(x, y) * h(y, x);
                    }
                }
                upscaled(j, i, 0) = to_uint8(pixelHR);
            } // int32_t j = margin
        }     // int32_t i = margin

        {
            std::filesystem::path filepath = result_directory;
            std::filesystem::path filename = images[count].filename();
            filename.replace_extension("png");
            filepath.append(filename.c_str());
            ycbcr2rgb(upscaled);
            stbi_write_png((const char*)filepath.u8string().c_str(), upscaled.w(), upscaled.h(), upscaled.c(), &upscaled(0, 0, 0), sizeof(stbi_uc) * upscaled.w() * upscaled.c());
        }

        ++image_count;
    } // for(size_t i
}

void print_help()
{
}

int main(int argc, char** argv)
{
    using namespace cppraisr;
    RAISRTrainer::FilterSet filters;
    const flags::args args(argc, argv);
    if(args.get<bool>("help", false)) {
        print_help();
        return 0;
    }

    {

        std::string filter("filter.bin");
        std::optional<std::string> f = args.get<std::string>("f");
        if(f) {
            filter = f.value();
        }
        std::filesystem::path filepath = std::filesystem::current_path();
        filepath.append(filter);
        std::ifstream file(filepath.c_str(), std::ios::binary);
        if(!file.is_open()) {
            return 0;
        }
        filters.read_matrix(file);
    }
    int32_t max_images = 1000000;
    bool measure_quality = false;
    {
        std::optional<int32_t> option = args.get<int32_t>("max");
        if(option) {
            max_images = option.value();
        }
        if(args.get<bool>("q", false)) {
            measure_quality = true;
        }
    }

    std::vector<std::filesystem::path> files = parse_directory("test_data",
                                                               [](const std::filesystem::directory_entry& entry) {
                                                                   std::filesystem::path extension = entry.path().extension();
                                                                   if(extension == ".jpg") {
                                                                       return true;
                                                                   }
                                                                   if(extension == ".png") {
                                                                       return true;
                                                                   }
                                                                   return false;
                                                               });
    test(files, filters, max_images, measure_quality);
    return 0;
}
