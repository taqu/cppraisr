#ifndef INC_CPPRAISR_H_
#define INC_CPPRAISR_H_
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
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>
#include <stb/stb_image.h>
#include "util.h"

namespace cppraisr
{

struct RAISRParam
{
    inline static constexpr uint8_t R = 2;
    inline static constexpr uint8_t PatchSize = 7;
    inline static constexpr uint8_t GradientSize = 5;
    inline static constexpr uint8_t Qangle = 24;
    inline static constexpr uint8_t Qstrength = 3;
    inline static constexpr uint8_t Qcoherence = 3;
};

using FilterSet = Array5d<double, RAISRParam::Qangle, RAISRParam::Qstrength, RAISRParam::Qcoherence, RAISRParam::R*RAISRParam::R, RAISRParam::PatchSize*RAISRParam::PatchSize>;

class RAISRTrainer
{
public:
    RAISRTrainer();
    ~RAISRTrainer();

    void train(const std::vector<std::filesystem::path>& images);

private:
    RAISRTrainer(const RAISRTrainer&) = delete;
    RAISRTrainer& operator=(const RAISRTrainer&) = delete;
    int32_t train_image(const Image<stbi_uc>& upscaledLR, const Image<stbi_uc>& original);

    ImageStatic<double, RAISRParam::PatchSize, RAISRParam::PatchSize> patch_image_;
    ImageStatic<double, RAISRParam::PatchSize, RAISRParam::PatchSize> patch_transposed_;
    ImageStatic<double, RAISRParam::GradientSize, RAISRParam::GradientSize> gradient_patch_;
    ImageStatic<double, RAISRParam::PatchSize*RAISRParam::PatchSize, RAISRParam::PatchSize*RAISRParam::PatchSize> ATA_;
    ImageStatic<double, RAISRParam::PatchSize, RAISRParam::PatchSize> ATb_;
    ImageStatic<double, RAISRParam::GradientSize, RAISRParam::GradientSize> weights_;

    FilterSet H_;
    BiCGStabSolver solver_;
};
} // namespace cppraisr

#endif //INC_CPPRAISR_H_

