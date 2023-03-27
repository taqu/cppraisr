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
#include "util.h"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stb/stb_image.h>
#include <thread>
#include <tuple>
#include <vector>

namespace cppraisr
{

struct RAISRParam
{
    inline static constexpr uint8_t R = 1;
    inline static constexpr uint8_t R2 = R * R;
    inline static constexpr uint8_t PatchSize = 7;
    inline static constexpr uint8_t PatchSize2 = PatchSize * PatchSize;
    inline static constexpr int32_t PatchSize4 = PatchSize2 * PatchSize2;
    inline static constexpr uint8_t GradientSize = 5;
    inline static constexpr uint8_t Qangle = 24;
    inline static constexpr uint8_t Qstrength = 3;
    inline static constexpr uint8_t Qcoherence = 3;
};

class RAISRTrainer
{
public:
    using VectorParamSize2 = Eigen::Matrix<double, RAISRParam::PatchSize2, 1>;
    using MatrixParamSize2 = Eigen::Matrix<double, RAISRParam::PatchSize2, RAISRParam::PatchSize2>;

    using FilterSet = Array5d<VectorParamSize2, RAISRParam::Qangle, RAISRParam::Qstrength, RAISRParam::Qcoherence, RAISRParam::R2, 1>;
    using ConjugateSet = Array5d<MatrixParamSize2, RAISRParam::Qangle, RAISRParam::Qstrength, RAISRParam::Qcoherence, RAISRParam::R2, 1>;

    RAISRTrainer();
    ~RAISRTrainer();

    void train(const std::vector<std::filesystem::path>& images, int32_t num_threads = 4, int32_t max_images = 2147483647);

private:
    RAISRTrainer(const RAISRTrainer&) = delete;
    RAISRTrainer& operator=(const RAISRTrainer&) = delete;

    using ImagePatch = ImageStatic<double, RAISRParam::PatchSize, RAISRParam::PatchSize>;
    using GradientPatch = ImageStatic<double, RAISRParam::GradientSize, RAISRParam::GradientSize>;
    using ConjugatePatch = ImageStatic<double, RAISRParam::PatchSize2, RAISRParam::PatchSize2>;

    struct SharedContext
    {
        std::optional<std::tuple<std::filesystem::path, int32_t>> next();
        void inject(int32_t count, const ConjugateSet& Q, const FilterSet& V);

        const std::vector<std::filesystem::path>* files_ = nullptr;
        size_t max_images_ = 0;
        size_t count_ = 0;
        size_t total_operations_ = 0;
        bool output_checkpoints_ = false;
        size_t checkpoint_cycle_ = 100;
        size_t checkpoint_count_ = 0;
        std::mutex mutex_;

        GradientPatch weights_;
        ConjugateSet Q_;
        FilterSet V_;
        FilterSet H_;
        std::filesystem::path model_directory_;
    };

    struct LocalContext
    {
        int32_t image_count_;
        ImagePatch patch_image_;
        GradientPatch gradient_patch_;
        ConjugateSet Q_;
        FilterSet V_;
    };

    static void train_thread(SharedContext& shared);
    static void train_image(const Image<stbi_uc>& upscaledLR, const Image<stbi_uc>& original, LocalContext& context, SharedContext& shared);
    void copy_examples(SharedContext& shared);
    void solve(SharedContext& shared);
};
} // namespace cppraisr

#endif // INC_CPPRAISR_H_
