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
#include <mutex>
#include <optional>
#include <stb/stb_image.h>
#include <tuple>
#include <vector>

namespace cppraisr
{

struct RAISRParam
{
    inline static constexpr uint8_t R = 1; //!< Number of pixel patterns for one axis
    inline static constexpr uint8_t R2 = R * R; //!< Number of pixel patterns
    inline static constexpr uint8_t PatchSize = 5; //! << Size of patch image
    inline static constexpr uint8_t PatchSize2 = PatchSize * PatchSize;
    inline static constexpr int32_t PatchSize4 = PatchSize2 * PatchSize2;
    inline static constexpr uint8_t GradientSize = 5; //!< Size of the area on witch calculate gradients
    inline static constexpr uint8_t Qangle = 24; //!< Resolution of angle patterns
    inline static constexpr uint8_t Qstrength = 3; //!< Resolution of strength
    inline static constexpr uint8_t Qcoherence = 3; //!< Resolution of coherence
    inline static constexpr double Sigma = 1.414; //!< Sigma of gaussian for weights matrix
};

/**
 *
 */
class FilterSet
{
public:
    using FilterType = Eigen::Matrix<double, RAISRParam::PatchSize2, 1>;
    inline static constexpr int32_t Count = RAISRParam::Qangle*RAISRParam::Qstrength*RAISRParam::Qcoherence*RAISRParam::R;

    FilterSet();
    ~FilterSet();
    const FilterType& operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type) const;
    FilterType& operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type);
    void write(std::ostream& os);
    void read(std::istream& is);
private:
    FilterSet(const FilterSet&) = delete;
    FilterSet& operator=(const FilterSet&) = delete;
    FilterType* filters_;
};

class MatrixSet
{
public:
    using MatrixType = Eigen::Matrix<double, RAISRParam::PatchSize2, RAISRParam::PatchSize2>;
    inline static constexpr int32_t Count = RAISRParam::Qangle*RAISRParam::Qstrength*RAISRParam::Qcoherence*RAISRParam::R;

    MatrixSet();
    ~MatrixSet();
    const MatrixType& operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type) const;
    MatrixType& operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type);
private:
    MatrixSet(const MatrixSet&) = delete;
    MatrixSet& operator=(const MatrixSet&) = delete;
    MatrixType* matrices_;
};

template<class T>
class MapTemplate
{
public:
    inline static constexpr int32_t Count = RAISRParam::Qangle*RAISRParam::Qstrength*RAISRParam::Qcoherence*RAISRParam::R;

    MapTemplate();
    ~MapTemplate();
    const T& operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type) const;
    T& operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type);
private:
    MapTemplate(const MapTemplate&) = delete;
    MapTemplate& operator=(const MapTemplate&) = delete;
    T* values_;
};

template<class T>
MapTemplate<T>::MapTemplate()
    : values_(nullptr)
{
    values_ = new T[Count];
    for(int32_t i = 0; i < Count; ++i) {
        values_[i] = 0;
    }
}

template<class T>
MapTemplate<T>::~MapTemplate()
{
    delete[] values_;
    values_ = nullptr;
}

template<class T>
const T& MapTemplate<T>::operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type) const
{
    int32_t index = ((angle * RAISRParam::Qstrength + strength) * RAISRParam::Qcoherence + coherence) * RAISRParam::R + pixel_type;
    return values_[index];
}

template<class T>
T& MapTemplate<T>::operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type)
{
    int32_t index = ((angle * RAISRParam::Qstrength + strength) * RAISRParam::Qcoherence + coherence) * RAISRParam::R + pixel_type;
    return values_[index];
}

class RAISRTrainer
{
public:
    inline static constexpr int32_t CheckStep = 1000;
    inline static constexpr int32_t MaxSum = 1000000;

    RAISRTrainer();
    ~RAISRTrainer();

    void train(
        const std::vector<std::filesystem::path>& images,
        const std::filesystem::path& path_q,
        const std::filesystem::path& path_v,
        const std::filesystem::path& path_o,
        int32_t max_images = 2147483647);

private:
    RAISRTrainer(const RAISRTrainer&) = delete;
    RAISRTrainer& operator=(const RAISRTrainer&) = delete;

    bool train(const std::filesystem::path& path);
    void train_image(const Image<stbi_uc>& upscaledLR, const Image<stbi_uc>& original);
    void copy_examples();
    bool solve();

    int32_t max_images_;
    int32_t check_count_;
    std::filesystem::path current_;
    std::filesystem::path model_name_;
    FilterSet V_;
    MatrixSet Q_;
    FilterSet H_;
    MapTemplate<bool> Checks_;
    MapTemplate<int32_t> Counts_;
    double weights_[RAISRParam::GradientSize*RAISRParam::GradientSize];
    double patch_image_[RAISRParam::PatchSize*RAISRParam::PatchSize];
    double gradient_image_[RAISRParam::GradientSize*RAISRParam::GradientSize];
};
} // namespace cppraisr

#endif // INC_CPPRAISR_H_
