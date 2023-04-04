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
#include <eigen/unsupported/Eigen/MatrixFunctions>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>

namespace cppraisr
{

FilterSet::FilterSet()
    : filters_(nullptr)
{
    filters_ = new FilterType[Count];
    for(int32_t i = 0; i < Count; ++i) {
        filters_[i].setZero();
    }
}

FilterSet::~FilterSet()
{
    delete[] filters_;
    filters_ = nullptr;
}

const FilterSet::FilterType& FilterSet::operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type) const
{
    int32_t index = ((angle * RAISRParam::Qstrength + strength) * RAISRParam::Qcoherence + coherence) * RAISRParam::R + pixel_type;
    return filters_[index];
}

FilterSet::FilterType& FilterSet::operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type)
{
    int32_t index = ((angle * RAISRParam::Qstrength + strength) * RAISRParam::Qcoherence + coherence) * RAISRParam::R + pixel_type;
    return filters_[index];
}

void FilterSet::write(std::ostream& os)
{
    for(int32_t i = 0; i < Count; ++i) {
        for(int32_t r = 0; r < filters_[i].rows(); ++r) {
            for(int32_t c = 0; c < filters_[i].cols(); ++c) {
                double x = filters_[i](r, c);
                os.write((const char*)&x, sizeof(x));
            }
        }
    }
}

void FilterSet::read(std::istream& is)
{
    for(int32_t i = 0; i < Count; ++i) {
        for(int32_t r = 0; r < filters_[i].rows(); ++r) {
            for(int32_t c = 0; c < filters_[i].cols(); ++c) {
                double x = 0;
                is.read((char*)&x, sizeof(x));
                filters_[i](r, c) = x;
            }
        }
    }
}

MatrixSet::MatrixSet()
    : matrices_(nullptr)
{
    matrices_ = new MatrixType[Count];
    for(int32_t i = 0; i < Count; ++i) {
        matrices_[i].setZero();
    }
}

MatrixSet::~MatrixSet()
{
    delete[] matrices_;
    matrices_ = nullptr;
}

const MatrixSet::MatrixType& MatrixSet::operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type) const
{
    int32_t index = ((angle * RAISRParam::Qstrength + strength) * RAISRParam::Qcoherence + coherence) * RAISRParam::R + pixel_type;
    return matrices_[index];
}

MatrixSet::MatrixType& MatrixSet::operator()(int32_t angle, int32_t strength, int32_t coherence, int32_t pixel_type)
{
    int32_t index = ((angle * RAISRParam::Qstrength + strength) * RAISRParam::Qcoherence + coherence) * RAISRParam::R + pixel_type;
    return matrices_[index];
}

RAISRTrainer::RAISRTrainer()
    : max_images_(0)
{
}

RAISRTrainer::~RAISRTrainer()
{
}

void RAISRTrainer::train(
    const std::vector<std::filesystem::path>& images,
    const std::filesystem::path& path_q,
    const std::filesystem::path& path_v,
    const std::filesystem::path& path_o,
    int32_t max_images)
{
    current_ = std::filesystem::current_path();
    max_images_ = std::min(max_images, static_cast<int32_t>(images.size()));
    gaussian2d(RAISRParam::GradientSize, weights_, RAISRParam::Sigma);

    if(!path_q.empty()) {
        std::ifstream file(path_q.c_str(), std::ios::binary);
        if(!file.is_open()) {
            return;
        }
    }
    if(!path_v.empty()) {
        std::ifstream file(path_v.c_str(), std::ios::binary);
        if(!file.is_open()) {
            return;
        }
    }

    if(!path_o.empty()) {
        model_name_ = path_o;
    } else {
        model_name_ = std::filesystem::current_path();
        model_name_.append("filters");
        time_t t = std::time(nullptr);
#ifdef _MSC_VER
        struct tm now;
        localtime_s(&now, &t);
#else
        struct tm now = *std::localtime(&t);
#endif
        now.tm_year += 1900;
        model_name_.append(std::format("filter_{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}",
                                       now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec));
        std::filesystem::path directory = model_name_.parent_path();
        if(!std::filesystem::exists(directory)) {
            std::filesystem::create_directory(directory);
        }
    }

    for(size_t i = 0; i < max_images_; ++i) {
        std::u8string path = images[i].u8string();
        std::cout << "[" << std::setfill('0') << std::right << std::setw(4) << (i + 1) << '/' << std::setfill('0') << std::right << std::setw(4) << images.size() << "] " << (char*)path.c_str() << std::endl;
        train(images[i]);
    }

    solve();
    {
        std::string filepath = model_name_.string();
        filepath.append(".bin");
        std::ofstream file(filepath.c_str(), std::ios::binary);
        if(file.is_open()) {
            H_.write(file);
            file.close();
        }

        filepath = model_name_.string();
        filepath.append(".json");
        file.open(filepath.c_str(), std::ios::binary);
        if(file.is_open()) {
            file << "{\"filter\":[\n";
            for(int32_t pixeltype = 0; pixeltype < RAISRParam::R2; ++pixeltype) {
                for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
                    for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
                        for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
                            const FilterSet::FilterType& filter = H_(angle, strength, coherence, pixeltype);
                            for(int32_t i = 0; i < filter.rows(); ++i) {
                                double x = filter(i, 0);
                                file << x << ',';
                            }
                            file << '\n';
                        }
                    }
                }
            }
            file.seekp(-2, std::ios::_Seekcur);
            file << "\n]}\n";
            file.close();
        }
    }
}

void RAISRTrainer::train(const std::filesystem::path& path)
{
    ::memset(patch_image_, 0, sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize);
    ::memset(gradient_image_, 0, sizeof(double) * RAISRParam::GradientSize * RAISRParam::GradientSize);
    Image<stbi_uc> original;
    {
        std::filesystem::path filepath = current_;
        filepath.append(path.c_str());
        int32_t w = 0, h = 0, c = 0;
        std::u8string u8path = filepath.generic_u8string();
        stbi_uc* pixels = stbi_load(reinterpret_cast<const char*>(u8path.c_str()), &w, &h, &c, STBI_default);
        if(nullptr == pixels) {
            return;
        }
        original.reset(w, h, c, pixels, stbi_image_free);
    }

    if(1 < original.c()) {
        Image<stbi_uc> grey(original.w(), original.h(), 1);
        if(!grey) {
            return;
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
        return;
    }
    {
        Image<stbi_uc> tmp(original.w() >> 1, original.h() >> 1, 1);
        if(!tmp) {
            return;
        }
        int r = stbir_resize_uint8_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(stbi_uc), &tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * sizeof(stbi_uc), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_CUBICBSPLINE, STBIR_COLORSPACE_LINEAR, nullptr);
        if(!r) {
            return;
        }

        r = stbir_resize_uint8_generic(&tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * tmp.c() * sizeof(stbi_uc), &upscaledLR(0, 0, 0), upscaledLR.w(), upscaledLR.h(), upscaledLR.w() * sizeof(stbi_uc), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_TRIANGLE, STBIR_COLORSPACE_LINEAR, nullptr);
        if(!r) {
            return;
        }
    }
    train_image(upscaledLR, original);
}

void RAISRTrainer::train_image(const Image<stbi_uc>& upscaledLR, const Image<stbi_uc>& original)
{
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
    Eigen::Matrix<double, 1, RAISRParam::PatchSize2> A;
    Eigen::Matrix<double, RAISRParam::PatchSize2, RAISRParam::PatchSize2> ATA;
    Eigen::Matrix<double, RAISRParam::PatchSize2, 1> ATb;

    for(int32_t i = margin; i < hend; ++i) {
        for(int32_t j = margin; j < wend; ++j) {
            for(int32_t y = patch_begin; y <= patch_end; ++y) {
                for(int32_t x = patch_begin; x <= patch_end; ++x) {
                    patch_image_[(y - patch_begin) * RAISRParam::PatchSize + x - patch_begin] = to_double(upscaledLR(j + x, i + y, 0));
                }
            }
            for(int32_t y = gradient_begin; y <= gradient_end; ++y) {
                for(int32_t x = gradient_begin; x <= gradient_end; ++x) {
                    gradient_image_[(y - gradient_begin) * RAISRParam::GradientSize + x - gradient_begin] = to_double(upscaledLR(j + x, i + y, 0));
                }
            }

            auto [angle, strength, coherence] = hashkey(RAISRParam::GradientSize, gradient_image_, weights_, RAISRParam::Qangle);
            double pixelHR = to_double(original(j, i, 0));

            int32_t pixeltype = ((i - margin) % RAISRParam::R) * RAISRParam::R + ((j - margin) % RAISRParam::R);

            A = Eigen::Map<Eigen::Matrix<double, 1, RAISRParam::PatchSize2>>(patch_image_);
            ATA = A.transpose() * A;
            ATb = A.transpose() * pixelHR;

            Q_(angle, strength, coherence, pixeltype) += ATA;
            V_(angle, strength, coherence, pixeltype) += ATb;
        } // int32_t j = margin
    }     // int32_t i = margin
}

#if 0
void RAISRTrainer::copy_examples()
{
    MatrixParamSize2* P = new MatrixParamSize2[7+4];
    MatrixParamSize2& rotate = P[7];
    MatrixParamSize2& flip = P[8];
    MatrixParamSize2& trotate = P[9];
    MatrixParamSize2& tflip = P[10];

    for(int32_t i=0; i<7; ++i){
        P[i].setZero();
    }
    rotate.setZero();
    flip.setZero();

    for(int32_t i = 0; i < (RAISRParam::PatchSize2); ++i) {
        int32_t i0 = i % RAISRParam::PatchSize;
        int32_t i1 = (i / RAISRParam::PatchSize);
        int32_t j = RAISRParam::PatchSize2 - RAISRParam::PatchSize + i1 - RAISRParam::PatchSize * i0;
        rotate(j, i) = 1;
        int32_t k = RAISRParam::PatchSize * (i1 + 1) - i0 - 1;
        flip(k, i) = 1;
    }
    for(int32_t i = 1; i < 8; ++i) {
        int32_t i0 = i % 4;
        int32_t i1 = i >> 2;
        trotate = rotate.pow(i0);
        tflip = flip.pow(i1);
        P[i-1] = tflip*trotate;
    }
    ConjugateSet QExt;
    FilterSet VExt;
    QExt.clear_matrix();
    VExt.clear_matrix();

    for(int32_t pixeltype = 0; pixeltype < RAISRParam::R2; ++pixeltype) {
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
                        const MatrixParamSize2& Q = *shared.Q_(angle, strength, coherence, pixeltype);
                        const VectorParamSize2& V = *shared.V_(angle, strength, coherence, pixeltype);

                        MatrixParamSize2& QE = *QExt(newangle, strength, coherence, pixeltype);
                        VectorParamSize2& VE = *VExt(newangle, strength, coherence, pixeltype);

                        QE += P[m-1].transpose() * Q * P[m-1];
                        VE += P[m-1].transpose() * V;
                    }
                }
            }
        }
    }

    for(int32_t i=0; i<shared.Q_.count(); ++i){
        shared.Q_(0,0,0,0)[i] += QExt(0,0,0,0)[i];
        shared.V_(0,0,0,0)[i] += VExt(0,0,0,0)[i];
    }

    delete[] P;
}
#endif

void RAISRTrainer::solve()
{
    // copy_examples();
    double gaussian[RAISRParam::PatchSize * RAISRParam::PatchSize];
    gaussian2d(RAISRParam::PatchSize, gaussian, 0.1f);

    for(int32_t pixeltype = 0; pixeltype < RAISRParam::R2; ++pixeltype) {
        for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
            for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
                for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
                    Eigen::Matrix<double, RAISRParam::PatchSize2, 1>& H = H_(angle, strength, coherence, pixeltype);
                    const Eigen::Matrix<double, RAISRParam::PatchSize2, RAISRParam::PatchSize2>& Q = Q_(angle, strength, coherence, pixeltype);
                    const Eigen::Matrix<double, RAISRParam::PatchSize2, 1>& V = V_(angle, strength, coherence, pixeltype);
                    const auto LDLT = Q.ldlt();
                    H = LDLT.solve(V);
                    if(Eigen::Success != LDLT.info()) {
                        H.setZero();
                    }
                    for(int32_t i = 0; i < RAISRParam::PatchSize2; ++i) {
                        H(i, 0) = gaussian[i];
                    }
                }
            }
        }
    }
}
} // namespace cppraisr
