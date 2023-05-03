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

void FilterSet::write_js(std::ostream& os)
{
    os << "var filter = [\n";
    for(int32_t i = 0; i < Count; ++i) {
        for(int32_t r = 0; r < filters_[i].rows(); ++r) {
            for(int32_t c = 0; c < filters_[i].cols(); ++c) {
                double x = filters_[i](r, c);
                os << std::setprecision(12);
                os << x << ',';
            }
        }
        os << "\n";
    }
    os.seekp(-2, std::ios::_Seekcur);
    os << "\n];\n";
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
    , check_count_(0)
    , weights_{}
    , patch_image_{}
    , gradient_image_{}
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
    box2d(RAISRParam::GradientSize, weights_);

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
        if(train(images[i])) {
            break;
        }
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
        filepath.append(".js");
        file.open(filepath.c_str(), std::ios::binary);
        if(file.is_open()) {
            H_.write_js(file);
            file.close();
        }
    }
}

bool RAISRTrainer::train(const std::filesystem::path& path)
{
    ::memset(patch_image_, 0, sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize);
    ::memset(gradient_image_, 0, sizeof(double) * RAISRParam::GradientSize * RAISRParam::GradientSize);
    Image<float> original;
    {
        std::filesystem::path filepath = current_;
        filepath.append(path.c_str());
        int32_t w = 0, h = 0, c = 0;
        std::u8string u8path = filepath.generic_u8string();
        float* pixels = stbi_loadf(reinterpret_cast<const char*>(u8path.c_str()), &w, &h, &c, STBI_default);
        if(nullptr == pixels) {
            return false;
        }
        original.reset(w, h, c, pixels, stbi_image_free);
    }

    if(1 < original.c()) {
        Image<float> grey(original.w(), original.h(), 1);
        if(!grey) {
            return false;
        }
        for(int j = 0; j < original.h(); ++j) {
            for(int k = 0; k < original.w(); ++k) {
                float r = original(k, j, 0);
                float g = original(k, j, 1);
                float b = original(k, j, 2);
                grey(k, j, 0) = to_grey(r, g, b);
            }
        }
        original.swap(grey);
    }
    Image<float> upscaledLR(original.w(), original.h(), 1);
    if(!upscaledLR) {
        return false;
    }
    {
        Image<float> tmp(original.w() >> 1, original.h() >> 1, 1);
        if(!tmp) {
            return false;
        }
        //int r = stbir_resize_float_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(float), &tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * sizeof(float), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_TRIANGLE, STBIR_COLORSPACE_LINEAR, nullptr);
        //int r = stbir_resize_float_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(float), &tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * sizeof(float), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_CUBICBSPLINE, STBIR_COLORSPACE_LINEAR, nullptr);
        //int r = stbir_resize_float_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(float), &tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * sizeof(float), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_CATMULLROM, STBIR_COLORSPACE_LINEAR, nullptr);
        int r = stbir_resize_float_generic(&original(0, 0, 0), original.w(), original.h(), original.w() * original.c() * sizeof(float), &tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * sizeof(float), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_MITCHELL, STBIR_COLORSPACE_LINEAR, nullptr);
        if(!r) {
            return false;
        }

        r = stbir_resize_float_generic(&tmp(0, 0, 0), tmp.w(), tmp.h(), tmp.w() * tmp.c() * sizeof(float), &upscaledLR(0, 0, 0), upscaledLR.w(), upscaledLR.h(), upscaledLR.w() * sizeof(float), 1, 0, 0, STBIR_EDGE_REFLECT, STBIR_FILTER_TRIANGLE, STBIR_COLORSPACE_LINEAR, nullptr);
        if(!r) {
            return false;
        }
    }
    train_image(upscaledLR, original);

    if(CheckStep <= ++check_count_) {
        check_count_ = 0;
        return solve();
    }
    return false;
}

void RAISRTrainer::train_image(const Image<float>& upscaledLR, const Image<float>& original)
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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(1, RAISRParam::PatchSize2);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ATA(RAISRParam::PatchSize2, RAISRParam::PatchSize2);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ATb(RAISRParam::PatchSize2, 1);

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

            auto [angle, strength, coherence] = hashkey(gradient_image_, RAISRParam::Qangle);
            double pixelHR = to_double(original(j, i, 0));

            int32_t pixeltype = ((i - margin) % RAISRParam::R) * RAISRParam::R + ((j - margin) % RAISRParam::R);
            Counts_(angle, strength, coherence, pixeltype) += 1;
            if(MaxSum <= Counts_(angle, strength, coherence, pixeltype)) {
                Counts_(angle, strength, coherence, pixeltype) = 0;
                Q_(angle, strength, coherence, pixeltype).setZero();
                V_(angle, strength, coherence, pixeltype).setZero();
                continue;
            }

            A = Eigen::Map<Eigen::Matrix<double, 1, RAISRParam::PatchSize2>>(patch_image_);
            ATA = A.transpose() * A;
            ATb = A.transpose() * pixelHR;

            Q_(angle, strength, coherence, pixeltype) += ATA;
            V_(angle, strength, coherence, pixeltype) += ATb;
        } // int32_t j = margin
    }     // int32_t i = margin
}

void RAISRTrainer::copy_examples()
{
    std::unique_ptr<MatrixSet::MatrixType> matrices(new MatrixSet::MatrixType[7 + 4]);
    MatrixSet::MatrixType* P = matrices.get();
    MatrixSet::MatrixType& rotate = P[7];
    MatrixSet::MatrixType& flip = P[8];
    MatrixSet::MatrixType& trotate = P[9];
    MatrixSet::MatrixType& tflip = P[10];

    for(int32_t i = 0; i < 7; ++i) {
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
        trotate = rotate;
        for(int32_t j = 1; j < i0; ++j) {
            trotate *= rotate;
        }
        tflip = flip;
        for(int32_t j = 1; j < i1; ++j) {
            tflip *= flip;
        }
        P[i - 1] = tflip * trotate;
    }
    MatrixSet QExt;
    FilterSet VExt;

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
                        const MatrixSet::MatrixType& Q = Q_(angle, strength, coherence, pixeltype);
                        const FilterSet::FilterType& V = V_(angle, strength, coherence, pixeltype);

                        MatrixSet::MatrixType& QE = QExt(newangle, strength, coherence, pixeltype);
                        FilterSet::FilterType& VE = VExt(newangle, strength, coherence, pixeltype);

                        QE += P[m - 1].transpose() * Q * P[m - 1];
                        VE += P[m - 1].transpose() * V;
                    }
                } // for(int32_t angle
            }     // for(int32_t strength
        }         // for(int32_t coherence
    }             // for(int32_t coherence

    for(int32_t pixeltype = 0; pixeltype < RAISRParam::R2; ++pixeltype) {
        for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
            for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
                for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
                    MatrixSet::MatrixType& Q = Q_(angle, strength, coherence, pixeltype);
                    FilterSet::FilterType& V = V_(angle, strength, coherence, pixeltype);

                    const MatrixSet::MatrixType& QE = QExt(angle, strength, coherence, pixeltype);
                    const FilterSet::FilterType& VE = VExt(angle, strength, coherence, pixeltype);
                    Q += QE;
                    V += VE;
                } // for(int32_t angle
            }     // for(int32_t strength
        }         // for(int32_t coherence
    }             // for(int32_t coherence
}

FilterSet::FilterType RAISRTrainer::solve(const MatrixSet::MatrixType& Q, const FilterSet::FilterType& V)
{
    FilterSet::FilterType H;
    H.setZero();
    MatrixSet::MatrixType I;
    I.setIdentity();
    MatrixSet::MatrixType A = Q;
    FilterSet::FilterType X;
    X.setZero();
    for(;;) {
        double sumA = A.sum();
        if(sumA < 10) {
            break;
        }
        if(A.determinant() < 1.0) {
            A = A + I * sumA * 000000005;
        } else {
            X = A.inverse() * V;
            break;
        }
    }
    return X;
}

bool RAISRTrainer::solve()
{
    // copy_examples();
    int32_t count = 0;
    double total = 0;
    for(int32_t pixeltype = 0; pixeltype < RAISRParam::R2; ++pixeltype) {
        for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
            for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
                for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
                    FilterSet::FilterType& H = H_(angle, strength, coherence, pixeltype);
                    MatrixSet::MatrixType& Q = Q_(angle, strength, coherence, pixeltype);
                    FilterSet::FilterType& V = V_(angle, strength, coherence, pixeltype);
                    int64_t& C = Counts_(angle, strength, coherence, pixeltype);
                    FilterSet::FilterType nH = solve(Q, V);
                    FilterSet::FilterType D = Q * H - V;
                    FilterSet::FilterType nD = Q * nH - V;
                    double d = 0.0;
                    double nd = 0.0;
                    for(int32_t r = 0; r < D.rows(); ++r) {
                        for(int32_t c = 0; c < D.cols(); ++c) {
                            d += D(r, c) * D(r, c);
                            nd += nD(r,c)*nD(r,c);
                        }
                    }
                    d /= (D.rows() * D.cols());
                    nd /= (D.rows()*D.cols());
                    int64_t c = C;
                    if(0<C && nd<d){
                        Q.setZero();
                        V.setZero();
                        C = 0;
                        d = nd;
                        H = nH;
                    }
                    total += d;
                    std::cout << "[" << count << "] " << d << " /" << c << std::endl;
                    ++count;
                } // for(int32_t angle
            }     // for(int32_t strength
        }         // for(int32_t coherence
    }             // for(int32_t coherence
    if(count <= 0) {
        return true;
    }
    total /= count;
    std::cout << "total: " << total << " / " << count << std::endl;
    return false;
}
} // namespace cppraisr
