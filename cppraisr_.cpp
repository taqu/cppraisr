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
#include <format>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <stb/stb_image_resize.h>
#include <stb/stb_image_write.h>

namespace cppraisr
{

std::optional<std::tuple<std::filesystem::path, int32_t>> RAISRTrainer::SharedContext::next()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(max_images_<=count_){
        return std::nullopt;
    }
    const std::filesystem::path& path = (*files_)[count_];
    size_t count = count_;
    ++count_;

    std::u8string p = path.u8string();
    std::cout << "[" << std::setfill('0') << std::right << std::setw(4) << count_ << '/' << max_images_ << "] " << (char*)p.c_str() << std::endl;
    return std::make_tuple(path, static_cast<int32_t>(count));
}

void RAISRTrainer::SharedContext::inject(int32_t count, const ConjugateSet& Q, const ConjugateSet& QC, const FilterSet& V, const FilterSet& VC)
{
    std::lock_guard<std::mutex> lock(mutex_);
    add_v(Q.count(), &Q_(0, 0, 0, 0, 0), &Q(0, 0, 0, 0, 0));
    add_v(V.count(), &V_(0, 0, 0, 0, 0), &V(0, 0, 0, 0, 0));

    {
        std::filesystem::path filepath = model_directory_;
        filepath.append(std::format("q_{0:06d}.bin", count_));
        std::ofstream file(filepath.c_str(), std::ios::binary);
        if(file.is_open()) {
            Q_.write(file);
        }
    }
    {
        std::filesystem::path filepath = model_directory_;
        filepath.append(std::format("v_{0:06d}.bin", count_));
        std::ofstream file(filepath.c_str(), std::ios::binary);
        if(file.is_open()) {
            V_.write(file);
        }
    }
}

RAISRTrainer::RAISRTrainer()
{
}

RAISRTrainer::~RAISRTrainer()
{
}

void RAISRTrainer::train(const std::vector<std::filesystem::path>& images, int32_t num_threads, int32_t max_images)
{
    int32_t hardware_threads = static_cast<int32_t>(std::thread::hardware_concurrency());
    num_threads = std::clamp(num_threads, 1, hardware_threads);

    H_.clear();
    std::unique_ptr<SharedContext> sharedContext = std::make_unique<SharedContext>();

    sharedContext->max_images_ = std::min(max_images, static_cast<int32_t>(images.size()));
    sharedContext->files_ = &images;
    sharedContext->Q_.clear();
    sharedContext->QC_.clear();
    sharedContext->V_.clear();
    sharedContext->VC_.clear();
    gaussian2d(RAISRParam::GradientSize, &sharedContext->weights_(0,0), 2.0);

    sharedContext->model_directory_ = std::filesystem::current_path();
    sharedContext->model_directory_.append("model");
    if(!std::filesystem::exists(sharedContext->model_directory_)){
        std::filesystem::create_directory(sharedContext->model_directory_);
    }

    std::vector<std::thread> threads;
    for(int32_t i=0; i<num_threads; ++i){
        threads.emplace_back([&sharedContext]{train_thread(*sharedContext);});
    }
    for(int32_t i=0; i<num_threads; ++i){
        threads[i].join();
    }

    solve(*sharedContext);
    {
        std::filesystem::path filepath = sharedContext->model_directory_;
        time_t t = std::time(nullptr);
        #ifdef _MSC_VER
        struct tm now;
        localtime_s(&now, &t);
        #else
        struct tm now = *std::localtime(&t);
        #endif
        now.tm_year += 1900;
        filepath.append(std::format("filter_{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}.bin",
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec));
        std::ofstream file(filepath.c_str(), std::ios::binary);
        if(file.is_open()) {
            H_.write(file);
        }
    }
}

void RAISRTrainer::train_thread(SharedContext& shared)
{
    std::unique_ptr<LocalContext> context = std::make_unique<LocalContext>();
    context->patch_image_.clear();
    context->patch_transposed_.clear();
    context->gradient_patch_.clear();
    context->ATA_.clear();
    context->ATb_.clear();
    for(;;){
        std::optional<std::tuple<std::filesystem::path, int32_t>> image = shared.next();
        if(!image){
            break;
        }
        Image<stbi_uc> original;
        context->image_count_ = std::get<1>(image.value());
        std::u8string path = std::get<0>(image.value()).u8string();
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
        train_image(upscaledLR, original, *context, shared);
    } // for(size_t i
}

void RAISRTrainer::train_image(const Image<stbi_uc>& upscaledLR, const Image<stbi_uc>& original, LocalContext& context, SharedContext& shared)
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

    ConjugateSet Q;
    ConjugateSet QC;
    FilterSet V;
    FilterSet VC;
    Q.clear();
    QC.clear();
    V.clear();
    VC.clear();

    int32_t half_patch_size = patch_size>>1;
    int32_t hend = upscaledLR.h() < margin ? 0 : upscaledLR.h() - margin;
    int32_t wend = upscaledLR.w() < margin ? 0 : upscaledLR.w() - margin;
    for(int32_t i = margin; i < hend; ++i) {
        for(int32_t j = margin; j < wend; ++j) {
            for(int32_t y = patch_begin; y <= patch_end; ++y) {
                for(int32_t x = patch_begin; x <= patch_end; ++x) {
                    context.patch_image_(x-patch_begin, y - patch_begin) = to_double(upscaledLR(j+x,i+y,0));
                }
            }
            for(int32_t y = gradient_begin; y <= gradient_end; ++y) {
                for(int32_t x = gradient_begin; x <= gradient_end; ++x) {
                    context.gradient_patch_(x-gradient_begin, y-gradient_begin) = to_double(upscaledLR(j+x,i+y,0));
                }
            }
            auto [angle, strength, coherence] = hashkey(RAISRParam::GradientSize, &context.gradient_patch_(0,0), &shared.weights_(0,0), RAISRParam::Qangle);
            double pixelHR = to_double(original(j,i,0));
            int32_t pixeltype = ((i-margin) % RAISRParam::R) * RAISRParam::R + ((j-margin) % RAISRParam::R);
            mul_mm(RAISRParam::PatchSize*RAISRParam::PatchSize, 1, RAISRParam::PatchSize*RAISRParam::PatchSize, &context.ATA_(0,0), &context.patch_image_(0,0), &context.patch_image_(0,0));
            mul_v(RAISRParam::PatchSize*RAISRParam::PatchSize, &context.ATb_(0,0), &context.patch_image_(0,0), pixelHR);

            double* TQ = Q(angle, strength, coherence, pixeltype);
            double* TQC = QC(angle, strength, coherence, pixeltype);
            double* TV = V(angle, strength, coherence, pixeltype);
            double* TVC = VC(angle, strength, coherence, pixeltype);
            add_v(RAISRParam::PatchSize*RAISRParam::PatchSize*RAISRParam::PatchSize*RAISRParam::PatchSize, TQ, &context.ATA_(0,0));
            add_v(RAISRParam::PatchSize*RAISRParam::PatchSize, TV, &context.ATb_(0,0));

        } // int32_t j = margin
    }     // int32_t i = margin

    shared.inject(context.image_count_, Q, QC, V, VC);
}

void RAISRTrainer::copy_examples(SharedContext& shared)
{
    double* P = static_cast<double*>(::malloc(sizeof(double) * 7 * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize));
    double* rotate = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize));
    double* flip = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize));

    double* trotate = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize));
    double* tflip = static_cast<double*>(::malloc(sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize));

    ::memset(P, 0, 7*sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize);
    ::memset(rotate, 0, sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize);
    ::memset(flip, 0, sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize);

    ::memset(trotate, 0, sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize);
    ::memset(tflip, 0, sizeof(double) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize);

    for(int32_t i = 0; i < (RAISRParam::PatchSize * RAISRParam::PatchSize); ++i) {
        int32_t i0 = i % RAISRParam::PatchSize;
        int32_t i1 = static_cast<int32_t>(floor(i / RAISRParam::PatchSize));
        int32_t j = RAISRParam::PatchSize * RAISRParam::PatchSize - RAISRParam::PatchSize + i1 - RAISRParam::PatchSize * i0;
        rotate[j * RAISRParam::PatchSize * RAISRParam::PatchSize + i] = 1;
        int32_t k = RAISRParam::PatchSize * (i1 + 1) - i0 - 1;
        flip[k * RAISRParam::PatchSize * RAISRParam::PatchSize + i] = 1;
    }
    for(int32_t i = 1; i < 8; ++i) {
        int32_t i0 = i % 4;
        int32_t i1 = i >> 2;

        power_m(RAISRParam::PatchSize * RAISRParam::PatchSize, tflip, flip, i1);
        power_m(RAISRParam::PatchSize * RAISRParam::PatchSize, trotate, rotate, i0);
        double* p = P + (i - 1) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize;
        mul_mm(RAISRParam::PatchSize * RAISRParam::PatchSize, p, tflip, trotate);
    }
    ConjugateSet QExt;
    FilterSet VExt;
    QExt.clear();
    VExt.clear();

    double* Pt = static_cast<double*>(::malloc(sizeof(double)*RAISRParam::PatchSize*RAISRParam::PatchSize*RAISRParam::PatchSize * RAISRParam::PatchSize));
    double* newQ0 = static_cast<double*>(::malloc(sizeof(double)*RAISRParam::PatchSize*RAISRParam::PatchSize*RAISRParam::PatchSize*RAISRParam::PatchSize));
    double* newQ1 = static_cast<double*>(::malloc(sizeof(double)*RAISRParam::PatchSize*RAISRParam::PatchSize*RAISRParam::PatchSize*RAISRParam::PatchSize));
    double* newV = static_cast<double*>(::malloc(sizeof(double)*RAISRParam::PatchSize*RAISRParam::PatchSize));

    for(int32_t pixeltype = 0; pixeltype < (RAISRParam::R * RAISRParam::R); ++pixeltype) {
        for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
            for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
                for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
                    for(int32_t m = 1; m<8; ++m){
                        int32_t m0 = m%4;
                        int32_t m1 = m>>2;
                        int32_t newangle = angle;
                        if(m1 == 1){
                            newangle = RAISRParam::Qangle - angle - 1;
                        }
                        newangle = newangle - RAISRParam::Qangle/2*m0;
                        while(newangle<0){
                            newangle += RAISRParam::Qangle;
                        }
                        const double* p = P + (m - 1) * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize * RAISRParam::PatchSize;
                        transpose(RAISRParam::PatchSize * RAISRParam::PatchSize, Pt, p);
                        const double* Q = shared.Q_(angle, strength, coherence, pixeltype);
                        const double* V = shared.V_(angle, strength, coherence, pixeltype);

                        double* q = QExt(angle, strength, coherence, pixeltype);
                        double* v = VExt(angle, strength, coherence, pixeltype);
                        mul_mm(RAISRParam::PatchSize*RAISRParam::PatchSize, newQ0, Pt, Q);
                        mul_mm(RAISRParam::PatchSize*RAISRParam::PatchSize, newQ1, newQ0, p);
                        mul_mv(RAISRParam::PatchSize*RAISRParam::PatchSize, RAISRParam::PatchSize*RAISRParam::PatchSize, newV, Pt, V);
                        add_m(RAISRParam::PatchSize*RAISRParam::PatchSize, q, newQ1);
                        add_m(RAISRParam::PatchSize, v, newV);
                    }
                }
            }
        }
    }
    add_v(shared.Q_.count(), &shared.Q_(0,0,0,0,0), &QExt(0,0,0,0,0));
    add_v(shared.V_.count(), &shared.V_(0,0,0,0,0), &VExt(0,0,0,0,0));

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

void RAISRTrainer::solve(SharedContext& shared)
{
    copy_examples(shared);
    for(int32_t pixeltype = 0; pixeltype < (RAISRParam::R * RAISRParam::R); ++pixeltype) {
        for(int32_t coherence = 0; coherence < RAISRParam::Qcoherence; ++coherence) {
            for(int32_t strength = 0; strength < RAISRParam::Qstrength; ++strength) {
                for(int32_t angle = 0; angle < RAISRParam::Qangle; ++angle) {
                    double* H = H_(angle, strength, coherence, pixeltype);
                    double* Q = shared.Q_(angle, strength, coherence, pixeltype);
                    double* V = shared.V_(angle, strength, coherence, pixeltype);
                    CGLSSolver::solve(RAISRParam::PatchSize*RAISRParam::PatchSize, H, Q, V);
                }
            }
        }
    }
}
