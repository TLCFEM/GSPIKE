// Shared host-side helpers for the CUDA implementation.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace gspike {
    namespace detail {

        constexpr int kBlock = 256;

        inline void check(cudaError_t e) {
            if(e != cudaSuccess)
                throw std::runtime_error(std::string("gspike: ") + cudaGetErrorString(e));
        }

        // Owning device allocation with a predictable lifetime.  Kernels still
        // take plain pointers through get().
        template<class T>
        class device_buffer {
        public:
            device_buffer() = default;
            ~device_buffer() { reset(); }
            device_buffer(const device_buffer&) = delete;
            device_buffer& operator=(const device_buffer&) = delete;
            device_buffer(device_buffer&&) = delete;
            device_buffer& operator=(device_buffer&&) = delete;

            void alloc(size_t count) {
                reset();
                if(count == 0) return;
                check(cudaMalloc(&p_, count * sizeof(T)));
                count_ = count;
            }

            void reset() noexcept {
                if(p_) cudaFree(p_);
                p_ = nullptr;
                count_ = 0;
            }

            T* get() const noexcept { return p_; }
            size_t size() const noexcept { return count_; }
            explicit operator bool() const noexcept { return p_ != nullptr; }

        private:
            T* p_ = nullptr;
            size_t count_ = 0;
        };

    } // namespace detail
} // namespace gspike
