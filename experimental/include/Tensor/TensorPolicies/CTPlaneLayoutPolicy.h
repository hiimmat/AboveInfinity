#pragma once

namespace AboveInfinity {

/**
 * Tensor specific policy used for accessing Plane class members
 * 
 * @tparam Tensor: CRTP template parameter 
 */
template<typename Tensor>
struct CTPlaneLayoutPolicy {
    template<std::size_t PlaneIndex = 0u>
    inline auto Buffer() noexcept {
        return static_cast<Tensor*>(this)->Planes().template At<PlaneIndex>().Buffer();
    }

    template<std::size_t PlaneIndex = 0u>
    inline const auto Buffer() const noexcept {
        return static_cast<const Tensor*>(this)->Planes().template At<PlaneIndex>().Buffer();
    }

    template<std::size_t PlaneIndex = 0u>
    inline constexpr const auto Shape() const noexcept {
        return static_cast<const Tensor*>(this)->Planes().template At<PlaneIndex>().Shape();
    }

    template<std::size_t PlaneIndex = 0u>
    inline constexpr std::size_t Channels() const noexcept {
        return static_cast<const Tensor*>(this)->Planes().template At<PlaneIndex>().Channels();
    }

    template<std::size_t PlaneIndex = 0u>
    inline constexpr int Offset() const noexcept {
        return static_cast<const Tensor*>(this)->Planes().template At<PlaneIndex>().Offset();
    }

    template<std::size_t PlaneIndex = 0u>
    inline constexpr std::size_t Rank() const noexcept {
        return static_cast<const Tensor*>(this)->Planes().template At<PlaneIndex>().Rank();
    }

    template<std::size_t PlaneIndex = 0u>
    inline constexpr int Total() const noexcept {
        return static_cast<const Tensor*>(this)->Planes().template At<PlaneIndex>().Total();
    }
};

} // namespace AboveInfinity