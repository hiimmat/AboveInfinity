#pragma once

namespace AboveInfinity {

/**
 * Tensor specific policy used for accessing and modifying the allocated memory of the specified plane
 * 
 * @tparam Tensor: CRTP template parameter
 */
template<typename Tensor>
struct CTMemoryAcessPolicy {
    template<std::size_t PlaneIndex, std::size_t... Offsets>
    inline auto SlicingPointer() const {
        return static_cast<const Tensor*>(this)
            ->Planes()
            .template At<PlaneIndex>()
            .template SlicingPointer<Offsets...>();
    }

    template<std::size_t PlaneIndex, typename... Offsets>
    inline auto SlicingPointer(Offsets&&... offsets) const {
        return static_cast<const Tensor*>(this)->Planes().template At<PlaneIndex>().template SlicingPointer(
            std::forward<Offsets>(offsets)...);
    }

    template<std::size_t PlaneIndex, std::size_t... Offsets, typename T>
    inline void SetValue(T value) {
        static_cast<Tensor*>(this)->Planes().template At<PlaneIndex>().template SetValue<Offsets...>(value);
    }

    template<std::size_t PlaneIndex, typename T, typename... Offsets>
    inline void SetValue(T value, Offsets... offsets) {
        static_cast<Tensor*>(this)->Planes().template At<PlaneIndex>().SetValue(value, offsets...);
    }
};

} // namespace AboveInfinity