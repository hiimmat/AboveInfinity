#pragma once

namespace AboveInfinity {

namespace internal {
/**
 * Helper function used as a compile-time substitute for the ternary operator 
 * 
 * @tparam res: condition determining the branching result
 * @tparam First: Type of the first function argument
 * @tparam Second: Type of the second function argument
 * @param first: First function argument
 * @param second: Second function argument
 * @return: first if res is true, second otherwise
 */
template<bool res, typename First, typename Second>
auto comparator(First&& first, Second&& second) {
    if constexpr(res) {
        return first;
    } else
        return second;
}
} // namespace internal

/**
 * Tensor specific policy used for manipulating the Tensor's memory layout
 * 
 * @tparam Tensor: CRTP template parameter
 */
template<typename Tensor>
struct PlaneManipulationPolicy {};

template<template<typename> typename PlaneList, typename... PlaneTypes, template<typename> typename... Policies>
struct PlaneManipulationPolicy<Tensor<PlaneList<PlaneTypes...>, Policies...>> {
    using _Tensor = Tensor<PlaneList<PlaneTypes...>, Policies...>;
    friend _Tensor;

private:
    template<int PlaneIndex, std::size_t... Order, std::size_t... is>
    inline auto FastPermuteImpl(std::index_sequence<is...>) const {
        if constexpr(PlaneIndex < 0) {
            auto planes = PlaneList{
                static_cast<const _Tensor*>(this)->Planes().template At<is>().template FastPermute<Order...>()...};
            return Tensor<decltype(planes), Policies...>(planes);
        } else {
            auto planes = PlaneList{(comparator<PlaneIndex == is>(
                static_cast<const _Tensor*>(this)->Planes().template At<is>().template FastPermute<Order...>(),
                static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
            return Tensor<decltype(planes), Policies...>(planes);
        }
    }

    template<int PlaneIndex, std::size_t... is>
    inline auto UndoPermutationImpl(std::index_sequence<is...>) const {
        auto planes = PlaneList{(comparator<(PlaneIndex == is || PlaneIndex < 0)>(
            static_cast<const _Tensor*>(this)->Planes().template At<is>().UndoPermutation(),
            static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
        return Tensor<decltype(planes), Policies...>(planes);
    }

    template<int PlaneIndex, std::size_t skipDim, int lenOffset, std::size_t... is>
    inline auto SliceImpl(std::index_sequence<is...>) const {
        auto planes = PlaneList{(comparator<(PlaneIndex == is || PlaneIndex < 0)>(
            static_cast<const _Tensor*>(this)->Planes().template At<is>().template Slice<skipDim, lenOffset>(),
            static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
        return Tensor<decltype(planes), Policies...>(planes);
    }

    template<int PlaneIndex, std::size_t skipDim, std::size_t... is>
    inline auto SliceImpl(std::index_sequence<is...>, int lenOffset) const {
        auto planes = PlaneList{(comparator<(PlaneIndex == is || PlaneIndex < 0)>(
            static_cast<const _Tensor*>(this)->Planes().template At<is>().template Slice<skipDim>(lenOffset),
            static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
        return Tensor<decltype(planes), Policies...>(planes);
    }

    template<std::size_t PlaneIndex, std::size_t offsetDim, int firstElem, int lastElem, std::size_t... is>
    inline auto SlabImpl(std::index_sequence<is...>) const {
        auto planes = PlaneList{(comparator<(PlaneIndex == is || PlaneIndex < 0)>(
            static_cast<const _Tensor*>(this)->Planes().template At<is>().template Slab<offsetDim, firstElem, lastElem>(),
            static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
        return Tensor<decltype(planes), Policies...>(planes);
    }

    template<std::size_t PlaneIndex, typename... pairs, std::size_t... is>
    inline auto SubspaceImpl(std::index_sequence<is...>) const {
        auto planes = PlaneList{(comparator<(PlaneIndex == is || PlaneIndex < 0)>(
            static_cast<const _Tensor*>(this)->Planes().template At<is>().template Subspace<pairs...>(),
            static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
        return Tensor<decltype(planes), Policies...>(planes);
    }

    template<std::size_t PlaneIndex, std::size_t N, std::size_t... is>
    inline auto NewAxisImpl(std::index_sequence<is...>) const {
        auto planes = PlaneList{(comparator<(PlaneIndex == is || PlaneIndex < 0)>(
            static_cast<const _Tensor*>(this)->Planes().template At<is>().template NewAxis<N>(),
            static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
        return Tensor<decltype(planes), Policies...>(planes);
    }

    template<std::size_t PlaneIndex, std::size_t... is>
    inline auto SqueezeImpl(std::index_sequence<is...>) const {
        auto planes = PlaneList{(comparator<(PlaneIndex == is || PlaneIndex < 0)>(
            static_cast<const _Tensor*>(this)->Planes().template At<is>().Squeeze(),
            static_cast<const _Tensor*>(this)->Planes().template At<is>()))...};
        return Tensor<decltype(planes), Policies...>(planes);
    }

public:
    template<int PlaneIndex, std::size_t... Order>
    inline auto FastPermute() const {
        return FastPermuteImpl<PlaneIndex, Order...>(std::make_index_sequence<sizeof...(PlaneTypes)>());
    }

    template<int PlaneIndex>
    inline auto UndoPermutation() const {
        return UndoPermutationImpl<PlaneIndex>(std::make_index_sequence<sizeof...(PlaneTypes)>());
    }

    template<int PlaneIndex, std::size_t skipDim, int lenOffset>
    inline auto Slice() const {
        return SliceImpl<PlaneIndex, skipDim, lenOffset>(std::make_index_sequence<sizeof...(PlaneTypes)>());
    }

    template<std::size_t PlaneIndex, std::size_t skipDim>
    inline auto Slice(int lenOffset) const {
        return SliceImpl<PlaneIndex, skipDim>(std::make_index_sequence<sizeof...(PlaneTypes)>(), lenOffset);
    }

    template<std::size_t PlaneIndex, std::size_t offsetDim, int firstElem, int lastElem>
    inline auto Slab() const {
        return SlabImpl<PlaneIndex, offsetDim, firstElem, lastElem>(std::make_index_sequence<sizeof...(PlaneTypes)>());
    }

    template<std::size_t PlaneIndex, typename... pairs>
    inline auto Subspace() const {
        return SubspaceImpl<PlaneIndex, pairs...>(std::make_index_sequence<sizeof...(PlaneTypes)>());
    }

    template<std::size_t PlaneIndex, std::size_t N>
    inline auto NewAxis() const {
        return NewAxisImpl<PlaneIndex, N>(std::make_index_sequence<sizeof...(PlaneTypes)>());
    }

    template<std::size_t PlaneIndex>
    inline auto Squeeze() const {
        return static_cast<const _Tensor*>(this)->Planes().template At<PlaneIndex>().Squeeze();
    }
};

} // namespace AboveInfinity