#include "HardwareFeatures.h"
#include "Requirements.h"

#include <array>
#include <iostream>
#include <tuple>
#include <utility>

namespace AboveInfinity {

namespace internal {

/* Checks that all elements of a sequence are unique */
template<auto head, auto... tail>
inline constexpr auto allUnique() noexcept {
    if constexpr(sizeof...(tail) > 0U) {
        requires(((head != tail) && ...));
        allUnique<tail...>();
    }

    if constexpr(sizeof...(tail) == 0U) return true;
}
}; // namespace internal

/* Class representing the lengths of a multidimensional array */
template<int... _Lengths>
class Lengths {
    requires(sizeof...(_Lengths) > 0U);
    requires(((_Lengths > 0) && ...));
    // Assure that not all lengths are ones
    requires(((_Lengths * ...) > 1));

private:
    template<std::size_t N, std::size_t... is>
    inline constexpr auto dimensionReductionImpl(std::index_sequence<is...>) const noexcept {
        requires(N >= 0U && N < sizeof...(_Lengths));
        requires(sizeof...(is) == sizeof...(_Lengths) - 1U);
        return Lengths<std::get<(is < N ? is : is + 1U)>(std::forward_as_tuple(_Lengths...))...>();
    }

    template<std::size_t N, int reductionSize, std::size_t... is>
    inline constexpr auto lengthReductionImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N >= 0U && N < sizeof...(_Lengths));
        requires(reductionSize >= 0 && reductionSize < std::get<N>(std::forward_as_tuple(_Lengths...)));
        requires(sizeof...(is) == sizeof...(_Lengths));
        return Lengths<(is == N ? std::get<is>(std::forward_as_tuple(_Lengths...)) - reductionSize :
                                  std::get<is>(std::forward_as_tuple(_Lengths...)))...>();
    }

    template<std::size_t N, int newLength, std::size_t... is>
    inline constexpr auto setLengthImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N >= 0U && N < sizeof...(_Lengths));
        requires(sizeof...(is) == sizeof...(_Lengths));
        return Lengths<(is == N ? newLength : std::get<is>(std::forward_as_tuple(_Lengths...)))...>();
    }

public:
    template<std::size_t... Order>
    inline constexpr auto fastPermute() const noexcept {
        requires(sizeof...(Order) == sizeof...(_Lengths));
        requires(((Order >= 0U && Order < sizeof...(_Lengths)) && ...));
        internal::allUnique<Order...>();
        return Lengths<(std::get<Order>(std::forward_as_tuple(_Lengths...)))...>();
    }

    template<std::size_t N>
    inline constexpr auto dimensionReduction() const noexcept {
        return dimensionReductionImpl<N>(std::make_index_sequence<sizeof...(_Lengths) - 1U>());
    }

    template<std::size_t N, int reductionSize>
    inline constexpr auto lengthReduction() const noexcept {
        return lengthReductionImpl<N, reductionSize>(std::make_index_sequence<sizeof...(_Lengths)>());
    }

    inline constexpr auto flatten() const noexcept { return Lengths<(_Lengths * ...)>(); }

    template<std::size_t N, int newLength>
    inline constexpr auto setLength() const noexcept {
        return setLengthImpl<N, newLength>(std::make_index_sequence<sizeof...(_Lengths)>());
    }

    template<std::size_t N>
    inline constexpr const int get() const noexcept {
        requires(N >= 0U && N < sizeof...(_Lengths));
        return std::get<N>(std::forward_as_tuple(_Lengths...));
    }

    inline constexpr const std::size_t size() const noexcept { return sizeof...(_Lengths); }

    inline constexpr auto tuple() const noexcept { return std::tuple(_Lengths...); }

    inline constexpr auto array() const noexcept { return std::array<int, sizeof...(_Lengths)>{_Lengths...}; }
};

/* Class representing the strides of a multidimensional array */
template<typename T, int... _Strides>
class Strides {
    requires(sizeof...(_Strides) > 0U);
    requires(((_Strides >= 0) && ...));

    template<std::size_t N, std::size_t... is>
    inline constexpr auto dimensionReductionImpl(std::index_sequence<is...>) const noexcept {
        requires(N >= 0U && N < sizeof...(_Strides));
        requires(sizeof...(is) == sizeof...(_Strides) - 1U);
        return Strides<T, std::get<(is < N ? is : is + 1U)>(std::forward_as_tuple(_Strides...))...>();
    }

public:
    using Type = T;

    template<std::size_t... Order>
    inline constexpr auto fastPermute() const noexcept {
        requires(sizeof...(Order) == sizeof...(_Strides));
        requires(((Order >= 0U && Order < sizeof...(_Strides)) && ...));
        internal::allUnique<Order...>();
        return Strides<T, (std::get<Order>(std::forward_as_tuple(_Strides...)))...>();
    }

    template<std::size_t N>
    inline constexpr auto dimensionReduction() const noexcept {
        return dimensionReductionImpl<N>(std::make_index_sequence<sizeof...(_Strides) - 1U>());
    }

    template<std::size_t N>
    inline constexpr const int get() const noexcept {
        requires(N >= 0U && N < sizeof...(_Strides));
        return std::get<N>(std::forward_as_tuple(_Strides...));
    }

    inline constexpr const std::size_t size() const noexcept { return sizeof...(_Strides); }

    inline constexpr auto tuple() const noexcept { return std::tuple(_Strides...); }

    inline constexpr auto array() const noexcept { return std::array<int, sizeof...(_Strides)>{_Strides...}; }
};

namespace internal {

/*
 * Function used to compute the aligned strides for a multidimensional array
 * WARNING
 * If the first length is set to 1, the second stride will contain a padding, even though it might not be needed.
 * Ideally, all the strides at the beginning whose previous length was 1 should be 1 as well. Just after a stride
 * is found that needs to be computed using a length that isn't equal to 1 should be computed using the computation
 * determining if there should be an offset between that and the next dimension for alignment purposes
 *
 * Another issue is that, if there's a dimension with the length equal to 1, its stride and the next dimensions
 * stride will be equal. This might affect the ordering in functions like undoPermutation
 */
template<typename T, class _Lengths, std::size_t... is>
inline constexpr auto computeAlignedStrides(std::index_sequence<is...>) noexcept {
    requires(sizeof...(is) > 0U);
    constexpr _Lengths lengths;
    requires(sizeof...(is) == lengths.size() - 2U);

    constexpr int firstLength = lengths.template get<0U>();
    constexpr int firstStride = 1;

    constexpr int alignMask = static_cast<int>(AIAlignment) / static_cast<int>(sizeof(T)) - 1;
    constexpr std::array<int, sizeof...(is) + 2U> strides{
        1, (firstLength + alignMask) & ~alignMask, (lengths.template get<is + 1U>() * strides[is + 1U])...};
    return Strides<T, firstStride, (firstLength + alignMask) & ~alignMask, (strides[is + 2U])...>();
}

template<typename T, class _Lengths>
inline constexpr auto computeAlignedStrides() noexcept {
    constexpr _Lengths lengths;
    constexpr int firstLength = lengths.template get<0U>();
    constexpr int firstStride = 1;
    // Strides are in number of elements, not in bytes
    constexpr int alignMask = AIAlignment / static_cast<int>(sizeof(T)) - 1;

    if constexpr(lengths.size() > 2U)
        return computeAlignedStrides<T, _Lengths>(std::make_index_sequence<lengths.size() - 2U>());
    else if constexpr(lengths.size() == 2U)
        return Strides<T, firstStride, (firstLength + alignMask) & ~alignMask>();
    else if constexpr(lengths.size() == 1U)
        return Strides<T, firstStride>();
}

template<typename T, class _Lengths, std::size_t... is>
inline constexpr auto computeUnalignedStrides(std::index_sequence<is...>) noexcept {
    requires(sizeof...(is) > 0U);
    constexpr _Lengths lengths;
    requires(sizeof...(is) == lengths.size() - 1U);

    constexpr int firstStride = 1;

    constexpr std::array<int, sizeof...(is) + 1U> strides{1, (lengths.template get<is>() * strides[is])...};
    return Strides<T, firstStride, (strides[is + 1U])...>();
}

/*
 * Computes unaligned strides for a multidimensional array
 * It's mostly used for modifying the Shape of already allocated multidimensional arrays
 * WARNING
 * If there's a dimension with the length equal to 1, its stride and the next dimensions
 * stride will be equal. This might affect the ordering in functions like undoPermutation
 */
template<typename T, class _Lengths>
inline constexpr auto computeUnalignedStrides() noexcept {
    constexpr _Lengths lengths;
    constexpr int firstLength = lengths.template get<0U>();
    constexpr int firstStride = 1;

    if constexpr(lengths.size() == 1U) return Strides<T, firstStride>();
    else if constexpr(lengths.size() == 2U)
        return Strides<T, firstStride, firstLength>();
    else {
        return computeUnalignedStrides<T, _Lengths>(std::make_index_sequence<lengths.size() - 1U>());
    }
}

template<class _Lengths, class _Strides, std::size_t... is, std::size_t... distance>
inline constexpr auto partiallyComputeStrides(std::index_sequence<is...>, std::index_sequence<distance...>) noexcept {
    constexpr _Lengths lengths;
    constexpr _Strides strides;
    requires(sizeof...(is) == strides.size());
    requires(lengths.size() - strides.size() == sizeof...(distance));
    constexpr auto lengthsArr = lengths.array();
    constexpr std::array<int, lengths.size()> stridesArr =
        std::array{strides.template get<is>()...,
                   (std::get<distance + strides.size() - 1U>(stridesArr) *
                    std::get<distance + strides.size() - 1U>(lengthsArr))...};
    return Strides<typename _Strides::Type,
                   strides.template get<is>()...,
                   std::get<distance + strides.size()>(stridesArr)...>();
}

/* Takes one or more existing strides, and computes the rest based on the lengths and strides it already has */
template<class _Lengths, class _Strides, bool aligned = true>
inline constexpr auto partiallyComputeStrides() noexcept {
    constexpr _Lengths lengths;
    constexpr _Strides strides;
    requires(strides.size() > 0U && strides.size() < lengths.size());

    constexpr int alignMask = AIAlignment / static_cast<int>(sizeof(typename _Strides::Type)) - 1;

    if constexpr(strides.size() == 1U && aligned) {
        constexpr auto partialStrides = Strides<typename _Strides::Type,
                                                strides.template get<0U>(),
                                                (lengths.template get<0U>() + alignMask) & ~alignMask>();

        return partiallyComputeStrides<_Lengths, decltype(partialStrides)>(
            std::make_index_sequence<partialStrides.size()>(),
            std::make_index_sequence<lengths.size() - partialStrides.size()>());
    } else {
        return partiallyComputeStrides<_Lengths, decltype(strides)>(
            std::make_index_sequence<strides.size()>(), std::make_index_sequence<lengths.size() - strides.size()>());
    }
}

/* Finds the index of the smallest stride */
template<auto... strides, std::size_t... is>
inline constexpr std::size_t minStrideIndex(std::index_sequence<is...>) noexcept {
    requires(sizeof...(strides) == sizeof...(is));
    constexpr auto stridesArr = std::array{strides...};
    std::size_t minStrideIdx = 0U;

    return ((stridesArr[is] < stridesArr[minStrideIdx] ? minStrideIdx = is : minStrideIdx), ...);
}

/* Returns the indexes of the strides in ascending order */
template<typename... pairs, std::size_t... is, std::size_t... os>
inline constexpr auto sortStrideIndexes(std::index_sequence<is...>, std::index_sequence<os...>) noexcept {
    requires(sizeof...(pairs) == sizeof...(is));
    requires(sizeof...(is) == sizeof...(os) + 1U);
    constexpr auto pairsTuple = std::tuple{pairs()...};
    constexpr std::size_t minStrideIdx = minStrideIndex<pairs::first...>(std::make_index_sequence<sizeof...(pairs)>());

    if constexpr(sizeof...(is) == 1U) return std::tuple(std::get<minStrideIdx>(pairsTuple).second);

    if constexpr(sizeof...(is) > 1U)
        return std::tuple_cat(
            std::tuple(std::get<minStrideIdx>(pairsTuple).second),
            sortStrideIndexes<std::decay_t<decltype(std::get<(os < minStrideIdx ? os : os + 1U)>(pairsTuple))>...>(
                std::make_index_sequence<sizeof...(os)>(), std::make_index_sequence<sizeof...(os) - 1U>()));
}

template<int... elems, class Compare>
inline constexpr const std::size_t count(Compare&& comp) noexcept {
    std::size_t count = 0U;
    return (((comp(elems)) ? ++count : count), ...);
}

template<typename A, std::size_t... is>
inline constexpr const std::size_t squeezeCount(A&& arr, std::index_sequence<is...>) noexcept {
    std::size_t count = 0U;
    return (((std::get<is>(arr) == 1) ? ++count : count), ...);
}

template<int Length, int Comp>
inline constexpr auto skipLength() noexcept {
    if constexpr(Length != Comp) return std::tuple(Length);
    if constexpr(Length == Comp) return std::tuple<>();
}

template<int Length, int Comp, int Stride>
inline constexpr auto skipStride() noexcept {
    if constexpr(Length != Comp) return std::tuple(Stride);
    if constexpr(Length == Comp) return std::tuple<>();
}

template<std::size_t idx, std::size_t... compIdxs>
inline constexpr auto skipIdx() noexcept {
    if constexpr(((idx != compIdxs) && ...)) return std::tuple(idx);
    if constexpr(((idx == compIdxs) || ...)) return std::tuple<>();
}

template<class _Strides, std::size_t... is>
inline constexpr std::size_t maxStrideIndex(std::index_sequence<is...>) noexcept {
    constexpr _Strides strides;
    constexpr auto strideArr = strides.array();
    requires(sizeof...(is) == strides.size());

    std::size_t maxStrideIdx = 0U;

    return ((strideArr[is] > strideArr[maxStrideIdx] ? maxStrideIdx = is : maxStrideIdx), ...);
}

template<class _Lengths, class _Strides, std::size_t... is>
inline constexpr auto stridesContainPadding(std::index_sequence<is...>) noexcept {
    constexpr _Lengths lengths;

    if constexpr(lengths.size() == 1U) return false;

    constexpr _Strides strides;
    constexpr std::size_t size = lengths.size();

    requires(sizeof...(is) == size);

    constexpr auto maxStrideIdx = maxStrideIndex<_Strides>(std::make_index_sequence<sizeof...(is)>());

    return ((lengths.template get<is>() * ...) !=
            lengths.template get<maxStrideIdx>() * strides.template get<maxStrideIdx>()) ?
               true :
               false;
}

}; // namespace internal

/* Helper data structure used for functions like subspace */
template<auto _first, auto _second>
struct pair {
    static constexpr auto first = _first;
    static constexpr auto second = _second;
};

/* Class representing the Shape of a multidimensional array */
template<typename T, class _Lengths, class _Strides = decltype(internal::computeAlignedStrides<T, _Lengths>())>
class Shape {
private:
    template<std::size_t... is>
    inline constexpr auto undoPermutationImpl(std::index_sequence<is...>) const noexcept {
        constexpr _Lengths lengths;
        constexpr _Strides strides;

        requires(sizeof...(is) == strides.size());
        /* Strides that are equal to zero would be set at the beginning */
        requires(((strides.template get<is>() > 0) && ...));
        constexpr auto order = internal::sortStrideIndexes<pair<std::get<is>(strides.array()), is>...>(
            std::make_index_sequence<strides.size()>(), std::make_index_sequence<strides.size() - 1U>());
        requires(std::tuple_size_v<decltype(order)> == sizeof...(is));
        return Shape<T,
                     decltype(Lengths<lengths.array()[std::get<is>(order)]...>()),
                     decltype(Strides<T, strides.array()[std::get<is>(order)]...>())>();
    }

    template<typename... pairs, std::size_t... is>
    inline constexpr void subspaceAssert(std::index_sequence<is...>&&) const noexcept {
        constexpr _Lengths lengths;
        requires(sizeof...(pairs) > 0U);
        requires(sizeof...(pairs) == sizeof...(is));
        requires(sizeof...(pairs) <= lengths.size());
        requires(((pairs::second - pairs::first >= 0) && ...));
        requires(((pairs::first >= 0 && pairs::first < lengths.template get<is>()) && ...));
        requires(((pairs::second >= 0 && pairs::second <= lengths.template get<is>()) && ...));
    }

    template<int... lengthsHS, std::size_t... slabIS, std::size_t... sliceIS>
    inline constexpr auto
        subspaceImpl(std::index_sequence<slabIS...>&&, std::index_sequence<sliceIS...>&&) const noexcept {
        constexpr _Strides strides;
        return Shape<
            T,
            decltype(Lengths<std::get<sliceIS>(std::tuple_cat(internal::skipLength<lengthsHS, 0>()...))...>()),
            decltype(Strides<T,
                             std::get<sliceIS>(std::tuple_cat(
                                 internal::skipStride<lengthsHS, 0, strides.template get<slabIS>()>()...))...>())>();
    }

    template<std::size_t N, std::size_t... is>
    inline constexpr auto newAxisImpl(std::index_sequence<is...>) const noexcept {
        constexpr _Lengths lengths;
        constexpr _Strides strides;

        requires(N < lengths.size());
        requires(sizeof...(is) == lengths.size() + 1U);

        const auto newLengths = Lengths<((is == N ? 1 : std::get<(is > N ? is - 1U : is)>(lengths.array())))...>();
        const auto newStrides = Strides<T, ((is == N ? 0 : std::get<(is > N ? is - 1U : is)>(strides.array())))...>();
        return Shape<T, decltype(newLengths), decltype(newStrides)>();
    }

    template<std::size_t... full, std::size_t... squeezed>
    inline constexpr auto squeeze_impl(std::index_sequence<full...>, std::index_sequence<squeezed...>) const noexcept {
        constexpr _Lengths lengths;
        constexpr _Strides strides;

        requires(sizeof...(full) == lengths.size());
        requires(sizeof...(squeezed) < sizeof...(full));
        return Shape<T,
                     decltype(Lengths<std::get<squeezed>(
                                  std::tuple_cat(internal::skipLength<std::get<full>(lengths.array()), 1>()...))...>()),
                     decltype(Strides<T,
                                      std::get<squeezed>(std::tuple_cat(
                                          internal::skipStride<std::get<full>(lengths.array()),
                                                               1,
                                                               std::get<full>(strides.array())>()...))...>())>();
    }

public:
    using Type = T;
    using Reference = T&;
    using Pointer = T*;

    template<std::size_t... Order>
    inline constexpr auto fastPermute() const noexcept {
        return Shape<T,
                     decltype(std::declval<_Lengths>().template fastPermute<Order...>()),
                     decltype(std::declval<_Strides>().template fastPermute<Order...>())>();
    }

    inline constexpr auto undoPermutation() const noexcept {
        constexpr _Lengths lengths;
        return undoPermutationImpl(std::make_index_sequence<lengths.size()>());
    }

    inline constexpr const auto lengths() const noexcept { return _Lengths(); }

    inline constexpr const auto strides() const noexcept { return _Strides(); }

    template<std::size_t N>
    inline constexpr const auto dimensionReduction() const noexcept {
        return Shape<T,
                     decltype(std::declval<_Lengths>().template dimensionReduction<N>()),
                     decltype(std::declval<_Strides>().template dimensionReduction<N>())>();
    }

    template<std::size_t N, int reductionSize>
    inline constexpr const auto lengthReduction() const noexcept {
        return Shape<T, decltype(std::declval<_Lengths>().template lengthReduction<N, reductionSize>()), _Strides>();
    }

    template<std::size_t N, int newLength>
    inline constexpr auto setLength() const noexcept {
        return Shape<T, decltype(std::declval<_Lengths>().template setLength<N, newLength>()), _Strides>();
    }

    template<typename... pairs>
    inline constexpr auto subspace() const noexcept {
        subspaceAssert<pairs...>(std::make_index_sequence<sizeof...(pairs)>());
        constexpr std::size_t sliceCount =
            internal::count<(pairs::second - pairs::first)...>([](int e) { return e == 0; });
        return subspaceImpl<(pairs::second - pairs::first)...>(
            std::make_index_sequence<sizeof...(pairs)>(), std::make_index_sequence<sizeof...(pairs) - sliceCount>());
    }

    template<std::size_t N>
    inline constexpr auto newAxis() const noexcept {
        constexpr _Lengths lengths;
        return newAxisImpl<N>(std::make_index_sequence<lengths.size() + 1U>());
    }

    inline constexpr auto squeeze() const noexcept {
        constexpr _Lengths lengths;
        constexpr std::size_t onesCount =
            internal::squeezeCount(lengths.array(), std::make_index_sequence<lengths.size()>());

        if constexpr(onesCount == 0U) return Shape<T, _Lengths, _Strides>();
        else
            return squeeze_impl(std::make_index_sequence<lengths.size()>(),
                                std::make_index_sequence<lengths.size() - onesCount>());
    }

    inline constexpr bool containsPadding() const noexcept {
        return internal::stridesContainPadding<_Lengths, _Strides>(
            std::make_index_sequence<std::tuple_size_v<decltype(std::declval<_Lengths>().tuple())>>());
    }

    inline constexpr auto ravel() const noexcept {
        constexpr _Lengths lengths;

        if constexpr(lengths.size() == 1U) return Shape<T, _Lengths, _Strides>();

        requires(!internal::stridesContainPadding<_Lengths, _Strides>(
            std::make_index_sequence<std::tuple_size_v<decltype(std::declval<_Lengths>().tuple())>>()));

        return Shape<T, decltype(lengths.flatten()), decltype(Strides<T, 1>())>();
    }

    inline constexpr const std::size_t rank() const noexcept {
        constexpr _Lengths lengths;
        return lengths.size();
    }
};
}; // namespace AboveInfinity
