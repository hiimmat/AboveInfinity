#include "HardwareFeatures.h"
#include "Requirements.h"

#include <array>
#include <tuple>
#include <utility>

namespace AboveInfinity {

namespace internal {

/* Checks that all elements of a sequence are unique */
template<auto head, auto... tail>
inline constexpr void allUnique() noexcept {
    if constexpr(sizeof...(tail) > 0U) {
        requires(((head != tail) && ...));
        allUnique<tail...>();
    }

    if constexpr(sizeof...(tail) == 0U) return;
}
} // namespace internal

/* Class representing the lengths of a multidimensional array */
template<int... _Lengths>
class Lengths {
    requires(sizeof...(_Lengths) > 0U);
    requires(((_Lengths > 0) && ...));
    // Assure that not all lengths are ones
    requires(((_Lengths * ...) > 1));

private:
    static constexpr auto ls = std::array{_Lengths...};

    template<std::size_t N, std::size_t... is>
    inline constexpr auto dimensionReductionImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N < sizeof...(_Lengths));
        requires(sizeof...(is) == sizeof...(_Lengths) - 1U);
        return Lengths<ls[is < N ? is : is + 1U]...>();
    }

    template<std::size_t N, int reductionSize, std::size_t... is>
    inline constexpr auto lengthReductionImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N < sizeof...(_Lengths));
        requires(reductionSize >= 0 && reductionSize < ls[N]);
        requires(sizeof...(is) == sizeof...(_Lengths));
        return Lengths<(is == N ? ls[is] - reductionSize : ls[is])...>();
    }

    template<std::size_t N, int newLength, std::size_t... is>
    inline constexpr auto setLengthImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N < sizeof...(_Lengths));
        requires(sizeof...(is) == sizeof...(_Lengths));
        requires(newLength > 0 && newLength <= ls[N]);
        return Lengths<(is == N ? newLength : ls[is])...>();
    }

public:
    template<std::size_t... Order>
    inline constexpr auto fastPermute() const noexcept {
        requires(sizeof...(Order) == sizeof...(_Lengths));
        requires(((Order < sizeof...(_Lengths)) && ...));
        internal::allUnique<Order...>();
        return Lengths<ls[Order]...>();
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
    inline constexpr int get() const noexcept {
        requires(N < sizeof...(_Lengths));
        return ls[N];
    }

    inline constexpr std::size_t size() const noexcept { return sizeof...(_Lengths); }

    inline constexpr auto tuple() const noexcept { return std::tuple(_Lengths...); }

    inline constexpr auto array() const noexcept { return std::array<int, sizeof...(_Lengths)>{_Lengths...}; }
};

/* Class representing the strides of a multidimensional array */
template<typename T, int... _Strides>
class Strides {
    requires(sizeof...(_Strides) > 0U);
    requires(((_Strides >= 0) && ...));

private:
    static constexpr auto ss = std::array{_Strides...};

    template<std::size_t N, std::size_t... is>
    inline constexpr auto dimensionReductionImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N < sizeof...(_Strides));
        requires(sizeof...(is) == sizeof...(_Strides) - 1U);
        return Strides<T, (is < N ? ss[is] : ss[is + 1U])...>();
    }

public:
    using Type = T;
    using Reference = T&;
    using Pointer = T*;

    template<std::size_t... Order>
    inline constexpr auto fastPermute() const noexcept {
        requires(sizeof...(Order) == sizeof...(_Strides));
        requires(((Order < sizeof...(_Strides)) && ...));
        internal::allUnique<Order...>();
        return Strides<T, ss[Order]...>();
    }

    template<std::size_t N>
    inline constexpr auto dimensionReduction() const noexcept {
        return dimensionReductionImpl<N>(std::make_index_sequence<sizeof...(_Strides) - 1U>());
    }

    template<std::size_t N>
    inline constexpr int get() const noexcept {
        requires(N < sizeof...(_Strides));
        return ss[N];
    }

    inline constexpr std::size_t size() const noexcept { return sizeof...(_Strides); }

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
template<typename T, typename _Lengths, std::size_t... is>
inline constexpr auto computeAlignedStrides(std::index_sequence<is...>&&) noexcept {
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

template<typename T, typename _Lengths>
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

template<typename T, typename _Lengths, std::size_t... is>
inline constexpr auto computeUnalignedStrides(std::index_sequence<is...>&&) noexcept {
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
template<typename T, typename _Lengths>
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

template<typename _Lengths, typename _Strides, std::size_t... is, std::size_t... distance>
inline constexpr auto
    partiallyComputeStrides(std::index_sequence<is...>&&, std::index_sequence<distance...>&&) noexcept {
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
template<typename _Lengths, typename _Strides, bool aligned = true>
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

template<auto... elems, std::size_t... is>
inline constexpr std::size_t findMinIndex(std::index_sequence<is...>&&) noexcept {
    requires(sizeof...(elems) >= sizeof...(is));
    constexpr auto elemsArr = std::array{elems...};
    std::size_t minIdx = std::get<0U>(std::forward_as_tuple(is...));

    return ((elemsArr[is] < elemsArr[minIdx] ? minIdx = is : minIdx), ...);
}

/* Returns the indexes of the strides in ascending order */
template<auto... elems, std::size_t... is, std::size_t... os>
inline constexpr auto sortedIndexesAsc(std::index_sequence<is...>&&, std::index_sequence<os...>&&) noexcept {
    requires(sizeof...(is) <= sizeof...(elems));
    requires(sizeof...(os) == sizeof...(is) - 1U);

    if constexpr(sizeof...(is) == 1U) return std::tuple(is...);

    if constexpr(sizeof...(is) > 1U) {
        constexpr std::size_t minElemIdx = findMinIndex<elems...>(std::index_sequence<is...>());
        constexpr auto inSeq = std::array{is...};
        return std::tuple_cat(
            std::tuple(minElemIdx),
            sortedIndexesAsc<elems...>(std::index_sequence<(inSeq[os] < minElemIdx ? inSeq[os] : inSeq[os + 1U])...>(),
                                       std::make_index_sequence<sizeof...(os) - 1U>()));
    }
}

template<int... elems, typename Compare>
inline constexpr std::size_t count(Compare&& comp) noexcept {
    std::size_t count = 0U;
    return (((comp(elems)) ? ++count : count), ...);
}

template<auto... elems>
inline constexpr std::size_t squeezeCount() noexcept {
    std::size_t count = 0U;
    return (((elems == 1) ? ++count : count), ...);
}

template<bool pack, auto... elems>
inline constexpr auto packToTuple() noexcept {
    if constexpr(pack) return std::tuple(elems...);
    else
        return std::tuple<>();
}

template<std::size_t idx, std::size_t... compIdxs>
inline constexpr auto skipIdx() noexcept {
    if constexpr(((idx != compIdxs) && ...)) return std::tuple(idx);
    else
        return std::tuple<>();
}

template<auto... strides, std::size_t... is>
inline constexpr std::size_t maxStrideIndex(std::index_sequence<is...>&&) noexcept {
    requires(sizeof...(is) == sizeof...(strides));
    constexpr auto stridesArr = std::array{strides...};
    std::size_t maxStrideIdx = 0U;

    return ((stridesArr[is] > stridesArr[maxStrideIdx] ? maxStrideIdx = is : maxStrideIdx), ...);
}

} // namespace internal

/* Helper data structure used for functions like subspace */
template<auto _first, auto _second>
struct pair {
    static constexpr auto first = _first;
    static constexpr auto second = _second;
};

template<typename, typename>
class Shape {
    ~Shape() = delete;
    Shape(Shape const&) = delete;
    void operator=(Shape const&) = delete;
};

/* Class representing the Shape of a multidimensional array */
template<typename T, int... ls, int... ss>
class Shape<Lengths<ls...>, Strides<T, ss...>> {
    requires(sizeof...(ls) == sizeof...(ss));

private:
    static constexpr auto _lengths = std::array{ls...};
    static constexpr auto _strides = std::array{ss...};

    template<std::size_t... is>
    inline constexpr auto undoPermutationImpl(std::index_sequence<is...>&&) const noexcept {
        requires(sizeof...(is) == sizeof...(ls));
        /* Strides that are equal to zero would be set at the beginning */
        requires(((_strides[is] > 0) && ...));
        constexpr auto order = internal::sortedIndexesAsc<ss...>(std::make_index_sequence<sizeof...(ss)>(),
                                                                 std::make_index_sequence<sizeof...(ss) - 1U>());
        requires(std::tuple_size_v<decltype(order)> == sizeof...(is));
        return Shape<Lengths<_lengths[std::get<is>(order)]...>, Strides<T, _strides[std::get<is>(order)]...>>();
    }

    template<std::size_t N, std::size_t... is>
    inline constexpr auto dimensionReductionImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N < sizeof...(ls));
        requires(sizeof...(is) == sizeof...(ls) - 1U);
        return Shape<Lengths<_lengths[is < N ? is : is + 1U]...>, Strides<T, _strides[is < N ? is : is + 1U]...>>();
    }

    template<std::size_t N, int reductionSize, std::size_t... is>
    inline constexpr auto lengthReductionImpl(std::index_sequence<is...>&&) const noexcept {
        requires(sizeof...(is) == sizeof...(ls));
        requires(N < sizeof...(ls));
        requires(reductionSize >= 0 && reductionSize < _lengths[N]);
        return Shape<Lengths<(is == N ? _lengths[is] - reductionSize : _lengths[is])...>, Strides<T, ss...>>();
    }

    template<std::size_t N, int newLength, std::size_t... is>
    inline constexpr auto setLengthImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N < sizeof...(ls));
        requires(newLength > 0 && newLength <= _lengths[N]);
        requires(sizeof...(is) == sizeof...(ls));
        return Shape<Lengths<(is == N ? newLength : _lengths[is])...>, Strides<T, ss...>>();
    }

    template<typename... pairs, std::size_t... is>
    inline constexpr void subspaceAssert(std::index_sequence<is...>&&) const noexcept {
        requires(sizeof...(pairs) > 0U);
        requires(sizeof...(pairs) == sizeof...(is));
        requires(sizeof...(pairs) <= sizeof...(ls));
        requires(((pairs::second - pairs::first >= 0) && ...));
        requires(((pairs::first >= 0 && pairs::first <= _lengths[is]) && ...));
        requires(((pairs::second >= 0 && pairs::second <= _lengths[is]) && ...));
    }

    template<int... lengthsHS, std::size_t... slabIS, std::size_t... sliceIS>
    inline constexpr auto
        subspaceImpl(std::index_sequence<slabIS...>&&, std::index_sequence<sliceIS...>&&) const noexcept {
        return Shape<Lengths<std::get<sliceIS>(
                         std::tuple_cat(internal::packToTuple<(lengthsHS != 0 ? true : false), lengthsHS>()...))...>,
                     Strides<T,
                             std::get<sliceIS>(std::tuple_cat(
                                 internal::packToTuple<(lengthsHS != 0 ? true : false), _strides[slabIS]>()...))...>>();
    }

    template<std::size_t N, std::size_t... is>
    inline constexpr auto newAxisImpl(std::index_sequence<is...>&&) const noexcept {
        requires(N <= sizeof...(ls));
        requires(sizeof...(is) == sizeof...(ls) + 1U);
        return Shape<Lengths<(is < N ? _lengths[is] : is == N ? 1 : _lengths[is - 1U])...>,
                     Strides<T, (is < N ? _strides[is] : is == N ? 0 : _strides[is - 1U])...>>();
    }

    template<std::size_t... sqIs>
    inline constexpr auto squeezeImpl(std::index_sequence<sqIs...>&&) const noexcept {
        constexpr auto lensTup = std::tuple_cat(internal::packToTuple<(ls != 1 ? true : false), ls>()...);
        constexpr auto stridesTup = std::tuple_cat(internal::packToTuple<(ls != 1 ? true : false), ss>()...);
        return Shape<Lengths<std::get<sqIs>(lensTup)...>, Strides<T, std::get<sqIs>(stridesTup)...>>();
    }

public:
    using Type = T;
    using Reference = T&;
    using Pointer = T*;

    Shape() = default;

    template<std::size_t... Order>
    inline constexpr auto fastPermute() const noexcept {
        requires(sizeof...(Order) == sizeof...(ls));
        requires(((Order < sizeof...(ls)) && ...));
        internal::allUnique<Order...>();
        return Shape<Lengths<_lengths[Order]...>, Strides<T, _strides[Order]...>>();
    }

    inline constexpr auto undoPermutation() const noexcept {
        return undoPermutationImpl(std::make_index_sequence<sizeof...(ls)>());
    }

    template<std::size_t N>
    inline constexpr auto dimensionReduction() const noexcept {
        requires(sizeof...(ls) > 1U);
        return dimensionReductionImpl<N>(std::make_index_sequence<sizeof...(ls) - 1U>());
    }

    template<std::size_t N, int reductionSize>
    inline constexpr auto lengthReduction() const noexcept {
        return lengthReductionImpl<N, reductionSize>(std::make_index_sequence<sizeof...(ls)>());
    }

    template<std::size_t N, int newLength>
    inline constexpr auto setLength() const noexcept {
        return setLengthImpl<N, newLength>(std::make_index_sequence<sizeof...(ls)>());
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
        return newAxisImpl<N>(std::make_index_sequence<sizeof...(ls) + 1U>());
    }

    inline constexpr auto squeeze() const noexcept {
        constexpr std::size_t onesCount = internal::squeezeCount<ls...>();

        if constexpr(onesCount == 0U) return Shape<Lengths<ls...>, Strides<T, ss...>>();
        else
            return squeezeImpl(std::make_index_sequence<sizeof...(ls) - onesCount>());
    }

    inline constexpr bool containsPadding() const noexcept {
        if constexpr(sizeof...(ss) == 1U) return false;

        constexpr std::size_t maxStrideIdx = internal::maxStrideIndex<ss...>(std::make_index_sequence<sizeof...(ss)>());
        return ((ls * ...) != _lengths[maxStrideIdx] * _strides[maxStrideIdx]) ? true : false;
    }

    inline constexpr auto ravel() const noexcept {
        if constexpr(sizeof...(ls) == 1U) return Shape<Lengths<ls...>, Strides<T, ss...>>();

        constexpr std::size_t maxStrideIdx = internal::maxStrideIndex<ss...>(std::make_index_sequence<sizeof...(ss)>());
        requires(((ls * ...) == _lengths[maxStrideIdx] * _strides[maxStrideIdx]));
        return Shape<Lengths<(ls * ...)>, Strides<T, 1>>();
    }

    inline constexpr const auto lengths() const noexcept { return Lengths<ls...>(); }

    template<std::size_t N>
    inline constexpr const auto length() const noexcept {
        requires(N < _lengths.size());
        return _lengths[N];
    }

    inline constexpr const auto strides() const noexcept { return Strides<T, ss...>(); }

    template<std::size_t N>
    inline constexpr const auto stride() const noexcept {
        requires(N < _strides.size());
        return _strides[N];
    }

    inline constexpr std::size_t rank() const noexcept { return sizeof...(ls); }
};

/* Tensor shape type */
template<typename T, int... ls>
using TShape = Shape<Lengths<ls...>, decltype(internal::computeAlignedStrides<T, Lengths<ls...>>())>;

/* Tensor shape type with provided Lengths type */
template<typename T, template<int...> typename _Lengths, int... ls>
using TLShape = Shape<_Lengths<ls...>, decltype(internal::computeAlignedStrides<T, _Lengths<ls...>>())>;

/* Unaligned tensor shape type */
template<typename T, int... ls>
using TUShape = Shape<Lengths<ls...>, decltype(internal::computeUnalignedStrides<T, Lengths<ls...>>())>;

/* Unaligned tensor shape type with provided Lengths type */
template<typename T, template<int...> typename _Lengths, int... ls>
using TULShape = Shape<_Lengths<ls...>, decltype(internal::computeUnalignedStrides<T, _Lengths<ls...>>())>;

/* Tensor shape type with partially computed strides */
template<typename _Lengths, typename _Strides, bool aligned = true>
using TPShape = Shape<_Lengths, decltype(internal::partiallyComputeStrides<_Lengths, _Strides, aligned>())>;

} // namespace AboveInfinity
