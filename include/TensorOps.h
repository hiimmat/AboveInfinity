#include "Tensor.h"

#include <charconv>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>

namespace AboveInfinity {

namespace internal {

/* Determines if the shape was permuted, and if it was, undoes the permutation */
template<template<int...> typename _Lengths, int... ls, template<typename, int...> typename _Strides, int... ss, typename T>
inline constexpr auto determineInitialShape(Shape<_Lengths<ls...>, _Strides<T, ss...>> shape) noexcept {
    if constexpr(std::is_same_v<std::decay_t<_Strides<T, ss...>>,
                                decltype(internal::computeAlignedStrides<T, std::decay_t<_Lengths<ls...>>>())>)
        return shape;
    else
        return shape.undoPermutation();
}

/* Copies the data of a multidimensional view into a flattened view */
template<typename _TensorView, typename _Pointer>
void flattenPermuted(_TensorView&& src, _Pointer&& ptr) {
    constexpr std::size_t rank = src.rank();

    if constexpr(rank > 1U)
        for(int i = 0; i < src.template length<rank - 1U>(); ++i)
            flattenPermuted(src.template slice<rank - 1U>(i), std::forward<_Pointer>(ptr));

    if constexpr(rank == 1U) {
        auto srcData = src.data();
        constexpr int innerSrcLength = src.template length<0U>();
        constexpr int innerSrcStride = src.template stride<0U>();
        constexpr int batchSize = static_cast<int>(SimultaneousVecOps<typename _TensorView::Type>());
        /* If the innermost stride is 1, we can optimize the copying of the data */
        if constexpr(innerSrcStride == 1) {
            if constexpr(innerSrcLength <= batchSize) {
                memcpy(static_cast<void*>(*ptr),
                       static_cast<void*>(srcData),
                       innerSrcLength * sizeof(typename _TensorView::Type));
                *ptr += innerSrcLength;
            } else {
                int iterSize = innerSrcLength;
                int currIter = 0;

                while(iterSize >= batchSize) {
                    memcpy(static_cast<void*>(*ptr),
                           static_cast<void*>(srcData + currIter),
                           static_cast<std::size_t>(batchSize) * sizeof(typename _TensorView::Type));
                    iterSize -= batchSize;
                    currIter += batchSize;
                    *ptr += batchSize;
                }

                memcpy(static_cast<void*>(*ptr),
                       static_cast<void*>(srcData + currIter),
                       iterSize * sizeof(typename _TensorView::Type));

                *ptr += iterSize;
            }
        } else {
            if constexpr(innerSrcLength <= batchSize) {
                for(int i = 0; i < innerSrcLength; ++i) {
                    **ptr = *(srcData + i * innerSrcStride);
                    ++*ptr;
                }
            } else {
                int iterSize = innerSrcLength;
                int currIter = 0;

                while(iterSize >= batchSize) {
                    for(int i = 0; i < batchSize; ++i) {
                        **ptr = *(srcData + (currIter + i) * innerSrcStride);
                        ++*ptr;
                    }
                    iterSize -= batchSize;
                    currIter += batchSize;
                    *ptr += batchSize;
                }

                for(int i = currIter; i < currIter + iterSize; ++i) {
                    **ptr = *(srcData + (currIter + i) * innerSrcStride);
                    ++*ptr;
                }
            }
        }
    }
}

/*
 * Checks if a view that contains padding can be reshaped in-place by checking
 * if one or more of the new reshaped lengths can match the first length of
 * the old shape
 * To reshape a padded view inplace, the first new length or a multiply of a
 * number of the new lengths must be equal to the first old length. This way,
 * the new padded stride can inherit the old padded stride
 */
template<int cmp, int head, int... tail>
inline constexpr bool canReshapePaddedInplaceImpl() noexcept {
    if constexpr(sizeof...(tail) == 0) return (cmp / head == 1 && cmp % head == 0);
    else if constexpr(cmp / head == 1)
        return (cmp % head == 0) ? true : false;
    else if constexpr(cmp / head != 1 && cmp % head == 0)
        return canReshapePaddedInplaceImpl<cmp / head, tail...>();
    else
        return false;
}

template<int firstOldLength, template<int...> typename _Lengths, int... ls>
inline constexpr bool canReshapePaddedInplace(_Lengths<ls...>) noexcept {
    return canReshapePaddedInplaceImpl<firstOldLength, ls...>();
}

/* Determines the number of new lengths we need to copy before we reach the padded stride */
template<int cmp, int head, int... tail>
inline constexpr std::size_t numElementsToCopyImpl() noexcept {
    if constexpr(sizeof...(tail) == 0) return 1U;
    else if constexpr(cmp / head == 1)
        return 1U;
    else
        return 1U + numElementsToCopyImpl<cmp / head, tail...>();
}

template<int firstOldLength, template<int...> typename _Lengths, int... ls>
inline constexpr std::size_t numElementsToCopy(_Lengths<ls...>) noexcept {
    requires(canReshapePaddedInplaceImpl<firstOldLength, ls...>());
    return numElementsToCopyImpl<firstOldLength, ls...>();
}

/*
 * Another helper function for reshaping a padded view in-place
 * Copies first N lengths to replace the stride of the first dimension
 * and then adds the padded stride that was before the stride to access the
 * 2nd dimension
 */
template<typename T, typename _Lengths, int offsetedStride, std::size_t... is>
inline constexpr auto partialReshapedStridesImpl(std::index_sequence<is...>&&) noexcept {
    constexpr _Lengths lengths;
    requires(sizeof...(is) <= lengths.size());
    if constexpr(sizeof...(is) == 0U) return Strides<T, 1, offsetedStride>();
    else
        return Strides<T, 1, lengths.template get<is>()..., offsetedStride>();
}

template<typename T, std::size_t N, typename _Lengths, int offsetedStride>
inline constexpr auto partialReshapedStrides() noexcept {
    return partialReshapedStridesImpl<T, _Lengths, offsetedStride>(std::make_index_sequence<N>());
}

/*
 * funcSignature and type_name are a combination
 * of these two stackoverflow answers:
 * https://stackoverflow.com/a/56766138
 * https://stackoverflow.com/a/59522794
 */
template<typename T>
inline constexpr const std::string_view funcSignature() {
#ifdef _MSC_VER
    return std::string_view(__FUNCSIG__);
#else
    return std::string_view(__PRETTY_FUNCTION__);
#endif
}

template<typename T>
inline constexpr std::string_view type_name() {
    std::string_view name;

    constexpr std::string_view signature = funcSignature<int>();

    /*
     * This can be statically asserted in GCC on Ubuntu, but on Windows, __PRETTY_FUNCTION__ isn't a constexpr
     * Even though it's a regular if statement, on Ubuntu, this might evaluate at compile-time and give us an
     * error message during compilation
     * Also, the find function is looking for "T = int", instead of just "int", because looking just for "int"
     * causes an issue where the namespace name "internal" affects the output of the funcSignature function,
     * and the detected type name ends up as "inter"
     * This makes the search for the substring "T = int" less ideal than just checking for "int", since it
     * depends on each compiler using this code to produce the same substring as a part of the
     * __PRETTY_FUNCTION__ macro, but it should also avoid unexpected outputs produced by name mangling
     */
    if(signature.find("T = int") == std::string_view::npos)
        throw std::runtime_error("Unsupported compiler - couldn't determine the name of the passed type\n");

    std::size_t prefixSize = signature.find("T = int") + 4U;     // 4 is the length of the substring "T = "
    std::size_t suffixSize = signature.size() - prefixSize - 3U; // 3 is the length of int

    name = funcSignature<T>();
    name.remove_prefix(prefixSize);
    name.remove_suffix(suffixSize);
    return name;
}

/* Stores the underlying data of a view in the format that it was used */
template<typename _TensorView>
inline void saveAsTextImpl(_TensorView&& view, std::ofstream& os) {
    constexpr std::size_t rank = view.rank();

    if constexpr(rank > 1U)
        for(int i = 0; i < view.template length<rank - 1U>(); ++i) {
            os << "[";
            saveAsTextImpl(view.template slice<rank - 1U>(i), os);
            os << "]\n";
        }

    if constexpr(rank == 1U) {
        auto data = view.data();
        constexpr int innerLength = view.template length<0U>();
        constexpr int innerStride = view.template stride<0U>();
        constexpr std::size_t batchSize = SimultaneousVecOps<typename std::decay_t<_TensorView>::Type>();
        if constexpr(innerLength <= batchSize) {
            int i = 0;
            os << "[";
            for(; i < innerLength - 1; ++i) os << *data + i * innerStride << ", ";
            os << *data + i * innerStride << "]";
        } else {
            int iterSize = innerLength;
            int currIter = 0;

            os << "[";

            while(iterSize > static_cast<int>(batchSize)) {
                for(int i = currIter; i < currIter + static_cast<int>(batchSize); i++)
                    os << *data + i * innerStride << ", ";
                iterSize -= batchSize;
                currIter += batchSize;
            }

            int i = currIter;
            for(; i < currIter + iterSize - 1; i++) os << *data + i * innerStride << ", ";
            os << *data + i * innerStride << "]";
        }
    }
}

/* Implementation of the loadFromText function */
template<typename _TensorView, std::size_t... is>
inline void loadFromTextImpl(_TensorView&& view, std::string&& fileName, std::index_sequence<is...>) {
    constexpr std::size_t rank = view.rank();

    requires(sizeof...(is) == rank);
    requires(((is < rank) && ...));
    allUnique<is...>();

    std::ifstream inStream(fileName, std::ios::in | std::ios::ate);
    std::ifstream::pos_type fileSize = inStream.tellg();

    if(fileSize == -1) throw std::runtime_error("Could not deduce the size of the input file " + fileName + ".\n");
    else {
        inStream.seekg(0, std::ios::beg);
        std::string buffer;
        buffer.resize(fileSize);
        buffer.assign((std::istreambuf_iterator<char>(inStream)), std::istreambuf_iterator<char>());
        inStream.close();

        std::string dataStream = buffer.substr(buffer.find('['));

        std::array<int, rank> lengths = {};
        std::size_t currentPlane = 0U;
        std::size_t currentRank = rank;
        bool planeBracketOpen = false;

        for(std::size_t i = 0; i < dataStream.size(); ++i) {
            if(isspace(dataStream[i]) || dataStream[i] == ',') continue;
            else if(isdigit(dataStream[i])) {
                std::size_t start = i;
                while(isdigit(dataStream[i]) || dataStream[i] == '.') i++;
                std::size_t end = i;
                std::string substr = dataStream.substr(start, end - start);
                i--;

                typename _TensorView::Type value;

                /* from_chars doesn't currently support floating point conversions */
                if constexpr(std::is_floating_point<typename _TensorView::Type>::value) {
                    if constexpr(std::is_same_v<typename _TensorView::Type, float>) value = std::stof(substr);
                    else if constexpr(std::is_same_v<typename _TensorView::Type, double>)
                        value = std::stod(substr);
                    else
                        value = std::stold(substr);
                } else
                    auto [p, ec] = std::from_chars(substr.data(), substr.data() + substr.size(), value);

                *(view.slicingPointer(currentPlane, lengths[is]...)) = value;
                ++lengths[0U];
            } else if(dataStream[i] == '[')
                planeBracketOpen ? (--currentRank) : (planeBracketOpen = true);
            else if(dataStream[i] == ']') {
                if(currentRank < rank)
                    currentRank > 0U ? (lengths[currentRank - 1U] = 0, ++lengths[currentRank++]) : (currentRank++);
                else if(planeBracketOpen) {
                    planeBracketOpen = false;
                    ++currentPlane;
                    std::memset(lengths.data(), 0, lengths.size() * sizeof(lengths[0U]));
                } else if(dataStream[i] == '\0')
                    break;
                else
                    throw std::runtime_error("Error while loading tensor from file caused by unexpected character " +
                                             std::to_string(dataStream[i]) + ".\n");
            }
        }
    }
}

/* Helper function that returns the first element of a sequence */
template<typename T, typename... Ts>
inline constexpr decltype(auto) front(T&& head, Ts&&...) noexcept {
    return std::forward<T>(head);
}

/* Implementation of the execute function */
template<std::size_t N, typename F, typename... TensorViews>
inline void executeImpl(F&& func, TensorViews&&... views) {
    auto front = internal::front(views...);
    constexpr std::size_t rank = front.rank();

    if constexpr(rank == N) {
        for(std::size_t p = 0U; p < front.planes(); ++p) {
            for(int i = 0; i < front.template length<rank - 1U>(); ++i)
                executeImpl<N>(std::forward<F>(func), views.keepPlane(p).template slice<rank - 1U>(i)...);
        }
    }

    if constexpr(rank < N && rank > 1U) {
        for(int i = 0; i < front.template length<rank - 1U>(); ++i)
            executeImpl<N>(std::forward<F>(func), views.template slice<rank - 1U>(i)...);
    }

    if constexpr(rank == 1U) {
        // I fucked up the strides
        constexpr int innerLength = front.template length<0U>();
        constexpr int batchSize = static_cast<int>(SimultaneousVecOps<typename decltype(front)::Type>());
        if constexpr(innerLength <= batchSize)
            for(int i = 0; i < innerLength; ++i) func(*(views.data() + i * views.template stride<0U>())...);
        else {
            int iterSize = innerLength;
            int currIter = 0;

            while(iterSize > batchSize) {
                for(int i = currIter; i < currIter + batchSize; i++)
                    func(*(views.data() + i * views.template stride<0U>())...);
                iterSize -= batchSize;
                currIter += batchSize;
            }
            for(int i = currIter; i < currIter + iterSize; i++)
                func(*(views.data() + i * views.template stride<0U>())...);
        }
    }
}

/*
 * Faster implementation of copying the data one view is looking at
 * into another. This function can be called from the function "copy"
 */
template<std::size_t N, typename SrcView, typename DestView>
inline void copyImpl(SrcView&& src, DestView&& dest) {
    constexpr std::size_t rank = src.rank();

    if constexpr(rank == N) {
        for(std::size_t p = 0U; p < src.planes(); ++p) {
            for(int i = 0; i < src.template length<rank - 1U>(); ++i)
                copyImpl<N>(src.keepPlane(p).template slice<rank - 1U>(i),
                            dest.keepPlane(p).template slice<rank - 1U>(i));
        }
    } else if constexpr(rank < N && rank > 1U) {
        for(int i = 0; i < src.template length<rank - 1U>(); ++i)
            copyImpl<N>(src.template slice<rank - 1U>(i), dest.template slice<rank - 1U>(i));
    } else if constexpr(rank == 1U) {
        auto destData = dest.data();
        auto srcData = src.data();
        constexpr int innerLength = src.template length<0U>();
        constexpr std::size_t batchSize = SimultaneousVecOps<typename SrcView::Type>();
        if constexpr(innerLength <= batchSize)
            memcpy(static_cast<void*>(destData),
                   static_cast<void*>(srcData),
                   innerLength * sizeof(typename SrcView::Type));
        else {
            int iterSize = innerLength;
            int currIter = 0;

            while(iterSize > static_cast<int>(batchSize)) {
                memcpy(static_cast<void*>(destData + currIter),
                       static_cast<void*>(srcData + currIter),
                       batchSize * sizeof(typename SrcView::Type));
                iterSize -= batchSize;
                currIter += batchSize;
            }

            memcpy(static_cast<void*>(destData + currIter),
                   static_cast<void*>(srcData + currIter),
                   iterSize * sizeof(typename SrcView::Type));
        }
    }
}

} // namespace internal

/* Changes the shape of a TensorView without changing the underlying data */
template<typename _ReshapedLengths, typename _TensorView>
constexpr decltype(auto) reshape(_TensorView&& view) {
    constexpr auto shape = view.shape();
    constexpr _ReshapedLengths reshapedLengths;

    /* Assure that the new lengths are compatible with the old lengths */
    requires(std::is_same_v<std::decay_t<decltype(shape.lengths().flatten())>,
                            std::decay_t<decltype(reshapedLengths.flatten())>>);

    /*
     * Ignore dimensions consisting only of ones as those don't matter and could
     * potentially decrease performance (especially if it's the innermost dimension)
     */
    constexpr auto squeezedShape = shape.squeeze();

    Tensor<Shape<_ReshapedLengths,
                 decltype(internal::computeAlignedStrides<typename std::decay_t<_TensorView>::Type, _ReshapedLengths>())>,
           view.planes()>
        reshaped;

    /* No need to do anything if the reshaped lengths match the current lengths */
    if constexpr(std::is_same_v<std::decay_t<decltype(shape.lengths())>, std::decay_t<_ReshapedLengths>>)
        return std::forward<_TensorView>(view);
    /* Check if there's a padding between the 1st and 2nd dimension needed for alignment */
    else if constexpr(squeezedShape.containsPadding()) {
        /* Undo permutation if there is one */
        constexpr auto initialShape = internal::determineInitialShape(squeezedShape);
        /* Try to avoid a new allocation */
        if constexpr(internal::canReshapePaddedInplace<initialShape.template length<0U>()>(_ReshapedLengths())) {
            constexpr std::size_t N =
                internal::numElementsToCopy<initialShape.template length<0U>()>(_ReshapedLengths());
            /* +1 for the dimension with the added offset */
            requires(N + 1U <= reshapedLengths.size());

            /*
             * The first stride will be 1
             * N is the number of lengths that covers the length of the first dimension
             * Use those lengths minus the last one to cover the stride for the 1st dimension
             * Skip the last length because the padded stride will be used instead
             */
            constexpr auto partialStrides = internal::partialReshapedStrides<typename std::decay_t<_TensorView>::Type,
                                                                             N - 1U,
                                                                             decltype(reshapedLengths),
                                                                             initialShape.template stride<1U>()>();

            return TensorView<
                decltype(
                    Shape<_ReshapedLengths,
                          decltype(internal::partiallyComputeStrides<_ReshapedLengths, decltype(partialStrides)>())>()),
                view.planes()>{view.pointers()};
        } else {
            /* Worst case scenario:
             * We can't reshape inplace since the dimensions don't match, and either one or both Tensors have a padding
             * between their first and second dimension caused by alignment requirements
             */
            requires(initialShape.template stride<0U>() == 1);

            requires(reshaped.template stride<0U>() == 1U);

            constexpr int innermostInLength = initialShape.template length<0U>();
            /* If the first dimension is 1 and it contains padding, the squeezed shape won't be aware of this.
             * The first stride of the squeezed shape will be equal to the padded stride, and subtracting the
             * innermost length of the squeezed shape from the next stride won't give us the correct padding.
             */
            constexpr int inOffset = initialShape.template stride<1U>() - innermostInLength;
            constexpr int innermostReshapedLength = reshapedLengths.template get<0U>();
            constexpr int reshapedOffset = reshaped.template stride<1U>() - innermostReshapedLength;
            constexpr std::size_t batchSize = SimultaneousVecOps<typename std::decay_t<_TensorView>::Type>();
            constexpr int lowerLength = (innermostInLength < innermostReshapedLength) ? innermostInLength :
                                                                                        innermostReshapedLength;
            constexpr int lowerStride = (innermostInLength < innermostReshapedLength) ?
                                            initialShape.template stride<1U>() :
                                            reshaped.template stride<1U>();
            constexpr int lowerOffset = (innermostInLength < innermostReshapedLength) ? inOffset : reshapedOffset;
            constexpr int higherLength = (innermostInLength > innermostReshapedLength) ? innermostInLength :
                                                                                         innermostReshapedLength;
            constexpr int higherOffset = (innermostInLength > innermostReshapedLength) ? inOffset : reshapedOffset;

            for(std::size_t p = 0U; p < view.planes(); ++p) {
                auto inData = view.data(p);
                auto reshapedData = reshaped.data(p);
                auto lowerData = (innermostInLength < innermostReshapedLength) ? &inData : &reshapedData;
                auto higherData = (innermostInLength > innermostReshapedLength) ? &inData : &reshapedData;
                std::size_t iteration = initialShape.lengths().flatten().template get<0U>();
                std::size_t firstBatch = 0U;
                while(iteration >= higherLength) {
                    if constexpr(lowerLength <= batchSize) {
                        int N = higherLength;
                        while(firstBatch > 0U) {
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   firstBatch * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += (firstBatch + lowerOffset);
                            *higherData += firstBatch;
                            N -= firstBatch;
                            firstBatch = 0U;
                        }

                        while(N >= lowerLength) {
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   lowerLength * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += lowerStride;
                            *higherData += lowerLength;
                            N -= lowerLength;
                        }
                        memcpy(static_cast<void*>(reshapedData),
                               static_cast<void*>(inData),
                               N * sizeof(typename std::decay_t<_TensorView>::Type));
                        *higherData += (N + higherOffset);
                        *lowerData += N;
                        firstBatch = lowerLength - N;
                    } else {
                        int N = higherLength;

                        while(firstBatch > 0U) {
                            while(firstBatch >= batchSize) {
                                memcpy(static_cast<void*>(reshapedData),
                                       static_cast<void*>(inData),
                                       batchSize * sizeof(typename std::decay_t<_TensorView>::Type));
                                *lowerData += batchSize;
                                *higherData += batchSize;
                                firstBatch -= batchSize;
                                N -= batchSize;
                            }
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   firstBatch * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += (firstBatch + lowerOffset);
                            *higherData += firstBatch;
                            N -= firstBatch;
                            firstBatch = 0U;
                        }

                        while(N >= lowerLength) {
                            int iterSize = lowerLength;

                            while(iterSize >= static_cast<int>(batchSize)) {
                                memcpy(static_cast<void*>(reshapedData),
                                       static_cast<void*>(inData),
                                       batchSize * sizeof(typename std::decay_t<_TensorView>::Type));
                                *lowerData += batchSize;
                                *higherData += batchSize;
                                iterSize -= batchSize;
                                N -= batchSize;
                            }
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   iterSize * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += (iterSize + lowerOffset);
                            *higherData += iterSize;
                            N -= iterSize;
                        }

                        while(N > static_cast<int>(batchSize)) {
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   batchSize * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += batchSize;
                            *higherData += batchSize;
                            N -= batchSize;
                        }

                        memcpy(static_cast<void*>(reshapedData),
                               static_cast<void*>(inData),
                               N * sizeof(typename std::decay_t<_TensorView>::Type));
                        *higherData += (N + higherOffset);
                        *lowerData += N;
                        firstBatch = lowerLength - N;
                    }
                    iteration -= higherLength;
                }
                if(iteration == 0U) continue;

                if constexpr(lowerLength <= batchSize) {
                    if(iteration >= firstBatch)
                        while(firstBatch > 0U) {
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   firstBatch * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += (firstBatch + lowerOffset);
                            *higherData += firstBatch;
                            iteration -= firstBatch;
                            firstBatch = 0U;
                        }

                    while(iteration >= lowerLength) {
                        memcpy(static_cast<void*>(reshapedData),
                               static_cast<void*>(inData),
                               lowerLength * sizeof(typename std::decay_t<_TensorView>::Type));
                        *lowerData += lowerStride;
                        *higherData += lowerLength;
                        iteration -= lowerLength;
                    }
                    memcpy(static_cast<void*>(reshapedData),
                           static_cast<void*>(inData),
                           iteration * sizeof(typename std::decay_t<_TensorView>::Type));
                } else {
                    if(iteration >= firstBatch)
                        while(firstBatch > 0U) {
                            while(firstBatch >= batchSize) {
                                memcpy(static_cast<void*>(reshapedData),
                                       static_cast<void*>(inData),
                                       batchSize * sizeof(typename std::decay_t<_TensorView>::Type));
                                *lowerData += batchSize;
                                *higherData += batchSize;
                                firstBatch -= batchSize;
                                iteration -= batchSize;
                            }
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   firstBatch * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += (firstBatch + lowerOffset);
                            *higherData += firstBatch;
                            firstBatch = 0U;
                            iteration -= firstBatch;
                        }

                    while(iteration >= lowerLength) {
                        int iterSize = lowerLength;

                        while(iterSize >= static_cast<int>(batchSize)) {
                            memcpy(static_cast<void*>(reshapedData),
                                   static_cast<void*>(inData),
                                   batchSize * sizeof(typename std::decay_t<_TensorView>::Type));
                            *lowerData += batchSize;
                            *higherData += batchSize;
                            iterSize -= batchSize;
                            iteration -= batchSize;
                        }
                        memcpy(static_cast<void*>(reshapedData),
                               static_cast<void*>(inData),
                               iterSize * sizeof(typename std::decay_t<_TensorView>::Type));
                        *lowerData += (iterSize + lowerOffset);
                        *higherData += iterSize;
                        iteration -= iterSize;
                    }

                    while(iteration >= batchSize) {
                        memcpy(static_cast<void*>(reshapedData),
                               static_cast<void*>(inData),
                               batchSize * sizeof(typename std::decay_t<_TensorView>::Type));
                        *lowerData += batchSize;
                        *higherData += batchSize;
                        iteration -= batchSize;
                    }

                    memcpy(static_cast<void*>(reshapedData),
                           static_cast<void*>(inData),
                           iteration * sizeof(typename std::decay_t<_TensorView>::Type));
                }
            }
            return reshaped;
        }
    } else {
        /*
         * If there's no padding between the first and second dimension, we can apply the new shape to the
         * TensorView by using the lengths passed to the function and by computing new strides. Use
         * computeUnalignedStrides to avoid out of bounds errors
         */
        return TensorView<decltype(
                              Shape<_ReshapedLengths,
                                    decltype(internal::computeUnalignedStrides<typename std::decay_t<_TensorView>::Type,
                                                                               _ReshapedLengths>())>()),
                          view.planes()>{view.pointers()};
    }
}

/* Returns a copy of the TensorView collapsed into a single dimension */
template<bool keepMemoryOrder, typename _TensorView>
decltype(auto) flatten(_TensorView&& view) {
    constexpr auto shape = view.shape();

    /*
     * I tried to put this tensor into the else statement that can be found lower. However, once it
     * was returned from the function, its data was garbage. Moving it outside of the else statement
     * did the trick */
    Tensor<Shape<decltype(shape.lengths().flatten()), Strides<typename std::decay_t<_TensorView>::Type, 1>>,
           view.planes()>
        flattened;

    /* If the rank of the view is already 1, don't attempt to flatten it */
    if constexpr(shape.rank() == 1U) return std::forward<_TensorView>(view);
    /*
     * If the rank of the view would be 1, but it isn't because there's one dimension that isn't 1, but all the other
     * ones are ones, and there's no padding, don't attempt to flatten it
     */
    else if constexpr(!shape.containsPadding() && shape.squeeze().rank() == 1U)
        return TensorView<
            Shape<decltype(shape.lengths().flatten()), Strides<typename std::decay_t<_TensorView>::Type, 1>>,
            view.planes()>(view.pointers());
    else {
        /* Ignore dimensions consisting only of ones */
        constexpr auto squeezedShape = shape.squeeze();

        /*
         * If we're keeping the memory order, or if there is no permutation, we can copy the
         * data in batches of size equal to the innermost dimension (or batchSize if the
         * innermost dimension is too large)
         */
        if constexpr(keepMemoryOrder ||
                     (std::is_same_v<
                         std::decay_t<decltype(squeezedShape.strides())>,
                         decltype(internal::computeAlignedStrides<typename std::decay_t<_TensorView>::Type,
                                                                  std::decay_t<decltype(squeezedShape.lengths())>>())>) ||
                     (std::is_same_v<std::decay_t<decltype(squeezedShape.strides())>,
                                     decltype(internal::computeUnalignedStrides<
                                              typename std::decay_t<_TensorView>::Type,
                                              std::decay_t<decltype(squeezedShape.lengths())>>())>)) {
            /*
             * Asserting that the first stride is 1 in these two cases because this might change when interleaved data
             * gets introduced and the algorithm will have to be adjusted
             */
            requires(flattened.template stride<0U>() == 1);
            constexpr auto initialShape = internal::determineInitialShape(squeezedShape);
            requires(initialShape.template stride<0U>() == 1);
            constexpr int innermostLength = initialShape.template length<0U>();
            constexpr int inOffset = initialShape.template stride<1U>() - innermostLength;
            constexpr int N = squeezedShape.lengths().flatten().template get<0U>() / innermostLength;
            constexpr int batchSize = static_cast<int>(SimultaneousVecOps<typename std::decay_t<_TensorView>::Type>());

            for(std::size_t p = 0U; p < view.planes(); ++p) {
                auto inData = view.data(p);
                auto flattenedData = flattened.data(p);

                for(int i = 0; i < N; ++i) {
                    if constexpr(innermostLength <= batchSize) {
                        memcpy(static_cast<void*>(flattenedData),
                               static_cast<void*>(inData),
                               innermostLength * sizeof(typename std::decay_t<_TensorView>::Type));
                        inData += initialShape.template stride<1U>();
                        flattenedData += innermostLength;
                    } else {
                        int leftSize = innermostLength;

                        while(leftSize >= batchSize) {
                            memcpy(static_cast<void*>(flattenedData),
                                   static_cast<void*>(inData),
                                   static_cast<std::size_t>(batchSize) *
                                       sizeof(typename std::decay_t<_TensorView>::Type));
                            inData += batchSize;
                            flattenedData += batchSize;
                            leftSize -= batchSize;
                        }
                        memcpy(static_cast<void*>(flattenedData),
                               static_cast<void*>(inData),
                               leftSize * sizeof(typename std::decay_t<_TensorView>::Pointer));
                        inData += (leftSize + inOffset);
                        flattenedData += leftSize;
                    }
                }
            }
        } else {
            /*
             * Worst case scenario:
             * We're not keeping the memory order, and we have a permuted view
             */
            requires(flattened.template stride<0U>() == 1);
            for(std::size_t p = 0U; p < view.planes(); ++p) {
                auto flattenedData = flattened.data(p);
                auto flattenedDataPtr = &flattenedData;
                internal::flattenPermuted(view.keepPlane(p), flattenedDataPtr);
            };
        }

        return flattened;
    }
}

/*
 * Attempts to return a flattened TensorView without copying the data
 * Copies the data only if it's necessary
 */
template<bool keepMemoryOrder, typename _TensorView>
constexpr decltype(auto) ravel(_TensorView&& view) {
    constexpr auto shape = view.shape();
    /* Ignore dimensions consisting only of ones */
    constexpr auto squeezedShape = shape.squeeze();

    /* If the rank of the view is already 1, don't attempt to flatten it */
    if constexpr(shape.rank() == 1U) return std::forward<_TensorView>(view);
    /* If there's no padding and the squeezed view has rank 1, don't flatten it */
    else if constexpr(!shape.containsPadding() && squeezedShape.rank() == 1U)
        return TensorView<std::decay_t<decltype(squeezedShape)>, view.planes()>{view.pointers()};
    /* If there's padding between the first and the second dimension, we have to make a copy */
    else if constexpr(squeezedShape.containsPadding()) {
        return flatten<keepMemoryOrder>(view);
    } else {
        /*
         * If there's no padding and we're either keeping the memory order or there's no permutation, we can represent
         * the existing view as a rank one view without making any additional copies
         */
        if constexpr(keepMemoryOrder ||
                     std::is_same_v<
                         std::decay_t<decltype(squeezedShape.strides())>,
                         decltype(internal::computeAlignedStrides<typename std::decay_t<_TensorView>::Type,
                                                                  std::decay_t<decltype(squeezedShape.lengths())>>())> ||
                     std::is_same_v<std::decay_t<decltype(squeezedShape.strides())>,
                                    decltype(internal::computeUnalignedStrides<
                                             typename std::decay_t<_TensorView>::Type,
                                             std::decay_t<decltype(squeezedShape.lengths())>>())>) {
            return TensorView<std::decay_t<decltype(squeezedShape.ravel())>, view.planes()>{view.pointers()};
        } else {
            /* Even if there's no padding between the first and second dimension, if we're not keeping the memory order
             * and there's a permutation, we have to copy the data from the view */
            return flatten<keepMemoryOrder>(view);
        }
    }
}

/* Saves a TensorView and its layout into a text file */
template<typename _TensorView>
inline void saveAsText(_TensorView&& view, std::string&& fileName) {
    std::ofstream os(fileName);
    /* Avoid squeezing ones as this has to represent the exact layout of the view */
    constexpr auto lengths = view.lengths().array();
    constexpr auto strides = view.strides().array();
    constexpr std::size_t rank = view.rank();
    constexpr std::size_t numPlanes = view.planes();

    os << "Type: " << internal::type_name<typename std::decay_t<_TensorView>::Type>() << std::endl;
    os << "Type size: " << sizeof(typename std::decay_t<_TensorView>::Type) << std::endl;
    /* Currently, there's only support Real and Planar data */
    os << "Memory type: " << (numPlanes == 1U ? "Real" : "Planar") << std::endl;
    os << "Number of planes: " << numPlanes << std::endl;

#ifdef _IS_BIG_ENDIAN
    os << "Bytes are ordered using " << _IS_BIG_ENDIAN ? << "big endian\n" : "small endian\n";
#endif

    os << "Lengths: ";
    std::size_t i = 0U;

    for(; i < rank - 1U; ++i) os << lengths[i] << ", ";
    os << lengths[i] << std::endl;

    os << "Strides: ";
    i = 0U;

    for(; i < rank - 1U; ++i) os << strides[i] << ", ";
    os << strides[i] << std::endl;

    for(std::size_t p = 0U; p < numPlanes; ++p) {
        os << "[";
        internal::saveAsTextImpl(view.keepPlane(p), os);
        os << "]"
           << "\n";
    }
    os.close();
}

/*
 * Loads a tensor from a text file. The function receives a view to a Tensor that will hold the data
 * and the absolute path to the file being read.
 * WARNING
 * Currently, there are no checks that assure that the planes, lengths and type between the instantiated
 * tensor and the tensor being loaded match.
 */
template<typename _TensorView>
inline void loadFromText(_TensorView&& view, std::string&& fileName) {
    internal::loadFromTextImpl(
        std::forward<_TensorView>(view), std::forward<std::string>(fileName), std::make_index_sequence<view.rank()>());
}

/*
 * Executes a function over one or more views
 * It can execute the same function over several view simultaneously, or one at a time
 * depending on the definition of the function that needs to be executed
 * This function expands into for loops during compile time that iterate one or more views simultaneously
 */
template<typename F, typename... TensorViews>
inline void execute(F&& func, TensorViews&&... views) {
    if constexpr(sizeof...(views) > 1U) {
        if constexpr(std::is_invocable_v<F, decltype((*views.data()))...>) {
            auto front = internal::front(views...);
            requires(((
                std::is_same_v<std::decay_t<decltype(front.lengths())>, std::decay_t<decltype(views.lengths())>>)&&...));
            requires(((front.planes() == views.planes()) && ...));
            internal::executeImpl<front.rank()>(std::forward<F>(func), std::forward<TensorViews>(views)...);
        } else {
            requires((std::is_invocable_v<F, decltype(*views.data())> && ...));
            ((internal::executeImpl<views.rank()>(std::forward<F>(func), std::forward<TensorViews>(views))), ...);
        }
    } else {
        requires(sizeof...(views) == 1U);
        requires(std::is_invocable_v<F, decltype((*views.data()))...>);
        auto front = internal::front(views...);
        internal::executeImpl<front.rank()>(std::forward<F>(func), std::forward<TensorViews>(views)...);
    }
}

/* Copies the data from one view into another */
template<typename SrcView, typename DestView>
inline void copy(SrcView&& src, DestView&& dest) {
    constexpr auto srcShape = src.shape().squeeze();
    constexpr auto destShape = dest.shape().squeeze();

    /* Assure that the source and destination view lengths match */
    requires(std::is_same_v<std::decay_t<decltype(srcShape.lengths())>, std::decay_t<decltype(destShape.lengths())>>);

    /* Assure that the numbers of planes match between the views */
    requires(src.planes() == dest.planes());

    /* If the innermost strides are equal to 1 (which currently covers the assumption that the innermost
     * dimensions weren't permuted) and the types match, use the optimized copy implementation */
    if constexpr((srcShape.template stride<0U>() == 1 && destShape.template stride<0U>() == 1) &&
                 std::is_same_v<typename std::decay_t<SrcView>::Type, typename std::decay_t<DestView>::Type>)
        internal::copyImpl<src.rank()>(src.squeeze(), dest.squeeze());
    else
        internal::executeImpl<src.rank()>(
            [](typename std::decay_t<SrcView>::Type source, typename std::decay_t<DestView>::Reference destination) {
                destination = source;
            },
            src.squeeze(),
            dest.squeeze());
}

} // namespace AboveInfinity
