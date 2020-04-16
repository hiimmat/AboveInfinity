#include "Tensor.h"

#include <fstream>
#include <string_view>

namespace AboveInfinity {

namespace internal {

/* Determines if the shape was permuted, and if it was, undoes the permutation */
template<typename _Shape>
inline constexpr auto determineInitialShape() noexcept {
    constexpr _Shape shape;

    if constexpr(std::is_same_v<std::decay_t<decltype(shape.strides())>,
                                decltype(internal::computeAlignedStrides<typename std::decay_t<_Shape>::Type,
                                                                         std::decay_t<decltype(shape.lengths())>>())>)
        return shape;
    else
        return shape.undoPermutation();
}

/* Copies the data of a multidimensional view into a flattened view */
template<class _TensorView, typename _Pointer>
auto flattenPermuted(_TensorView&& src, _Pointer&& ptr) {
    constexpr auto srcLengths = src.lengths().array();
    constexpr auto size = srcLengths.size();

    if constexpr(size > 1U)
        for(int i = 0; i < srcLengths[size - 1U]; ++i)
            flattenPermuted(src.template slice<size - 1U>(i), std::forward<_Pointer>(ptr));

    if constexpr(size == 1U) {
        auto srcData = src.data();
        constexpr auto srcStrides = src.strides().array();
        constexpr std::size_t batchSize = SimultaneousVecOps<typename _TensorView::Type>();
        /* If the innermost stride is 1, we can optimize the copying of the data */
        if constexpr(srcStrides[0U] == 1) {
            if constexpr(srcLengths[0U] <= batchSize) {
                memcpy(static_cast<void*>(*ptr),
                       static_cast<void*>(srcData),
                       srcLengths[0U] * sizeof(typename _TensorView::Type));
                *ptr += srcLengths[0U];
            } else {
                int iterSize = srcLengths[0U];
                int currIter = 0;

                while(iterSize >= batchSize) {
                    memcpy(static_cast<void*>(*ptr),
                           static_cast<void*>(srcData + currIter),
                           batchSize * sizeof(typename _TensorView::Type));
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
            if constexpr(srcLengths[0U] <= batchSize) {
                for(auto i = 0; i < srcLengths[0U]; ++i) {
                    **ptr = *(srcData + i * srcStrides[0U]);
                    ++*ptr;
                }
            } else {
                int iterSize = srcLengths[0U];
                int currIter = 0;

                while(iterSize >= batchSize) {
                    for(int i = 0; i < batchSize; ++i) {
                        **ptr = *(srcData + (currIter + i) * srcStrides[0U]);
                        ++*ptr;
                    }
                    iterSize -= batchSize;
                    currIter += batchSize;
                    *ptr += batchSize;
                }

                for(int i = currIter; i < currIter + iterSize; ++i) {
                    **ptr = *(srcData + (currIter + i) * srcStrides[0U]);
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

template<int firstOldLength, class _ReshapedLengths, std::size_t... is>
inline constexpr bool canReshapePaddedInplace(std::index_sequence<is...>) noexcept {
    constexpr _ReshapedLengths lengths;
    requires(sizeof...(is) == lengths.size());
    return canReshapePaddedInplaceImpl<firstOldLength, lengths.template get<is>()...>();
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

template<int firstOldLength, class _ReshapedLengths, std::size_t... is>
inline constexpr std::size_t numElementsToCopy(std::index_sequence<is...>) noexcept {
    constexpr _ReshapedLengths lengths;
    requires(canReshapePaddedInplace<firstOldLength, _ReshapedLengths>(std::make_index_sequence<lengths.size()>()));
    return numElementsToCopyImpl<firstOldLength, lengths.template get<is>()...>();
}

/*
 * Another helper function for reshaping a padded view in-place
 * Copies first N lengths as strides and then adds the padded stride
 */
template<typename T, class _Lengths, int offsetedStride, std::size_t... is>
inline constexpr auto partialReshapedStridesImpl(std::index_sequence<is...>) noexcept {
    constexpr _Lengths lengths;
    requires(sizeof...(is) > 0U && sizeof...(is) <= lengths.size());
    return Strides<T, lengths.template get<is>()..., offsetedStride>();
}

template<typename T, std::size_t N, class _Lengths, int offsetedStride>
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
template<typename Shape, std::size_t Planes>
inline void saveAsTextImpl(TensorView<Shape, Planes> view, std::ofstream& os) {
    constexpr Shape shape;
    constexpr auto lengths = shape.lengths().array();
    constexpr auto size = shape.lengths().size();

    if constexpr(size > 1U)
        for(int i = 0; i < lengths[size - 1U]; ++i) {
            os << "[";
            saveAsTextImpl(view.template slice<size - 1U>(i), os);
            os << "]\n";
        }

    if constexpr(size == 1U) {
        auto data = view.data();
        constexpr auto strides = shape.strides().array();
        constexpr std::size_t batchSize = SimultaneousVecOps<typename decltype(view)::Type>();
        if constexpr(lengths[0U] <= batchSize) {
            int i = 0;
            os << "[";
            for(; i < lengths[0U] - 1; ++i) os << *data + i * strides[0U] << ", ";
            os << *data + i * strides[0U] << "]";
        } else {
            int iterSize = lengths[0U];
            int currIter = 0;

            os << "[";

            while(iterSize > batchSize) {
                for(int i = currIter; i < currIter + batchSize; i++) os << *data + i * strides[0U] << ", ";
                iterSize -= batchSize;
                currIter += batchSize;
            }

            int i = currIter;
            for(; i < currIter + iterSize - 1; i++) os << *data + i * strides[0U] << ", ";
            os << *data + i * strides[0U] << "]";
        }
    }
}

/* Helper function that returns the first element of a sequence */
template<typename T, typename... Ts>
inline constexpr decltype(auto) front(T&& head, Ts&&... tail) noexcept {
    return std::forward<T>(head);
}

/* The logic behind execute is implemented here */
template<std::size_t N, typename F, typename... TensorViews>
inline void executeImpl(F&& func, TensorViews&&... views) {
    auto front = internal::front(views...);
    constexpr auto lengths = front.lengths().array();
    constexpr auto size = lengths.size();

    if constexpr(size == N) {
        for(std::size_t p = 0U; p < front.planes(); ++p) {
            for(int i = 0; i < lengths[size - 1U]; ++i)
                executeImpl<N>(std::forward<F>(func), views.keepPlane(p).template slice<size - 1U>(i)...);
        }
    }

    if constexpr(size < N && size > 1U) {
        for(int i = 0; i < lengths[size - 1U]; ++i)
            executeImpl<N>(std::forward<F>(func), views.template slice<size - 1U>(i)...);
    }

    if constexpr(size == 1U) {
        constexpr auto strides = front.strides().array();
        constexpr std::size_t batchSize = SimultaneousVecOps<typename decltype(front)::Type>();
        if constexpr(lengths[0U] <= batchSize)
            for(int i = 0; i < lengths[0U]; ++i) func(*(views.data() + i * strides[0U])...);
        else {
            int iterSize = lengths[0U];
            int currIter = 0;

            while(iterSize > batchSize) {
                for(int i = currIter; i < currIter + batchSize; i++) func(*(views.data() + i * strides[0U])...);
                iterSize -= batchSize;
                currIter += batchSize;
            }
            for(int i = currIter; i < currIter + iterSize; i++) func(*(views.data() + i * strides[0U])...);
        }
    }
}

/*
 * Faster implementation of copying one view into another that
 * possibly gets called from the copy function
 */
template<std::size_t N, typename SrcView, typename DestView>
inline void copyImpl(SrcView&& src, DestView&& dest) {
    constexpr auto lengths = src.lengths().array();
    constexpr auto size = lengths.size();

    if constexpr(size == N) {
        for(std::size_t p = 0U; p < src.planes(); ++p) {
            for(int i = 0; i < lengths[size - 1U]; ++i)
                copyImpl<N>(src.keepPlane(p).template slice<size - 1U>(i),
                            dest.keepPlane(p).template slice<size - 1U>(i));
        }
    }

    if constexpr(size < N && size > 1U) {
        for(int i = 0; i < lengths[size - 1U]; ++i)
            copyImpl<N>(src.template slice<size - 1U>(i), dest.template slice<size - 1U>(i));
    }

    if constexpr(size == 1U) {
        auto destData = dest.data();
        auto srcData = src.data();
        constexpr auto strides = src.strides().array();
        constexpr std::size_t batchSize = SimultaneousVecOps<typename SrcView::Type>();
        if constexpr(lengths[0U] <= batchSize)
            memcpy(
                static_cast<void*>(destData), static_cast<void*>(srcData), lengths[0] * sizeof(typename SrcView::Type));
        else {
            int iterSize = lengths[0U];
            int currIter = 0;

            while(iterSize > batchSize) {
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

}; // namespace internal

/* Changes the shape of a TensorView without changing the underlying data */
template<class _ReshapedLengths, class _TensorView>
constexpr auto reshape(_TensorView&& view) {
    constexpr auto shape = view.shape();
    constexpr _ReshapedLengths reshapedLengths;

    /* No need to do anything if the reshaped lengths match the current lengths */
    if constexpr(std::is_same_v<std::decay_t<decltype(shape.lengths())>, std::decay_t<_ReshapedLengths>>) return view;

    /* Assure that the new lengths are compatible with the old lengths */
    requires(std::is_same_v<std::decay_t<decltype(shape.lengths().flatten())>,
                            std::decay_t<decltype(reshapedLengths.flatten())>>);

    /*
     * Ignore dimensions consisting only of ones as those don't matter and could
     * potentially decrease performance (especially if it's the innermost dimension)
     */
    constexpr auto squeezedShape = shape.squeeze();

    /* Check if there's a padding between the 1st and 2nd dimension needed for alignment */
    if constexpr(squeezedShape.containsPadding()) {
        /* Undo permutation if there is one */
        constexpr auto initialShape = internal::determineInitialShape<std::decay_t<decltype(squeezedShape)>>();
        /* Try to avoid a new allocation */
        if constexpr(internal::canReshapePaddedInplace<initialShape.lengths().template get<0U>(), _ReshapedLengths>(
                         std::make_index_sequence<reshapedLengths.size()>())) {
            constexpr std::size_t N =
                internal::numElementsToCopy<initialShape.lengths().template get<0U>(), _ReshapedLengths>(
                    std::make_index_sequence<reshapedLengths.size()>());
            /* +1 for the dimension with the added offset */
            requires(N + 1U <= reshapedLengths.size());
            constexpr auto partialStrides =
                internal::partialReshapedStrides<typename std::decay_t<_TensorView>::Type,
                                                 N,
                                                 decltype(reshapedLengths),
                                                 initialShape.strides().template get<1U>()>();

            return TensorView<
                decltype(
                    Shape<typename std::decay_t<_TensorView>::Type,
                          _ReshapedLengths,
                          decltype(internal::partiallyComputeStrides<_ReshapedLengths, decltype(partialStrides)>())>()),
                view.planes()>{view.pointers()};
        } else {
            /* Worst case scenario:
             * We can't reshape inplace since the dimensions don't match, and either one or both Tensors have a padding
             * between their first and second dimension caused by alignment requirements
             */
            requires(initialShape.strides().template get<0U>() == 1);
            Tensor<decltype(Shape<typename std::decay_t<_TensorView>::Type, _ReshapedLengths>()), view.planes()>
                reshaped;

            requires(reshaped.strides().template get<0U>() == 1U);

            constexpr int innermostInLength = initialShape.lengths().template get<0U>();
            constexpr auto inStrides = initialShape.strides();
            constexpr int inOffset = initialShape.strides().template get<1U>() - innermostInLength;
            constexpr int innermostReshapedLength = reshapedLengths.template get<0U>();
            constexpr auto reshapedStrides = reshaped.strides();
            constexpr int reshapedOffset = reshaped.strides().template get<1U>() - innermostReshapedLength;
            constexpr std::size_t batchSize = SimultaneousVecOps<typename std::decay_t<_TensorView>::Type>();
            constexpr int lowerLength = (innermostInLength < innermostReshapedLength) ? innermostInLength :
                                                                                        innermostReshapedLength;
            constexpr int lowerStride = (innermostInLength < innermostReshapedLength) ?
                                            inStrides.template get<1U>() :
                                            reshapedStrides.template get<1U>();
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

                            while(iterSize >= batchSize) {
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

                        while(N > batchSize) {
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
                if(iteration == 0U) break;

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

                        while(iterSize >= batchSize) {
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
                              Shape<typename std::decay_t<_TensorView>::Type,
                                    _ReshapedLengths,
                                    decltype(internal::computeUnalignedStrides<typename std::decay_t<_TensorView>::Type,
                                                                               _ReshapedLengths>())>()),
                          view.planes()>{view.pointers()};
    }
}

/* Returns a copy of the TensorView collapsed into a single dimension */
template<bool keepMemoryOrder, class _TensorView>
auto flatten(_TensorView&& view) {
    constexpr auto shape = view.shape();

    /* If the rank of the view is already 1, don't attempt to flatten it */
    if constexpr(shape.rank() == 1U) return view;

    Tensor<Shape<typename std::decay_t<_TensorView>::Type,
                 decltype(shape.lengths().flatten()),
                 Strides<typename std::decay_t<_TensorView>::Type, 1>>,
           view.planes()>
        flattened;

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
                 ||
                 (std::is_same_v<
                     std::decay_t<decltype(squeezedShape.strides())>,
                     decltype(internal::computeUnalignedStrides<typename std::decay_t<_TensorView>::Type,
                                                                std::decay_t<decltype(squeezedShape.lengths())>>())>)) {
        requires(flattened.strides().template get<0U>() == 1);
        /* Assure that we're not using permuted lengths and strides */
        constexpr auto initialShape = internal::determineInitialShape<std::decay_t<decltype(squeezedShape)>>();
        requires(initialShape.strides().template get<0U>() == 1);
        constexpr int innermostLength = initialShape.lengths().template get<0U>();
        constexpr int inOffset = initialShape.strides().template get<1U>() - innermostLength;
        constexpr int N = squeezedShape.lengths().flatten().template get<0U>() / innermostLength;
        constexpr std::size_t batchSize = SimultaneousVecOps<typename std::decay_t<_TensorView>::Type>();

        for(std::size_t p = 0U; p < view.planes(); ++p) {
            auto inData = view.data(p);
            auto flattenedData = flattened.data(p);

            for(int i = 0; i < N; ++i) {
                if constexpr(innermostLength <= batchSize) {
                    memcpy(static_cast<void*>(flattenedData),
                           static_cast<void*>(inData),
                           innermostLength * sizeof(typename std::decay_t<_TensorView>::Type));
                    inData += initialShape.strides().template get<1U>();
                    flattenedData += innermostLength;
                } else {
                    int leftSize = innermostLength;

                    while(leftSize >= batchSize) {
                        memcpy(static_cast<void*>(flattenedData),
                               static_cast<void*>(inData),
                               batchSize * sizeof(typename std::decay_t<_TensorView>::Type));
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
        requires(flattened.strides().template get<0U>() == 1);
        for(std::size_t p = 0U; p < view.planes(); ++p) {
            auto flattenedData = flattened.data(p);
            auto flattenedDataPtr = &flattenedData;
            internal::flattenPermuted(view.keepPlane(p), flattenedDataPtr);
        };
    }

    return flattened;
}

/*
 * Attempts to return a flattened TensorView without copying the data
 * Copies the data only if it's necessary
 */
template<bool keepMemoryOrder, class _TensorView>
constexpr auto ravel(_TensorView&& view) {
    constexpr auto shape = view.shape();

    /* If the rank of the view is already 1, don't attempt to flatten it */
    if constexpr(shape.rank() == 1U) return view;

    /* Ignore dimensions consisting only of ones */
    constexpr auto squeezedShape = shape.squeeze();

    /* If there's padding between the first and the second dimension, we have to make a copy */
    if constexpr(squeezedShape.containsPadding()) {
        return flatten<keepMemoryOrder>(view);
    } else {
        /*
         * If there's no padding and we're either keeping the memory order or there's no permutation, we can represent
         * the existing view as a rank one view without making any additional copies
         */
        if constexpr(keepMemoryOrder ||
                     std::is_same_v<std::decay_t<decltype(squeezedShape.strides())>,
                                    decltype(internal::computeAlignedStrides<typename std::decay_t<_TensorView>::Type,
                                                                             decltype(squeezedShape.lengths())>())> ||
                     std::is_same_v<std::decay_t<decltype(squeezedShape.strides())>,
                                    decltype(internal::computeUnalignedStrides<typename std::decay_t<_TensorView>::Type,
                                                                               decltype(squeezedShape.lengths())>())>) {
            return TensorView<decltype(squeezedShape.ravel()), view.planes()>{view.pointers()};
        } else {
            /* Even if there's no padding between the first and second dimension, if we're not keeping the memory order
             * and there's a permutation, we have to copy the data from the view */
            return flatten<keepMemoryOrder>(view);
        }
    }
}

/* Saves a TensorView and its layout into a text file */
template<typename _Shape, std::size_t Planes>
inline void saveAsText(TensorView<_Shape, Planes> view, std::string&& fileName) {
    std::ofstream os(fileName, std::ios::out);
    /* Avoid squeezing ones as this has to represent the exact layout of the view */
    constexpr _Shape shape;
    constexpr auto lengths = shape.lengths().array();
    constexpr auto strides = shape.strides().array();
    constexpr std::size_t size = shape.lengths().size();

    os << "Type: " << internal::type_name<typename _Shape::Type>() << std::endl;
    os << "Type size: " << sizeof(typename _Shape::Type) << std::endl;
    /* Currently, there's only support Real and Planar data */
    os << "Memory type: " << (Planes == 1U ? "Real" : "Planar") << std::endl;

#ifdef _IS_BIG_ENDIAN
    os << "Bytes are ordered using " << (_IS_BIG_ENDIAN == 1) ? << "big endian\n" : "small endian\n";
#endif

    os << "Lengths: ";
    std::size_t i = 0U;

    for(; i < size - 1U; ++i) os << lengths[i] << ", ";
    os << lengths[i] << std::endl;

    os << "Strides: ";
    i = 0U;

    for(; i < size - 1U; ++i) os << strides[i] << ", ";
    os << strides[i] << std::endl;

    os << "Number of planes: " << Planes << std::endl << std::endl;

    for(std::size_t p = 0U; p < Planes; ++p) {
        os << "Plane " << p + 1U << ":\n";
        internal::saveAsTextImpl(view.keepPlane(p), os);
        os << std::endl;
    }
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
            requires(((std::is_same_v<decltype(front.lengths()), decltype(views.lengths())>)&&...));
            requires(((front.planes() == views.planes()) && ...));
            internal::executeImpl<front.lengths().size()>(std::forward<F>(func), std::forward<TensorViews>(views)...);
        } else {
            requires((std::is_invocable_v<F, decltype(*views.data())> && ...));
            ((internal::executeImpl<views.lengths().size()>(std::forward<F>(func), std::forward<TensorViews>(views))),
             ...);
        }
    } else {
        requires(sizeof...(views) == 1U);
        requires(std::is_invocable_v<F, decltype((*views.data()))...>);
        auto front = internal::front(views...);
        internal::executeImpl<front.lengths().size()>(std::forward<F>(func), std::forward<TensorViews>(views)...);
    }
}

/* Copies the data from one view into another */
template<class SrcView, class DestView>
inline void copy(SrcView&& src, DestView&& dest) {
    constexpr auto srcShape = src.shape().squeeze();
    constexpr auto destShape = dest.shape().squeeze();

    /* Assure that the source and destination view lengths match */
    requires(std::is_same_v<decltype(srcShape.lengths()), decltype(destShape.lengths())>);
    /* If the innermost strides are equal to 1 (which currently covers the assumption that the innermost
     * dimensions weren't permuted) and the types match, use the optimized copy implementation */
    if constexpr((srcShape.strides().template get<0U>() == 1 && destShape.strides().template get<0U>() == 1) &&
                 std::is_same_v<typename std::decay_t<SrcView>::Type, typename std::decay_t<DestView>::Type>)
        internal::copyImpl<src.lengths().size()>(src.squeeze(), dest.squeeze());
    else
        internal::executeImpl<src.lengths().size()>(
            [](typename std::decay_t<SrcView>::Reference source,
               typename std::decay_t<DestView>::Reference destination) { destination = source; },
            src.squeeze(),
            dest.squeeze());
}

} // namespace AboveInfinity
