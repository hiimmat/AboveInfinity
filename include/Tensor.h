#include "AlignedMemory.h"
#include "TensorView.h"

namespace AboveInfinity {

namespace internal {

/* Computes the size used for allocating a Tensor (not including the size of the underlying type) */
template<typename _Shape, std::size_t Planes>
inline constexpr int TensorSize() noexcept {
    requires(Planes > 0U);
    constexpr _Shape shape;
    constexpr std::size_t N = shape.rank() - 1U;
    return shape.template length<N>() * shape.template stride<N>() * static_cast<int>(Planes);
}

} // namespace internal

/*
 * This class represents a multidimensional array of fixed size items
 * It's described using a Shape consisting of the type of
 * each element, Lengths (number of elements in each dimension),
 * Strides (offset needed to access each dimension) and the
 * underlying memory layout (currently either Real or Planar)
 * This representation is specifically used for describing a
 * move only memory block on the stack
 */
template<typename, std::size_t>
class StackTensor {
    ~StackTensor() = delete;
    StackTensor(StackTensor const&) = delete;
    void operator=(StackTensor const&) = delete;
};

template<typename T, int... ls, int... ss, std::size_t Planes>
class StackTensor<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes> {
    requires(Planes > 0U);

private:
    alignas(AIAlignment) std::array<
        T,
        static_cast<std::size_t>(internal::TensorSize<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes>())> memory;
    std::array<T*, Planes> _ptrs;
    static constexpr auto _lengths = std::array{ls...};
    static constexpr auto _strides = std::array{ss...};

    template<std::size_t Plane, int... lenOffsets, std::size_t... is>
    inline T* slicingPointerImpl(std::index_sequence<is...>&&) const {
        constexpr std::size_t offsetsSize = sizeof...(lenOffsets);
        constexpr std::size_t lengthsSize = sizeof...(ls);
        constexpr auto offsets = std::array{lenOffsets...};

        requires(Plane < Planes);
        requires(sizeof...(is) == offsetsSize);
        requires(offsetsSize <= lengthsSize);
        requires(((lenOffsets >= 0) && ...));
        requires(((offsets[offsetsSize - is - 1U] < _lengths[lengthsSize - is - 1U]) && ...));

        T* _memory = _ptrs[Plane];

        return _memory + ((((offsets[offsetsSize - is - 1U] * _strides[lengthsSize - is - 1U])) + ...));
    }

    template<std::size_t... is, typename... Offsets>
    inline T* slicingPointerImpl(std::size_t Plane, std::index_sequence<is...>&&, Offsets&&... offsets) const {
        requires(((std::is_same_v<std::decay_t<Offsets>, int>)&&...));
        constexpr std::size_t nOffsets = sizeof...(offsets);
        constexpr std::size_t rank = sizeof...(ls);
        auto offsetsArr = std::array{offsets...};

        requires(sizeof...(is) == nOffsets);
        requires(nOffsets <= rank);

        T* _memory = _ptrs[Plane];

        return _memory + ((((offsetsArr[nOffsets - is - 1U] * _strides[rank - is - 1U])) + ...));
    }

public:
    using Type = T;
    using Reference = T&;
    using Pointer = T*;

    inline StackTensor() {
        _ptrs[0U] = memory.data();

        if constexpr(Planes > 1U)
            for(std::size_t i = 1U; i < Planes; ++i)
                _ptrs[i] = _ptrs[i - 1U] + _lengths[sizeof...(ls) - 1U] * _strides[sizeof...(ss) - 1U];
    }

    StackTensor(const StackTensor&) = delete;
    StackTensor& operator=(const StackTensor&) = delete;

    inline StackTensor(StackTensor&&) noexcept = default;
    inline StackTensor& operator=(StackTensor&&) noexcept = default;

    /* Returns a TensorView pointing to the data of the StackTensor */
    inline auto view() const noexcept { return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes>{_ptrs}; }

    /*
     * Retrieves the pointer to a dimension on the specified plane
     * The same result can be achieved by calling slice several times and later
     * retrieving the pointer for the specified plane through the data function
     */
    template<std::size_t Plane, int... lenOffsets>
    inline T* slicingPointer() const {
        return slicingPointerImpl<Plane, lenOffsets...>(std::make_index_sequence<sizeof...(lenOffsets)>());
    }

    /*
     * Runtime version of the previously defined "slicingPointer" function
     * WARNING
     * This function has no bound checks checking the passed offsets
     */
    template<typename... Offsets>
    inline constexpr T* slicingPointer(std::size_t Plane, Offsets&&... offsets) const {
        return slicingPointerImpl(Plane, std::make_index_sequence<sizeof...(offsets)>(), offsets...);
    }

    /* Returns the Shape of the StackTensor */
    inline constexpr const auto shape() const noexcept { return Shape<Lengths<ls...>, Strides<T, ss...>>(); }

    /* Returns the underlying lengths of the StackTensor */
    inline constexpr const auto lengths() const noexcept { return Lengths<ls...>(); }

    /* Returns the length of the requested dimension */
    template<std::size_t N>
    inline constexpr int length() const noexcept {
        requires(N < sizeof...(ls));
        return _lengths[N];
    }

    /* Returns the underlying strides of the StackTensor */
    inline constexpr const auto strides() const noexcept { return Strides<T, ss...>(); }

    /* Returns the stride of the requested dimension */
    template<std::size_t N>
    inline constexpr int stride() const noexcept {
        requires(N < sizeof...(ss));
        return _strides[N];
    }

    /* Returns the number of planes of the StackTensor */
    inline constexpr std::size_t planes() const noexcept { return Planes; }

    /* Returns the rank of the StackTensor */
    inline constexpr std::size_t rank() noexcept { return sizeof...(ls); }

    /* Returns a pointer to the beginning of the data for a specified plane of the StackTensor */
    template<std::size_t Plane = 0U>
    inline T* data() noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /* Returns a const pointer to the beginning of the data for a specified plane of the StackTensor */
    template<std::size_t Plane = 0U>
    inline const T* data() const noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /*
     * Returns a pointer to the beginning of the data for a specified plane of the StackTensor
     * WARNING
     * No bound checks are performed
     */
    inline T* data(std::size_t plane = 0U) { return _ptrs[plane]; }

    /*
     * Returns a const pointer to the beginning of the data for a specified plane of the Tensor
     * WARNING
     * No bound checks are performed
     */
    inline const T* data(std::size_t plane = 0U) const { return _ptrs[plane]; }
};

/*
 * This class represents a multidimensional array of fixed size items
 * It's described using a Shape consisting of the type of
 * each element, Lengths (number of elements in each dimension),
 * Strides (offset needed to access each dimension) and the
 * underlying memory layout (currently either Real or Planar)
 * This representation is specifically used for describing a
 * move only memory block on the heap
 */
template<typename, std::size_t>
class HeapTensor {
    ~HeapTensor() = delete;
    HeapTensor(HeapTensor const&) = delete;
    void operator=(HeapTensor const&) = delete;
};

template<typename T, int... ls, int... ss, std::size_t Planes>
class HeapTensor<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes> {
    requires(Planes > 0U);

private:
    AlignedMemory<T> memory;
    std::array<T*, Planes> _ptrs;
    static constexpr auto _lengths = std::array{ls...};
    static constexpr auto _strides = std::array{ss...};

    template<std::size_t Plane, int... lenOffsets, std::size_t... is>
    inline T* slicingPointerImpl(std::index_sequence<is...>&&) const {
        constexpr std::size_t offsetsSize = sizeof...(lenOffsets);
        constexpr std::size_t lengthsSize = sizeof...(ls);
        constexpr auto offsets = std::array{lenOffsets...};

        requires(Plane < Planes);
        requires(sizeof...(is) == offsetsSize);
        requires(offsetsSize <= lengthsSize);
        requires(((lenOffsets >= 0) && ...));
        requires(((offsets[offsetsSize - is - 1U] < _lengths[lengthsSize - is - 1U]) && ...));

        T* _memory = _ptrs[Plane];

        return _memory + ((((offsets[offsetsSize - is - 1U] * _strides[lengthsSize - is - 1U])) + ...));
    }

    template<std::size_t... is, typename... Offsets>
    inline T* slicingPointerImpl(std::size_t Plane, std::index_sequence<is...>&&, Offsets&&... offsets) const {
        requires(((std::is_same_v<std::decay_t<Offsets>, int>)&&...));
        constexpr std::size_t nOffsets = sizeof...(offsets);
        constexpr std::size_t rank = sizeof...(ls);
        auto offsetsArr = std::array{offsets...};

        requires(sizeof...(is) == nOffsets);
        requires(nOffsets <= rank);

        T* _memory = _ptrs[Plane];

        return _memory + ((((offsetsArr[nOffsets - is - 1U] * _strides[rank - is - 1U])) + ...));
    }

public:
    using Type = T;
    using Reference = T&;
    using Pointer = T*;

    HeapTensor() {
        memory = AlignedMemory<T>(
            static_cast<std::size_t>(internal::TensorSize<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes>()),
            AIAlignment);
        _ptrs[0U] = memory.data();

        if constexpr(Planes > 1U)
            for(std::size_t i = 1U; i < Planes; ++i)
                _ptrs[i] = _ptrs[i - 1U] + _lengths[sizeof...(ls) - 1U] * _strides[sizeof...(ss) - 1U];
    }

    /* Returns a TensorView pointing to the data of the HeapTensor */
    inline auto view() const noexcept { return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes>{_ptrs}; }

    /*
     * Retrieves the pointer to a dimension on the specified plane
     * The same result can be achieved by calling slice several times and later
     * retrieving the pointer for the specified plane through the data function
     */
    template<std::size_t Plane, int... lenOffsets>
    inline T* slicingPointer() const {
        return slicingPointerImpl<Plane, lenOffsets...>(std::make_index_sequence<sizeof...(lenOffsets)>());
    }

    /*
     * Runtime version of the previously defined "slicingPointer" function
     * WARNING
     * This function has no bound checks checking the passed offsets
     */
    template<typename... Offsets>
    inline constexpr T* slicingPointer(std::size_t Plane, Offsets&&... offsets) const {
        return slicingPointerImpl(Plane, std::make_index_sequence<sizeof...(offsets)>(), offsets...);
    }

    /* Returns the Shape of the HeapTensor */
    inline constexpr const auto shape() const noexcept { return Shape<Lengths<ls...>, Strides<T, ss...>>(); }

    /* Returns the underlying lengths of the HeapTensor */
    inline constexpr const auto lengths() const noexcept { return Lengths<ls...>(); }

    /* Returns the length of the requested dimension */
    template<std::size_t N>
    inline constexpr int length() const noexcept {
        requires(N < sizeof...(ls));
        return _lengths[N];
    }

    /* Returns the underlying strides of the HeapTensor */
    inline constexpr const auto strides() const noexcept { return Strides<T, ss...>(); }

    /* Returns the stride of the requested dimension */
    template<std::size_t N>
    inline constexpr int stride() const noexcept {
        requires(N < sizeof...(ss));
        return _strides[N];
    }

    /* Returns the number of planes of the HeapTensor */
    inline constexpr std::size_t planes() const noexcept { return Planes; }

    /* Returns the rank of the HeapTensor */
    inline constexpr std::size_t rank() noexcept { return sizeof...(ls); }

    /* Returns a pointer to the beginning of the data for a specified plane of the HeapTensor */
    template<std::size_t Plane = 0U>
    inline T* data() noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /* Returns a const pointer to the beginning of the data for a specified plane of the HeapTensor */
    template<std::size_t Plane = 0U>
    inline const T* data() const noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /*
     * Returns a pointer to the beginning of the data for a specified plane of the HeapTensor
     * WARNING
     * No bound checks are performed
     */
    inline T* data(std::size_t plane = 0U) { return _ptrs[plane]; }

    /*
     * Returns a const pointer to the beginning of the data for a specified plane of the HeapTensor
     * WARNING
     * No bound checks are performed
     */
    inline const T* data(std::size_t plane = 0U) const { return _ptrs[plane]; }
};

/*
 * Creates an alias for the StackTensor and HeapTensor and determines which one of those two will be used
 * when creating a new Tensor. This is based on the Tensor size multiplied with the size of its underlying
 * type. If the size exceeds the maxStackAllocSize, a HeapTensor will be constructed. Otherwise, a
 * StackTensor will be created
 */
template<typename _Shape, std::size_t Planes = 1U>
using Tensor = typename std::conditional_t<
    (internal::TensorSize<_Shape, Planes>() * static_cast<int>(sizeof(typename _Shape::Type)) <= maxStackAllocSize),
    StackTensor<_Shape, Planes>,
    HeapTensor<_Shape, Planes>>;

} // namespace AboveInfinity
