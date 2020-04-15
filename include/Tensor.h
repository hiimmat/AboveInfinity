#include "AlignedMemory.h"
#include "TensorView.h"

namespace AboveInfinity {

namespace internal {

/* Computes the size used for allocating a Tensor (not including the size of the underlying type) */
template<class _Shape, std::size_t Planes>
inline constexpr int TensorSize() noexcept {
    requires(Planes > 0U);
    constexpr _Shape shape;
    return shape.lengths().template get<shape.lengths().size() - 1U>() *
           shape.strides().template get<shape.strides().size() - 1U>() * static_cast<int>(Planes);
}

}; // namespace internal

/*
 * This class represents a multidimensional array of fixed size items
 * It's described using a Shape consisting of the type of
 * each element, Lengths (number of elements in each dimension),
 * Strides (offset needed to access each dimension) and the
 * underlying memory layout (currently either Real or Planar)
 * This representation is specifically used for describing a
 * move only memory block on the stack
 */
template<class _Shape, std::size_t Planes = 1U>
class StackTensor {
    requires(std::is_object_v<typename _Shape::Type>);
    requires(Planes > 0U);

public:
    using Type = typename _Shape::Type;
    using Reference = typename _Shape::Reference;
    using Pointer = typename _Shape::Pointer;

private:
    alignas(AIAlignment) std::array<Type, static_cast<std::size_t>(internal::TensorSize<_Shape, Planes>())> memory;
    std::array<Pointer, Planes> _ptrs;

    template<std::size_t Plane, int... lenOffsets, std::size_t... is>
    inline Pointer slicingPointerImpl(std::index_sequence<is...>) const {
        constexpr _Shape shape;

        requires(Plane >= 0U && Plane < Planes);
        requires(sizeof...(is) == sizeof...(lenOffsets));
        requires(sizeof...(lenOffsets) <= shape.lengths().size());
        requires(((lenOffsets >= 0) && ...));
        requires(((std::get<sizeof...(lenOffsets) - is - 1U>(std::tuple(lenOffsets...)) <
                   shape.lengths().template get<shape.lengths().size() - is - 1U>()) &&
                  ...));
        Pointer _memory = _ptrs[Plane];

        return _memory + ((((std::get<sizeof...(lenOffsets) - is - 1U>(std::tuple(lenOffsets...)) *
                             shape.strides().template get<shape.strides().size() - is - 1U>())) +
                           ...));
    }

public:
    inline StackTensor() {
        _ptrs[0U] = memory.data();

        if constexpr(Planes > 1U) {
            _Shape shape;
            constexpr int length = shape.lengths().template get<shape.lengths().size() - 1U>();
            constexpr int stride = shape.strides().template get<shape.strides().size() - 1U>();

            for(std::size_t i = 1U; i < Planes; ++i) _ptrs[i] = _ptrs[i - 1U] + length * stride;
        }
    }

    StackTensor(const StackTensor&) = delete;
    StackTensor& operator=(const StackTensor&) = delete;

    inline StackTensor(StackTensor&&) noexcept = default;
    inline StackTensor& operator=(StackTensor&&) noexcept = default;

    /* Returns a TensorView pointing to the data of the StackTensor */
    inline auto view() const noexcept { return TensorView<_Shape, Planes>{_ptrs}; }

    /*
     * Retrieves the pointer to a dimension on the specified plane
     * The same result can be achieved by calling slice several times
     * retrieving the pointer for the specified plane through the data function
     * NOTE
     * The function receives its parameters in a reverse order
     * The first element matches the outtermost length of the StackTensor, etc.
     */
    template<std::size_t Plane, int... lenOffsets>
    inline Pointer slicingPointer() const {
        return slicingPointerImpl<Plane, lenOffsets...>(std::make_index_sequence<sizeof...(lenOffsets)>());
    }

    /* Returns the Shape of the StackTensor */
    inline constexpr const auto shape() const noexcept { return _Shape(); }

    /* Returns the underlying lengths of the StackTensor */
    inline constexpr const auto lengths() const noexcept {
        constexpr _Shape shape;
        return shape.lengths();
    }

    /* Returns the length of the requested dimension */
    template<std::size_t N>
    inline constexpr const int length() const noexcept {
        _Shape shape;
        return shape.lengths().template get<N>();
    }

    /* Returns the underlying strides of the StackTensor */
    inline constexpr const auto strides() const noexcept {
        constexpr _Shape shape;
        return shape.strides();
    }

    /* Returns the stride of the requested dimension */
    template<std::size_t N>
    inline constexpr const int stride() const noexcept {
        _Shape shape;
        return shape.strides().template get<N>();
    }

    /* Returns the number of planes of the StackTensor */
    inline constexpr const std::size_t planes() const noexcept { return Planes; }

    /* Returns the rank of the StackTensor */
    inline constexpr const std::size_t rank() noexcept {
        constexpr _Shape shape;
        return shape.lengths().size();
    }

    /* Returns a pointer to the beginning of the data for a specified plane of the StackTensor */
    template<std::size_t Plane = 0U>
    inline Pointer data() noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /* Returns a const pointer to the beginning of the data for a specified plane of the StackTensor */
    template<std::size_t Plane = 0U>
    inline const Pointer data() const noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /*
     * Returns a pointer to the beginning of the data for a specified plane of the StackTensor
     * WARNING
     * No bound checks are performed
     */
    inline Pointer data(std::size_t plane = 0U) { return _ptrs[plane]; }

    /*
     * Returns a const pointer to the beginning of the data for a specified plane of the Tensor
     * WARNING
     * No bound checks are performed
     */
    inline const Pointer data(std::size_t plane = 0U) const { return _ptrs[plane]; }
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
template<class _Shape, std::size_t Planes = 1U>
class HeapTensor {
    requires(std::is_object_v<typename _Shape::Type>);
    requires(Planes > 0U);

public:
    using Type = typename _Shape::Type;
    using Reference = typename _Shape::Reference;
    using Pointer = typename _Shape::Pointer;

private:
    AlignedMemory<Type> memory;
    std::array<Pointer, Planes> _ptrs;

    template<std::size_t Plane, int... lenOffsets, std::size_t... is>
    inline Pointer slicingPointerImpl(std::index_sequence<is...>) const {
        constexpr _Shape shape;

        requires(Plane >= 0U && Plane < Planes);
        requires(sizeof...(is) == sizeof...(lenOffsets));
        requires(sizeof...(lenOffsets) <= shape.lengths().size());
        requires(((lenOffsets >= 0) && ...));
        requires(((std::get<sizeof...(lenOffsets) - is - 1U>(std::tuple(lenOffsets...)) <
                   shape.lengths().template get<shape.lengths().size() - is - 1U>()) &&
                  ...));
        Pointer _memory = _ptrs[Plane];

        return _memory + ((((std::get<sizeof...(lenOffsets) - is - 1U>(std::tuple(lenOffsets...)) *
                             shape.strides().template get<shape.strides().size() - is - 1U>())) +
                           ...));
    }

public:
    HeapTensor() {
        _Shape shape;

        memory = AlignedMemory<Type>(static_cast<std::size_t>(internal::TensorSize<_Shape, Planes>()), AIAlignment);

        _ptrs[0U] = memory.get();

        if constexpr(Planes > 1U) {
            constexpr int length = shape.lengths().template get<shape.lengths().size() - 1U>();
            constexpr int stride = shape.strides().template get<shape.strides().size() - 1U>();

            for(std::size_t i = 1U; i < Planes; ++i) _ptrs[i] = _ptrs[i - 1U] + length * stride;
        }
    }

    /* Returns a TensorView pointing to the data of the HeapTensor */
    inline auto view() const noexcept { return TensorView<_Shape, Planes>{_ptrs}; }

    /*
     * Retrieves the pointer to a dimension on the specified plane
     * The same result can be achieved by calling slice several times
     * retrieving the pointer for the specified plane through the data function
     * NOTE
     * The function receives its parameters in a reverse order
     * The first element matches the outtermost length of the HeapTensor, etc.
     */
    template<std::size_t Plane, int... lenOffsets>
    inline Pointer slicingPointer() const {
        return slicingPointerImpl<Plane, lenOffsets...>(std::make_index_sequence<sizeof...(lenOffsets)>());
    }

    /* Returns the Shape of the HeapTensor */
    inline constexpr const auto shape() const noexcept { return _Shape(); }

    /* Returns the underlying lengths of the HeapTensor */
    inline constexpr const auto lengths() const noexcept {
        constexpr _Shape shape;
        return shape.lengths();
    }

    /* Returns the length of the requested dimension */
    template<std::size_t N>
    inline constexpr const int length() const noexcept {
        _Shape shape;
        return shape.lengths().template get<N>();
    }

    /* Returns the underlying strides of the HeapTensor */
    inline constexpr const auto strides() const noexcept {
        constexpr _Shape shape;
        return shape.strides();
    }

    /* Returns the stride of the requested dimension */
    template<std::size_t N>
    inline constexpr const int stride() const noexcept {
        _Shape shape;
        return shape.strides().template get<N>();
    }

    /* Returns the number of planes of the HeapTensor */
    inline constexpr const std::size_t planes() const noexcept { return Planes; }

    /* Returns the rank of the HeapTensor */
    inline constexpr const std::size_t rank() noexcept {
        constexpr _Shape shape;
        return shape.lengths().size();
    }

    /* Returns a pointer to the beginning of the data for a specified plane of the HeapTensor */
    template<std::size_t Plane = 0U>
    inline Pointer data() noexcept {
        requires(Planes > 1U || Plane == 0U);
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /* Returns a const pointer to the beginning of the data for a specified plane of the HeapTensor */
    template<std::size_t Plane = 0U>
    inline const Pointer data() const noexcept {
        requires(Planes > 1U || Plane == 0U);
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /*
     * Returns a pointer to the beginning of the data for a specified plane of the HeapTensor
     * WARNING
     * No bound checks are performed
     */
    inline Pointer data(std::size_t plane = 0U) { return _ptrs[plane]; }

    /*
     * Returns a const pointer to the beginning of the data for a specified plane of the HeapTensor
     * WARNING
     * No bound checks are performed
     */
    inline const Pointer data(std::size_t plane = 0U) const { return _ptrs[plane]; }
};

/*
 * Creates an alias for the StackTensor and HeapTensor and determines which one of those two will be used
 * when creating a new Tensor. This is based on the Tensor size multiplied with the size of its underlying
 * type. If the size exceeds the maxStackAllocSize, a HeapTensor will be constructed. Otherwise, a
 * StackTensor will be created
 */
template<class _Shape, std::size_t Planes = 1U>
using Tensor = typename std::conditional_t<
    (internal::TensorSize<_Shape, Planes>() * static_cast<int>(sizeof(typename _Shape::Type)) <= maxStackAllocSize),
    StackTensor<_Shape, Planes>,
    HeapTensor<_Shape, Planes>>;

} // namespace AboveInfinity
