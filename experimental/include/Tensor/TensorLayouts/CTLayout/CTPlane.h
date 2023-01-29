#pragma once

#include "TensorLayout.h"

namespace AboveInfinity {

/**
 * @tparam: TBuffer: Type of the buffer used by the plane
 * @tparam: Shape: Lengths and strides defining the plane's shape
 * @tparam: channels: Number of channels (used for interleaved and semi-interleaved data)
 *
 * This class represents a single plane of a Tensor. It's designed to allow for custom memory buffers that can be sparse
 * or dense, owning or non-owning memory, performing different allocations and deallocations depending on the buffer's policy.
 * Giving the plane its own shape and channels, it allows the Tensors to have different dimensions and layouts (real or
 * interleaved) for all of their planes. While the offset allows them to reuse the same buffer for some or all of the planes
 * by skipping the required number of elements to access the requested plane's data.
 * This kind of design allows for interleaved, semi-interleaved, planar and packed Tensor memory formats.
 *
 * This is what a single plane looks like in a more readable format:
 *
 * template<typename BufferType>
 * struct plane {
 *     BufferType buffer;
 *     array<size_t, N> lengths;
 *     array<size_t, N> strides;
 *     int offset;
 * };
 *
 * The buffer can be either a class or a pointer.
 */
template<typename TBuffer, typename Shape, std::size_t channels = 1u>
class CTPlane {
    ~CTPlane() = delete;
    CTPlane(CTPlane const&) = delete;
    void operator=(CTPlane const&) = delete;
};

template<typename TBuffer, int... ls, int... ss, std::size_t channels>
class CTPlane<TBuffer, Shape<Lengths<ls...>, Strides<typename TBuffer::value_type, ss...>>, channels> {
    requires(std::is_class_v<TBuffer>);
    requires(channels > 0u);

public:
    using value_type = typename TBuffer::value_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

private:
    TBuffer _buffer{};
    int _offset{};
    static constexpr Shape<Lengths<ls...>, Strides<value_type, ss...>> _shape{};
    static constexpr auto _lengths = _shape.lengths().array();
    static constexpr auto _strides = _shape.strides().array();
    static constexpr auto _rank = _shape.rank();

    template<std::size_t... offsets, std::size_t... is>
    inline const value_type* SlicingPointerImpl(std::index_sequence<is...>&& inSeq) const {
        constexpr std::size_t nOffsets = sizeof...(offsets);
        constexpr auto offsetsArr = std::array{offsets...};

        requires(nOffsets <= _rank + 1u);

        if constexpr(channels == 1u) {
            requires(((offsetsArr[nOffsets - is - 1U] < _lengths[_rank - is - 1U]) && ...));
            return _buffer.At(_offset +
                              (((offsetsArr[nOffsets - is - 1u] * _strides[_rank - is - 1u]) + ...) * channels));
        } else {
            requires(((offsetsArr[nOffsets - is - 2U] < _lengths[_rank - is - 1U]) && ...));
            requires(offsetsArr[nOffsets - 1u] < channels);
            return _buffer.At(_offset +
                              (((offsetsArr[nOffsets - is - 2u] * _strides[_rank - is - 1u]) + ...) * channels) +
                              offsetsArr[nOffsets - 1u]);
        }
    }

    template<std::size_t... is, typename... Offsets>
    inline const value_type* SlicingPointerImpl(std::index_sequence<is...>&& inSeq, Offsets&&... offsets) const {
        requires(((std::is_same_v<std::decay_t<Offsets>, int>)&&...));
        constexpr std::size_t nOffsets = sizeof...(offsets);
        auto offsetsArr = std::array{offsets...};

        requires(nOffsets <= _rank + 1u);

        if constexpr(channels == 1u) {
            return _buffer.At(_offset +
                              (((offsetsArr[nOffsets - is - 1u] * _strides[_rank - is - 1u]) + ...) * channels));
        } else {
            return _buffer.At(_offset +
                              (((offsetsArr[nOffsets - is - 2u] * _strides[_rank - is - 1u]) + ...) * channels) +
                              offsetsArr[nOffsets - 1u]);
        }
    }

    template<std::size_t... offsets, std::size_t... is>
    inline void SetValueImpl(value_type value, std::index_sequence<is...>&& inSeq) {
        constexpr std::size_t nOffsets = sizeof...(offsets);
        constexpr auto offsetsArr = std::array{offsets...};

        requires(nOffsets <= _rank + 1u);

        if constexpr(channels == 1u) {
            requires(((offsetsArr[nOffsets - is - 1U] < _lengths[_rank - is - 1U]) && ...));
            _buffer.Set(_offset + (((offsetsArr[nOffsets - is - 1u] * _strides[_rank - is - 1u]) + ...) * channels),
                        value);
        } else {
            requires(((offsetsArr[nOffsets - is - 2U] < _lengths[_rank - is - 1U]) && ...));
            requires(offsetsArr[nOffsets - 1u] < channels);
            _buffer.Set(_offset + (((offsetsArr[nOffsets - is - 2u] * _strides[_rank - is - 1u]) + ...) * channels) +
                            offsetsArr[nOffsets - 1u],
                        value);
        }
    }

    template<std::size_t... is, typename... Offsets>
    inline void SetValueImpl(value_type value, std::index_sequence<is...>&& inSeq, Offsets&&... offsets) {
        requires(((std::is_same_v<std::decay_t<Offsets>, int>)&&...));
        constexpr std::size_t nOffsets = sizeof...(offsets);
        auto offsetsArr = std::array{offsets...};

        requires(nOffsets <= _rank + 1u);

        if constexpr(channels == 1u) {
            _buffer.Set(_offset + (((offsetsArr[nOffsets - is - 1u] * _strides[_rank - is - 1u]) + ...) * channels),
                        value);
        } else {
            _buffer.Set(_offset + (((offsetsArr[nOffsets - is - 2u] * _strides[_rank - is - 1u]) + ...) * channels) +
                            offsetsArr[nOffsets - 1u],
                        value);
        }
    }

    template<typename... pairs, std::size_t... pairsIS>
    inline auto SubspaceImpl(std::index_sequence<pairsIS...>&& pairsSeq) const {
        constexpr int pointerOffset =
            (((((pairs::first == pairs::second && pairs::first > 0) ? (pairs::first - 1) : pairs::first) *
               _strides[pairsIS]) +
              ...)) *
            channels;

        return CTPlane<TBuffer, decltype(_shape.template subspace<pairs...>()), channels>{_buffer,
                                                                                          _offset + pointerOffset};
    }

public:
    /**
     * Constructs a new CTPlane object and sets the offset to the first valid buffer element
     * 
     * @param offset: offset: Offset to the first valid buffer element
     */
    inline explicit CTPlane(int offset = 0) : _buffer{_shape.total() * channels}, _offset{offset} {}

    /**
     * Construct a new CTPlane object using an existing buffer and an offset to the first valid buffer element
     * 
     * @param buffer: Buffer used to represent the plane's allocated memory
     * @param offset: Offset to the first valid buffer element
     */
    inline CTPlane(const TBuffer& buffer, int offset = 0) : _buffer{buffer}, _offset(offset) {}

    /**
     * Construct a new CTPlane object using an existing rvalue buffer and an offset to the first valid buffer element
     * 
     * @param buffer: Buffer used to represent the plane's allocated memory
     * @param offset: Offset to the first valid buffer element
     */
    inline CTPlane(TBuffer&& buffer, int offset = 0) : _buffer{buffer}, _offset(offset) {}

    /**
     * SlicingPointer represents the main method for memory reads. It receives the offsets for each dimension, starting
     * from the outermost one and uses it to compute the index of the requested element in the buffer. If the plane is
     * interleaved, the last offset it receives is used as the channel index instead.
     * This method has also a compile-time assert that assures that the requested index is not out of bounds.
     * Since it doesn't access the memory buffer directly but only computes the index of the element that is requested
     * it can work with sparse buffers as well as with dense buffers.
     * 
     * @tparam offsets: Offset for each dimension starting from the outermost one
     * @return: const pointer to the specified element
     */
    template<std::size_t... offsets>
    inline const value_type* SlicingPointer() const {
        if constexpr(channels > 1u)
            return SlicingPointerImpl<offsets...>(std::make_index_sequence<sizeof...(offsets) - 1u>());
        else
            return SlicingPointerImpl<offsets...>(std::make_index_sequence<sizeof...(offsets)>());
    }

    /**
     * Runtime version of the SlicingPointer method
     * 
     * @tparam Offsets: Type of each offset
     * @param offsets: Offset for each dimension starting from the outermost one
     * @return: Pointer to the specified element
     */
    template<typename... Offsets>
    inline const value_type* SlicingPointer(Offsets&&... offsets) const {
        if constexpr(channels > 1u)
            return SlicingPointerImpl(std::make_index_sequence<sizeof...(offsets) - 1u>(), offsets...);
        else
            return SlicingPointerImpl(std::make_index_sequence<sizeof...(offsets)>(), offsets...);
    }

    /**
     * SetValue represents the main method for memory writes. It receives the offsets for each dimension, starting
     * from the outermost one and uses it to compute the index of the element in the buffer whose value needs to be changed.
     * If the plane is interleaved, the last offset it receives is used as the channel index instead.
     * This method has also a compile-time assert that assures that the requested index is not out of bounds.
     * Since it doesn't access the memory buffer directly but only computes the index of the element that is requested
     * it can work with sparse buffers as well as with dense buffers.
     * 
     * @tparam offsets: Offset for each dimension starting from the outermost one 
     * @param value: New value of the specified element
     */
    template<std::size_t... offsets>
    inline void SetValue(value_type value) {
        if constexpr(channels > 1u)
            SetValueImpl<offsets...>(value, std::make_index_sequence<sizeof...(offsets) - 1u>());
        else
            SetValueImpl<offsets...>(value, std::make_index_sequence<sizeof...(offsets)>());
    }

    /**
     * Purely runtime version of the SetValue method
     * 
     * @tparam Offsets
     * @param value 
     * @param offsets 
     */
    template<typename... Offsets>
    inline void SetValue(value_type value, Offsets&&... offsets) {
        if constexpr(channels > 1u)
            SetValueImpl(value, std::make_index_sequence<sizeof...(offsets) - 1u>(), offsets...);
        else
            SetValueImpl(value, std::make_index_sequence<sizeof...(offsets)>(), offsets...);
    }

    /**
     * Reorders the dimensions of this plane using the given order
     * 
     * @tparam Order: New order of the planes' dimensions
     * @return: Plane object with the reordered dimensions
     */
    template<std::size_t... Order>
    inline auto FastPermute() const {
        return CTPlane<TBuffer, decltype(_shape.template fastPermute<Order...>()), channels>{_buffer, _offset};
    }

    /**
     * Undoes the permutation of the dimensions of this plane
     * 
     * @return: Plane object with undone dimension permutation
     */
    inline auto UndoPermutation() const {
        return CTPlane<TBuffer, decltype(_shape.template undoPermutation()), channels>{_buffer, _offset};
    }

    /**
     * Performs a hyperplane over the specified dimension
     * 
     * @tparam skipDim: Dimension that's being removed
     * @tparam lenOffset: Offset of the removed dimension
     * @return: Plane object with one less dimension
     */
    template<std::size_t skipDim, int lenOffset>
    inline auto Slice() const {
        requires(_rank > 1U);
        requires(skipDim < _rank);
        requires(lenOffset >= 0 && lenOffset < _lengths[skipDim]);

        constexpr int pointerOffset = lenOffset * _strides[skipDim] * channels;

        return CTPlane<TBuffer, decltype(_shape.template dimensionReduction<skipDim>()), channels>{
            _buffer, _offset + pointerOffset};
    }

    /**
     * Partial runtime version of the Slice method
     * 
     * @tparam skipDim: Dimension that's being removed
     * @param lenOffset: Offset of the removed dimension
     * @return: Plane object with one less dimension 
     */
    template<std::size_t skipDim>
    inline auto Slice(int lenOffset) const {
        requires(_rank > 1U);
        requires(skipDim < _rank);

        int pointerOffset = lenOffset * _strides[skipDim] * channels;

        return CTPlane<TBuffer, decltype(_shape.template dimensionReduction<skipDim>()), channels>{
            _buffer, _offset + pointerOffset};
    }

    /**
     * Performs a hyperslab over the specified dimension 
     * 
     * @tparam offsetDim: Dimension that's being shortened
     * @tparam firstElem: First element used in the shortened dimension
     * @tparam lastElem: Last element used in the shortened dimension
     * @return auto: Plane object with a shortened dimension
     */
    template<std::size_t offsetDim, int firstElem, int lastElem>
    inline auto Slab() const {
        requires(offsetDim < _rank);
        requires(firstElem >= 0 && firstElem < _lengths[offsetDim]);
        requires(lastElem > 0 && lastElem <= _lengths[offsetDim]);
        requires(lastElem - firstElem > 0);

        constexpr int pointerOffset = firstElem * _strides[offsetDim] * channels;

        return CTPlane<TBuffer, decltype(_shape.template setLength<offsetDim, lastElem - firstElem>()), channels>{
            _buffer, _offset + pointerOffset};
    }

    /**
     * Performs either a hyperplane or a hyperslab for each specified element
     * 
     * @tparam pairs: A pair of integers representing the new starting and ending point of each dimension starting
     * from the innermost one
     * @return: Plane object with modified dimensions
     */
    template<typename... pairs>
    inline auto Subspace() const {
        return SubspaceImpl<pairs...>(std::make_index_sequence<sizeof...(pairs)>());
    }

    /**
     * Adds a new dimension of length 1 at the specified position
     * 
     * @tparam N: position on which the new dimension is added
     * @return: Plane object with an increased rank
     */
    template<std::size_t N>
    inline auto NewAxis() const {
        return CTPlane<TBuffer, decltype(_shape.template newAxis<N>()), channels>{_buffer, _offset};
    }

    /**
     * Removes all dimensions with length equal to 1
     * 
     * @return: Plane object reduced for dimensions of length equal to 1
     */
    inline auto Squeeze() const {
        return CTPlane<TBuffer, decltype(_shape.template squeeze()), channels>{_buffer, _offset};
    }

    /**
     * Retrieves the buffer used by the plane
     * 
     * @return: reference to planes' buffer
     */
    inline TBuffer& Buffer() noexcept { return _buffer; }

    /**
     * Retrieves the buffer used by the plane
     * 
     * @return: const reference to the planes' buffer
     */
    inline const TBuffer& Buffer() const noexcept { return _buffer; }

    /**
     * Retrieves the shape of the plane
     * 
     * @return: planes' shape
     */
    inline constexpr const auto Shape() const noexcept { return _shape; }

    /**
     * Returns the number of chanenls the plane consists of
     * 
     * @return: planes' number of channels
     */
    inline constexpr std::size_t Channels() const noexcept { return channels; }

    /**
     * Retrieves the buffer offset to the first valid element
     * 
     * @return: planes' buffer offset to the first valid element
     */
    inline constexpr int Offset() const noexcept { return _offset; }

    /**
     * Retrieves the rank of the plane
     * 
     * @return: planes' rank
     */
    inline constexpr std::size_t Rank() const noexcept { return _rank; }

    /**
     * Retrieves the total number of elements
     * 
     * @return: total number of elements
     */
    inline constexpr int Total() const noexcept { return _shape.total() * channels; }
};

} // namespace AboveInfinity
