#include "TensorLayout.h"

namespace AboveInfinity {

/*
 * Class representing a view into the data of a Tensor
 * WARNING
 * Even though this class receives the same Shape type as the Tensor and uses bound checks on it,
 * it can't guarantee that there will be no out of bounds errors, since it doesn't allocate the
 * memory it uses. It's either relying on functions that create views or on the user to assure
 * that the memory being used won't go out of scope or that it won't use already released memory.
 * I was considering making it safer either by using shared pointers in Tensors or by using
 * some kind of a design pattern (mediator?).
 * I decided against the shared pointer implementation mostly as I didn't want the transferable
 * ownership, which could end up with a Tensor staying alive longer than what the user would expect
 */

template<typename, std::size_t>
class TensorView {
    ~TensorView() = delete;
    TensorView(TensorView const&) = delete;
    void operator=(TensorView const&) = delete;
};

template<typename T, int... ls, int... ss, std::size_t Planes>
class TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes> {
    requires(Planes > 0U);

private:
    std::array<T*, Planes> _ptrs;
    static constexpr auto _lengths = std::array{ls...};
    static constexpr auto _strides = std::array{ss...};

    template<std::size_t Plane, int... offsets, std::size_t... is>
    inline T* slicingPointerImpl(std::index_sequence<is...>&&) const {
        constexpr std::size_t nOffsets = sizeof...(offsets);
        constexpr std::size_t rank = sizeof...(ls);
        constexpr auto offsetsArr = std::array{offsets...};

        requires(Plane < Planes);
        requires(sizeof...(is) == nOffsets);
        requires(nOffsets <= rank);
        requires(((offsets >= 0) && ...));
        requires(((offsetsArr[nOffsets - is - 1U] < _lengths[rank - is - 1U]) && ...));

        T* _memory = _ptrs[Plane];

        return _memory + ((((offsetsArr[nOffsets - is - 1U] * _strides[rank - is - 1U])) + ...));
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

    template<std::size_t skipDim, int lenOffset, std::size_t... is>
    inline auto sliceImpl(std::index_sequence<is...>&&) const {
        requires(sizeof...(ls) > 1U);
        requires(skipDim < sizeof...(ls));
        requires(lenOffset >= 0 && lenOffset < _lengths[skipDim]);
        requires(sizeof...(is) == Planes);

        constexpr int pointerOffset = lenOffset * _strides[skipDim];

        return TensorView<decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().template dimensionReduction<skipDim>()),
                          Planes>{(std::get<is>(_ptrs) + pointerOffset)...};
    }

    template<std::size_t skipDim, std::size_t... is>
    inline auto sliceImpl(int lenOffset, std::index_sequence<is...>&&) const {
        requires(sizeof...(ls) > 1U);
        requires(skipDim < sizeof...(ls));
        requires(sizeof...(is) == Planes);

        int pointerOffset = lenOffset * _strides[skipDim];

        return TensorView<decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().template dimensionReduction<skipDim>()),
                          Planes>{(_ptrs[is] + pointerOffset)...};
    }

    template<std::size_t offsetDim, int firstElem, int lastElem, std::size_t... is>
    inline constexpr auto slabImpl(std::index_sequence<is...>&&) const {
        requires(offsetDim < sizeof...(ls));
        requires(firstElem >= 0 && firstElem < _lengths[offsetDim]);
        requires(lastElem > 0 && lastElem <= _lengths[offsetDim]);
        requires(lastElem - firstElem > 0);
        requires(sizeof...(is) == Planes);

        constexpr int pointerOffset = firstElem * _strides[offsetDim];

        return TensorView<
            decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().template setLength<offsetDim, lastElem - firstElem>()),
            Planes>{(_ptrs[is] + pointerOffset)...};
    }

    template<typename... pairs, std::size_t... pairsIS, std::size_t... planesIS>
    inline constexpr auto subspaceImpl(std::index_sequence<pairsIS...>&&, std::index_sequence<planesIS...>&&) const {
        requires(sizeof...(pairsIS) == sizeof...(pairs));
        requires(sizeof...(planesIS) == Planes);
        constexpr int pointerOffset =
            (((((pairs::first == pairs::second && pairs::first > 0) ? (pairs::first - 1) : pairs::first) *
               _strides[pairsIS]) +
              ...));

        return TensorView<decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().template subspace<pairs...>()), Planes>{
            (_ptrs[planesIS] + pointerOffset)...};
    }

    template<std::size_t... is, typename... Pointers>
    inline auto addPlanesImpl(std::index_sequence<is...>&&, Pointers&&... pointers) {
        requires(((std::is_same_v<std::decay_t<Pointers>, T*>)&&...));
        requires(sizeof...(is) == Planes);

        const auto planePointers = std::array{_ptrs[is]..., pointers...};
        return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes + sizeof...(Pointers)>{planePointers};
    }

    template<std::size_t N, typename _Pointer, std::size_t... is>
    inline auto addPlaneImpl(_Pointer&& ptr, std::index_sequence<is...>&&) {
        requires(N <= Planes);
        requires(std::is_same_v<std::decay_t<_Pointer>, T*>);
        requires(sizeof...(is) == Planes + 1U);

        return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes + 1U>{
            (is < N ? _ptrs[is] : is == N ? ptr : _ptrs[is - 1U])...};
    }

    template<std::size_t N, typename _Pointer, std::size_t... is>
    inline auto replacePlaneImpl(_Pointer&& ptr, std::index_sequence<is...>&&) {
        requires(N < Planes);
        requires(std::is_same_v<std::decay_t<_Pointer>, T*>);
        requires(sizeof...(is) == Planes);

        return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, Planes>{(is == N ? ptr : _ptrs[is])...};
    }

    template<std::size_t... planesToRemove, std::size_t... planesIS, std::size_t... outIS>
    inline constexpr auto
        removePlanesImpl(std::index_sequence<planesIS...>&&, std::index_sequence<outIS...>&&) const noexcept {
        requires(sizeof...(planesIS) == Planes);
        requires(sizeof...(planesIS) - sizeof...(outIS) > 0U && sizeof...(planesIS) - sizeof...(outIS) < Planes);

        constexpr auto planesToKeep = std::tuple_cat(internal::skipIdx<planesIS, planesToRemove...>()...);
        return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, sizeof...(outIS)>{
            _ptrs[std::get<outIS>(planesToKeep)]...};
    }

public:
    using Type = T;
    using Reference = T&;
    using Pointer = T*;

    inline constexpr TensorView() = default;

    inline constexpr explicit TensorView(std::array<T*, Planes> ptrs) noexcept : _ptrs{ptrs} {}

    template<typename... pointers>
    inline constexpr explicit TensorView(pointers&&... ptrs) noexcept : _ptrs{ptrs...} {
        requires(((std::is_same_v<std::decay_t<pointers>, T*>)&&...));
        requires(sizeof...(pointers) == Planes);
    }

    /*
     * Permutes the shape of the view in the requested order
     * WARNING
     * This just reorders the lengths and the strides of the view, so it avoids creating a copy
     * It might be a fast way to get a permuted Tensor, but accessing the elements in the new
     * order will be slower
     */
    template<std::size_t... Order>
    inline constexpr auto fastPermute() const noexcept {
        return TensorView<decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().template fastPermute<Order...>()), Planes>{
            _ptrs};
    }

    /*
     * Undoes the permutation if there is one
     * WARNING
     * This function doesn't allow dimensions added through newaxis, since their stride would be equal to zero
     * which would affect the ordering significantly
     * Dimensions with equal strides might come in the wrong ordering as well, since the function relies on the
     * strides to determine what the right ordering should be
     */
    inline constexpr auto undoPermutation() const noexcept {
        return TensorView<decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().undoPermutation()), Planes>{_ptrs};
    }

    /*
     * Retrieves the pointer to a dimension on the specified plane
     * The same result can be achieved by calling slice several times and later
     * retrieving the pointer for the specified plane through the data function
     */
    template<std::size_t Plane, int... offsets>
    inline constexpr T* slicingPointer() const {
        return slicingPointerImpl<Plane, offsets...>(std::make_index_sequence<sizeof...(offsets)>());
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

    /* Performs a hyperplane */
    template<std::size_t skipDim, int lenOffset>
    inline auto slice() const {
        return sliceImpl<skipDim, lenOffset>(std::make_index_sequence<Planes>());
    }

    /*
     * Performs a hyperplane
     * WARNING
     * The variable lenOffset isn't asserted, as it would introduce overhead
     */
    template<std::size_t skipDim>
    inline auto slice(int lenOffset) const {
        return sliceImpl<skipDim>(lenOffset, std::make_index_sequence<Planes>());
    }

    /* Performs a hyperslab */
    template<std::size_t offsetDim, int firstElem, int lastElem>
    inline constexpr auto slab() const {
        return slabImpl<offsetDim, firstElem, lastElem>(std::make_index_sequence<Planes>());
    }

    /*
     * This function is a combination of the slice and slab functions
     * Depending on the passed parameters, it performs either a hyperplane or a hyperslab
     * It receives a sequence of pairs as an input, where the first element of a pair
     * represents the beginning of the length, while the second element represents the end
     * of the length for a given dimension. If the result of subtracting the two elements of
     * a pair (end - beginning) results in a 0, the dimension will be removed. If it's 1,
     * it's the same as performing a hyperplane, however, the dimension will be kept.
     * This is due to the fact that a user might want to keep the dimension as 1, rather than
     * wanting to completely remove it. In all other cases, a hyperslab is performed.
     * Each pair represents a single dimension starting from the innermost one (this is
     * opposite to the slicingPointer)
     * If a dimension isn't specified, it will be ignored, and it will be the same as performing
     * hyperplane on the first element of the length of the dimension
     */
    template<typename... pairs>
    inline constexpr auto subspace() const {
        return subspaceImpl<pairs...>(std::make_index_sequence<sizeof...(pairs)>(), std::make_index_sequence<Planes>());
    }

    /*
     * Adds a dimension of length 1 at the specified position
     * WARNING
     * This function might affect functions as undoPermutation, functions that are performing iterations
     * over Tensors or TensorViews and different checks, in example ones that are checking if the dimensions
     * were permuted
     * undoPermutation might be affected because the stride added through newAxis is 0, and the ordering in
     * undoPermutation depends on the stride.
     * Functions iterating over Tensors or TensorViews might have a reduced performance if the innermost
     * length is equal to 1, as that would most likely cause operations to occur only on one element at a time
     */
    template<std::size_t N>
    inline constexpr auto newAxis() const noexcept {
        return TensorView<decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().template newAxis<N>()), Planes>{_ptrs};
    }

    /* Removes single dimensions from the Shape */
    inline constexpr auto squeeze() const noexcept {
        return TensorView<decltype(Shape<Lengths<ls...>, Strides<T, ss...>>().squeeze()), Planes>{_ptrs};
    }

    /* Keeps the plane/s specified in the input, and discards the rest from the TensorView */
    template<std::size_t... planeIdxs>
    inline constexpr auto keepPlanes() const noexcept {
        requires(Planes > 1U);
        requires(sizeof...(planeIdxs) > 0U && sizeof...(planeIdxs) <= Planes);
        requires(((planeIdxs < Planes) && ...));
        internal::allUnique<planeIdxs...>();

        return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, sizeof...(planeIdxs)>{_ptrs[planeIdxs]...};
    }

    /*
     * Keeps only the plane that was specified in the input
     * WARNING
     * No bound checks are performed
     */
    inline auto keepPlane(std::size_t planeIdx) const {
        return TensorView<Shape<Lengths<ls...>, Strides<T, ss...>>, 1U>{_ptrs[planeIdx]};
    }

    /* Adds one or more planes to the TensorView */
    template<typename... Pointers>
    inline auto addPlanes(Pointers&&... pointers) {
        return addPlanesImpl(std::make_index_sequence<Planes>(), pointers...);
    }

    /*
     * Adds a single plane at the specified position to the TensorView
     * WARNING
     * No bound checks are performed
     */
    template<std::size_t N, typename _Pointer>
    inline auto addPlane(_Pointer&& ptr) {
        return addPlaneImpl<N>(std::forward<_Pointer>(ptr), std::make_index_sequence<Planes + 1U>());
    }

    /*
     * Replaces the plane at the specified position from the TensorView
     * WARNING
     * No bound checks are performed
     */
    template<std::size_t N, typename _Pointer>
    inline auto replacePlane(_Pointer&& ptr) {
        return replacePlaneImpl<N>(std::forward<_Pointer>(ptr), std::make_index_sequence<Planes>());
    }

    /* Removes one or more planes from the TensorView */
    template<std::size_t... planesToRemove>
    inline constexpr auto removePlanes() const noexcept {
        requires(Planes > 1U);
        requires(sizeof...(planesToRemove) >= 0U && sizeof...(planesToRemove) < Planes);
        requires(((planesToRemove < Planes) && ...));
        internal::allUnique<planesToRemove...>();

        return removePlanesImpl<planesToRemove...>(std::make_index_sequence<Planes>(),
                                                   std::make_index_sequence<Planes - sizeof...(planesToRemove)>());
    }

    /* Returns the Shape of the TensorView */
    inline constexpr const auto shape() const noexcept { return Shape<Lengths<ls...>, Strides<T, ss...>>(); }

    /* Returns the underlying lengths of the TensorView */
    inline constexpr const auto lengths() const noexcept { return Lengths<ls...>(); }

    /* Returns the length of the requested dimension */
    template<std::size_t N>
    inline constexpr int length() const noexcept {
        requires(N < sizeof...(ls));
        return _lengths[N];
    }

    /* Returns the underlying strides of the TensorView */
    inline constexpr const auto strides() const noexcept { return Strides<T, ss...>(); }

    /* Returns the stride of the requested dimension */
    template<std::size_t N>
    inline constexpr int stride() const noexcept {
        requires(N < sizeof...(ss));
        return _strides[N];
    }

    /* Returns the number of planes of the TensorView */
    inline constexpr std::size_t planes() const noexcept { return Planes; }

    /* Returns the rank of the TensorView */
    inline constexpr std::size_t rank() noexcept { return sizeof...(ls); }

    /* Returns a pointer to the beginning of the data for a specified plane of the TensorView */
    template<std::size_t Plane = 0U>
    inline T* data() noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /* Returns a const pointer to the beginning of the data for a specified plane of the TensorView */
    template<std::size_t Plane = 0U>
    inline const T* data() const noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /*
     * Returns a pointer to the beginning of the data for a specified plane of the TensorView
     * WARNING
     * No bound checks are performed
     */
    inline T* data(std::size_t plane = 0U) { return _ptrs[plane]; }

    /*
     * Returns a const pointer to the beginning of the data for a specified plane of the TensorView
     * WARNING
     * No bound checks are performed
     */
    inline const T* data(std::size_t plane = 0U) const { return _ptrs[plane]; }

    /* Returns a std::array containing all the pointers representing the planes of the TensorView */
    inline std::array<T*, Planes> pointers() noexcept { return _ptrs; }

    /* Returns a const std::array containing all the pointers representing the planes of the TensorView */
    inline const std::array<const T*, Planes> pointers() const noexcept { return _ptrs; }
};

} // namespace AboveInfinity
