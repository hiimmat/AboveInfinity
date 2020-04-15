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
template<class _Shape, std::size_t Planes = 1U>
class TensorView {
    requires(std::is_object_v<typename _Shape::Type>);
    requires(Planes > 0U);

public:
    using Type = typename _Shape::Type;
    using Reference = typename _Shape::Reference;
    using Pointer = typename _Shape::Pointer;

private:
    std::array<Pointer, Planes> _ptrs;

    template<std::size_t Plane, int... lenOffsets, std::size_t... is>
    inline constexpr Pointer slicingPointerImpl(std::index_sequence<is...>) const {
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

    template<std::size_t skipDim, int lenOffset, std::size_t... is>
    inline auto sliceImpl(std::index_sequence<is...>) const {
        constexpr _Shape shape;

        requires(shape.rank() > 1U);
        requires(skipDim >= 0U && skipDim < shape.rank());
        requires(lenOffset >= 0 && lenOffset < shape.lengths().template get<skipDim>());
        requires(sizeof...(is) == Planes);

        constexpr int pointerOffset = lenOffset * shape.strides().template get<skipDim>();

        return TensorView<decltype(shape.template dimensionReduction<skipDim>()), Planes>{
            (std::get<is>(_ptrs) + pointerOffset)...};
    }

    template<std::size_t skipDim, std::size_t... is>
    inline auto sliceImpl(int lenOffset, std::index_sequence<is...>) const {
        constexpr _Shape shape;

        requires(shape.rank() > 1U);
        requires(skipDim >= 0U && skipDim < shape.rank());
        requires(sizeof...(is) == Planes);

        int pointerOffset = lenOffset * shape.strides().template get<skipDim>();

        return TensorView<decltype(shape.template dimensionReduction<skipDim>()), Planes>{
            (std::get<is>(_ptrs) + pointerOffset)...};
    }

    template<std::size_t offsetDim, int firstElem, int lastElem, std::size_t... is>
    inline constexpr auto slabImpl(std::index_sequence<is...>) const {
        constexpr _Shape shape;
        requires(offsetDim >= 0U && offsetDim < shape.rank());
        requires(firstElem >= 0 && firstElem < static_cast<int>(shape.lengths().template get<offsetDim>()));
        requires(lastElem >= 0 && lastElem <= static_cast<int>(shape.lengths().template get<offsetDim>()));
        requires(lastElem - firstElem > 0);
        requires(sizeof...(is) == Planes);

        constexpr int pointerOffset = firstElem * shape.strides().template get<offsetDim>();

        return TensorView<decltype(shape.template setLength<offsetDim, lastElem - firstElem>()), Planes>{
            (std::get<is>(_ptrs) + pointerOffset)...};
    }

    template<typename... pairs, std::size_t... pairsIS, std::size_t... planesIS>
    inline constexpr auto subspaceImpl(std::index_sequence<pairsIS...>, std::index_sequence<planesIS...>) const {
        constexpr _Shape shape;

        constexpr int pointerOffset =
            (((((pairs::first == pairs::second && pairs::first > 0) ? (pairs::first - 1) : pairs::first) *
               shape.strides().template get<pairsIS>()) +
              ...));

        return TensorView<decltype(shape.template subspace<pairs...>()), Planes>{
            (std::get<planesIS>(_ptrs) + pointerOffset)...};
    }

    template<std::size_t... is, typename... Pointers>
    inline auto addPlanesImpl(std::index_sequence<is...>, Pointers&&... pointers) {
        requires(((std::is_same_v<std::decay_t<Pointers>, Pointer>)&&...));
        requires(sizeof...(is) == Planes);

        const auto planePointers = std::array{_ptrs[is]..., pointers...};
        return TensorView<_Shape, Planes + sizeof...(Pointers)>{planePointers};
    }

    template<std::size_t N, typename _Pointer, std::size_t... is>
    inline auto addPlaneImpl(_Pointer ptr, std::index_sequence<is...>) {
        requires(N <= Planes);
        requires(std::is_same_v<std::decay_t<_Pointer>, Pointer>);
        requires(sizeof...(is) == Planes + 1U);

        if constexpr(N == Planes) return TensorView<_Shape, Planes + 1U>{(is != N ? _ptrs[is] : ptr)...};
        else
            return TensorView<_Shape, Planes + 1U>{(is < N ? _ptrs[is] : is == N ? ptr : _ptrs[is - 1U])...};
    }

    template<std::size_t N, typename _Pointer, std::size_t... is>
    inline auto replacePlaneImpl(_Pointer ptr, std::index_sequence<is...>) {
        requires(N < Planes);
        requires(std::is_same_v<std::decay_t<_Pointer>, Pointer>);
        requires(sizeof...(is) == Planes);

        return TensorView<_Shape, Planes>{(is == N ? ptr : _ptrs[is])...};
    }

    template<std::size_t... planesToRemove, std::size_t... planesIS, std::size_t... outIS>
    inline constexpr auto
        removePlanesImpl(std::index_sequence<planesIS...>, std::index_sequence<outIS...>) const noexcept {
        requires(sizeof...(planesIS) == Planes);
        requires(sizeof...(planesIS) - sizeof...(outIS) > 0U && sizeof...(planesIS) - sizeof...(outIS) < Planes);

        constexpr auto planesToKeep = std::tuple_cat(internal::skipIdx<planesIS, planesToRemove...>()...);
        return TensorView<_Shape, sizeof...(outIS)>{_ptrs[std::get<outIS>(planesToKeep)]...};
    }

public:
    inline constexpr TensorView() = default;

    inline constexpr explicit TensorView(std::array<Pointer, Planes> ptrs) noexcept : _ptrs{ptrs} {}

    template<typename... pointers>
    inline constexpr explicit TensorView(pointers&&... ptrs) noexcept : _ptrs{ptrs...} {
        requires(((std::is_same_v<std::decay_t<pointers>, Pointer>)&&...));
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
        return TensorView<decltype(std::declval<_Shape>().template fastPermute<Order...>()), Planes>{_ptrs};
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
        return TensorView<decltype(std::declval<_Shape>().undoPermutation()), Planes>{_ptrs};
    }

    /*
     * Retrieves the pointer to a dimension on the specified plane
     * The same result can be achieved by calling slice several times
     * retrieving the pointer for the specified plane through the data function
     * NOTE
     * The function receives its parameters in a reverse order
     * The first element matches the outtermost length of the TensorView, etc.
     */
    template<std::size_t Plane, int... lenOffsets>
    inline constexpr Pointer slicingPointer() const {
        return slicingPointerImpl<Plane, lenOffsets...>(std::make_index_sequence<sizeof...(lenOffsets)>());
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
        constexpr _Shape shape;
        return TensorView<decltype(shape.template newAxis<N>()), Planes>{_ptrs};
    }

    /* Removes single dimensions from the Shape */
    inline constexpr auto squeeze() const noexcept {
        constexpr _Shape shape;
        return TensorView<decltype(shape.squeeze()), Planes>{_ptrs};
    }

    /* Keeps the plane/s specified in the input, and discards the rest from the TensorView */
    template<std::size_t... planeIdxs>
    inline constexpr auto keepPlanes() const noexcept {
        requires(Planes > 1U);
        requires(sizeof...(planeIdxs) > 0U && sizeof...(planeIdxs) <= Planes);
        requires(((planeIdxs < Planes) && ...));
        internal::allUnique<planeIdxs...>();

        return TensorView<_Shape, sizeof...(planeIdxs)>{_ptrs[planeIdxs]...};
    }

    /*
     * Keeps only the plane that was specified in the input
     * WARNING
     * No bound checks are performed
     */
    inline auto keepPlane(std::size_t planeIdx) const { return TensorView<_Shape, 1U>{_ptrs[planeIdx]}; }

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
        return addPlaneImpl<N>(ptr, std::make_index_sequence<Planes + 1U>());
    }

    /*
     * Replaces the plane at the specified position from the TensorView
     * WARNING
     * No bound checks are performed
     */
    template<std::size_t N, typename _Pointer>
    inline auto replacePlane(_Pointer&& ptr) {
        return replacePlaneImpl<N>(ptr, std::make_index_sequence<Planes>());
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
    inline constexpr const auto shape() const noexcept { return _Shape(); }

    /* Returns the underlying lengths of the TensorView */
    inline constexpr const auto lengths() const noexcept {
        constexpr _Shape shape;
        return shape.lengths();
    }

    /* Returns the length of the requested dimension */
    template<std::size_t N>
    inline constexpr const int length() const noexcept {
        constexpr _Shape shape;
        return shape.lengths().template get<N>();
    }

    /* Returns the underlying strides of the TensorView */
    inline constexpr const auto strides() const noexcept {
        constexpr _Shape shape;
        return shape.strides();
    }

    /* Returns the stride of the requested dimension */
    template<std::size_t N>
    inline constexpr const int stride() const noexcept {
        constexpr _Shape shape;
        return shape.strides().template get<N>();
    }

    /* Returns the number of planes of the TensorView */
    inline constexpr const std::size_t planes() const noexcept { return Planes; }

    /* Returns the rank of the TensorView */
    inline constexpr const std::size_t rank() noexcept {
        constexpr _Shape shape;
        return shape.lengths().size();
    }

    /* Returns a pointer to the beginning of the data for a specified plane of the TensorView */
    template<std::size_t Plane = 0U>
    inline Pointer data() noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /* Returns a const pointer to the beginning of the data for a specified plane of the TensorView */
    template<std::size_t Plane = 0U>
    inline const Pointer data() const noexcept {
        requires(Plane < Planes);
        return _ptrs[Plane];
    }

    /*
     * Returns a pointer to the beginning of the data for a specified plane of the TensorView
     * WARNING
     * No bound checks are performed
     */
    inline Pointer data(std::size_t plane = 0U) { return _ptrs[plane]; }

    /*
     * Returns a const pointer to the beginning of the data for a specified plane of the TensorView
     * WARNING
     * No bound checks are performed
     */
    inline const Pointer data(std::size_t plane = 0U) const { return _ptrs[plane]; }

    /* Returns a std::array containing all the pointers representing the planes of the TensorView */
    inline std::array<Pointer, Planes> pointers() noexcept { return _ptrs; }

    /* Returns a const std::array containing all the pointers representing the planes of the TensorView */
    inline const std::array<const Pointer, Planes> pointers() const noexcept { return _ptrs; }
};

} // namespace AboveInfinity
