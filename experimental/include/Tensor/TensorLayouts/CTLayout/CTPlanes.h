#pragma once

#include <tuple>

namespace AboveInfinity {

/**
 * Template class that can hold any number of plane objects irrelevant of their type
 */
template<typename... Planes>
class CTPlanes {
private:
    std::tuple<Planes...> _planes;
    static constexpr std::size_t N = sizeof...(Planes);

public:
    /**
     * Default constructor
     */
    inline CTPlanes() = default;

    /**
     * Constructs a plane list using a variadic number of lvalue plane objects
     *
     * @param planes: Planes used to construct the list 
     */
    inline CTPlanes(const Planes&... planes) { _planes = std::make_tuple(planes...); }

    /**
     * Constructs a plane list using a variadic number of rvalue plane objects
     *
     * @param planes: Planes used to construct the list 
     */
    inline CTPlanes(Planes&&... planes) { _planes = std::make_tuple(planes...); }

    /**
     * Returns a reference to the specified plane
     * 
     * @tparam position: Index of the specified plane
     * @return: Reference to the specified plane
     */
    template<std::size_t position>
    inline constexpr decltype(auto) At() const noexcept {
        requires(position < N);
        return std::get<position>(_planes);
    }
};


/**
 * CTPlanes empty class specialization
 */
template<>
class CTPlanes<> {};

} // namespace AboveInfinity