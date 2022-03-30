#pragma once

#include "CTPlane.h"
#include "CTPlanes.h"
#include "CustomTraits.h"

namespace AboveInfinity {

/**
 * Class representing an N-dimensional tensor
 *
 * All of the Tensor's functionalities come from its policies that it inherits through CRTP. This design has many
 * advantages, such as compile-time policy based design with a variadic number of policies that can be easily
 * added/removed and completely modular plane design. The biggest disadvantage is not being able to declare policies as
 * friends and aliasing them. It is possible though to declare the first, last or Nth policy as a friend. However,
 * declaring all of them as friends isn't possible. I've taken a different approach to solve this issue. The solution that
 * I came up with can be found in the file TensorWithFriendPolicies.h. The biggest disadvantage (and the reason I'm not using
 * it here) is that it requires a lot of casting that can't be evaluated during compile-time.
 *
 * @tparam _Planes: Class representing the planes of the Tensor
 * @tparam Policies: Policies that the Tensor inherits through CRTP
 */
template<typename _Planes = CTPlanes<>, template<typename> typename... Policies>
class Tensor : public Policies<Tensor<_Planes, Policies...>>... {
private:
    _Planes planes;

    template<template<typename> typename... _Policies>
    auto ExtractPoliciesFromList(TTypeList<_Policies...>) {
        return Tensor<_Planes, _Policies...>();
    }

public:
    /**
     * Default constructor
     */
    Tensor() = default;

    /**
     * Construct a new Tensor object using existing planes
     * 
     * @tparam Planes: Types of planes used to construct the Tensor
     * @param planes: Planes used by the Tensor
     */
    template<typename Planes>
    Tensor(Planes&& planes) : planes{std::forward<Planes>(planes)} {
        requires(std::is_same_v<std::decay_t<Planes>, std::decay_t<_Planes>>,
                 "Planes given to Tensors' constructor must have the same type as the ones used for its declaration");
    }

    /**
     * Default copy constructor
     */
    Tensor(Tensor&) = default;
    
    /**
     * Default copy assignment operator
     */
    Tensor& operator=(Tensor&) = default;

    /**
     * Default move constructor
     */
    Tensor(Tensor&&) = default;

    /**
     * Default move assignment operator
     */
    Tensor& operator=(Tensor&&) = default;

    /**
     * Returns the list of planes used by the tensor
     * 
     * @return: reference to the list of planes used by the tensor 
     */
    inline _Planes& Planes() { return planes; }

    /**
     * Returns the list of planes used by the tensor
     * 
     * @return: const reference to the list of planes used by the tensor 
     */
    inline const _Planes& Planes() const { return planes; }

    /**
     * Adds a new policy to the Tensor
     * 
     * @tparam NewPolicy: Type of the newly added policy
     * @tparam position: position at which the policy should be added.
     * If the position is negative, the policy is added at the end of the list
     * @return: Tensor object with the newly added policy 
     */
    template<template<typename> typename NewPolicy, int position = -1>
    auto AddPolicy() {
        return ExtractPoliciesFromList(InsertType<TTypeList<Policies...>, NewPolicy, position>());
    }

    /**
     * Removes an existing policy from the Tensor
     * 
     * @tparam _Policy: Type of the policy that's being removed
     * @return: Tensor object without the specified policy
     */
    template<template<typename> typename _Policy>
    auto RemovePolicy() {
        return ExtractPoliciesFromList(EraseType<TTypeList<Policies...>, _Policy>());
    }

    /**
    * Removes the policy on the specified position. If the position is negative, it removes the last policy from the list
    * 
    * @tparam N: position of the specified policy
    * @return: Tensor object without the specified policy
    */
    template<int N = -1>
    auto RemoveNthPolicy() {
        return ExtractPoliciesFromList(RemoveType<TTypeList<Policies...>, N>());
    }
};

} // namespace AboveInfinity
