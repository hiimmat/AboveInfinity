#pragma once

#include <iostream>

#include "planes.hpp"

namespace ntensor {

// Tensor-specific helper methods
inline namespace internal {

template <typename...>
struct first_policy;

/*
 * first_policy struct specialization that retrieves the first policy from a list of policies
 * Parameters:
 * @tparam Policy: First policy found in a list of policies
 * @tparam PolicyArgs: Template parameters of the first policy
 * @tparam Policies: Rest of the policies
 */
template <template <typename...> typename Policy, typename... PolicyArgs, typename... Policies>
struct first_policy<Policy<PolicyArgs...>, Policies...> {
  using type = Policy<PolicyArgs...>;
};

/*
 * Retrieves the first policy that the Tensor inherits
 * Parameters:
 * @tparam Policies: List of policies that the Tensor inherits
 */
template <typename... Policies>
using first_policy_t = typename first_policy<Policies...>::type;

template <typename, typename>
struct append;

/*
 * append struct specialization that appends a policy to a Tensor
 * Parameters:
 * @tparam Policy: Policy that needs to be inherited by the Tensor
 * @tparam PolicyArgs: Policy template parameters
 * @tparam Tensor: Type of the Tensor
 * @tparam Planes: Planes types that represent the Tensor
 * @tparam Policies: List of policies that the Tensor inherits
 */
template <template <typename...> typename Policy, typename... PolicyArgs,
          template <typename, template <typename...> typename...> typename Tensor, typename Planes,
          template <typename...> typename... Policies>
struct append<Policy<PolicyArgs...>, Tensor<Planes, Policies...>> {
  using type = Tensor<Planes, Policy, Policies...>;
};

/*
 * append struct specialization that appends a policy to a Tensor
 * Parameters:
 * @tparam Tensor: Type of the Tensor
 * @tparam Planes: Planes types that represent the Tensor
 * @tparam Policies: List of policies that the Tensor inherits
 * @tparam Policy: Policy that needs to be inherited by the Tensor
 * @tparam PolicyArgs: Policy template parameters
 */
template <template <typename, template <typename...> typename...> typename Tensor, typename Planes,
          template <typename...> typename... Policies, template <typename...> typename Policy, typename... PolicyArgs>
struct append<Tensor<Planes, Policies...>, Policy<PolicyArgs...>> {
  using type = Tensor<Planes, Policies..., Policy>;
};

/*
 * append struct specialization that concatenates the policies of two Tensors into a new Tensor type
 * Parameters:
 * @tparam Tensor: Type of the two Tensors
 * @tparam Planes: Planes types that represent the two Tensors
 * @tparam FirstPoliciesList: Policies inherited by the first tensor
 * @tparam SecondPoliciesList: Policies inherited by the second tensor
 */
template <template <typename, template <typename...> typename...> typename Tensor, typename Planes,
          template <typename...> typename... FirstPoliciesList, template <typename...> typename... SecondPoliciesList>
struct append<Tensor<Planes, FirstPoliciesList...>, Tensor<Planes, SecondPoliciesList...>> {
  using type = Tensor<Planes, FirstPoliciesList..., SecondPoliciesList...>;
};

/*
 * Appends a Policy to the defined Tensor
 * Parameters:
 * @tparam Policy: Policy that needs to be appended
 * @tparam Tensor: Type of the Tensor to which the policy is appended
 */
template <typename Policy, typename Tensor>
using append_t = typename append<Policy, Tensor>::type;

template <typename, typename>
struct erase;

/*
 * erase struct specialization in which the first policy found in the Tensor is removed
 * Parameters:
 * @tparam Policy: First policy that the Tensor inherits
 * @tparam PolicyArgs: Template arguments of the first policy
 * @tparam Tensor: Type of the Tensor
 * @tparam Planes: Planes types that represent the Tensor
 * @tparam Policies: Rest of the policies that the Tensor inherits
 */
template <template <typename...> typename Policy, typename... PolicyArgs,
          template <typename, template <typename...> typename...> typename Tensor, typename Planes,
          template <typename...> typename... Policies>
struct erase<Policy<PolicyArgs...>, Tensor<Planes, Policy, Policies...>> {
  using type = Tensor<Planes, Policies...>;
};

/*
 * erase struct specialization in which the policy is completely ignored, and just the Tensor type is returned
 * Parameters:
 * @tparam Policy: First policy that the Tensor inherits
 * @tparam PolicyArgs: Template arguments of the first policy
 * @tparam Tensor: Type of the Tensor
 * @tparam Planes: Planes types that represent the Tensor
 * @tparam Policies: Rest of the policies that the Tensor inherits
 */
template <template <typename...> typename Policy, typename... PolicyArgs,
          template <typename, template <typename...> typename...> typename Tensor, typename Planes,
          template <typename...> typename... Policies>
struct erase<Policy<PolicyArgs...>, Tensor<Planes, Policies...>> {
  using type = Tensor<Planes, Policies...>;
};

/*
 * erase struct specialization in which the policy that needs to be removed from the Tensor isn't found
 * In this case the recursion continues until we iterate all policies or until we find the policy that needs to be
 * removed Parameters:
 * @tparam Policy: Current policy that is being checked
 * @tparam PolicyArgs: Template arguments of the current policy
 * @tparam Tensor: Type of the Tensor
 * @tparam Planes: Planes types that represent the Tensor
 * @tparam FirstPolicy: First policy that the Tensor inherits
 * @tparam Policies: Rest of the policies that the Tensor inherits
 */
template <template <typename...> typename Policy, typename... PolicyArgs,
          template <typename, template <typename...> typename...> typename Tensor, typename Planes,
          template <typename...> typename FirstPolicy, template <typename...> typename... Policies>
struct erase<Policy<PolicyArgs...>, Tensor<Planes, FirstPolicy, Policies...>> {
  using type = typename append<Tensor<Planes, FirstPolicy>,
                               typename erase<Policy<PolicyArgs...>, Tensor<Planes, Policies...>>::type>::type;
};

/*
 * Removes a Policy from the Tensor
 * Parameters:
 * @tparam Policy: Policy that needs to be removed
 * @tparam Tensor: Type of the Tensor on which the removal is performed
 */
template <typename Policy, typename Tensor>
using erase_t = typename erase<Policy, Tensor>::type;

template <typename>
struct reorder_as_first_policy;

/*
 * Reorders the policies that the Tensor inherits
 * Parameters:
 * @tparam Policy: Policy that needs to be placed on the first position inside the policies sequence
 * @tparam Tensor: Type of the Tensor
 * @tparam Planes: Planes types that represent the Tensor
 * @tparam Policies: Sequence of policies that the Tensor inherits
 */
template <template <typename...> typename Policy,
          template <typename, template <typename...> typename...> typename Tensor, typename Planes,
          template <typename...> typename... Policies>
struct reorder_as_first_policy<Policy<Tensor<Planes, Policies...>>> {
  using type = Policy<append_t<Policy<Tensor<Planes, Policies...>>,
                               erase_t<Policy<Tensor<Planes, Policies...>>, Tensor<Planes, Policies...>>>>;
};

/*
 * Reorders a sequence of policies, putting the specified policy on the first position in the sequence
 * Parameters:
 * @tparam Policy: Policy that should be placed on the first position in the sequence
 */
template <typename Policy>
using reorder_as_first_policy_t = typename reorder_as_first_policy<Policy>::type;

}  // namespace internal

/*
 * A multidimensional array representation with adjustable planes and variadic policy based design
 * The idea behind this design is to allow easily switching out different types of planes and adjusting features through
 * policies as needed Parameters:
 * @tparam Planes: Planes representing the Tensor
 * @tparam Policies: Policies defining the features of the Tensor type
 */
template <typename Planes, template <typename> typename... Policies>
class Tensor : public reorder_as_first_policy_t<Policies<Tensor<Planes, Policies...>>>... {
 private:
  friend first_policy_t<Policies<Tensor<Planes, Policies...>>...>;
  Planes _planes;

 public:
  /*
   * Default constructor
   */
  Tensor() noexcept = default;

  /*
   * Constructs the tensor containing the specified planes using copy semantics
   * Parameters:
   * @param planes: Planes representing the tensor
   */
  explicit Tensor(const Planes& planes) : _planes{planes} {}

  /*
   * Constructs the tensor containing the specified planes using move semantics
   * Parameters:
   * @param planes: Planes representing the tensor
   */
  explicit Tensor(Planes&& planes) : _planes{std::move(planes)} {}

  /*
   * Compares two tensors for equality
   * Parameters:
   * param lhs: first (left-hand side) tensor
   * param rhs: second (right-hand side) tensor
   * True if the tensors are equal, false otherwise
   */
  [[nodiscard]] friend bool operator==(const Tensor& lhs, const Tensor& rhs) noexcept {
    return lhs._planes == rhs._planes;
  }

  /*
   * Creates a new Tensor object with the same planes, but with new policies appended to the existing ones
   * @tparam Policies: new policies that will be inherited by the Tensor object
   * @return: new Tensor object
   */
  template <template <typename> typename... OtherPolicies>
  [[nodiscard]] inline auto like() const {
    return Tensor<Planes, Policies..., OtherPolicies...>{_planes};
  }

  /*
   * Creates a new Tensor object with new planes, and optionally with new policies appended to the existing ones
   * @tparam OtherPolicies: new policies that will be inherited by the Tensor object
   * @tparam OtherPlanes: type of the planes the Tensor will be created with
   * @param planes: planes object used to create the Tensor
   * @return: new Tensor object
   */
  template <template <typename> typename... OtherPolicies, typename OtherPlanes>
  [[nodiscard]] inline auto like(OtherPlanes planes) const {
    return Tensor<OtherPlanes, Policies..., OtherPolicies...>{planes};
  }
};

/*
 * Specialization of the Tensor class containing no policies
 */
template <typename Planes>
class Tensor<Planes> {
 private:
  Planes _planes;

 public:
  /*
   * Default constructor
   */
  Tensor() noexcept = default;

  /*
   * Constructs the tensor containing the specified planes using copy semantics
   * Parameters:
   * @param planes: Planes representing the tensor
   */
  explicit Tensor(const Planes& planes) : _planes{planes} {}

  /*
   * Constructs the tensor containing the specified planes using move semantics
   * Parameters:
   * @param planes: Planes representing the tensor
   */
  explicit Tensor(Planes&& planes) : _planes{std::move(planes)} {}

  /*
   * Compares two tensors for equality
   * Parameters:
   * param lhs: first (left-hand side) tensor
   * param rhs: second (right-hand side) tensor
   * True if the tensors are equal, false otherwise
   */
  [[nodiscard]] friend bool operator==(const Tensor& lhs, const Tensor& rhs) noexcept {
    return lhs._planes == rhs._planes;
  }

  /*
   * Creates a new Tensor object with the same planes, but with new policies
   * @tparam Policies: policies that will be inherited by the Tensor object
   * @return: new Tensor object
   */
  template <template <typename> typename... Policies>
  [[nodiscard]] inline auto like() const {
    return Tensor<Planes, Policies...>{_planes};
  }

  /*
   * Creates a new Tensor object with new planes, and optionally with new policies
   * @tparam OtherPolicies: policies that will be inherited by the Tensor object
   * @tparam OtherPlanes: type of the planes the Tensor will be created with
   * @param planes: planes object used to create the Tensor
   * @return: new Tensor object
   */
  template <template <typename> typename... OtherPolicies, typename OtherPlanes>
  [[nodiscard]] inline auto like(OtherPlanes planes) const {
    return Tensor<OtherPlanes, OtherPolicies...>{planes};
  }
};

/*
 * Helper method used for creating a tensor
 * Parameters:
 * @tparam Policies: Policies that the tensor object has to inherit
 * @tparam Planes: Planes of the tensor
 * @return: new Tensor object
 */
template <template <typename> typename... Policies, typename... Planes>
[[nodiscard]] inline auto create_tensor(Planes&&... planes) {
  return Tensor<ntensor::Planes<std::decay_t<Planes>...>, Policies...>{ntensor::Planes{planes...}};
}

/*
 * Creates a functor which calls a function with each index found in an index sequence
 * Parameters:
 * @tparam is: index sequence
 * @return: functor that calls a passed input function with each index found in the given index sequence
 */
template <std::size_t... is>
[[nodiscard]] consteval auto make_index_dispatcher(std::index_sequence<is...>) noexcept {
  return [](auto&& f) { (f(std::integral_constant<std::size_t, is>{}), ...); };
}

/*
 * Calls an invocable with each plane of a tensor separately
 * Parameters:
 * @param invocable: invocable that's called with each plane
 * @param tensor: Tensor object over whose planes the invocable is called
 */
template <typename Invocable, typename Tensor>
void for_each_plane(Invocable&& invocable, Tensor&& tensor) {
  static constexpr std::size_t N = std::decay_t<decltype(tensor.planes())>::size();
  static constexpr auto dispatcher = make_index_dispatcher(std::make_index_sequence<N>{});
  dispatcher([&tensor, &invocable](auto idx) { invocable(tensor.planes().template plane<idx>()); });
}

/*
 * Calls the invocable with a plane from each tensor. All tensors must have the same number of planes. During each
 * iteration the invocable gets called with the index of the current iterated plane of each tensor Parameters:
 * @param invocable: invocable that's called with all of the planes
 * @param tensors: Sequence of Tensor objects over whose planes the invocable is called
 */
template <typename Invocable, typename... Tensors>
void for_all_planes(Invocable&& invocable, Tensors&&... tensors) {
  using first_tensor = fts_t<Tensors...>;
  static constexpr std::size_t N = std::decay_t<decltype(std::declval<first_tensor>().planes())>::size();

  // Assure that all tensors have the same number of planes
  static_assert(((N == std::decay_t<decltype(tensors.planes())>::size()) && ...));

  static constexpr auto dispatcher = make_index_dispatcher(std::make_index_sequence<N>{});
  dispatcher([&tensors..., &invocable](auto idx) { invocable(tensors.planes().template plane<idx>()...); });
}

}  // namespace ntensor
