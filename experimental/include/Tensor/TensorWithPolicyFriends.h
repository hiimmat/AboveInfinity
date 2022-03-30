#pragma once

#include "CustomTraits.h"

#include <iostream>
#include <type_traits>

/*
 * This file contains an example of a Tensor with a variadic number of policies of which all of them can access its
 * private members.
 *
 * Unfortunately, C++ doesn't allow us to use the friend declaration with a variadiac parameter pack, so I had to come
 * up with a different solution.
 *
 * The solution that I came up with requires making a template specialization of the Tensor class that inherits and
 * befriends the first policy in the parameter pack, giving it direct access to its private members. The first policy
 * does the same with the second policy, and the second policy does the same thing with the third. And it goes on like
 * this until we reach the last policy in the pack. Using this design, we can access Tensor's private elements in two
 * ways (at least in two ways that I'm aware of).
 *
 * The first way is through recursive method calls where each policy knows its predecessor and has access to its private
 * members. Here each policy has to have methods that allow them access to the previous policy's private methods, with
 * the first policy being an exception and providing methods for accessing the Tensor's private members instead.
 *
 * The second way is much simpler and it requires significantly less casting. All we have to do is either swap out the
 * positions of the first policy and the Nth policy (where the Nth policy represents the policy that requires
 * access of the Tensor's private members) or move the Nth policy in front of the first policy. In both cases, we can
 * perform a direct cast to the Tensor instead of backtracking to the first policy and casting it to the Tensor.
 * Undoing the policy reordering is also simple, because we already know the previous policy order. Here's an example.
 * Let's say that we have a Tensor like this:
 *
 * Tensor<Planes, FirstPolicy, SecondPolicy, ThirdPolicy, FourthPolicy> tensor;
 *
 * And we want to write a method in the ThirdPolicy that accesses a private Tensors' method called "print". For
 * simplicity, let's assume that the Tensor's print method doesn't receive any arguments. This is how we can write it:
 *
 * void PrivateTensorPrint() {
 *     static_cast<Tensor<Planes, ThirdPolicy, SecondPolicy, FirstPolicy, FourthPolicy>*>(this)->print();
 * }
 *
 * Or even simpler, by just moving the ThirdPolicy to the first position:
 *
 * void PrivateTensorPrint() {
 *     static_cast<Tensor<Planes, ThirdPolicy, FirstPolicy, SecondPolicy, FourthPolicy>*>(this)->print();
 * }
 *
 * To get the initial policy order, we can just read it from ThirdPolicy's class template specialization.
 *
 * Another benefit from this design is that each policy can can access any other policy's members and methods.
 * The downsides of this design are that it's quite complex and it requires casting.
 * The downside of creating new elements due to casting might be partially solved within C++20, since we'll have
 * transient dynamic allocations available during compile-time. I'm saying transient dynamic allocations because they'll
 * have to be deallocated before the compilation finishes.
 * Also, it's worth noting that the last policy shouldn't inherit the next policy it sees in the list, since the next
 * policy will be the first policy, which indirectly inherits the last policy, which inherits the first policy, and so
 * on.
 *
 * The following code is an example of two different approaches of accessing the Tensor's private elements through
 * his policies.
 */

template<typename Planes, template<typename> typename First, template<typename> typename... Rest>
class Tensor : public First<Tensor<Planes, Rest..., First>> {
private:
    friend First<Tensor<Planes, Rest..., First>>;

    void print() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
};

template<typename, typename>
struct ExtractTensorType;

template<typename Planes, template<typename> typename... Ts>
struct ExtractTensorType<Planes, TTypeList<Ts...>> {
    using type = Tensor<Planes, Ts...>;
};

template<typename T>
class FirstPolicy {};

// Specialization used in case if the first policy is the only policy
// We could get away without this policy if we wrote a type trait that returns the next policy in the list. And if this
// is the only policy in the list, or if this is the last policy in the list, the type trait would return an empty base instead
template<typename Planes>
class FirstPolicy<Tensor<Planes, FirstPolicy>> {};

template<typename Planes, template<typename> typename Next, template<typename> typename... Rest>
class FirstPolicy<Tensor<Planes, Next, Rest...>> : public Next<Tensor<Planes, Rest..., Next>> {
public:
    using currentLast = typename SelectLast<TTypeList<Rest...>>::type;
    using lastRemoved = typename RemoveLastType<Next, Rest...>::type;
    using updatedList = AppendType<currentLast, lastRemoved>;
    using previous = typename ExtractTensorType<Planes, updatedList>::type;

private:
    friend Next<Tensor<Planes, Rest..., Next>>;

    void print() {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        static_cast<previous*>(this)->print();
    }
};

template<typename T>
class SecondPolicy {};

template<typename Planes, template<typename> typename Next, template<typename> typename... Rest>
class SecondPolicy<Tensor<Planes, Next, Rest...>> : public Next<Tensor<Planes, Rest..., Next>> {
public:
    using currentLast = typename SelectLast<TTypeList<Rest...>>::type;
    using lastRemoved = typename RemoveLastType<Next, Rest...>::type;
    using updatedList = AppendType<currentLast, lastRemoved>;
    using newLast = typename SelectLast<updatedList>::type;
    using previous = typename ExtractType<newLast, typename ExtractTensorType<Planes, updatedList>::type>::type;

private:
    friend Next<Tensor<Planes, Rest..., Next>>;

    void print() {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        static_cast<previous*>(this)->print();
    }
};

template<typename T>
class ThirdPolicy {};

template<typename Planes, template<typename> typename Next, template<typename> typename... Rest>
class ThirdPolicy<Tensor<Planes, Next, Rest...>> : public Next<Tensor<Planes, Rest..., Next>> {
public:
    using currentLast = typename SelectLast<TTypeList<Rest...>>::type;
    using lastRemoved = typename RemoveLastType<Next, Rest...>::type;
    using updatedList = AppendType<currentLast, lastRemoved>;
    using newLast = typename SelectLast<updatedList>::type;
    using previous = typename ExtractType<newLast, typename ExtractTensorType<Planes, updatedList>::type>::type;

private:
    friend Next<Tensor<Planes, Rest..., Next>>;

    void print() {
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        static_cast<previous*>(this)->print();
    }
};

template<typename T>
class FourthPolicy {};

/*
 * Last policy specialization could be generalized by writing a type trait that checks if the current policy is the last
 * policy in the circular list. Or by checking if this or a previous policy is derived from the next policy. Then it would
 * be possible to use that same type trait with std::conditional_t and to either inherit the next policy in the list or an
 * empty base class.
 */
template<typename Planes, template<typename> typename Next, template<typename> typename... Rest>
class FourthPolicy<Tensor<Planes, Next, Rest...>> {
public:
    using currentLast = typename SelectLast<TTypeList<Rest...>>::type;
    using lastRemoved = typename RemoveLastType<Next, Rest...>::type;
    using updatedList = AppendType<currentLast, lastRemoved>;
    using newLast = typename SelectLast<updatedList>::type;
    using previous = typename ExtractType<newLast, typename ExtractTensorType<Planes, updatedList>::type>::type;

    void recursivePrintExample() {
        std::cout << "Recursive print example:\n";
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        static_cast<previous*>(this)->print();
        std::cout << std::endl;
    }

    void tensorCastExample() {
        std::cout << "Tensor cast example:\n";
        std::cout << __PRETTY_FUNCTION__ << std::endl;
        static_cast<typename ExtractTensorType<Planes, updatedList>::type*>(this)->print();
        std::cout << std::endl;
    }
};

struct Planes {};

int main() {
    Tensor<Planes, FirstPolicy, SecondPolicy, ThirdPolicy, FourthPolicy> t;
    t.recursivePrintExample();
    t.tensorCastExample();

    return 0;
}
