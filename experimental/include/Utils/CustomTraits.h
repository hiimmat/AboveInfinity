#pragma once

namespace AboveInfinity {

/**
 * Type list that can hold template template parameters
 * 
 * @tparam: list of template template parameters
 */
template<template<typename> typename...>
class TTypeList {};

template<typename FTTList, typename STTList>
struct _AppendType;

template<template<typename> typename... TList, template<typename> typename T>
struct _AppendType<TTypeList<TList...>, TTypeList<T>> {
    using result_type = TTypeList<TList..., T>;
};

template<template<typename> typename T, template<typename> typename... TList>
struct _AppendType<TTypeList<T>, TTypeList<TList...>> {
    using result_type = TTypeList<T, TList...>;
};

template<template<typename> typename... FTTList, template<typename> typename... STTList>
struct _AppendType<TTypeList<FTTList...>, TTypeList<STTList...>> {
    using result_type = TTypeList<FTTList..., STTList...>;
};

/**
 * Type trait that appends a type to a list
 * 
 * @tparam FTTList: Typelist of template template parameters
 * @tparam STTList: Type that needs to be added to the list
 */
template<typename FTTList, typename STTList>
using AppendType = typename _AppendType<FTTList, STTList>::result_type;

template<typename TTList, template<typename> typename T>
struct _EraseType;

/**
 * Type trait that removes a type from the list
 * 
 * @tparam TTList: Typelist of template template parameters
 * @tparam T: Type that needs to be removed from the list
 */
template<typename TTList, template<typename> typename T>
using EraseType = typename _EraseType<TTList, T>::result_type;

template<template<typename> typename... Tail, template<typename> typename T>
struct _EraseType<TTypeList<T, Tail...>, T> {
    using result_type = TTypeList<Tail...>;
};

// If the type is not a part of the parameter list, return the existing parameter list
template<template<typename> typename... Tail, template<typename> typename T>
struct _EraseType<TTypeList<Tail...>, T> {
    using result_type = TTypeList<Tail...>;
};

template<template<typename> typename Head, template<typename> typename... Tail, template<typename> typename T>
struct _EraseType<TTypeList<Head, Tail...>, T> {
    using result_type = AppendType<TTypeList<Head>, EraseType<TTypeList<Tail...>, T>>;
};

/**
 * Type trait that adds a type to a Typelist
 * 
 * @tparam T: Type that needs to be added to the TypeList
 */
template<template<typename> typename T>
struct Tag {
    using type = TTypeList<T>;
};

/**
 * Extracts the last type of a list
 * 
 * @tparam typename: Typelist from which the last type is extracted from
 */
template<typename>
struct SelectLast;

template<template<typename> typename... Ts>
struct SelectLast<TTypeList<Ts...>> {
    using type = typename decltype((Tag<Ts>{}, ...))::type;
};

/**
 * Removes the last type from a Typelist
 * 
 * @tparam Ts: Typelist from which the last type needs to be removed
 */
template<template<typename> typename... Ts>
struct RemoveLastType;

template<template<typename> typename T>
struct RemoveLastType<T> {
    using type = TTypeList<>;
};

/**
 * Concatenates two Typelists
 * 
 * @tparam typename: First typelist that's being concatenated
 * @tparam typename: Second typelist that's being concatenated
 */
template<typename, typename>
struct ConcatenateLists {};

template<template<typename> typename... Ts1, template<typename> typename... Ts2>
struct ConcatenateLists<TTypeList<Ts1...>, TTypeList<Ts2...>> {
    using type = TTypeList<Ts1..., Ts2...>;
};

template<template<typename> typename T, template<typename> typename... Ts>
struct RemoveLastType<T, Ts...> {
    using type = typename ConcatenateLists<TTypeList<T>, typename RemoveLastType<Ts...>::type>::type;
};

/**
 * Extracts a type from a typelist
 * 
 * @tparam T: Type to extract from the typelist
 * @tparam U: Typelist used for the extraction
 */
template<typename T, typename U>
struct ExtractType;

template<template<typename> typename T, typename U>
struct ExtractType<TTypeList<T>, U> {
    using type = T<U>;
};

template<typename FirstList, typename SecondList, typename NewElement, int N, int position>
struct _InsertType;

template<template<typename> typename... Ts,
         template<typename>
         typename Head,
         template<typename>
         typename... Us,
         template<typename>
         typename NewElement,
         int N,
         int position>
struct _InsertType<TTypeList<Ts...>, TTypeList<Head, Us...>, TTypeList<NewElement>, N, position> {
    using type =
        typename _InsertType<TTypeList<Ts..., Head>, TTypeList<Us...>, TTypeList<NewElement>, N + 1, position>::type;
};

template<template<typename> typename... Ts,
         template<typename>
         typename Head,
         template<typename>
         typename... Us,
         template<typename>
         typename NewElement,
         int N>
struct _InsertType<TTypeList<Ts...>, TTypeList<Head, Us...>, TTypeList<NewElement>, N, -1> {
    using type = TTypeList<Ts..., Head, Us..., NewElement>;
};

template<template<typename> typename... Ts,
         template<typename>
         typename Head,
         template<typename>
         typename... Us,
         template<typename>
         typename NewElement,
         int N>
struct _InsertType<TTypeList<Ts...>, TTypeList<Head, Us...>, TTypeList<NewElement>, N, N> {
    using type = TTypeList<Ts..., NewElement, Head, Us...>;
};

template<template<typename> typename NewElement, int position>
struct _InsertType<TTypeList<>, TTypeList<>, TTypeList<NewElement>, 0, position> {
    requires(position <= 0);
    using type = TTypeList<NewElement>;
};

/**
 * Inserts a type into a Typelist at the specified position
 * 
 * @tparam List: Typelist in which the new type is inserted
 * @tparam NewElement: Type that's being inserted into the list
 * @tparam position: Position on which the new type will be inserted
 */
template<typename List, template<typename> typename NewElement, int position>
using InsertType = typename _InsertType<TTypeList<>, List, TTypeList<NewElement>, 0, position>::type;

template<typename FirstList, typename SecondList, int N, int position>
struct _RemoveType;

template<template<typename> typename... Ts,
         template<typename>
         typename Head,
         template<typename>
         typename... Us,
         int N,
         int position>
struct _RemoveType<TTypeList<Ts...>, TTypeList<Head, Us...>, N, position> {
    using type = typename _RemoveType<TTypeList<Ts..., Head>, TTypeList<Us...>, N + 1, position>::type;
};

template<template<typename> typename... Ts, template<typename> typename Head, template<typename> typename... Us, int N>
struct _RemoveType<TTypeList<Ts...>, TTypeList<Head, Us...>, N, -1> {
    using type = typename _RemoveType<TTypeList<Ts...>, TTypeList<Head, Us...>, N, sizeof...(Us)>::type;
};

template<template<typename> typename... Ts, template<typename> typename Head, template<typename> typename... Us, int N>
struct _RemoveType<TTypeList<Ts...>, TTypeList<Head, Us...>, N, N> {
    using type = TTypeList<Ts..., Us...>;
};

/**
 * Removes the type at the specified position from the typelist
 * 
 * @tparam List: Typelist from which the type is removed
 * @tparam position: Position of the type that will be removed
 */
template<typename List, int position>
using RemoveType = typename _RemoveType<TTypeList<>, List, 0, position>::type;

} // namespace AboveInfinity