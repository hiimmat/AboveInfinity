#include <iostream>


template<typename...>
struct front {};

template<template<typename...> typename T, typename... Ts, typename... Rest>
struct front<T<Ts...>, Rest...> {
    using type = T<Ts...>;
};

template<typename... Ts>
using front_t = typename front<Ts...>::type;


template<typename, typename>
struct append {
};

template<template<typename...> typename T, typename... Ts, template<template<typename...> typename...> typename U, template<typename...> typename... Us>
struct append<T<Ts...>, U<Us...>> {
    using type = U<T, Us...>;
};

template<template<template<typename...> typename...> typename U, template<typename...> typename... Us, template<typename...> typename T, typename...Ts>
struct append<U<Us...>, T<Ts...>> {
    using type = U<Us..., T>;
};

template<template<template<typename...> typename...> typename U, template<typename...> typename... Us, template<typename...> typename... Ts>
struct append<U<Us...>, U<Ts...>> {
    using type = U<Us..., Ts...>;
};

template<typename T, typename U>
using append_t = typename append<T, U>::type;


template<typename, typename>
struct erase {
};

template<template<typename...> typename T, typename...Ts, template<template<typename...> typename...> typename U, template<typename...> typename... Us>
struct erase<T<Ts...>, U<T, Us...>> {
    using type = U<Us...>;
};

template<template<typename...> typename T, typename...Ts, template<template<typename...> typename...> typename U, template<typename...> typename... Us>
struct erase<T<Ts...>, U<Us...>> {
    using type = U<Us...>;
};

template<template<typename...> typename T, typename...Ts, template<template<typename...> typename...> typename U, template<typename...> typename Head, template<typename...> typename... Tail>
struct erase<T<Ts...>, U<Head, Tail...>> {
    using type = typename append<U<Head>, typename erase<T<Ts...>, U<Tail...>>::type>::type;
};

template<typename T, typename U>
using erase_t = typename erase<T, U>::type;


template<typename>
struct move_to_front {
};

template<template<typename...> typename Derived, template<template<typename...> typename...> typename Base, template<typename...> typename... BaseTs>
struct move_to_front<Derived<Base<BaseTs...>>> {
   using type = Derived<append_t<Derived<Base<BaseTs...>>, erase_t<Derived<Base<BaseTs...>>, Base<BaseTs...>>>>;
};

template<typename T>
using move_to_front_t = typename move_to_front<T>::type;


template<template<typename> typename... Policies>
class Tensor : public move_to_front_t<Policies<Tensor<Policies...>>>... {

private:
    friend front_t<Policies<Tensor<Policies...>>...>;

    int x{0};
    
    void private_print() {
        std::cout << "In tensors' private method\n";
    }

public:

    void public_print() {
        std::cout << "In tensors' public method\n";
    }

};

template<>
class Tensor<>{};


template<typename T>
class FirstPolicy {
public:

    void first_print() {
       static_cast<T*>(this)->private_print();
       static_cast<T*>(this)->public_print();
    }

};

template<typename T>
class SecondPolicy {
public:

void second_print() {
    std::cout << "Value of x: " << static_cast<T*>(this)->x << std::endl;
    static_cast<T*>(this)->x = 5;
    std::cout << "Value of x: " << static_cast<T*>(this)->x << std::endl;
}

};

template<typename T>
class ThirdPolicy {
public:

    void third_print() {
        std::cout << "Value of x: " << static_cast<T*>(this)->x << std::endl;
    }

};


int main() {
    Tensor<FirstPolicy, SecondPolicy, ThirdPolicy> t;

    t.first_print();
    t.second_print();
    t.third_print();

    return 0;
}
