#pragma once
#include <utility>


//usign for debugging purposes only
template<class F, class ... As>
void for_each_arg(F f, As&& ... as)
{
	using IntArr = int[sizeof...(As)];

	//initializer list which initializes array to all zeroes and as side effect
	//we apply function f to all args
	//fill up the array with zeroes (as many zeroes as args in template)
	(void)IntArr {
		(f(std::forward<As>(as)), 0)...
	};
}


#include <ostream>

namespace helpers
{

	//function specialization if there are no additional args, so we don't have undefined behavior
	void print_call_impl(std::ostream&)
	{}

	//funtion specialization if there are actually args present
	template<class H, class ... Ts>
	void print_call_impl(std::ostream& out, H const& h, Ts const& ... ts)
	{
		out << h; //h in this case is a function and Ts are function parameters
		for_each_arg([&](auto const& x) { out << ", " << x; }
		, ts...);
	}

} // helpers

template<class FN, class ... As>
void print_call(std::ostream& out, FN function_name, As const& ... as)
{
	out << function_name << "(";
	helpers::print_call_impl(out, as...);
	out << ")\n";
}

#define PRINT_CALL0(out) print_call(out, __FUNCTION__)
#define PRINT_CALL(out, ...) print_call(out, __FUNCTION__, __VA_ARGS__)




// USAGE :

//#include <iostream>
//#include <string>
//
//void empty_fn()
//{
//	PRINT_CALL0(std::cout); //for void parameters of functions
//}
//
//void some_fn(int x, float y, std::string str)
//{
//	PRINT_CALL(std::cout, x, y, str);
//}
