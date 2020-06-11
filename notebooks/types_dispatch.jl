# # Functions
# 
# In Julia, a function is an object that maps a tuple of argument values to a return value. In the following example we define function that

function f(x,y)
    x * y
end

# This function accepts two arguments `x` and `y` and returns the value of the last expression evaluated, which is `x * y`.

[f(2, 3), f(2, -3)]

# Sometimes it is useful to return something other than the last expression.  For such a case there is an `return` keyword:

function g(x,y)
    val = x * y
    if val < 0
        return -val
    else
        return val
    end
end

# This function accepts two arguments `x` and `y` and computes `val = x * y`. Then if `val` is less than zero, it returns` -val`, otherwise it returns `val`.

[g(2, 3), g(2, -3)]
 
# The traditional function declaration syntax demonstrated above is equivalent to the following compact form, which is very common in Julia:

f(x,y) = x * y

# ### Optional and keyword arguments

# Other very useful things are optional and keyword arguments, which can be added in a very easy way

function f_hello(x, y, a = 0; sayhello = false)
    sayhello && println("Hello everyone 👋" )
    x * y + a
end

# This function accepts two arguments `x` and `y`, one optional argument `a` and one keyword argument `sayhello`. If the function is called only with mandatory arguments, then it returns `x * y + 0` 

f_hello(2,3)

# The change of the optional argument `a` will change the output value to `x * y + a`

f_hello(2,3,2)

# Since `f_hello` is a function with good manners (as opposed to `f`), it says hello if the keyword argument `sayhello` is true

f_hello(2,3; sayhello = true)


# ### Anonymous functions

# It is also common to use anonymous functions, ie functions without specified name. Such a function can be defined in almost the same way as a normal function:

h1 = function (x)
    x^2 + 2x - 1
end

h2 = x ->  x^2 + 2x - 1

#  Those two function declarations create functions with automatically generated names. Then variables `h1` and `h2` only refers to these functions. The primary use for anonymous functions is passing them to functions which take other functions as arguments. A classic example is `map`, which applies a function to each value of an array and returns a new array containing the resulting values:

map(x -> x^2 + 2x - 1, [1,3,-1])

# For more complicated functions, the `do` blocks can be used

map([1,3,-1]) do x
    x^2 + 2x - 1
end

# # Types, methods and multiple-dispatch

# So far we did not mention any types. The default behavior in Julia when types are omitted is to allow values to be of any type. Thus, one can write many useful Julia functions without ever explicitly using types. When additional expressiveness is needed, however, it is easy to gradually introduce explicit type annotations into previously "untyped" code. 
# 
#  In Julia, functions consist of multiple methods. The choice of which method to execute when a function is applied is called dispatch. Julia allows the dispatch process to choose which of a function's methods to call based on
# * the number of arguments given
# * types of all of the function's arguments.
# 
# Using all of a function's arguments to choose which method should be invoked is known as **multiple dispatch**. 
# 
# Until now, we have defined only functions with a single method having unconstrained argument types.  

f(x, y) = x * y

# We can easily check which methods are defined for this function using the `methods` function

methods(f)

# Each function can be easily extended by new methods

f(x, y, z) = x * y * z
f(x, y, z, q) = x * y * z * q
f(x...) = reduce(*, x)

methods(f)

# Since we did not specify what types of arguments are allowed, function `f` will work for all types

[
    f(2, 3),
    f(2.0, 3),
    f(2, 3.0),
    f("a", "b")
]

# However, some combinations of arguments will result in an error

f(:a, :b)

#  When using types we can be extremely conservative and we can set a specific type for each function argument

foo(x::Int64, y::Int64) = x * y

# This function definition applies only to calls where `x` and `y` are both values of type Int64:

foo(2,3)

# Applying it to any other types of arguments will result in a `MethodError`:

foo(2.0,3)

# It is better to use abstract types like `Number` or` Real` instead of concrete types like `Float64`,` Float32`, `Int64` ... .  To find an super type for a specific type, we can use  `supertype` function

supertype(Int64)

# or we can create a simple recursive function that prints the entire tree of supertypes for a given type

function supertypes_tree(::Type{T}, k::Int = 0) where {T <: Any}
    T === Any && return 
    col = isabstracttype(T) ? :blue : :green 
    printstyled(repeat("   ", k)..., T, "\n"; bold = true, color = col)    
    supertypes_tree(supertype(T), k + 1)
    return
end

supertypes_tree(Int64)

#  All abstract types are printed in blue and all concrete types are printed in green. There is also `subtypes` function, which returns all subtypes for a given type.  

subtypes(Number)

# As with supertypes, we can create a simple recursive function that prints the entire tree of subtypes for a given type.

function subtypes_tree(::Type{T}, k::Int = 0) where {T <: Any}
    col = isabstracttype(T) ? :blue : :green 
    printstyled(repeat("   ", k)..., T; bold = true, color = col)
    println()
    subtypes_tree.(subtypes(T), k + 1)
    return
end


subtypes_tree(Number)

# From the tree of all subtypes of the abstract type "Number," we see the whole structure of numerical types in Julia. So if we really want to specify the argument types of a function, we should use some abstract type, such as `Real`

foo(x::Real, y::Real) = x * y

# This function definition applies to calls where `x` and `y` are both values of any subtype of `Real`.

Real[
    foo(2.0, 3)
    foo(2.0, 3.0)
    foo(Int32(2), Int16(3.0))
    foo(Int32(2), Float32(3.0))
]

# Now we can check again how many methods are defined for `foo`

methods(foo)

# ### Method Ambiguities
# 
# It is possible to define a set of function methods such that there is no unique most specific method applicable to some combinations of arguments:

goo(x::Float64, y) = x * y
goo(x, y::Float64) = x + y

#  Here, the `goo` function has two methods. The first method applies if the first argument is of type `Float64`. 

goo(2.0, 3)

# The second method applies if the second argument is of type `Float64`. 

goo(2, 3.0)

# The case, where both arguments are of type `Float64` can be handled by both methods. The problem is that neither method is more specific than the other. In such cases, Julia raises a `MethodError` rather than arbitrarily picking a method.

goo(2.0, 3.0)

#  We can avoid method ambiguities by specifying an appropriate method for the intersection case:

goo(x::Float64, y::Float64) = x - y
goo(2.0, 3.0)

# If we can check again how many methods are defined for `goo`, there will be three methods

methods(goo)


# ### Composite types
# 
# Composite types are called records, structs, or objects in various languages. A composite type is a collection of named fields, an instance of which can be treated as a single value. In many languages, composite types are the only kind of user-definable type, and they are by far the most commonly used user-defined type in Julia as well.

abstract type Food{T<:Real} end
abstract type Fruit{T<:Real} <: Food{T} end
abstract type Vegetable{T<:Real} <: Food{T} end

struct Apple{T<:Real} <: Fruit{T}  weight::T end
struct Orange{T<:Real} <: Fruit{T} weight::T end
struct Banana{T<:Real} <: Fruit{T} weight::T end

struct Cucumber{T<:Real} <: Vegetable{T} weight::T end
struct Carrot{T<:Real} <: Vegetable{T} weight::T end
struct Lettuce{T<:Real} <: Vegetable{T} weight::T end

# Using the `subtypes_tree` function, we can easily check the type hierarchy

subtypes_tree(Food)

# In Julia, it is not possible to set mandatory fields for all subtypes of a given abstract type. For example, each food subtype should have a specified color. However, we can easily define general properties using multiple-dispatch

color(::Type{<:Apple}) = "red"
color(::Type{<:Orange}) = "orange"
color(::Type{<:Banana}) = "yellow"
color(::Type{<:Cucumber}) = "green"
color(::Type{<:Carrot}) = "orange"
color(::Type{<:Lettuce}) = "green"

# However, these methods can be applied only to the type itself. 

[color(Apple), color(Lettuce)]

# To apply `color` function to a specific instance of any `Food` subtype, we must do the following

a = Apple(123)
color(typeof(a))

# However, it can be also done in a more elegant way using a new method. 

color(::T) where {T<:Food} = color(T)

[color(Apple), color(Apple(123))]

# Now we can define two other functions:
# * `weight` function return the rounded weight of given food
# * `description` function prints some basic information about a given food

weight(x::Food) = ceil(Int64, x.weight)
description(x::T) where {T <: Food} =
    println("$(supertype(T).name): $(T.name), color: $(color(x)), weight: $(weight(x))g")

fruits = [
    Apple(150),
    Orange(235.4),
    Banana(186.6),
    Cucumber(246.1),
    Carrot(120),
    Lettuce(169)
]

description.(fruits);

# It is very useful to know which functions are defined for a particular type. We can use the `methodswith` function to get such information.

methodswith(Apple; supertypes = true)


# # Complex example: Temperatures ([original source](https://medium.com/@Jernfrost/defining-custom-units-in-julia-and-python-513c34a4c971))
#
# In this example, we will show how to deal with temperatures in different units (*Celsius*, *Kelvin*, *Fahrenheit*). We have following goals:
# 
# 1. define types that represent temperature units
# 2. define functions for conversion between temperature types
# 3. define basic arithmetic operations for temperature types
#     * `Temperature + Temperature`
#     * `Temperature - Temperature`
#     * `number * Temperature`
#     * `Temperature / number`
# 
# First, we define the abstract type `Temperature`. All of the above functions can be implemented without the use of the `Temperature` type, but it will be much more complicated.

abstract type Temperature end

# We can now define a `Celsius` structure that will represent the temperature in degree *Celsius*.

struct Celsius <: Temperature
    value::Float64
    
    function Celsius(x::Real)
        x < -273.15 && throw(ArgumentError("input temperature smaller than absolute zero"))
        return new(x)
    end
end

# The previous definition is valid for any real number greater than or equal to *-273.15*, which is absolute zero in degrees Celsius.

(Celsius(-273.15), Celsius(0), Celsius(100)) 

# Since Julia supports multiple-dispatch, we can easily extend existing functions to support newly defined types. For example, we can extend the `show` function from the` Base` module to change the way `Celsius` type is printed in REPL.

Base.show(io::IO, t::Celsius) = print(io, t.value, "°C")

# Using the same values as above, we get the following output

(Celsius(-273.15), Celsius(0), Celsius(100)) 

# In  the same way, we can easily define another temperature scale.

struct Kelvin <: Temperature
    value::Float64

    function Kelvin(x::Real)
        x < 0 && throw(ArgumentError("input temperature smaller than absolute zero"))
        return new(x)
    end
end

Base.show(io::IO, t::Kelvin) = print(io, t.value, "K")

# ## Conversion
# 
# We are now able to express temperatures in two different units, but we are not able to convert from one unit to another. In order to convert between units, we need to create a conversion function.

Celsius2kelvin(t::Celsius) = Kelvin(t.value + 273.15)
Kelvin2Celsius(t::Kelvin) = Celsius(t.value - 273.15)

[
    Celsius2kelvin(Celsius(0)),
    Kelvin2Celsius(Kelvin(0))
]

# However, the better way is to extend the `convert` function from the` Base` module and combine it with outer constructors for temperature types. In the first step, we define conversion rules using outer constructors

Kelvin(t::Celsius) = Kelvin(t.value + 273.15)
Celsius(t::Kelvin) = Celsius(t.value - 273.15)

# In the second step, we define `convert` function for any subtype of the abstract type `Temperature`.

Base.convert(::Type{T}, t::T) where {T<:Temperature} = t
Base.convert(::Type{T}, t::Temperature) where {T<:Temperature} = T(t)

# The first method is only a temperature identity. The second method is an auxiliary function that passes the given temperature `t` to the constructor of the given type of temperature `T`. Using the same example as for the `temp_convert` function above results in 

[
    convert(Celsius, Celsius(0)),
    convert(Kelvin, Kelvin(0)),
    convert(Kelvin, Celsius(0)),
    convert(Celsius, Kelvin(0))
]

# ### Basic arithmetic operations
# 
# Before defining any arithmetic operation, we must define the right way to deal with cases where we have to deal with temperatures in different temperature scales. To do that, we have to define  `promote_rule` for our types

Base.promote_rule(::Type{Kelvin}, ::Type{Celsius}) = Kelvin

[
    promote_type(Celsius, Kelvin)
    promote(Celsius(-273.15), Kelvin(0))
]

# We can now define basic arithmetic operations in two easy steps as can be seen in the following code

import Base: +, -, *, /

+(x::Temperature, y::Temperature) = +(promote(x,y)...)
+(x::T, y::T) where {T<:Temperature} = T(x.value + y.value)

-(x::Temperature, y::Temperature) = -(promote(x,y)...)
-(x::T, y::T) where {T<:Temperature} = T(x.value - y.value)

# Now we are able to add and subtract temperatures in different temperature scales

[
    Celsius(-273.15) + Kelvin(0),
    Kelvin(0) + Celsius(-273.15),
    Celsius(-273.15) - Kelvin(0),
    Kelvin(0) - Celsius(-273.15)
]

# We can also define the multiplication of temperature by a number and rounding function

*(x::Number, y::T) where {T <: Temperature} = T(x * y.value)
*(x::T, y::Number) where {T <: Temperature} = T(y * x.value)

Base.round(t::T, args...; kwargs...) where {T<:Temperature} = T(round(t.value, args...; kwargs...))

# In Julia, it is possible to apply given function `f(x)` to each element of an array `A` to yield a new array via `f.(A)`. We can use this syntax to obtain a random temperature vector in degrees Kelvin as follows

temps_K = Kelvin.(273.15 .+ 20 .* rand(10))

#  In the same way, we can convert this vector to degrees Celsius and round it to two digits

temps_C = round.(Celsius.(temps_K); digits = 2)

# Finally, we can compute for example the sum of this vector 

sum(temps_C)

# ### Adding new temperature scale
# 
# To add a new temperature scale, we have to: 
# * define new type
# * extend `Base.show` (otpional)
# * define outer constructors for `Kelvin` and `Celsius`
# * define promote rules for `Kelvin` and `Celsius`

struct Fahrenheit <: Temperature
    value::Float64

    function Fahrenheit(x::Real)
        x < -459.67 && throw(ArgumentError("input temperature smaller than absolute zero"))
        return new(x)
    end
end

Base.show(io::IO, t::Fahrenheit) = print(io, t.value, "°F")

Celsius(t::Fahrenheit) = Celsius((t.value - 32)*5/9)
Fahrenheit(t::Celsius) = Fahrenheit(t.value*9/5 + 32)
Kelvin(t::Fahrenheit) = Kelvin(Celsius(t))
Fahrenheit(t::Kelvin) = Fahrenheit(Celsius(t))

Base.promote_rule(::Type{Fahrenheit}, ::Type{Celsius}) = Celsius
Base.promote_rule(::Type{Fahrenheit}, ::Type{Kelvin}) = Kelvin

# Now all the functions defined for `Kelvin` and `Celsius` will work for `Fahrenheit` as well

Temperature[
    Fahrenheit(Celsius(0)),
    Celsius(Fahrenheit(32)),
    Fahrenheit(Kelvin(0)),
    Kelvin(Fahrenheit(-459.67)),
]

Fahrenheit(32) + Celsius(0) - Kelvin(273.15)

# To obtain even more user-friendly behavior, we can define constants representing 1 degree in each temperature scale

const °C = Celsius(1)
const K = Kelvin(1)
const °F = Fahrenheit(1)

#  With these constants and the fact, that `*` operator can be omitted in some cases, we can work with temperatures as follows

2°C + 4K - 2°F