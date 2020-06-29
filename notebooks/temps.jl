abstract type Temperature end

# We can now define a `Celsius` structure that will represent the temperature in degree *Celsius*.

struct Celsius <: Temperature
    value::Float64
    
    function Celsius(x::Real)
        # x < -273.15 && throw(ArgumentError("input temperature smaller than absolute zero"))
        return new(x)
    end
end


Base.show(io::IO, t::Celsius) = print(io, value(t), "°C")


struct Kelvin <: Temperature
    value::Float64

    function Kelvin(x::Real)
        # x < 0 && throw(ArgumentError("input temperature smaller than absolute zero"))
        return new(x)
    end
end

value(x::Temperature) = x.value
Zygote.@adjoint value(x::T) where T<:Temperature = x.value, x̄ -> (T(x̄),)

Base.show(io::IO, t::Kelvin) = print(io, value(t), "K")

Celsius2kelvin(t::Celsius) = Kelvin(value(t) + 273.15)
Kelvin2Celsius(t::Kelvin) = Celsius(value(t) - 273.15)


Kelvin(t::Celsius) = Kelvin(value(t) + 273.15)
Celsius(t::Kelvin) = Celsius(value(t) - 273.15)


Base.convert(::Type{T}, t::T) where {T<:Temperature} = t
Base.convert(::Type{T}, t::Temperature) where {T<:Temperature} = T(t)


Base.promote_rule(::Type{Kelvin}, ::Type{Celsius}) = Kelvin

import Base: +, -, *, /

+(x::Temperature, y::Temperature) = +(promote(x,y)...)
+(x::T, y::T) where {T<:Temperature} = T(value(x) + value(y))

-(x::Temperature, y::Temperature) = -(promote(x,y)...)
-(x::T, y::T) where {T<:Temperature} = T(value(x) - value(y))

*(x::Number, y::T) where {T <: Temperature} = T(x * value(y))
*(x::T, y::Number) where {T <: Temperature} = T(y * value(x))

Base.round(t::T, args...; kwargs...) where {T<:Temperature} = T(round(value(t), args...; kwargs...))


struct Fahrenheit <: Temperature
    value::Float64

    function Fahrenheit(x::Real)
        x < -459.67 && throw(ArgumentError("input temperature smaller than absolute zero"))
        return new(x)
    end
end

Base.show(io::IO, t::Fahrenheit) = print(io, value(t), "°F")

Celsius(t::Fahrenheit) = Celsius((value(t) - 32)*5/9)
Fahrenheit(t::Celsius) = Fahrenheit(value(t)*9/5 + 32)
Kelvin(t::Fahrenheit) = Kelvin(Celsius(t))
Fahrenheit(t::Kelvin) = Fahrenheit(Celsius(t))

Base.promote_rule(::Type{Fahrenheit}, ::Type{Celsius}) = Celsius
Base.promote_rule(::Type{Fahrenheit}, ::Type{Kelvin}) = Kelvin

const °C = Celsius(1)
const K = Kelvin(1)
const °F = Fahrenheit(1)


import Base: zero
zero(t::Kelvin) = Kelvin(0.0)


*(t1::Kelvin, t2::Kelvin) = Kelvin(value(t1)*value(t2))

import Base: /

/(x::Number, y::T) where {T <: Temperature} = T(x / value(y))
/(x::T, y::Number) where {T <: Temperature} = T(value(x) / y)

Base.conj!(x::Array{Kelvin,2}) = x
Base.conj(x::Kelvin) = x
Base.adjoint(x::Kelvin) = x