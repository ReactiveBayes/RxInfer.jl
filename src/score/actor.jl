using Rocket

mutable struct ScoreActorProps
    count       :: Int
    is_released :: Bool
end

struct ScoreActor{L} <: Rocket.Actor{L}
    score :: Vector{L}
    inds  :: Vector{Int}
    props :: ScoreActorProps
end

ScoreActor()                            = ScoreActor(Real)
ScoreActor(::Type{L}) where {L <: Real} = ScoreActor{L}(Vector{L}(), Vector{Int}(), ScoreActorProps(0, true))

Base.show(io::IO, ::ScoreActor) = print(io, "ScoreActor()")

function Rocket.on_next!(actor::ScoreActor{L}, data::L) where {L}
    actor.props.count += 1
    actor.props.is_released = false
    push!(actor.score, data)
end

function Rocket.on_error!(actor::ScoreActor, err)
    error(err)
end

function Rocket.on_complete!(actor::ScoreActor)
    Rocket.release!(actor)
    nothing
end

function Rocket.release!(actor::ScoreActor)
    if !actor.props.is_released
        push!(actor.inds, actor.props.count)
        actor.props.is_released = true
    end
end

function Rocket.getvalues(actor::ScoreActor)
    if length(actor.inds) <= 1
        return actor.score
    end

    n       = length(actor.inds)
    indices = Iterators.flatten((0, actor.inds))

    # Allocation free pairwise
    pairwise = zip(Iterators.take(indices, n), Iterators.take(Iterators.drop(indices, 1), n))

    columns = map(pairwise) do (left, right)
        return view(actor.score, (left+1):right)
    end

    maxlength = maximum(length.(columns))

    result = zeros(eltype(actor.score), maxlength)

    foreach(columns) do column
        broadcast!(+, result, result, Iterators.flatten((column, Iterators.repeated(last(column), maxlength - length(column)))))
    end

    return result
end
