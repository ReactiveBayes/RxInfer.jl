# Rocket.jl message passing specific extensions

using Rocket

mutable struct LimitStackSchedulerProps
    soft_depth::Int
    hard_depth::Int
end

struct LimitStackScheduler <: Rocket.AbstractScheduler
    soft_limit :: Int
    hard_limit :: Int
    props      :: LimitStackSchedulerProps
end

LimitStackScheduler(soft_limit::Int)                  = LimitStackScheduler(soft_limit, typemax(Int) - 1)
LimitStackScheduler(soft_limit::Int, hard_limit::Int) = LimitStackScheduler(soft_limit, hard_limit, LimitStackSchedulerProps(0, 0))

get_soft_limit(scheduler::LimitStackScheduler) = scheduler.soft_limit
get_hard_limit(scheduler::LimitStackScheduler) = scheduler.hard_limit

function increase_depth!(scheduler::LimitStackScheduler)
    scheduler.props.soft_depth = scheduler.props.soft_depth + 1
    scheduler.props.hard_depth = scheduler.props.hard_depth + 1
end

function decrease_depth!(scheduler::LimitStackScheduler)
    scheduler.props.soft_depth = scheduler.props.soft_depth - 1
    scheduler.props.hard_depth = scheduler.props.hard_depth - 1
end

get_soft_depth(scheduler::LimitStackScheduler)     = scheduler.props.soft_depth
set_soft_depth!(scheduler::LimitStackScheduler, v) = scheduler.props.soft_depth = v

get_hard_depth(scheduler::LimitStackScheduler)     = scheduler.props.hard_depth
set_hard_depth!(scheduler::LimitStackScheduler, v) = scheduler.props.hard_depth = v

Base.show(io::IO, scheduler::LimitStackScheduler) = print(
    io,
    "LimitStackScheduler(soft_limit = $(get_soft_limit(scheduler)), hard_limit = $(get_hard_limit(scheduler)))"
)

Base.similar(scheduler::LimitStackScheduler) = LimitStackScheduler(get_soft_limit(scheduler), get_hard_limit(scheduler))

Rocket.makeinstance(::Type, scheduler::LimitStackScheduler) = scheduler

Rocket.instancetype(::Type, ::Type{<:LimitStackScheduler}) = LimitStackScheduler

function limitstack(callback::Function, instance::LimitStackScheduler)
    increase_depth!(instance)
    if get_hard_depth(instance) >= get_hard_limit(instance)
        error("Hard limit in LimitStackScheduler exceeded")
    end
    result = if get_soft_depth(instance) < get_soft_limit(instance)
        callback()
    else
        previous_soft_depth = get_soft_depth(instance)
        set_soft_depth!(instance, 0)
        condition = Base.Condition()
        @async begin
            try
                notify(condition, callback())
            catch exception
                notify(condition, exception, error = true)
            end
        end
        r = wait(condition) # returns `callback()`
        set_soft_depth!(instance, previous_soft_depth)
        r
    end
    decrease_depth!(instance)
    return result
end

struct LimitStackSubscription <: Teardown
    instance     :: LimitStackScheduler
    subscription :: Teardown
end

Rocket.as_teardown(::Type{<:LimitStackSubscription}) = UnsubscribableTeardownLogic()

Rocket.on_unsubscribe!(scheduler::LimitStackSubscription) =
    limitstack(() -> Rocket.unsubscribe!(scheduler.subscription), scheduler.instance)

Rocket.scheduled_subscription!(source, actor, instance::LimitStackScheduler) =
    limitstack(instance) do
        return LimitStackSubscription(instance, Rocket.on_subscribe!(source, actor, instance))
    end

Rocket.scheduled_next!(actor, value, instance::LimitStackScheduler) = limitstack(() -> Rocket.on_next!(actor, value), instance)
Rocket.scheduled_error!(actor, err, instance::LimitStackScheduler)  = limitstack(() -> Rocket.on_error!(actor, err), instance)
Rocket.scheduled_complete!(actor, instance::LimitStackScheduler)    = limitstack(() -> Rocket.on_complete!(actor), instance)
