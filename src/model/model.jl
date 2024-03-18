
export ProbabilisticModel
export getoptions, getconstraints, getmeta
export getnodes, getvariables, getrandom, getconstant, getdata

import Base: push!, show, getindex, haskey, firstindex, lastindex
import ReactiveMP: getaddons, AbstractFactorNode
import Rocket: getscheduler

struct ProbabilisticModel{M}
    model::M
end

getmodel(model::ProbabilisticModel) = model.model

getvardict(model::ProbabilisticModel) = getvardict(getmodel(model))
getrandomvars(model::ProbabilisticModel) = getrandomvars(getmodel(model))
getdatavars(model::ProbabilisticModel) = getdatavars(getmodel(model))
getconstantvars(model::ProbabilisticModel) = getconstantvars(getmodel(model))
getfactornodes(model::ProbabilisticModel) = getfactornodes(getmodel(model))