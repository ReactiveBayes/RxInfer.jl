using RxInfer

struct MockModel
    fe_stream
end

function RxInfer.score(model::MockModel, ::RxInfer.BetheFreeEnergy, _)
    return model.fe_stream
end