# Tests for inference over *partially-referenced* (sparse) conditioned data tensors.
#
# When a model conditions on a data array but only references some of its indices in `~`
# statements (e.g. masked / missing observations, or a sub-model that only touches the
# observed entries), GraphPPL materializes data-variable labels only at those indices,
# leaving the rest of the bounding box as `#undef` holes. Result collection
# (`getvardict`/`getvarref`), the random/data/anonymous predicates, and the per-iteration
# data feed must all tolerate such sparse arrays — iterating only the assigned entries —
# and the dense data the user supplies must be fed to the materialized variables by index.

@testitem "Inference with a 1-D sparse (partially-referenced) data tensor" begin
    @model function partial_1d(y)
        x ~ NormalMeanVariance(0.0, 100.0)
        y[1] ~ NormalMeanVariance(x, 1.0)
        y[3] ~ NormalMeanVariance(x, 1.0) # y[2] is never referenced -> a hole
    end

    # y[2] = 999.0 is supplied but never referenced; it must be ignored entirely.
    result = infer(
        model = partial_1d(), data = (y = [1.0, 999.0, 3.0],), iterations = 1
    )
    q = last(result.posteriors[:x])

    # x ~ N(0, 100) with two unit-variance observations 1.0 and 3.0:
    @test isapprox(precision(q), 1 / 100 + 2; atol = 1e-8)
    @test isapprox(weightedmean(q), 0 / 100 + (1.0 + 3.0); atol = 1e-8)
end

@testitem "Inference with a 2-D sparse data tensor" begin
    @model function partial_2d(y)
        x ~ NormalMeanVariance(0.0, 100.0)
        y[1, 1] ~ NormalMeanVariance(x, 1.0)
        y[2, 2] ~ NormalMeanVariance(x, 1.0)
        y[1, 3] ~ NormalMeanVariance(x, 1.0) # (2,1),(1,2),(2,3) are holes
    end

    data = reshape(collect(1.0:6.0), 2, 3) # [1 3 5; 2 4 6]
    result = infer(model = partial_2d(), data = (y = data,), iterations = 1)
    q = last(result.posteriors[:x])

    observed = [data[1, 1], data[2, 2], data[1, 3]] # 1.0, 4.0, 5.0
    @test isapprox(precision(q), 1 / 100 + length(observed); atol = 1e-8)
    @test isapprox(weightedmean(q), sum(observed); atol = 1e-8)
end

@testitem "Inference with a sub-model that references data only at masked positions" begin
    # `x` is the missing interface, so `x ~ observe(...)` binds the shared latent.
    @model function observe(y_, mask_, x)
        for j in eachindex(mask_)
            if mask_[j]
                y_[j] ~ NormalMeanVariance(x, 1.0)
            end
        end
    end

    @model function masked_chain(y, mask)
        J, T = size(mask)
        x ~ NormalMeanVariance(0.0, 100.0)
        for t in 1:T
            x ~ observe(y_ = y[1:J, t:t], mask_ = mask[1:J, t:t])
        end
    end

    J, T = 2, 3
    mask = Bool[1 0 1; 0 1 0] # observed: (1,1), (2,2), (1,3)
    data = reshape(collect(1.0:6.0), J, T) # [1 3 5; 2 4 6]
    result = infer(
        model = masked_chain(mask = mask), data = (y = data,), iterations = 1
    )
    q = last(result.posteriors[:x])

    observed = [data[idx] for idx in findall(mask)] # 1.0, 4.0, 5.0
    @test isapprox(precision(q), 1 / 100 + length(observed); atol = 1e-8)
    @test isapprox(weightedmean(q), sum(observed); atol = 1e-8)
end

@testitem "Dense data tensors are unaffected by the sparse-array handling" begin
    @model function dense_model(y)
        x ~ NormalMeanVariance(0.0, 100.0)
        for i in eachindex(y)
            y[i] ~ NormalMeanVariance(x, 1.0)
        end
    end

    ys = [1.0, 2.0, 3.0]
    result = infer(model = dense_model(), data = (y = ys,), iterations = 1)
    q = last(result.posteriors[:x])

    @test isapprox(precision(q), 1 / 100 + length(ys); atol = 1e-8)
    @test isapprox(weightedmean(q), sum(ys); atol = 1e-8)
end

@testitem "Sparse-array helpers (_is_densely_assigned / _map_sparse)" begin
    import RxInfer: _is_densely_assigned, _map_sparse
    import GraphPPL: ResizableArray

    # Plain arrays are always dense.
    @test _is_densely_assigned([1, 2, 3])
    @test _is_densely_assigned(reshape(1:6, 2, 3))

    # A densely-built ResizableArray is dense; `_map_sparse` matches `Base.map` (returns Array).
    dense = ResizableArray(Int, Val(1))
    dense[1] = 1;
    dense[2] = 2;
    dense[3] = 3
    @test _is_densely_assigned(dense)
    mapped = _map_sparse(x -> x + 10, dense)
    @test mapped isa Array
    @test mapped == [11, 12, 13]

    # A 1-D ResizableArray with a hole (use a non-bitstype so the hole is a true `#undef`).
    sparse1 = ResizableArray(String, Val(1))
    sparse1[1] = "a";
    sparse1[3] = "c" # index 2 is a hole
    @test !_is_densely_assigned(sparse1)
    res1 = _map_sparse(uppercase, sparse1)
    @test res1 isa ResizableArray
    @test size(res1) == (3,)
    @test isassigned(res1, 1) && !isassigned(res1, 2) && isassigned(res1, 3)
    @test res1[1] == "A" && res1[3] == "C"

    # A 2-D ResizableArray with holes preserves shape and assigned positions.
    sparse2 = ResizableArray(String, Val(2))
    sparse2[1, 1] = "x";
    sparse2[2, 2] = "y";
    sparse2[1, 3] = "z"
    @test !_is_densely_assigned(sparse2)
    res2 = _map_sparse(uppercase, sparse2)
    @test res2 isa ResizableArray
    @test res2[1, 1] == "X" && res2[2, 2] == "Y" && res2[1, 3] == "Z"
    @test !isassigned(res2, 2, 1) && !isassigned(res2, 1, 2)
end

@testitem "Streaming inference with a partially-referenced (sparse) data tensor" begin
    import RxInfer: from

    @model function ssm_sparse(y, xm, xv)
        x ~ NormalMeanVariance(xm, xv)
        y[1] ~ NormalMeanVariance(x, 1.0)
        y[3] ~ NormalMeanVariance(x, 1.0) # y[2] never referenced -> hole
    end

    autoupdates = @autoupdates begin
        xm, xv = mean_var(q(x))
    end

    run_stream(y2a, y2b) = begin
        engine = infer(
            model          = ssm_sparse(),
            datastream     = from([(y = [1.0, y2a, 3.0],), (y = [2.0, y2b, 4.0],)]),
            autoupdates    = autoupdates,
            initialization = (@initialization begin
                q(x) = NormalMeanVariance(0.0, 100.0)
            end),
            historyvars    = (x = KeepLast(),),
            keephistory    = 10,
            autostart      = true,
        )
        return engine.history[:x]
    end

    # Runs at all (previously: UndefRefError in `getvardict` before the stream started).
    h1 = run_stream(999.0, -777.0)
    @test length(h1) == 2
    @test all(isfinite ∘ mean, h1)

    # The unreferenced entry y[2] must have no effect on the inferred trajectory.
    h2 = run_stream(-12345.0, 54321.0)
    @test all(isapprox.(mean.(h1), mean.(h2); atol = 1e-10))
    @test all(isapprox.(var.(h1), var.(h2); atol = 1e-10))
end

@testitem "Conditioning on offset-indexed (OffsetArray) data with 1-based model indexing" begin
    using OffsetArrays

    # Model indexes 1-based explicitly.
    @model function obs_1based(y)
        x ~ NormalMeanVariance(0.0, 100.0)
        y[1] ~ NormalMeanVariance(x, 1.0)
        y[2] ~ NormalMeanVariance(x, 1.0)
        y[3] ~ NormalMeanVariance(x, 1.0)
    end
    # Model indexes via eachindex (after normalization, eachindex(y) is 1-based).
    @model function obs_eachindex(y)
        x ~ NormalMeanVariance(0.0, 100.0)
        for i in eachindex(y)
            y[i] ~ NormalMeanVariance(x, 1.0)
        end
    end

    ys = [10.0, 20.0, 30.0]
    # `warn = false`: the offset-copy warning is expected here and is asserted separately in
    # the "Offset data emits a `warn`-gated copy warning (batch)" testitem below. This testitem
    # only checks inference correctness, so the warning is suppressed to keep CI logs clean.
    check(model, ydata, observed) = begin
        q = last(
            infer(
                model = model,
                data = (y = ydata,),
                iterations = 1,
                warn = false,
            ).posteriors[:x],
        )
        @test isapprox(precision(q), 1 / 100 + length(observed); atol = 1e-8)
        @test isapprox(weightedmean(q), sum(observed); atol = 1e-8)
    end

    # Baseline (1-based) is unchanged.
    check(obs_1based(), ys, ys)
    # 0-based OffsetArray: previously errored at construction; now normalized to 1-based.
    check(obs_1based(), OffsetArray(ys, 0:2), ys)
    # Negative-based OffsetArray.
    check(obs_1based(), OffsetArray(ys, -1:1), ys)
    # `eachindex` over offset data now yields 1-based indices.
    check(obs_eachindex(), OffsetArray(ys, 0:2), ys)
end

@testitem "Offset-indexed data combined with sparse (partial) referencing" begin
    using OffsetArrays

    @model function obs_sparse(y)
        x ~ NormalMeanVariance(0.0, 100.0)
        y[1] ~ NormalMeanVariance(x, 1.0)
        y[3] ~ NormalMeanVariance(x, 1.0) # y[2] (1-based) unreferenced -> sparse
    end

    # `warn = false` for the same reason as above: the offset-copy warning is intentional and
    # asserted in the dedicated warning testitem; here we only check inference correctness.
    check(ydata) = begin
        q = last(
            infer(
                model = obs_sparse(),
                data = (y = ydata,),
                iterations = 1,
                warn = false,
            ).posteriors[:x],
        )
        @test isapprox(precision(q), 1 / 100 + 2; atol = 1e-8)        # two observations
        @test isapprox(weightedmean(q), 10.0 + 30.0; atol = 1e-8)     # 1st and 3rd values, 2nd ignored
    end

    check([10.0, 999.0, 30.0])                       # standard, sanity
    check(OffsetArray([10.0, 999.0, 30.0], 0:2))     # offset + sparse
    check(OffsetArray([10.0, 999.0, 30.0], -1:1))    # negative-based + sparse
end

@testitem "__normalize_data_indexing leaves standard data untouched and rebases offset data" begin
    import RxInfer: __normalize_data_indexing
    using OffsetArrays

    # Non-arrays and 1-based arrays are returned unchanged (same object, no copy).
    @test __normalize_data_indexing(3.0) === 3.0
    v = [1.0, 2.0, 3.0]
    @test __normalize_data_indexing(v) === v
    m = [1.0 2.0; 3.0 4.0]
    @test __normalize_data_indexing(m) === m

    # Offset arrays are rebased to 1-based, preserving values and order.
    o = OffsetArray([10.0, 20.0, 30.0], 0:2)
    n = __normalize_data_indexing(o)
    @test axes(n) == (Base.OneTo(3),)
    @test n == [10.0, 20.0, 30.0]

    o2 = OffsetArray([1.0 2.0; 3.0 4.0], 0:1, 0:1)
    n2 = __normalize_data_indexing(o2)
    @test axes(n2) == (Base.OneTo(2), Base.OneTo(2))
    @test n2 == [1.0 2.0; 3.0 4.0]
end

@testitem "Offset data emits a `warn`-gated copy warning (batch)" begin
    using OffsetArrays, Logging

    @model function coin(y)
        θ ~ Beta(1.0, 1.0)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    flips = [1.0, 0.0, 1.0, 1.0, 0.0]
    offset = OffsetArray(flips, 0:4)
    has_offset_warn(logs) =
        any(l -> l.level == Logging.Warn && occursin("offset", l.message), logs)

    # standard 1-based data: no copy, no warning
    logs, _ = Test.collect_test_logs() do
        infer(model = coin(), data = (y = flips,))
    end
    @test !has_offset_warn(logs)

    # offset data, default warn = true: warning emitted
    logs, _ = Test.collect_test_logs() do
        infer(model = coin(), data = (y = offset,))
    end
    @test has_offset_warn(logs)

    # offset data, warn = false: suppressed
    logs, _ = Test.collect_test_logs() do
        infer(model = coin(), data = (y = offset,), warn = false)
    end
    @test !has_offset_warn(logs)
end

@testitem "Offset data emits a `warn`-gated copy warning (streaming, sparse var)" begin
    using OffsetArrays, Logging
    import RxInfer: from

    @model function ssm(y, xm, xv)
        x ~ NormalMeanVariance(xm, xv)
        y[1] ~ NormalMeanVariance(x, 1.0)
        y[3] ~ NormalMeanVariance(x, 1.0) # sparse: y[2] unreferenced
    end
    autoupdates = @autoupdates begin
        xm, xv = mean_var(q(x))
    end

    run_stream(warnflag) = Test.collect_test_logs() do
        infer(
            model          = ssm(),
            datastream     = from([(y = OffsetArray([1.0, 999.0, 3.0], 0:2),)]),
            autoupdates    = autoupdates,
            initialization = (@initialization begin
                q(x) = NormalMeanVariance(0.0, 100.0)
            end),
            keephistory    = 1,
            autostart      = true,
            warn           = warnflag,
        )
    end
    has_offset_warn(logs) =
        any(l -> l.level == Logging.Warn && occursin("offset", l.message), logs)

    logs_on, _  = run_stream(true)
    logs_off, _ = run_stream(false)
    @test has_offset_warn(logs_on)
    @test !has_offset_warn(logs_off)
end

@testitem "new_observation_indexed! feeds sparse data variables by index" begin
    import RxInfer: new_observation_indexed!
    import GraphPPL: ResizableArray
    const RMP = RxInfer.ReactiveMP

    readback(dv) = RMP.getdata(RxInfer.Rocket.getrecent(dv.messageout))

    # Build a sparse array of data variables with a hole at index 2.
    vars = ResizableArray(typeof(RMP.datavar()), Val(1))
    vars[1] = RMP.datavar()
    vars[3] = RMP.datavar() # index 2 is a hole

    # The provided dense data has 3 entries; index 2 (999.0) must be ignored.
    new_observation_indexed!(vars, [10.0, 999.0, 30.0])

    @test readback(vars[1]) == PointMass(10.0)
    @test readback(vars[3]) == PointMass(30.0)
end
