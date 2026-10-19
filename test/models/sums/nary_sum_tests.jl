@testitem "A sum of n terms agrees with a chain of binary additions" begin
    @model function nary_sum(y, n)
        local x
        for i in 1:n
            x[i] ~ Normal(mean = i / 2, variance = i)
        end
        total := +(x...)
        y ~ Normal(mean = total, variance = 0.5)
    end

    @model function chain_sum(y, n)
        local x
        for i in 1:n
            x[i] ~ Normal(mean = i / 2, variance = i)
        end
        s[1] := x[1] + x[2]
        for i in 3:n
            s[i - 1] := s[i - 2] + x[i]
        end
        y ~ Normal(mean = s[n - 1], variance = 0.5)
    end

    for n in (2, 3, 5), y in (-1.0, 3.0)
        nary = infer(
            model = nary_sum(n = n), data = (y = y,), free_energy = true
        )
        chain = infer(
            model = chain_sum(n = n), data = (y = y,), free_energy = true
        )

        for i in 1:n
            @test collect(mean_var(nary.posteriors[:x][i])) ≈
                collect(mean_var(chain.posteriors[:x][i]))
        end
        @test collect(mean_var(nary.posteriors[:total])) ≈
            collect(mean_var(chain.posteriors[:s][n - 1]))
        @test last(nary.free_energy) ≈ last(chain.free_energy)
    end
end

@testitem "A sum with fixed terms agrees with a chain of binary additions" begin
    @model function sum_three(out, a, b, c)
        out := a + b + c
    end

    @model function nary_shifted(y, c1, c2)
        x ~ Normal(mean = 0.5, variance = 1.0)
        total ~ sum_three(a = c1, b = x, c = c2)
        y ~ Normal(mean = total, variance = 0.5)
    end

    @model function chain_shifted(y, c1, c2)
        x ~ Normal(mean = 0.5, variance = 1.0)
        s := c1 + x
        total := s + c2
        y ~ Normal(mean = total, variance = 0.5)
    end

    for data in
        ((y = 2.0, c1 = 2, c2 = -1.0f0), (y = -0.5, c1 = 1.5, c2 = 0.25))
        nary = infer(model = nary_shifted(), data = data, free_energy = true)
        chain = infer(model = chain_shifted(), data = data, free_energy = true)

        @test collect(mean_var(nary.posteriors[:x])) ≈
            collect(mean_var(chain.posteriors[:x]))
        @test collect(mean_var(nary.posteriors[:total])) ≈
            collect(mean_var(chain.posteriors[:total]))
        @test last(nary.free_energy) ≈ last(chain.free_energy)
    end
end
