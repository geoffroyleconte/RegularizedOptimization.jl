for (mod, mod_name) ∈ ((x -> x, "exact"), (LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
    for solver_sym ∈ (:R2, :TR)
      solver_sym == :TR && mod_name == "exact" && continue
      h_name == "B0" && continue  # FIXME
      solver_name = string(solver_sym)
      solver = eval(solver_sym)
      @testset "bpdn-with-bounds-$(mod_name)-$(solver_name)-$(h_name)" begin
        x0 = zeros(bpdn2.meta.nvar)
        p = randperm(bpdn2.meta.nvar)[1:nz]
        x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
        args = solver_sym == :R2 ? () : (NormLinf(1.0),)
        @test has_bounds(mod(bpdn2))
        out = solver(mod(bpdn2), h, args..., options, x0 = x0)
        @test typeof(out.solution) == typeof(bpdn2.meta.x0)
        @test length(out.solution) == bpdn2.meta.nvar
        @test typeof(out.solver_specific[:Fhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:Hhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:Hhist])
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:SubsolverCounter])
        @test obj(bpdn2, out.solution) == out.solver_specific[:Fhist][end]
        @test h(out.solution) == out.solver_specific[:Hhist][end]
        @test out.status == :first_order
      end
    end
  end
end

for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
  for solver_sym ∈ (:LM, :LMTR)
    solver_name = string(solver_sym)
    solver = eval(solver_sym)
    h_name == "B0" && continue  # FIXME
    @testset "bpdn-with-bounds-ls-$(solver_name)-$(h_name)" begin
      x0 = zeros(bpdn_nls2.meta.nvar)
      p = randperm(bpdn_nls2.meta.nvar)[1:nz]
      x0[p[1:nz]] = abs.(sign.(randn(nz)))  # initial guess with nz nonzeros (necessary for h = B0)
      args = solver_sym == :LM ? () : (NormLinf(1.0),)
      @test has_bounds(bpdn_nls2)
      out = solver(bpdn_nls2, h, args..., options, x0 = x0)
      @test typeof(out.solution) == typeof(bpdn_nls2.meta.x0)
      @test length(out.solution) == bpdn_nls2.meta.nvar
      @test typeof(out.solver_specific[:Fhist]) == typeof(out.solution)
      @test typeof(out.solver_specific[:Hhist]) == typeof(out.solution)
      @test typeof(out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:Hhist])
      @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:SubsolverCounter])
      @test obj(bpdn_nls2, out.solution) == out.solver_specific[:Fhist][end]
      @test h(out.solution) == out.solver_specific[:Hhist][end]
      @test out.status == :first_order
    end
  end
end
