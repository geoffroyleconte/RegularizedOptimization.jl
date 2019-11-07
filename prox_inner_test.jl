# Julia Testing function
# Generate Compressive Sensing Data
using Plots, Printf, Random, LinearAlgebra, Convex, SCS, IterativeSolvers
include("DescentMethods.jl")
include("minconf_spg/oneProjector.jl")
include("ProxProj.jl")


#Here we just try to solve an easy example 
#######
# min_s gᵀs + 1/2sᵀBks + λ||s+x||_1	s.t. ||s||_1⩽1
function prox_inner_test()
m,n = 200,200 # let's try something not too hard 
g = randn(n)
B = rand(n,n)
Bk = B'*B
x  = rand(n)
λ = 1.0


S = Variable(n)
problem = minimize(g'*S + sumsquares(B*S)+λ*norm(vec(S+x), 1), norm(vec(S),1)<=1)
solve!(problem, SCSSolver())

function proxp(z, α)
    return sign.(z).*max(abs.(z).-(α)*ones(size(z)), zeros(size(x)))
end
projq(z) = oneProjector(z, 1.0, 1.0)
#input β, λ
w1_options=s_options(norm(Bk)^2;maxIter=100, verbose=3, restart=10, λ=λ, η =.1, η_factor=.9, 
    gk = g, Bk = Bk, xk=x)
s,w = prox_split_1w(proxp, zeros(size(x)), projq, w1_options)

w2_options=s_options(norm(Bk)^2;maxIter=10, verbose=3, restart=200, λ=λ, η =.1, η_factor=.9, 
    gk = g, Bk = Bk, xk=x)
s2,w12,w22 = prox_split_2w(proxp, zeros(size(x)), projq, w2_options)


# x1 = rand(n)
# xp, hispg, fevalpg = PG(funcF, x1, funProj,options)
# x2 = rand(n)
# xf, hisf, fevalf = FISTA(funcF, x2, funProj, options)
@printf("l2-norm CVX: %5.5e\n", norm(S.value - s)/norm(S.value))
@printf("l2-norm CVX: %5.5e\n", norm(S.value - w)/norm(S.value))
@printf("l2-norm CVX: %5.5e\n", norm(S.value - s2)/norm(S.value))
@printf("l2-norm CVX: %5.5e\n", norm(S.value - w12)/norm(S.value))
@printf("l2-norm CVX: %5.5e\n", norm(S.value - w22)/norm(S.value))
# @printf("l2-norm| PG: %5.5e | FISTA: %5.5e\n", norm(xp - xt), norm(xf-xt))
end
