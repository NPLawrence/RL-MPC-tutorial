using Pkg
Pkg.add(["ControlSystems","LinearAlgebra","Plots","LaTeXStrings","RollingFunctions"])

using ControlSystems
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using RollingFunctions

P = c2d(tf([1.0],[1.0,1.0])^3, 1.0)
# C = pid(2.0, 0.0, 0.0, form=:parallel, Ts=1.0)
C = pid(1.0, 0.5, 0.0, form=:parallel, Ts=1.0)
G = feedback(P*C)

y,t,x,δ = lsim(G, (x,t)-> (t >= 0), -4:20)
u,_,_,_ = lsim(C*sensitivity(P,C), (x,t)-> (t >= 0), -4:20)

y_extend = kron(y,ones(1,10))
δ_extend = kron(u,ones(1,10))

t_extend = range(t[1],t[end],length(y_extend)+1)

anim_u = @animate for i ∈ 1:length(δ_extend)
    plot(t_extend[1:i], δ_extend[1:i], xlims=(t_extend[1],t_extend[end]), ylims=(-0.1, 2.2),
    linetype=:step,
    legend=false, lw=4, xlabel="", ylabel="", axis=([], false))
end

anim_y = @animate for i ∈ 1:length(δ_extend)
    plot(t_extend[1:i], y_extend[1:i], xlims=(t_extend[1],t_extend[end]), ylims=(-0.1, 1.5),
    linetype=:step,
    legend=false, lw=4, xlabel="", ylabel="", axis=([], false), color=:red4)
    plot!(t, t->(t>=0), color=:grey, linestyle=:dash, lw=2, linetype=:step)
end

gif(anim_u, "figs/anim_PI_u.gif", fps = 45)
gif(anim_y, "figs/anim_PI_y.gif", fps = 45)
