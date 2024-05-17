include("colloquium.jl")

P = tf([1.4],[1.0,1.0])^3

# Q(ζ) = tf([1.0],[1,2ζ,1])
Q(x) = 0.85tf([float(-2sin(4π*x)),1.0], [1.0float(sin(π*x)),1.0])*tf([1.0],[1.0float(abs(sin(2π*x))),1.0])
# plot(tq, y')
# plot!(impulse(P*Q(10),tq))
# plot!(impulse(P,tq))

ζ = reverse(range(0.0005,0.9995,length=200))

anim = @animate for z in ζ
    plot(impulse(Q(z)),
    xlims=(0,8), ylims=(-1,1), axis=([], false), lw=4, linetype=:step, xlabel="", ylabel="")
end

anim_yk = @animate for z in ζ
    plot(impulse(P*Q(z)),
    xlims=(0,8), ylims=(-1,1), axis=([], false), lw=4, linetype=:step, xlabel="", ylabel="")
end

# anim_Q = @animate for z in ζ
#     plot(impulse(Q(z),10.0), linetype=:step, xticks=[], yticks=[], legend=false, lw=4)
# end
# anim_yk = @animate for z in ζ
#     plot(impulse(P*Q(z),10),
#     linetype=:step, xticks=[], yticks=[], legend=false, lw=4)
# end

gif(anim, "anim_Q.gif", fps = 15)
gif(anim_yk, "anim_YK.gif", fps = 15)
