# ---------------------------------------------------
##          含时演化，Bloch振荡
# ---------------------------------------------------
using Revise
using SpinHall
using OrderedCollections,FileIO

includet("src/RK.jl")
using .RK

using CairoMakie
set_theme!(;size=(600,400))
const cm = 72/2.54

function set_lattice(v0, m0, gg, mz)
    g = [0.35,0.3].*gg
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end

#=
# ------------------- 集群测试 -----------------------
# machines=readlines("hostlist")  # 节点列表
# machines_and_workers=[(machine,2) for machine in machines] # 每个节点2个进程
# addprocs(machines_and_workers,exeflags="-t2") # -t2 每个进程2个线程
# ----------------------------------------------------

# ------------------- 本地测试 -----------------------
# using Distributed
# addprocs(2,exeflags="-t4")
# ---------------------------------------------------

using LinearAlgebra
using MKL
@everywhere include("common.jl")
@everywhere using .Common
include("Bloch.jl")
using .Bloch: Lattice
include("GroundState.jl")
@everywhere include("RK.jl")
using JLD2,FileIO
=#

# -------------------------------------------------------------
##                   single particle states
# -------------------------------------------------------------
lat = set_lattice(4.0,2.0,1.0,0.0)
Γ = [0.0,0.0]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],256)
en = eigband(lat,kl.k, 1:20)
xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
fig= series(kl.r,en[1:12,:].-en[1]; axis=(;xticks=xt,yticks=0:2:12),color=repeat(Makie.wong_colors(),4))


# ------------------------------------------------------------
##                  Ground state
# ------------------------------------------------------------
lat = set_lattice(4.0,0.65,1.0,0.0)
Γ = [0.0,0.0]
Nopt = 10
E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
init_gs=[1,cispi(-0.25), zeros(Nopt-2)...]

ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=init_gs, Nstep=10^5)
mat = calmat(lat, Γ)
@time ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=10^4)
SpinHall.gaugephi0!(ϕG, ϕ0)
ϕ0'*ϕG|>expshow


# ------------------------------------------------------------
##              time evolution
# ------------------------------------------------------------
function rkw2(
    hk0::Array{ComplexF64,2},
    ψ0::Array{ComplexF64,1},
    lat::Lattice
)
    w = 1.0
    T = 48pi/w         # 总时长
    dt= 0.0002         # 计算步长
    step = 50          # 采样间隔,i.e.每50步保存一个结果
    Δt = dt*step
    Nsample = round(Int,T/Δt)
    Tlist = [Δt*i for i in 1:Nsample]
    r = [cospi(0.0),sinpi(0.0)].*0.02
    k = [0.0,0.0]
    ψt,klist = RK.rkw(r,k,hk0,ψ0,lat.Kvec,dt,step,Nsample,lat.v0,lat.mz)

    return ψt,Tlist,klist
end

##
E0,ϕ0=eigenband(lat, Γ, 1:10)
mat = calmat(lat, Γ)
ψt,Tlist,klist = rkw2(mat,ϕG,lat)

##
px = [RK.momentum(ψt[:,ii],klist[:,ii],lat.Kvec,θ=0.0)[1] for ii in eachindex(Tlist)]
py = [RK.momentum(ψt[:,ii],klist[:,ii],lat.Kvec,θ=0.0)[2] for ii in eachindex(Tlist)]

series(Tlist./Er,[px py]')
scatter(py)
scatter(px,py)

##
ii = 5380
en,ev = eigenband(lat,klist[:,ii], 1:8)
ev[:,2]'*ψt[:,ii]|>abs

##
pu1 = [RK.momentum_sp(ψt[:,ii],klist[:,ii],lat.Kvec;θ=pi/4,spu=true)[1] for ii in eachindex(Tlist)]
pu2 = [RK.momentum_sp(ψt[:,ii],klist[:,ii],lat.Kvec;θ=pi/4,spu=true)[2] for ii in eachindex(Tlist)]
pd1 = [RK.momentum_sp(ψt[:,ii],klist[:,ii],lat.Kvec;θ=pi/4,spu=false)[1] for ii in eachindex(Tlist)]
pd2 = [RK.momentum_sp(ψt[:,ii],klist[:,ii],lat.Kvec;θ=pi/4,spu=false)[2] for ii in eachindex(Tlist)]

series([pu1 pd1]')
series([pu2 pd2]')