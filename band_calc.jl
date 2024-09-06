# 首次运行时须按[option]+[enter]执行以下3行，清除包的具体版本依赖，以后就不用了，在其他文件也不需要
rm("Manifest.toml")
using Pkg
Pkg.instantiate()
##

using Revise
using SpinHall

using CairoMakie
set_theme!(;size=(500,400))
const cm = 72/2.54

function set_lattice(v0, m0, gg)
    mz = 0.0
    g =[0.35,0.3].*gg # [g_{↑↑}, g_{↑↓}]
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]

    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end


# -------------------------------------------------------------
##                   single particle states
# -------------------------------------------------------------
lat = set_lattice(8.0,1.5,1.0)
Γ = [0.0,0.1]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],128)
en = eigband(lat,kl.k, 1:20)
xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
fig= series(kl.r,en.-en[1]; axis=(;xticks=xt,yticks=0:2:12),color=repeat(Makie.wong_colors(),4))

## some 2D plot

M = -0.5.*(lat.b[:,1].+lat.b[:,2])
bz = mymesh([M, M.+lat.b[:,1], M.+lat.b[:,2]].*0.5, [24,24])
x = eigen2D(lat,bz,1:12)

fig,ax,hm = heatmap(x.bcav[1,:,:])
Colorbar(fig[1,2],hm)
fig|>display

x.bcav[:,12,12].-cal_Bcav(lat, Γ, 1:12)



# -------------------------------------------------------------
##                   Ground State
# -------------------------------------------------------------
Nopt = 10
E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
gs=[1.0,cispi(0.24), zeros(Nopt-2)...]
ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=gs)


##
mat = calmat(lat, Γ)
ϕG,u0=imag_time_evl(mat, ϕG, lat)
SpinHall.gaugephi0!(ϕG, ϕ0)
ϕ0'*ϕG|>expshow

##

x = range(-pi,pi,128)
y = x
ψup = cal_bloch_wave(Γ, ϕG[1:lat.NK], lat, x, y)
ψdn = cal_bloch_wave(Γ, ϕG[lat.NK+1:end], lat, x, y)

fig,ax,hm=heatmap(x,y,abs.(ψup),colormap=:thermal,axis=(;title="ψ↑",aspect=1))
Colorbar(fig[1,2],hm)
fig|>display

fig,ax,hm=heatmap(x,y,abs.(ψdn),colormap=:thermal,axis=(;title="ψ↓",aspect=1))
Colorbar(fig[1,2],hm)
fig|>display


# -------------------------------------------------------------
##                   BdG sortperm
# -------------------------------------------------------------
ben = main_BdG(lat,ϕG,u0,kl.k,20)
fig=series(kl.r, ben;
    color=repeat(Makie.wong_colors(),3),
    figure=(;size=(600,600*0.63)),
    linewidth=1.5,
    axis=(;xticks=xt,yticks=range(0,10,6),ygridvisible=false)
)


