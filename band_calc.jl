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
Γ = [0.0,0.0]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],256)
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
gs=[1.0,cispi(-0.245), zeros(Nopt-2)...]
ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=gs,Nstep=2*10^5)

mat = calmat(lat, Γ)
ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=10^4)
SpinHall.gaugephi0!(ϕG, ϕ0)
ϕ0'*ϕG|>expshow

##

myint(ϕ0[:,1],ϕ0[:,2],lat.Kvec,"PT")|>expshow
myint(ϕG,ϕG,lat.Kvec,"S3")|>expshow

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

x = range(-pi,pi,24)
y = x
sp = cal_bloch_spin(Γ, ϕG, lat, x, y)
arrows(x, y, sp[1], sp[2], arrowsize = 8, lengthscale = 1,
    arrowcolor = vec(sp[3]), linecolor = vec(sp[3])
)

# -------------------------------------------------------------
##                   BdG sortperm
# -------------------------------------------------------------
ben = main_BdG(lat,ϕG,u0,kl.k,20)
fig=series(kl.r, ben[1:12,:];
    color=repeat(Makie.wong_colors(),3),
    figure=(;size=(400,400*0.63)),
    linewidth=1.5,
    axis=(;xticks=xt,yticks=range(0,10,6),ygridvisible=false)
)

##
Mk0,tz = cal_BdG(lat,ϕG,u0,Γ)
Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=1)
Jy,Dhy = cal_Ju(ϕG,Γ,lat.Kvec; u=2,sp=1)

w = range(-0.8,0.8,256) #[range(0,1.5,100); range(1.6,4.4,18); range(4.45,6.0,120)]
Xw1 = Green1(Mk0,w,Jx,Jy,η=0.01)./lat.Sunit
fig = series(w,hcat(reim(Xw1)...)',marker=:circle,axis=(;limits=(nothing,(-0.1,0.1))))

##  谱分解计算 Spin Hall
ben,bev=eigBdG(Mk0)
Xw2 = Xspec1(w,Dhx,Dhy,ben,bev,ϕG,η=0.02)./lat.Sunit
series(w,hcat(reim(Xw2)...)',marker=:circle,linestyle=:dash)
# fig