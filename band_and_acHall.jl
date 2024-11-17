# -------------------------------------------------------------
##     单粒子能谱，贝里曲率，BdG能谱，有限频Spin Hall
# -------------------------------------------------------------
using Revise
using SpinHall
using OrderedCollections,FileIO
using CairoMakie
set_theme!(;size=(600,400))
const cm = 72/2.54

function set_lattice(v0, m0, gg)
    mz = 0.0
    g = [0.35,0.3].*gg
    Kmax = 6
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end

# -------------------------------------------------------------
##                   单粒子能谱
# -------------------------------------------------------------
lat = set_lattice(4.0,1.5,1.0)
Γ = [0.0,0.0]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],256)
en = eigband(lat,kl.k, 1:20)
xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
fig= series(kl.r,en.-en[1]; axis=(;xticks=xt,yticks=0:2:12),color=repeat(Makie.wong_colors(),4))


## --- Berry curvature ---
M = -0.5.*(lat.b[:,1].+lat.b[:,2])
bz = mymesh([M, lat.b[:,1], lat.b[:,2]].*0.5, [24,24])
x = eigen2D(lat,bz,1:12)

fig,ax,hm = heatmap(x.bcav[1,:,:])
Colorbar(fig[1,2],hm)
fig|>display
x.bcav[:,12,12].-cal_Bcav(lat, Γ, 1:12)


##  nonabelian berry curvature
using LinearAlgebra: dot,diagm,tr
##
k0 = [0.1,0.2]#./2
Nb = 2
dk = hcat([[cos(0pi/3+i*pi/2),sin(0pi/3+i*pi/2)] for i in 0:3]...).*(0.001*sqrt(2))
dS = 2*(dk[1]^2+dk[2]^2)
en,ev = eigenband(lat,dk.+k0,1:12)
A = Array{ComplexF64}(undef,Nb,Nb,4)
for i in 1:4,n in 1:Nb,m in 1:Nb
    j = i==4 ? 1 : i+1
    A[m,n,i] = dot(ev[:,m,i],ev[:,n,j])
end
B = diagm(fill(1.0+0im,Nb))
for i in 1:4
    B.= B*A[:,:,i]
end
(dB=B.-diagm(fill(1.0+0im,Nb)))|>myfilter|>display
dB./dS|>myfilter|>tr

##
s = 1.0+0im
for i in 1:4
    s*= A[1,1,i]
end
(s-1)/dS|>display
cal_Bcav(lat, k0, 1:2)



# ---------------------------------------------------
##               BdG spectrum
# ---------------------------------------------------
Nopt = 10
E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
init_gs=[1,cispi(-0.25), zeros(Nopt-2)...]

ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=init_gs, Nstep=10^5)
mat = calmat(lat, Γ)
@time ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=10^4)
SpinHall.gaugephi0!(ϕG, ϕ0)
ϕ0'*ϕG|>expshow

# ϕG = (ϕ0[:,1].+cispi(0.25).*ϕ0[:,2])./√2  # nointeracting ground state


## ----  plot ground state ----
x = range(-pi,pi,20)
sp = cal_bloch_spin(Γ, ϕG, lat, x, x)
nsp= vec(sqrt.(sp[1].^2+sp[2].^2))
arrows(x, x, sp[1], sp[2], arrowsize = 8, lengthscale = 1,
    arrowcolor = nsp, linecolor = nsp, axis=(;aspect=1)
)

## ----  计算BdG能谱 ------
@time ben = eig_BdG(lat,ϕG,u0,kl.k,20); ## 55.078
fig=series(kl.r, ben[1:20,:];
    #color=repeat(Makie.wong_colors(),3),
    solid_color = :blue,
    figure=(size=(1,0.63).*600,),
    linewidth=1.5,
    axis=(;xticks=xt,yticks=range(0,10,6),ygridvisible=false)
)

## --- spectrum near Γ point ---
kl2 = BzLine([Γ, 0.013.*lat.b[:,1]],20)
@time ben2 = eig_BdG(lat,ϕG,u0,kl2.k,2); ## 55.078
fig=series(kl2.r, ben2;
    marker=:circle,
    color=Makie.wong_colors(),
    figure=(;size=(1,0.63).*400),
    linewidth=1.5,
)


# --------------------------------------------------- 
##              Spin Hall
# --------------------------------------------------- 
Mk0,tz = cal_BdG(lat,ϕG,u0,Γ)
Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=-1)
Jy,Dhy = cal_Ju(ϕG,Γ,lat.Kvec; u=2,sp=1)

w = [range(0,1.5,100); range(1.6,4.4,18); range(4.45,6.0,120)]
Xw1 = Green1(Mk0,w,Jx,Jy,η=0.0)./lat.Sunit
fig = series(w,hcat(reim(Xw1)...)',marker=:circle,axis=(;limits=(nothing,(-0.1,0.1))))


## --- 谱分解计算 Spin Hall ---
ben,bev=eigBdG(Mk0)
Xw2 = Xspec1(w,Dhx,Dhy,ben,bev,ϕG)./lat.Sunit
series!(w,hcat(reim(Xw2)...)',solid_color=:red,linestyle=:dash)
fig



## --- symmetry of BdG state ---
myint(ϕG,ϕG,lat.Kvec,"T")|>expshow   # T symmetry

Nm = round(Int,length(ben)/2)
phs = 1/myint(ϕG,ϕG,lat.Kvec,"S1")   # \tilde D1 symmetry
for ii in 0:11
    v1 = SpinHall.normalize(bev[1:Nm,ii+1])
    a = myint(v1,v1,lat.Kvec,"S1")*phs
    if abs(a-1)<1e-5
        println(ii,", B1, ", expshow(a))
    else
        println(ii,", B2, ", expshow(a))
    end
end

phs = 1/myint(ϕG,ϕG,lat.Kvec,"TxC2")   # Λₓr₂ symmetry
for ii in 0:11
    v1 = SpinHall.normalize(bev[1:Nm,ii+1])
    a = myint(v1,v1,lat.Kvec,"TxC2")*phs
    if abs(a-1)<1e-5
        println(ii,", ", real(a))
    else
        println(ii,", ", real(a))
    end
end


## --- 任意角度 spin hall -----
Mk0,tz = cal_BdG(lat,ϕG,u0,Γ)
Jx = cal_Jθ(ϕG,Γ,lat.Kvec,pi/2; sp=-1)
Jy = cal_Jθ(ϕG,Γ,lat.Kvec,0.0; sp=1)

w = [range(0,1.5,100); range(1.6,4.4,18); range(4.45,6.0,120)]
Xw1 = Green1(Mk0,w,Jx,Jy)./lat.Sunit
fig,_,_ = series(w,hcat(reim(Xw1)...)',marker=:circle,axis=(limits=(nothing,(-0.1,0.1)),title=L"\sigma_{j_\theta,j_{\theta+\pi/2}^s}(\omega),\quad \theta=0.0"),labels=["Re","Im"])
axislegend()
fig



## ---- 含时 Hall ----
Er = 2.3272e4
Mk0,tz = cal_BdG(lat,ϕG,u0,Γ)
ben,bev=eigBdG(Mk0)
J1s= cal_Dθ(Γ,lat.Kvec,pi/2; sp=-1)
J2s= cal_Dθ(Γ,lat.Kvec,0.0; sp=-1)
J2 = cal_Dθ(Γ,lat.Kvec,0.0; sp=1)

t = range(0,1.5,950)
sxy = [Hall_time(ϕG,ben,bev,J1s,J2,i,η=0.0) for i in t]./lat.Sunit
sxx = [Hall_time(ϕG,ben,bev,J2s,J2,i,η=0.01) for i in t]./lat.Sunit

fig,_,_=lines(t./Er,real.(sxy), axis=(xlabel=L"t",))
# lines!([0,0.1],real.([Xw1[1],Xw1[1]]))
fig
##

fig,_,_=series(t./Er,real.([sxy sxx]'),marker=:circle,
    labels=[L"σ^s_{yx}(t)",L"σ^s_{xx}(t)"],axis=(xlabel=L"t",))
axislegend(position=:lb)
fig




## --------- 2阶 SpinHall -----------
sg2(x::Real)=sign(x)*x^2
ktmp = sg2.(range(-1,1,64)).*0.5.+0.5
bz = mymesh([-0.5.*(lat.b[:,1].+lat.b[:,2]),lat.b[:,1],lat.b[:,2]],ktmp,ktmp)
scatter(reshape(bz,2,:),axis=(aspect=1,),markersize=4,figure=(size=(600,500),))
##

@time ben,bev = eigen_BdG(lat,ϕG,u0,bz,2*lat.NK);
# using JLD2,FileIO
# save("data/BdGsol.jld2","en",ben,"ev",bev)

sxy = Hall2ed(ben,bev,bz,lat.Kvec;θ=(pi/2,0.0),sp=(1,1))
SpinHall.trapz((ktmp,ktmp),sxy)./lat.Sunit

tmp = real.(sxy)
tmp[abs.(tmp).>25].=NaN64
fig,_,hm=heatmap(ktmp,ktmp,tmp,axis=(aspect=1,))
Colorbar(fig[1,2],hm)
fig

##
ndep = SpinHall.Qudep(bev)
tmp = real.(ndep)
tmp[abs.(tmp).>55].=NaN64
fig,_,hm=heatmap(ktmp,ktmp,tmp,axis=(aspect=1,))
Colorbar(fig[1,2],hm)
fig

SpinHall.trapz((ktmp,ktmp),ndep)

##
ħ=6.62607015e-34/2pi
ω=30*2pi
m=87*1.66e-27
F= 2.2732e-26
kL = 2pi/(787e-9)
Er = (ħ*kL)^2/2m
F*kL/Er

m*ω^2*1e-6