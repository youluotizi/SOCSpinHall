using Revise
using SpinHall
using OrderedCollections,FileIO
using CairoMakie
set_theme!(;size=(600,400))
const cm = 72/2.54

function set_lattice(v0, m0, gg)
    mz = 0.0
    g = [0.35,0.3].*gg
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end


# ---------------------------------------------------
##          Ground state
# ---------------------------------------------------
lat = set_lattice(8.0,1.5,1.0)
Γ = [0.0,0.0]
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

## ----  another ground state ----
PTϕG = PTtransform(ϕG)
Vg=[ϕG PTϕG]
SpinHall.zeeman_split!(Vg)
(ci = Vg'*ϕG)|>expshow|>println
abs.(ci[1].*Vg[:,1].+ci[2].*Vg[:,2].-ϕG)|>findmax|>println

ϕG1=Vg[:,1].*ci[1]
ϕG2=Vg[:,2].*ci[2]

## --- hall separation ----
Jx1,Dhx1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=1,sp=-1)
Jy1,Dhy1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=2)
Jx2,Dhx2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=1,sp=-1)
Jy2,Dhy2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=2)

Xw11 = Green1(Mk0,w,Jx1,Jy1)./lat.Sunit
Xw22 = Green1(Mk0,w,Jx2,Jy2)./lat.Sunit
Xw12 = Green1(Mk0,w,Jx1,Jy2)./lat.Sunit
Xw21 = Green1(Mk0,w,Jx2,Jy1)./lat.Sunit


##
fig=series(w,hcat(reim(Xw11.+Xw22)...)',marker=:circle,solid_color=:black,axis=(;limits=(nothing,(-0.1,0.1))), labels=["re 11+22","im 11+22"])
series!(w,hcat(reim(Xw12.+Xw21)...)',linestyle=:dash,solid_color=:red,labels=["re 12+21","im 12+21"])
series!(w,hcat(reim(Xw12.+Xw21.+Xw11.+Xw22)...)',linestyle=:dash,solid_color=:blue,labels=["re 11+12+..","im 11+12+.."])
axislegend(position=:ct)
fig|>display

##
fig = series(w,[imag.(Xw2) real.(Xw2)]',marker=:circle,solid_color=:blue,labels=["re σs","im σs"],axis=(;limits=(nothing,(-0.1,0.1))))
series!(w,hcat(reim(Xw12.+Xw21.+Xw11.+Xw22)...)',linestyle=:dash,labels=["re 11+12+..","im 11+12+.."],solid_color=:red)
axislegend(position=:ct)
fig