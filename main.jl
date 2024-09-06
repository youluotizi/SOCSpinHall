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
    g = [0.35,0.3].*gg
    Kmax = 6
    b = [[1.0,1.0] [-1.0,1.0]]

    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end


## ---------------------------------------------------


lat = set_lattice(8.0,1.5,1.0)
Γ = [0.0,0.0]
Nopt = 10
E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
gs=[1,cispi(0.25), zeros(Nopt-2)...]

ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=gs)
mat = calmat(lat, Γ)
ϕG,u0=imag_time_evl(mat, ϕG, lat)
SpinHall.gaugephi0!(ϕG, ϕ0)
ϕ0'*ϕG|>expshow


##
set_theme!(;
    Figure=(;size=(500,400)),
    Axis=(;limits=(nothing,(-0.1,0.1)))
)

PTϕG = PTtransform(ϕG)
Vg=[ϕG PTϕG]
SpinHall.zeeman_split!(Vg)
ϕG1=Vg[:,1]./√2
ϕG2=Vg[:,2].*(cispi(0.25)/√2)
abs.(ϕG1.+ϕG2.-ϕG)|>findmax|>println

Mk0,tz = cal_BdG(lat,ϕG,u0,Γ)
Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=-1)
Jy,Dhy = cal_Ju(ϕG,Γ,lat.Kvec; u=2)

w = range(0,6,128)
Xw1 = Green1(Mk0,w,Jx,Jy)./lat.Sunit
fig = series(w,hcat(reim(Xw1)...)',marker=:circle,color=Makie.wong_colors())

ben,bev=eigBdG(Mk0)
Xw2 = Xspec2(w,Dhx,Dhy,ben,bev)./lat.Sunit
series!(w,hcat(reim(Xw2)...)',solid_color=:blue,linestyle=:dash)
fig

##


Jx1,Dhx1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=1,sp=-1)
Jy1,Dhy1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=2)
Jx2,Dhx2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=1,sp=-1)
Jy2,Dhy2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=2)

w = range(0,6,128)
Xw11 = Green1(Mk0,w,Jx1,Jy1)./lat.Sunit
Xw22 = Green1(Mk0,w,Jx2,Jy2)./lat.Sunit
Xw12 = Green1(Mk0,w,Jx1,Jy2)./lat.Sunit
Xw21 = Green1(Mk0,w,Jx2,Jy1)./lat.Sunit
##

fig=series(w,hcat(reim(Xw11.+Xw22)...)',marker=:circle,solid_color=:black)
series!(w,hcat(reim(Xw12.+Xw21)...)',linestyle=:dash,solid_color=:red)
series!(w,hcat(reim(Xw12.+Xw21.+Xw11.+Xw22)...)',linestyle=:dash,solid_color=:blue)
fig

fig = series(w,[imag.(Xw2) real.(Xw2)]',marker=:circle,solid_color=:blue)
series!(w,hcat(reim(Xw12.+Xw21.+Xw11.+Xw22)...)',linestyle=:dash,solid_color=:red)
fig






## Hall conductivity

function spinhall_M0(g0::Float64)
    t=time()
    Nopt = 8
    m0list=range(0.1,2.6,25) 
    Nm0=length(m0list)
    Xw=Array{ComplexF64}(undef,9,Nm0)
    Γ=[0.0,0.0]
 
    for ii in eachindex(m0list)
        lat = set_lattice(8.0,m0list[ii],g0)
        E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
        gs=[1.0,0.0,reim(cispi(0.25))..., zeros(2*Nopt-4)...]

        ϕG,u0,xopt=main_opt(gs, E0, ϕ0, lat; Nstep=120000)
        mat= calmat(lat,Γ)
        ϕG,u0=imag_time_evl(mat, ϕG, lat)
        SpinHall.gaugephi0!(ϕG, ϕ0)
        
        PTϕG = PTtransform(ϕG)
        Vg=[ϕG PTϕG]
        SpinHall.zeeman_split!(Vg)
        ci = Vg'*ϕG
        ci|>expshow|>println
        
        ϕG1=Vg[:,1].*ci[1]
        ϕG2=Vg[:,2].*ci[2]
        
        Mk0,tz=cal_BdG(lat,ϕG,u0,Γ)

        Jx1,Dhx1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=1,sp=-1)
        Jy1,Dhy1 = cal_Ju(ϕG1,Γ,lat.Kvec; u=2)
        Jx2,Dhx2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=1,sp=-1)
        Jy2,Dhy2 = cal_Ju(ϕG2,Γ,lat.Kvec; u=2)

        Xw[1,ii] = Green1(Mk0,Jx1,Jy1)/lat.Sunit
        Xw[2,ii] = Green1(Mk0,Jx2,Jy2)/lat.Sunit
        Xw[3,ii] = Green1(Mk0,Jx1,Jy2)/lat.Sunit
        Xw[4,ii] = Green1(Mk0,Jx2,Jy1)/lat.Sunit

        Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=-1)
        Jy,Dhy = cal_Ju(ϕG,Γ,lat.Kvec; u=2)
        Xw[5,ii] = Green1(Mk0,Jx,Jy)/lat.Sunit

        tmp = cal_Bcav(lat,Γ,1:2)./lat.Sunit
        
        Xw[6,ii] = dot_sz(ϕG1)*tmp[1]
        Xw[7,ii] = dot_sz(ϕG2)*tmp[2]
        Xw[8,ii] = -dot_sz(ϕG1)*tmp[2]
        Xw[9,ii] = -dot_sz(ϕG2)*tmp[1]
    end

    println("time_used: ",time()-t)
    return (;m0list,Xw)
end

begin
    m1,Xw1 = spinhall_M0(1.0)
    m2,Xw2 = spinhall_M0(0.5)
    m3,Xw3 = spinhall_M0(0.2)
end


begin
    fig,ax,plt=series(m1,real.(Xw1[1:5,:]),color=Makie.wong_colors(),
        marker=:circle,
        labels=["⟨ϕ₁...ϕ₁⟩","⟨ϕ₂...ϕ₂⟩","⟨ϕ₁...ϕ₂⟩","⟨ϕ₂...ϕ₁⟩","σˢ",]
    )
    series!(m1,real.(Xw1[6:9,:]),color=Makie.wong_colors()[4:7],
        marker=:rtriangle,linestyle=:dash,
        labels=["⟨ϕ₁σ₃ϕ₁⟩B₁","⟨ϕ₂σ₃ϕ₂⟩B₂","-⟨ϕ₁σ₃ϕ₁⟩B₂","-⟨ϕ₂σ₃ϕ₁⟩B₁"]
    )
    axislegend(ax; position = :lb, labelsize = 15)
    fig|>display

    fig,ax,plt=series(m1,real.([Xw1[1,:].+Xw1[2,:] Xw1[3,:].+Xw1[4,:] Xw1[5,:] (Xw1[6,:].+Xw1[7,:].+Xw1[8,:].+Xw1[9,:])])',
        color=Makie.wong_colors(),
        labels=["⟨ϕ₁...ϕ₁⟩+⟨ϕ₂...ϕ₂⟩","⟨ϕ₁...ϕ₂⟩+⟨ϕ₂...ϕ₁⟩","σˢ","⟨σ₃⟩₁B₁+⟨σ₃⟩₂B₂-⟨σ₃⟩₁B₂-⟨σ₃⟩₂B₁"],
        marker=:circle
    )
    axislegend(ax; position = :lb, labelsize = 15)
    fig
end
