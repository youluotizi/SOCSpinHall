# 首次运行时须按[option]+[enter]执行以下3行，清除包的具体版本依赖，以后就不用了，在其他文件也不需要
rm("Manifest.toml")
using Pkg
Pkg.instantiate()
##
using OrderedCollections,FileIO
using Revise
using SpinHall

using CairoMakie
set_theme!(;size=(500,400))

const cm = 72/2.54

function set_lattice(v0, m0, gg)
    mz = 0.0
    g = [0.35,0.3].*gg
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]

    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end

##
set_theme!(;
    Figure=(;size=(500,400)),
    Axis=(;limits=(nothing,(-0.1,0.1)))
)
## ---------------------------------------------------


lat = set_lattice(8.0,1.5,1.0)
Γ = [0.0,0.0]
Nopt = 12
E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
gs=[1,cispi(0.25), zeros(Nopt-2)...]

ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=gs, Nstep=10^5)
mat = calmat(lat, Γ)
ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=10^4)
SpinHall.gaugephi0!(ϕG, ϕ0)
ϕ0'*ϕG|>expshow

# ϕG = (ϕ0[:,1].+cispi(0.25).*ϕ0[:,2])./√2
##

PTϕG = PTtransform(ϕG)
Vg=[ϕG PTϕG]
SpinHall.zeeman_split!(Vg)
(ci = Vg'*ϕG)|>expshow|>println
abs.(ci[1].*Vg[:,1].+ci[2].*Vg[:,2].-ϕG)|>findmax|>println
SpinHall.dot(ci[1].*Vg[:,1].-ci[2].*Vg[:,2],ϕG)


Mk0,tz = cal_BdG(lat,ϕG,u0,Γ)
Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=-1)
Jy,Dhy = cal_Ju(ϕG,Γ,lat.Kvec; u=2)

w = [range(0,1.5,100);
    range(1.6,4.4,18)#;
    range(4.45,6.0,120)
] #range(0,6,128)
Xw1 = Green1(Mk0,w,Jx,Jy)./lat.Sunit
fig = series(w,hcat(reim(Xw1)...)',marker=:circle,solid_color=:blue)

# ben,bev=eigBdG(Mk0)
# Xw2 = Xspec1(w,Dhx,Dhy,ben,bev,ϕG)./lat.Sunit
# series!(w,hcat(reim(Xw2)...)',solid_color=:blue,linestyle=:dash)

# Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1)
# Xw3 = Xspec2(w,tz*Dhx,tz*Dhy,ben,bev,ϕG)./lat.Sunit
# series!(w,hcat(reim(Xw3)...)',solid_color=:gray,linestyle=:dot)
# fig

##
ϕ2=ci[1].*Vg[:,1].-ci[2].*Vg[:,2]
SpinHall.dot(bev[1:226,2]./sqrt(5.240386180751463),PTϕG)
SpinHall.dot(conj.(bev[227:end,2])./sqrt(4.240386180751464),PTϕG)
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
# spectrum decomposition

function spinhall_M0(g0::Float64)
    t=time()
    m0 = [range(0.03,1.73,18); range(1.76,1.99,10); range(2.02,3.2,13)]
    Nm0 = length(m0)
    Xw = Array{ComplexF64}(undef,7,Nm0)
    Mspin = Array{Float64}(undef,3,Nm0)
    Γ = [0.0,0.0]
 
    for ii in eachindex(m0)
        println("--------------------------------------------------")
        println("m0 = ",m0[ii])

        lat = set_lattice(8.0,m0[ii],g0)
        E0,ϕ0=eigenband(lat, Γ, 1:20)
        
        if abs(g0)>1e-5
            if 1.679<m0[ii]<1.93
                Nopt=16; Nstep1=3*10^5; Nstep2=10^4
            else
                Nopt=8; Nstep1=10^5; Nstep2=5000
            end
            gs=[1.0,cispi(0.25), zeros(Nopt-2)...]
            ϕG,u0,xopt=main_opt(E0, ϕ0, lat;gs=gs, Nstep=Nstep1)
            mat= calmat(lat,Γ)

            ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=Nstep2)
            SpinHall.gaugephi0!(ϕG, ϕ0)
        else
            if m0[ii]<1.87
                ϕG = (ϕ0[:,1].+cispi(0.25).*ϕ0[:,2])./√2
                u0 = E0[1]
            else
                ϕG = SpinHall.normalize(ϕ0[:,1].+0.5.*ϕ0[:,1])
                u0 = E0[1]
            end
        end
        
        Mspin[:,ii].= real.(dot_spin(ϕG))
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

        tmp = -1.0.*cal_Bcav(lat,Γ,1:2)./lat.Sunit
        Xw[6,ii] = dot_sz(ϕG1)*tmp[1]+dot_sz(ϕG2)*tmp[2]

        ben,bev=eigBdG(Mk0)
        Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=1)
        Xw[7,ii] = Xspec2(Dhx,Dhy,ben,bev,ϕG,PTϕG)/lat.Sunit
    end

    println("time_used: ",time()-t,"\n\n")
    return (;m0,Xw,Mspin)
end

g0=spinhall_M0(0.0)
save("data/SpinHall_int.h5", OrderedDict(
    "g0/m0"=>g0.m0, "g0/Xw"=>real.(g0.Xw), "g0/Mspin"=>g0.Mspin,
    "g1/m0"=>g2.m0, "g1/Xw"=>real.(g2.Xw), "g1/Mspin"=>g2.Mspin,
    "sw/ben"=>real.(ben[1:20]),"sw/w"=>w,"sw/Re"=>real.(Xw1),"sw/Im"=>imag.(Xw1)
))

begin
    fig,_,_ = scatterlines(m,abs.(Mspin[3,:]))
    scatterlines!(m,sqrt.(Mspin[1,:].^2 .+Mspin[2,:].^2))
    fig
end

scatterlines(m,real.(Xw[3,:].+Xw[4,:]))

begin
    g=g2
    fig = Figure(size=(800,600))
    ax = Axis(fig[1,1],limits=(nothing,(-0.01,0.105)),title=L"V_0=8",xlabel=L"M_0")
    scatterlines!(fig[1,1],g.m0,real.(g.Xw[5,:]),color=:black,label=L"\sigma_{xy}^s",linewidth=3)
    scatterlines!(g.m0,-1.0.*real.(g.Xw[6,:]),marker=:circle,color=Makie.wong_colors()[1],
        label=L"\langle\sigma_z\rangle_1B_1+\langle\sigma_z\rangle_2B_2")
    series!(g.m0,real.([g.Xw[7,:] g.Xw[1,:].+g.Xw[2,:] g.Xw[3,:].+g.Xw[4,:]])',
        color=Makie.wong_colors()[2:end],
        labels=[L"\langle g|\sigma_z|g'\rangle\langle g'|j_x,j_y]|g\rangle",
            L"\langle\phi_1\cdots\phi_1\rangle+\langle\phi_2\cdots\phi_2\rangle",
            L"\langle\phi_1\cdots\phi_2\rangle+\langle\phi_2\cdots\phi_1\rangle"
        ],
        linestyle=:dash, marker=:ltriangle
    )
    axislegend(ax; position = :lt, labelsize = 15)
    fig
end


function spinhall_V0(g0::Float64)
    t=time()
    Nopt = 8
    V0list=range(0.6,8.0,38)
    Nv0=length(V0list)
    Xw=Array{ComplexF64}(undef,9,Nv0)
    Γ=[0.0,0.0]
 
    for ii in eachindex(V0list)
        lat = set_lattice(V0list[ii],1.5,g0)
        E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
        #=
        gs=[1.0,cispi(0.25), zeros(Nopt-2)...]
        ϕG,u0,xopt=main_opt(E0, ϕ0, lat;gs=gs, Nstep=120000)
        mat= calmat(lat,Γ)
        ϕG,u0=imag_time_evl(mat, ϕG, lat)
        SpinHall.gaugephi0!(ϕG, ϕ0)
        =#
        ϕG = (ϕ0[:,1].+cispi(0.25).*ϕ0[:,2])./√2
        u0 = E0[1]
        
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

        tmp = -1.0.*cal_Bcav(lat,Γ,1:2)./lat.Sunit
        Xw[6,ii] = dot_sz(ϕG1)*tmp[1]+dot_sz(ϕG2)*tmp[2]

        ben,bev=eigBdG(Mk0)
        Jx,Dhx = cal_Ju(ϕG,Γ,lat.Kvec; u=1,sp=1)
        Xw[7,ii] = Xspec2(Dhx,Dhy,ben,bev,ϕG,PTϕG)/lat.Sunit
    end

    println("time_used: ",time()-t)
    return (;V0list,Xw)
end

V0,Xw2 = spinhall_V0(0.0)

begin
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1],limits=(nothing,(-0.01,0.12)),title=L"M_0=1.5",xlabel=L"V_0")
    scatterlines!(V0,real.(Xw2[5,:]),color=:black,label=L"\sigma_{xy}^s",linewidth=3)
    scatterlines!(V0,-1.0.*real.(Xw2[6,:]),marker=:circle,color=Makie.wong_colors()[1],
        label=L"\langle\sigma_z\rangle_1B_1+\langle\sigma_z\rangle_2B_2"
    )
    series!(V0,real.([Xw2[7,:] Xw2[1,:].+Xw2[2,:] Xw2[3,:].+Xw2[4,:]])',
        color=Makie.wong_colors()[2:end],linestyle=:dash, marker=:ltriangle,
        labels=[L"\langle g|\sigma_z|g'\rangle\langle g'|j_x,j_y]|g\rangle",
            L"\langle\phi_1\cdots\phi_1\rangle+\langle\phi_2\cdots\phi_2\rangle",
            L"\langle\phi_1\cdots\phi_2\rangle+\langle\phi_2\cdots\phi_1\rangle"
        ],
    )
    axislegend(ax; position = :lt, labelsize = 15, nbanks=1)
    fig
end