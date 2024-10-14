using Revise
using SpinHall
using OrderedCollections,FileIO

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

# -------------------------------------------------------------
##                   single particle states
# -------------------------------------------------------------
lat = set_lattice(8.0,1.5,1.0)
Γ = [0.0,0.0]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],256)
en = eigband(lat,kl.k, 1:20)
xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
fig= series(kl.r,en.-en[1]; axis=(;xticks=xt,yticks=0:2:12),color=repeat(Makie.wong_colors(),4))


## --- some 2D plot ---
M = -0.5.*(lat.b[:,1].+lat.b[:,2])
bz = mymesh([M, lat.b[:,1], lat.b[:,2]].*0.5, [24,24])
x = eigen2D(lat,bz,1:12)

fig,ax,hm = heatmap(x.bcav[1,:,:])
Colorbar(fig[1,2],hm)
fig|>display
x.bcav[:,12,12].-cal_Bcav(lat, Γ, 1:12)


# ---------------------------------------------------
##          Ground state
# ---------------------------------------------------
lat = set_lattice(8.0,1.5,1.0)
Γ = [0.0,0.0]
Nopt = 10
E0,ϕ0=eigenband(lat, Γ, 1:Nopt)
gs=[1,cispi(-0.25), zeros(Nopt-2)...]

ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=gs, Nstep=10^5)
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


# ---------------------------------------------------
##               BdG spectrum
# ---------------------------------------------------
@time ben = main_BdG(lat,ϕG,u0,kl.k,20); ## 55.078
fig=series(kl.r, ben[1:12,:];
    color=repeat(Makie.wong_colors(),3),
    figure=(;size=(400,400*0.63)),
    linewidth=1.5,
    axis=(;xticks=xt,yticks=range(0,10,6),ygridvisible=false)
)

## --- spectrum near Γ point ---
kl2 = BzLine([Γ, 0.013.*lat.b[:,1]],50)
@time ben2 = main_BdG(lat,ϕG,u0,kl2.k,2); ## 55.078
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
Xw1 = Green1(Mk0,w,Jx,Jy)./lat.Sunit
fig = series(w,hcat(reim(Xw1)...)',marker=:circle,axis=(;limits=(nothing,(-0.1,0.1))))


## --- 谱分解计算 Spin Hall ---
ben,bev=eigBdG(Mk0)
Xw2 = Xspec1(w,Dhx,Dhy,ben,bev,ϕG)./lat.Sunit
series!(w,hcat(reim(Xw2)...)',solid_color=:red,linestyle=:dash)
fig


## --- symmetry of BdG state ---
myint(ϕG,ϕG,lat.Kvec,"T")|>expshow  # T symmetry

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



# ---------------------------------------------------
##  Hall conductivity as function of M_0
# ---------------------------------------------------
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
            ϕG,u0,xopt=main_opt(E0[1:Nopt], ϕ0[:,1:Nopt], lat;gs=gs, Nstep=Nstep1)
            mat= calmat(lat,Γ)

            ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=Nstep2)
            SpinHall.gaugephi0!(ϕG, ϕ0)
        else
            if m0[ii]<1.87
                ϕG = (ϕ0[:,1].+cispi(0.25).*ϕ0[:,2])./√2
                u0 = E0[1]
            else
                ϕG = ϕ0[:,1] # SpinHall.normalize(ϕ0[:,1].+0.5.*ϕ0[:,2])
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

@time g0=spinhall_M0(0.0); # g=(0.35,0.3)*0, nointeracting limits
@time g1=spinhall_M0(1.0); # g=(0.35,0.3)*1, cost several minites

##
#=
save("data/SpinHall_int.h5", OrderedDict(
    "g0/m0"=>g0.m0, "g0/Xw"=>real.(g0.Xw), "g0/Mspin"=>g0.Mspin,
    "g1/m0"=>g1.m0, "g1/Xw"=>real.(g1.Xw), "g1/Mspin"=>g1.Mspin,
    "sw/ben"=>real.(ben),"sw/w"=>w,"sw/Re"=>real.(Xw1),"sw/Im"=>imag.(Xw1),
    "k/pt"=>kl.pt, "k/r"=>kl.r,
))
# d = load("data/SpinHall_int.h5",dict=OrderedDict())
# save("data/SpinHall_int.h5", d)
=#

begin
    g=g1
    fig,_,_ = scatterlines(g.m0,abs.(g.Mspin[3,:]))
    scatterlines!(g.m0,sqrt.(g.Mspin[1,:].^2 .+g.Mspin[2,:].^2))
    fig
end

begin
    g=g0
    fig = Figure(size=(800,600))
    ax = Axis(fig[1,1],limits=(nothing,(-0.01,maximum(real.(g.Xw))+0.005)),title=L"V_0=8",xlabel=L"M_0")
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


# ---------------------------------------------------
##     Hall conductivity as function of V_0
# ---------------------------------------------------
function spinhall_V0(m0::Float64,g0::Float64)
    t=time()
    V0=range(0.6,8.0,38)
    Nv0 = length(V0)
    Xw = Array{ComplexF64}(undef,7,Nv0)
    Mspin = Array{Float64}(undef,3,Nv0)
    Γ = [0.0,0.0]
 
    for ii in eachindex(V0)
        println("--------------------------------------------------")
        println("V0 = ",V0[ii])

        lat = set_lattice(V0[ii],m0,g0)
        E0,ϕ0=eigenband(lat, Γ, 1:20)
        
        if abs(g0)>1e-5
            if 1.679<m0<1.93
                Nopt=16; Nstep1=3*10^5; Nstep2=10^4
            else
                Nopt=8; Nstep1=10^5; Nstep2=5000
            end
            gs=[1.0,cispi(0.25), zeros(Nopt-2)...]
            ϕG,u0,xopt=main_opt(E0[1:Nopt], ϕ0[:,1:Nopt], lat;gs=gs, Nstep=Nstep1)
            mat= calmat(lat,Γ)

            ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=Nstep2)
            SpinHall.gaugephi0!(ϕG, ϕ0)
        else
            if m0[ii]<1.87
                ϕG = (ϕ0[:,1].+cispi(0.25).*ϕ0[:,2])./√2
                u0 = E0[1]
            else
                ϕG = ϕ0[:,1] # SpinHall.normalize(ϕ0[:,1].+0.5.*ϕ0[:,2])
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
    return (;V0,Xw,Mspin)
end

gv = spinhall_V0(1.5,1.0)

begin
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1],limits=(nothing,extrema(real.(gv.Xw)).+(0.005,0.005)),title=L"M_0=1.5",xlabel=L"V_0")
    scatterlines!(gv.V0,real.(gv.Xw[5,:]),color=:black,label=L"\sigma_{xy}^s",linewidth=3)
    scatterlines!(gv.V0,-1.0.*real.(gv.Xw[6,:]),marker=:circle,color=Makie.wong_colors()[1],
        label=L"\langle\sigma_z\rangle_1B_1+\langle\sigma_z\rangle_2B_2"
    )
    series!(gv.V0,real.([gv.Xw[7,:] gv.Xw[1,:].+gv.Xw[2,:] gv.Xw[3,:].+gv.Xw[4,:]])',
        color=Makie.wong_colors()[2:end],linestyle=:dash, marker=:ltriangle,
        labels=[L"\langle g|\sigma_z|g'\rangle\langle g'|j_x,j_y]|g\rangle",
            L"\langle\phi_1\cdots\phi_1\rangle+\langle\phi_2\cdots\phi_2\rangle",
            L"\langle\phi_1\cdots\phi_2\rangle+\langle\phi_2\cdots\phi_1\rangle"
        ],
    )
    axislegend(ax; position = :lt, labelsize = 15, nbanks=1)
    fig
end