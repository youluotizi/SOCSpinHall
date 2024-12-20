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

# ---------------------------------------------------
##          任意角度 spin hall 
# ---------------------------------------------------
function hall_theta(lat,ϕG,u0,Γ,N)
    Mk0,_ = cal_BdG(lat,ϕG,u0,Γ)
    Xw = Array{ComplexF64}(undef,2,N)
    θ = range(0,2pi,N)
    w = [range(0,1.5,20); range(1.6,3,10)]#; range(4.45,6.0,50)]
    for i in 1:N
        J1s = cal_Jθ(ϕG,Γ,lat.Kvec,θ[i]; sp=-1)
        J2 = cal_Jθ(ϕG,Γ,lat.Kvec,θ[i]+pi/2; sp=1)
        J2s = cal_Jθ(ϕG,Γ,lat.Kvec,θ[i]+pi/2; sp=-1)
    
        Xw1 = Green1(Mk0,w,J1s,J2)./lat.Sunit
        fig = Figure(size=(400,700))

        str = myfilter(θ[i]/pi)
        series(fig[1,1],w,hcat(reim(Xw1)...)',marker=:circle,axis=(limits=(0,3,-0.1,0.1),
                title=L"\sigma_{xy}^s,\quad \theta=%$(str)\pi")
        )

        Xw2 = Green1(Mk0,w,J2s,J2)./lat.Sunit
        series(fig[2,1],w,hcat(reim(Xw2)...)',marker=:circle,axis=(limits=(0,3,-0.1,0.1),
                title=L"\sigma_{yy}^s,\quad \theta=%$(str)\pi")
        )
        display(fig)
        Xw[1,i]=Xw1[1]
        Xw[2,i]=Xw2[1]
    end
    return (;θ, Xw)
end
Xθ = hall_theta(lat,ϕG,u0,Γ,21)
xt = ([range(0,2pi,9);],[L"%$(i)\pi/4" for i in 0:8])
fig2,_,_=series(Xθ.θ,real.(Xθ.Xw),marker=:circle, axis=(xticks=xt,),labels=[L"\sigma_{\theta,\theta+\pi/2}^s",L"\sigma_{\theta+\pi/2,\theta+\pi/2}^s"])
axislegend()
fig2

save("data/anisotropy.h5",OrderedDict("Xtheta"=>real.(Xθ.Xw),"theta"=>collect(Xθ.θ)))


## 2ed Spin hall
ben = load("data/BdGsol.jld2","en")
bev = load("data/BdGsol.jld2","ev")

sg2(x::Real)=sign(x)*x^2
ktmp = sg2.(range(-1,1,size(ben,3))).*0.5.+0.5
bz = mymesh([-0.5.*(lat.b[:,1].+lat.b[:,2]),lat.b[:,1],lat.b[:,2]],ktmp,ktmp)
function hall_theta(lat,ben,bev,bz,ktmp,N)
    Xw = Array{ComplexF64}(undef,N)
    θ = range(0,1pi,N)
    t = time()
    for i in 1:N
        sxy = Hall2ed(ben,bev,bz,lat.Kvec;θ=(θ[i],θ[i]),sp=(-1,1));
        Xw[i] = SpinHall.trapz((ktmp,ktmp),sxy)./lat.Sunit

        tmp = real.(sxy)
        tmp[abs.(tmp).>30].=NaN64
        str = myfilter(θ[i]/pi)
        fig,_,hm=heatmap(ktmp,ktmp,tmp,axis=(aspect=1,title=L"\theta=%$(str)\pi"))
        Colorbar(fig[1,2],hm)
        display(fig)
        println("time,used: ",time()-t)
        sleep(3)
    end
    return (;θ, Xw)
end
X2 = hall_theta(lat,ben,bev,bz,ktmp,21)

xt = ([range(0,pi,5);],[L"%$(i)\pi/4" for i in 0:4])
fig2,_,_=scatterlines(X2.θ,real.(X2.Xw),marker=:circle, axis=(xticks=xt,))

##
# x = load("data/anisotropy.h5")
xt = ([range(0,2pi,9);],[L"%$(i)\pi/4" for i in 0:8])
fig,_,_=scatterlines(x["theta"],x["Xtheta"][1,:],label=L"\sigma_{yx}",axis=(xticks=xt,))
scatterlines!(x["theta"],x["Xtheta"][2,:],label=L"\sigma_{xx}")
scatterlines!(x["theta"],x["Xtheta"][3,:],linestyle=:dash,label=L"$\sigma_{yx}$-qdep",marker=:xcross)
scatterlines!(x["theta"],x["Xtheta"][4,:],linestyle=:dash,label=L"$\sigma_{xx}$-qdep",marker=:xcross)
axislegend()
fig
# ---------------------------------------------------
##  Hall conductivity as function of M_0
# ---------------------------------------------------
function spinhall_M0(g0::Float64)
    t=time()
    m0 = range(0.03,1.5,8)#[range(0.03,1.73,18); range(1.76,1.99,10); range(2.02,3.2,13)]
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