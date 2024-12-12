# ---------------------------------------------------
##              spin hall of fermi gases
# ---------------------------------------------------
using Revise
using SpinHall
using OrderedCollections,FileIO
using FFTW: fftshift
using Trapz
using CairoMakie
set_theme!(;size=(600,400))
const cm = 72/2.54

function set_lattice(v0, m0, gg, Kmax::Int=7)
    mz = 0.0
    g = [0.35,0.3].*gg
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g[1],g[2],Kmax)
end

## --------- 网格均匀划分计算 -----------
lat = set_lattice(4.0,1.5,1.0);
bz = mymesh([(lat.b[:,1].+lat.b[:,2])./(-2),lat.b[:,1],lat.b[:,2]],[128,128])
ktmp=range(0,1,size(bz,3))
scatter(reshape(bz,2,:),axis=(aspect=1,),markersize=3,figure=(size=(600,600),))


## --------- 网格非均匀划分计算 -----------
lat = set_lattice(4.0,1.5,1.0);
sg2(x::Real)=sign(x)*x^2
ktmp = sg2.(range(-1,1,64)).*0.5.+0.5
kplt = sort([i>0.5 ? i-1 : i for i in ktmp])
bz = mymesh([[0.0,0.0],lat.b[:,1],lat.b[:,2]],ktmp,ktmp)
scatter(reshape(bz,2,:),axis=(aspect=1,),markersize=3,figure=(size=(600,600),))



##  验证各向异性
function fermin_theta(ktmp,bz,lat,θ)
    Nθ = length(θ)
    s = Array{Float64}(undef,4,Nθ)
    for i in 1:Nθ
        stmp = FermiHall(bz,lat,θ[i])

        fig = Figure(size=(450,800))
        ftmp = stmp[1,:,:]./lat.Sunit
        ftmp[abs.(ftmp).>50].=NaN64
        _,hm1 = heatmap(fig[1,1],ktmp,ktmp,ftmp,axis=(aspect=1,))
        Colorbar(fig[1,2],hm1)

        ftmp = stmp[3,:,:]./lat.Sunit
        ftmp[abs.(ftmp).>10].=NaN64
        _,hm2 = heatmap(fig[2,1],ktmp,ktmp,ftmp,axis=(aspect=1,))
        Colorbar(fig[2,2],hm2)
        fig|>display

        for j in 1:4
            s[j,i]=SpinHall.trapz((ktmp,ktmp),stmp[j,:,:])/lat.Sunit
        end
    end
    return s
end

ktmp = sg2.(range(-1,1,32)).*0.5.+0.5
bz = mymesh([[0.0,0.0],lat.b[:,1],lat.b[:,2]],ktmp,ktmp)
scatter(reshape(bz[1,:,:],:),reshape(bz[2,:,:],:),axis=(aspect=1,),markersize=3)|>display

θ = range(0,pi,5)
s = fermin_theta(ktmp,bz,lat,θ)
series(θ,s, marker=:circle)


##
function Fermi_v0(v0,bz)
    s_arr = Array{Float64}(undef,4,length(v0))
    for i in eachindex(v0)
        lat = set_lattice(v0[i],1.5,1.0)
        s0 = FermiHall(bz,lat,0.0)

        ftmp = s0[1,:,:]./lat.Sunit
        ftmp[abs.(ftmp).>20].=NaN64
        fig = Figure(size=(800,780))
        _,hm = heatmap(fig[1,1],ktmp,ktmp,ftmp,axis=(aspect=1,title=string(i)))
        Colorbar(fig[1,2],hm)

        ftmp = s0[3,:,:]./lat.Sunit
        ftmp[abs.(ftmp).>20].=NaN64
        _,hm = heatmap(fig[1,3],ktmp,ktmp,ftmp,axis=(aspect=1,title=string(i)))
        Colorbar(fig[1,4],hm)

        ftmp = s0[5,:,:]./lat.Sunit
        ftmp[abs.(ftmp).>20].=NaN64
        _,hm = heatmap(fig[2,1],ktmp,ktmp,ftmp,axis=(aspect=1,))
        Colorbar(fig[2,2],hm)

        ftmp = s0[7,:,:]./lat.Sunit
        ftmp[abs.(ftmp).>20].=NaN64
        _,hm = heatmap(fig[2,3],ktmp,ktmp,ftmp,axis=(aspect=1,))
        Colorbar(fig[2,4],hm)
        display(fig)

        for j in 1:4
            s_arr[j,i] = SpinHall.trapz((ktmp,ktmp),s0[j,:,:])/lat.Sunit
        end
    end
    return s_arr
end

v0 = range(4.0,8.0,11)
s_v0 = Fermi_v0(v0,bz)
# save("data/fermigase.jld2","v0",v1,"s_v0",s)
s_v0 = load("data/fermigase.jld2","s_v0")
save("data/fermigase.h5",OrderedDict("v0"=>[x["v0"];],"s_v0"=>x["s_v0"]))
##
fig,_,_ = scatterlines(v1,s[3,:],label=L"\sigma_1^s",marker=:utriangle)
scatterlines!(v1,s[4,:],label=L"\sigma_2^s",marker=:utriangle,linestyle=:dash)
# scatterlines!(v1,s[3,:].*(-1),label=L"\langle \sigma_z\rangle_1 B_1")
# scatterlines!(v1,s[4,:].*(-1),label=L"\langle \sigma_z\rangle_2 B_2",linestyle=:dash)
axislegend(position=:lb)
fig



##
Nb = 4
lat = set_lattice(4.0,2.0,0.0)
Γ = [0.0,0.0]
M = (lat.b[:,1].+lat.b[:,2])./2
en = eigband(lat,[Γ M],1:4)
mu = [range(en[1,1],en[1,2],5)[2:end];range(en[1,2],en[4,1]+1e-3,5)[2:end]]
@time x = SpinHall.FermiHall_mu(bz,lat,Nb,mu);

##
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],128)
en2 = eigband(lat,kl.k, 1:4)
μ = myfilter(mu.-en2[1],digit=2)
xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
yt = [L"\mu_{%$(i)}:%$(μ[i])" for i in eachindex(mu)]
fig= series(kl.r,en2.-en2[1], color=Makie.wong_colors(),
    axis=(xticks=xt,yticks=(μ,yt),title=L"V_0=%$(lat.v0),\,M_0=%$(lat.m0)",limits=(nothing,(-3e-3,en2[4,end]-en2[1]+0.01))),
    figure=(size=(1,0.83).*500,)
)

##
# ftmp = zeros(size(x.sz,2),size(x.sz,3))
# for i in 3:3
#     ftmp.+=x.s1[i,:,:,6]
# end
iu=1
ib=3
# ftmp=dropdims(sum(x.s1[:,:,:,iu],dims=1),dims=1)
ftmp=fftshift(x.s3[ib,:,:,iu])
ftmp[abs.(ftmp).>20].=NaN
fig,_,hm = heatmap(kplt,kplt,ftmp,axis=(aspect=1,title=L"\mu_{%$(iu)},\,n=%$ib"),figure=(size=(500,430),))
Colorbar(fig[1,2],hm)
display(fig)

##
stmp = dropdims(sum(x.s1,dims=1),dims=1)
[trapz((ktmp,ktmp),stmp[:,:,i]) for i in eachindex(mu)]./lat.Sunit

s1=[sum([trapz((ktmp,ktmp),x.s1[i,:,:,iu]) for i in 1:Nb]) for iu in eachindex(mu)]./lat.Sunit
s2=[sum([trapz((ktmp,ktmp),x.s2[i,:,:,iu].*x.sz[i,:,:]) for i in 1:Nb]) for iu in eachindex(mu)]./lat.Sunit
s3=[sum([trapz((ktmp,ktmp),x.s3[i,:,:,iu].*x.sz[i,:,:]) for i in 1:Nb]) for iu in eachindex(mu)]./lat.Sunit

begin
    fig = Figure(size=(1,0.83).*500)
    xt=(μ,[L"\mu_{%$(i)}" for i in eachindex(mu)])
    ax = Axis(fig[1,1],xlabel=L"\mu/E_r",title=L"V_0=%$(lat.v0),\,M_0=%$(lat.m0)",xticks=xt)
    scatterlines!(μ,s1,label=L"\sigma^s_{xy}",)
    scatterlines!(μ,s2,label=L"\langle \sigma_z\rangle B'",marker=:utriangle,linestyle=:dash)
    scatterlines!(μ,s3,label=L"\langle \sigma_z\rangle B",marker=:star5,linestyle=:dot)
    axislegend(position=:rc)
    fig
end



##
function cal_mu_M0(M0,v0,bz,Nb)
    mu = Array{Float64}(undef,3)
    
    lat = set_lattice(v0,M0,0.0,6)
    en = eigband(lat,bz,1:Nb)
    mu[1] = maximum(en[Nb,:,:])+1e-12

    Dos = sort(reshape(en[Nb,:,:],:))
    Ne = length(Dos)
    mid=div(Ne,2)
    mu[2] = iseven(Ne) ? (Dos[mid]+Dos[mid+1])/2 : Dos[mid+1]+1e-13
    mu[3] = minimum(en[Nb,:,:])-1e-12

    return mu
end

function Fermi_M0(M0,bz,ktmp)
    Nb = 4
    v0=4.0
    s_M0 = Array{Float64}(undef,4,length(M0))
    mu=Array{Float64}(undef,3,length(M0))
    t=time()
    for im in eachindex(M0)
        print("$(im)/$(length(M0)),")
        mu[:,im].= cal_mu_M0(M0[im],v0,bz,Nb)
        lat = set_lattice(v0,M0[im],0.0)
        x = SpinHall.FermiHall_mu(bz,lat,Nb,mu[1:2,im])

        fig = Figure(size=(800,720),title=L"m_0=%$(M0[im])")
        str="=$(myfilter(mu[1,im]))"
        _,hm1 = heatmap(fig[1,1],ktmp,ktmp,x.s1[1,:,:,1],axis=(aspect=1,title=L"n=1,\mu=%$(str)"))
        Colorbar(fig[1,2], hm1)
        _,hm2 = heatmap(fig[1,3],ktmp,ktmp,x.s1[3,:,:,1],axis=(aspect=1,title=L"n=3,\mu%$(str)"))
        Colorbar(fig[1,4], hm2)
        str="=$(myfilter(mu[2,im]))"
        _,hm3 = heatmap(fig[2,1],ktmp,ktmp,x.s1[1,:,:,2],axis=(aspect=1,title=L"n=1,\mu%$(str)"))
        Colorbar(fig[2,2], hm3)
        _,hm4 = heatmap(fig[2,3],ktmp,ktmp,x.s1[3,:,:,2],axis=(aspect=1,title=L"n=3,\mu%$(str)"))
        Colorbar(fig[2,4], hm4)
        display(fig)
       
        stmp = zeros(length(ktmp),length(ktmp))
        for iu in 1:2
            stmp.= 0.0
            for i in 1:Nb
                stmp.+=x.s1[i,:,:,iu]
            end
            s_M0[iu,im] = trapz((ktmp,ktmp),stmp)/lat.Sunit
        end

        for iu in 1:2
            stmp.= 0.0
            for i in 1:Nb
                stmp.+=x.s2[i,:,:,iu].*x.sz[i,:,:]
            end
            s_M0[iu+2,im] = trapz((ktmp,ktmp),stmp)/lat.Sunit
        end
        println("time_used:",time()-t)
    end
    return (;s_M0, mu)
end
M0 = range(0,2,21)
M1 = M0[1:2:end]
M2 = M0[2:2:end]
s_M2,mu2 = Fermi_M0(M2,bz,ktmp);
save("data/Fermi_M2.h5",
    OrderedDict("M2"=>collect(M2),
                "s_M2"=>s_M2,
                "mu2"=>mu2
    )
)
##
x1 = load("data/Fermi_M1.h5")
s_M0 = Array{Float64}(undef,4,21)
mu = Array{Float64}(undef,3,21)
for i in 1:10
    s_M0[:,2*i-1].=x1["s_M1"][:,i]
    s_M0[:,2*i].=s_M2[:,i]
    mu[:,2*i-1].=x1["mu1"][:,i]
    mu[:,2*i].=mu2[:,i]
end
s_M0[:,21].=x1["s_M1"][:,11]
mu[:,21].=x1["mu1"][:,11]

save("data/Fermi_M0.h5",
    OrderedDict("M0"=>collect(M0),
                "s_M0"=>s_M0,
                "mu"=>mu
    )
)

load("data/Fermi_M0.h5")
##
ib = 1
title=["insulator","metal"]
fig=Figure(size=(1,1.6).*400)
ax = Axis(fig[1,1],title=L"%$(title[ib]),$V_0=%$(lat.v0)$",xlabel=L"M_0/E_r")
scatterlines!(M0,s_M0[ib,:],label=L"\sigma_{xy}^s",marker=:circle)
scatterlines!(M0,s_M0[ib+2,:],label=L"\langle \sigma_z\rangle B'",marker=:utriangle,linestyle=:dash)
axislegend(position=:lt)

ib=2
ax = Axis(fig[2,1],title=L"%$(title[ib]),$V_0=%$(lat.v0)$",xlabel=L"M_0/E_r")
scatterlines!(M0,s_M0[ib,:],label=L"\sigma_{xy}^s",marker=:circle)
scatterlines!(M0,s_M0[ib+2,:],label=L"\langle \sigma_z\rangle B'",marker=:utriangle,linestyle=:dash)
axislegend(position=:lt)
fig


##
lat = set_lattice(4.0,0.5,0.0)
@time s0,sz = FermiHall(bz,lat)
ftmp = s0[1,:,:]./lat.Sunit
ftmp[abs.(ftmp).>10].=NaN64
fig = Figure(size=(700,680))
_,hm = heatmap(fig[1,1],ktmp,ktmp,ftmp,axis=(aspect=1,))
Colorbar(fig[1,2],hm)
fig
##

[SpinHall.trapz((ktmp,ktmp),s0[i,:,:]) for i in 1:4]./lat.Sunit
##
v0 = range(4.0,8.0,11)
v01= v0[1:end-1].+(0.5*(v0[2]-v0[1]))
s_v01= Fermi_v0(v01,bz)

##
s = Array{Float64}(undef,4,length(v0)+length(v01))
v1 = range(4.0,8.0,size(s,2))
for i in 1:length(v01)
    s[:,2i-1].=s_v0[:,i]
    s[:,2i].=s_v01[:,i]
end
s[:,end].=s_v0[:,end]


##
using LinearAlgebra: dot,Diagonal
Γ = [0.0,0.9]
lat = set_lattice(4.0,1.5,0.0,7)
@time en,ev = eigenband(lat,Γ,1:2*lat.NK);
Jx = cal_Jθ(Γ,lat.Kvec,0.0;sp=1)
Jsx= cal_Jθ(Γ,lat.Kvec,0.0;sp=-1)
Jy = cal_Jθ(Γ,lat.Kvec,pi/2,sp=1)
H = calmat(lat,Γ)
tz= Diagonal([ones(lat.NK);fill(-1.0,lat.NK)])
tau = ((H*tz).-(tz*H)).*(1im)
;
##
@time SpinHall._fermiHall(en,ev,Jsx,Jy,2)
SpinHall.cal_tau0(en,ev,Jx,tau)
SpinHall.cal_tau1(en,ev,Jx,Jy,tau)
SpinHall.cal_tau2(en,ev,Jx,Jy,tau)
SpinHall.xcsz(en,ev,Jx,Jy,tz)

##
n = 1
sz = Array{ComplexF64}(undef,30)
sm = similar(sz)
hall1 = similar(sz,Float64)
@views for l in eachindex(sz)
    s = 0.0
    for m in 3:length(en)
        s+=(dot(ev[:,l],Jx,ev[:,m])*dot(ev[:,m],Jy,ev[:,n]))/(en[m]-en[n])^2
    end
    sm[l]=s
    sz[l]=dot_sz(ev[:,n],ev[:,l])
    hall1[l]=2*imag(sz[l]*sm[l])
end

begin
    fig,_,_=scatterlines(real.(sz))
    scatterlines!(imag.(sz))
    fig
end
begin
    fig,_,_=scatterlines(real.(sm))
    scatterlines!(imag.(sm))
    fig
end
scatterlines(hall1)
# sum(hall1)

##
Bm = Array{Float64}(undef,12,4)
for n in axes(Bm,2), m in axes(Bm,1)
    if m==n
        Bm[m,n] = 0.0
        continue
    end
    Bm[m,n]=2*imag(dot(ev[:,n],Jx,ev[:,m])*dot(ev[:,m],Jy,ev[:,n]))/(en[m]-en[n])^2
end


##
fig = Figure()
ax = Axis(fig[1,1],xlabel="m",xticks=range(1,12,12))
marks = [:circle,:dtriangle,:star5,:xcross]
for n in 1:4 #axes(Bm,2)
    scatterlines!(Bm[:,n],label=L"\frac{2\text{Im}\langle %$n|j_x|m\rangle\langle m|j_y|%$n\rangle}{(E_{%$n}-E_m)^2}",marker=marks[n],color=Makie.wong_colors()[n])
end
axislegend(nbanks=2)
fig
# sum(hall1)
