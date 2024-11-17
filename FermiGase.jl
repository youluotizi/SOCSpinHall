# ---------------------------------------------------
##              spin hall of fermi gases
# ---------------------------------------------------
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

## --------- 网格均匀划分计算 -----------
lat = set_lattice(8.0,1.5,1.0)
bz = mymesh([[0.0,0.0],lat.b[:,1],lat.b[:,2]],[128,128])
@time s2 = FermiHall(bz,lat,0.0)

sum(s2)/128^2/lat.Sunit|>println
krn = range(0,1,128)
SpinHall.trapz((krn,krn),s2[1,:,:])/lat.Sunit|>println
fig = Figure(size=(450,400))
ax,hm = heatmap(fig[1,1],s2[2,:,:]./lat.Sunit,axis=(aspect=1,))
Colorbar(fig[1,2],hm)
fig



## --------- 网格非均匀划分计算 -----------
sg2(x::Real)=sign(x)*x^2
ktmp = sg2.(range(-1,1,64)).*0.5.+0.5
bz = mymesh([[0.0,0.0],lat.b[:,1],lat.b[:,2]],ktmp,ktmp)
scatter(reshape(bz[1,:,:],:),reshape(bz[2,:,:],:),axis=(aspect=1,),markersize=3)|>display
@time s0 = FermiHall(bz,lat,0.0)
SpinHall.trapz((ktmp,ktmp),s0[2,:,:])*2/lat.Sunit|>println

ftmp = s0[1,:,:]./lat.Sunit
ftmp[abs.(ftmp).>50].=NaN64
fig,_,hm = heatmap(fig[1,1],ktmp,ktmp,ftmp,axis=(aspect=1,),figure=(size=(450,400),))
Colorbar(fig[1,2],hm)
fig



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