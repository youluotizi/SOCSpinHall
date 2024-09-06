export main_BdG,main_BdG2D,cal_BdG

function _intBdgM!(
    matH::Array{ComplexF64,2},
    ψ::Array{ComplexF64,1},
    Kl::Array{Int,2},
    g1::Float64,
    g2::Float64
)
    Nv=length(ψ)
    NQ = round(Int,Nv/2)
    NKl=size(Kl,2)

    @inbounds for ii in 1:NKl
        t1,t2,t3,t4 = view(Kl,:,ii)

        tmp = 2*conj(ψ[t2])*ψ[t3]*g1 + conj(ψ[t2+NQ])*ψ[t3+NQ]*g2
        matH[t1,t4] += tmp
        matH[t1+Nv,t4+Nv] += conj(tmp)

        tmp = conj(ψ[t2+NQ])*ψ[t3]*g2
        matH[t1,t4+NQ] += tmp
        matH[t1+Nv,t4+NQ+Nv] += conj(tmp)

        tmp = conj(ψ[t2])*ψ[t3+NQ]*g2
        matH[t1+NQ,t4] += tmp
        matH[t1+NQ+Nv,t4+Nv] += conj(tmp)
        
        tmp = 2*conj(ψ[t2+NQ])*ψ[t3+NQ]*g1+conj(ψ[t2])*ψ[t3]*g2
        matH[t1+NQ,t4+NQ] += tmp
        matH[t1+NQ+Nv,t4+NQ+Nv] += conj(tmp)
        
        # ---------------- off diag ----------------------
        tmp = ψ[t3]*ψ[t4]*g1
        matH[t1,t2+Nv] += tmp
        matH[t2+Nv,t1] += conj(tmp)

        tmp = ψ[t3+NQ]*ψ[t4]*g2
        matH[t1,t2+NQ+Nv] += tmp
        matH[t2+NQ+Nv,t1] += conj(tmp)

        tmp = ψ[t3]*ψ[t4+NQ]*g2
        matH[t1+NQ,t2+Nv] += tmp
        matH[t2+Nv,t1+NQ] += conj(tmp)

        tmp = ψ[t3+NQ]*ψ[t4+NQ]*g1
        matH[t1+NQ,t2+NQ+Nv] += tmp
        matH[t2+NQ+Nv,t1+NQ] += conj(tmp)
    end
    nothing
end
# BdG 矩阵的相互作用部分，未左乘tz
function intBdgM!(
    matH::Array{ComplexF64,2},
    ev0::Array{ComplexF64,1},
    lat::Lattice
)
    Kl=cal_phi4_coe(lat.Kcoe,lat.Kmax)
    _intBdgM!(matH,ev0,Kl,lat.g1,lat.g2)
    nothing
end

# BdG 矩阵的非对角部分，已左乘tz
function matHu0!(matH::Array{ComplexF64,2},lat::Lattice;pband::Int=0)
    Nm=2*lat.NK
    Mii=diag(matH)
    tauz=Diagonal([ones(ComplexF64,Nm); fill(-1.0+0.0im,Nm)])

    mat=calmat(lat,[0.0,0.0])
    matH[1:Nm,1:Nm].+=mat
    matH[Nm+1:end,Nm+1:end].+=conj.(mat)

    ben = eigvals(Hermitian(matH),1:20)
    u0::Float64 = ben[1+2*pband] 

    lmul!(tauz,matH)
    return Mii,u0
end

# BdG 矩阵的对角部分，已左乘tz
function calmatH!(
    mtmp::Matrix{ComplexF64},
    Mii::Vector{ComplexF64},
    kk::Vector{Float64},
    Kvec::Array{Float64,2},
    NK::Int,
    mz::Float64,
    u0::Float64
)
    for iQ in 1:NK
        tmp=(kk[1]+Kvec[1,iQ])^2+(kk[2]+Kvec[2,iQ])^2+u0
        mtmp[iQ,iQ]=Mii[iQ]+tmp+mz
        mtmp[iQ+NK,iQ+NK]=Mii[iQ+NK]+tmp-mz

        tmp=(-kk[1]+Kvec[1,iQ])^2+(-kk[2]+Kvec[2,iQ])^2+u0
        mtmp[iQ+2*NK,iQ+2*NK]=-1*(Mii[iQ+2*NK]+tmp+mz)
        mtmp[iQ+3*NK,iQ+3*NK]=-1*(Mii[iQ+3*NK]+tmp-mz)
    end
    nothing
end

function main_BdG(lat::Lattice,ϕ::Vector{ComplexF64},uu::Float64,k::Array{Float64,2},Nb::Int; mu::Bool=true)
    (;NK,v0,mz,Kvec) = lat
    matH=zeros(ComplexF64,4*NK,4*NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0=matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0 = mu ? v0-u0 : uu    #  使用平均场基态求得的化学势或无能隙要求的化学势，默认后者

    en=Array{Float64,2}(undef,Nb,size(k,2))
    mtmp=similar(matH)
    for ik in axes(k,2)
        mtmp.=matH
        calmatH!(mtmp,Mii,k[:,ik],Kvec,NK,mz,u0)
        entmp=real.(eigvals!(mtmp))
        pt=partialsortperm(entmp,2*NK+1:2*NK+Nb)
        en[:,ik].=entmp[pt]
    end

    return en
end

# 计算k点的BdG矩阵，不左乘tz
function cal_BdG(lat::Lattice,ϕ::Vector{ComplexF64},uu::Float64,kk::Vector{Float64})
    matH=zeros(ComplexF64,4*lat.NK,4*lat.NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0=matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0=lat.v0-u0

    calmatH!(matH,Mii,kk,lat.Kvec,lat.NK,lat.mz,u0)
    tz=Diagonal([ones(2*lat.NK);fill(-1.0,2*lat.NK)])
    lmul!(tz,matH)

    return matH,tz
end

function main_BdG2D(lat::Lattice,uu::Float64,ϕ::Vector{ComplexF64},bz::Array{Float64,3},Nb::Int)
    _,Nx,Ny=size(bz)
    matH=zeros(ComplexF64,4*lat.NK,4*lat.NK)
    intBdgM!(matH,ϕ,lat)
    Mii,u0=matHu0!(matH,lat)
    println("check_u0: ",uu-u0)
    u0=lat.v0-u0
    
    en=Array{Float64,3}(undef,Nb,Nx,Ny)
    ev=Array{ComplexF64,4}(undef,4*lat.NK,Nb,Nx,Ny)
    bcav=Array{Float64,3}(undef,Nb,Nx-1,Ny-1)
    @inbounds for iy in 1:Ny
        mtmp=similar(matH)
        for ix in 1:Nx
            mtmp.=matH
            calmatH!(mtmp,Mii,bz[:,ix,iy],lat.Kvec,lat.NK,lat.mz,u0)
            entmp,evtmp=eigen!(mtmp)
            pt=partialsortperm(real.(entmp),2*lat.NK+1:2*lat.NK+Nb)
            en[:,ix,iy].=real.(entmp[pt])
            ev[:,:,ix,iy].=evtmp[:,pt]
        end
    end

    ds=abs(det([bz[:,2,1].-bz[:,1,1] bz[:,1,2].-bz[:,1,1]]))
    tz=Diagonal([ones(2*lat.NK);fill(-1.0,2*lat.NK)])
    Threads.@threads for ix=1:Nx-1
        @views for iy in 1:Ny-1,ib=1:Nb
            tmp=dot(ev[:,ib,iy,ix],tz,ev[:,ib,iy,ix+1])
            tmp*=dot(ev[:,ib,iy,ix+1],tz,ev[:,ib,iy+1,ix+1])
            tmp*=dot(ev[:,ib,iy+1,ix+1],tz,ev[:,ib,iy+1,ix])
            tmp*=dot(ev[:,ib,iy+1,ix],tz,ev[:,ib,iy,ix])
            bcav[ib,iy,ix]=-angle(tmp)/ds
        end
    end
    return en,bcav
end

