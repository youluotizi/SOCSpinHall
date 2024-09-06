#---------------------------------------------------------
#           Hall conductivity
#---------------------------------------------------------
export cal_Ju,Green1,eigBdG,Xspec1,Xspec2

function cal_Ju(
    ϕ::Vector{ComplexF64},
    kk::Vector{Float64},
    Kvec::Array{Float64,2};
    u::Int=1, # u=1,2 分别为x，y方向
    sp::Int=1 # sp=±1 代表粒子流和自旋流
)
    NQ=size(Kvec,2)
    Nm=NQ*2
    Ju=Vector{ComplexF64}(undef,Nm*2)
    Dhu=similar(Ju)

    for iQ in 1:NQ
        Dhu[iQ]=2*(kk[u]+Kvec[u,iQ])
        Dhu[iQ+NQ]=Dhu[iQ]*sp

        Dhu[iQ+Nm]=-2*(-kk[u]+Kvec[u,iQ])
        Dhu[iQ+Nm+NQ]=Dhu[iQ+Nm]*sp
    end
    for iQ in 1:Nm
        Ju[iQ]=Dhu[iQ]*ϕ[iQ]
        Ju[iQ+Nm]=Dhu[iQ+Nm]*(-1)*conj(ϕ[iQ])
    end

    return Ju,Diagonal(Dhu)
end

function Green1(
    Mk0::Matrix{ComplexF64},
    w::AbstractVector{Float64},
    Jx::AbstractVector{ComplexF64},
    Jy::AbstractVector{ComplexF64};
    η::Float64=0.0
)
    Nm=round(Int,size(Mk0,1)/2)
    Nw=length(w)
    Xw=Vector{ComplexF64}(undef,Nw)

    Gw=-1.0.*Mk0
    for iw in 1:Nw
        ww=w[iw]+1im*η
        abs(ww)<1e-7 && (ww=(w[iw+1]-w[iw])/2+1im*η)
        for iQ in 1:Nm
            Gw[iQ,iQ]=ww-Mk0[iQ,iQ]
            Gw[iQ+Nm,iQ+Nm]=-ww-Mk0[iQ+Nm,iQ+Nm]
        end
        Xw[iw]=dot(Jx,inv(Gw),Jy)/ww
    end
    return Xw
end

function Green1(
    Mk0::Matrix{ComplexF64},
    Jx::AbstractVector{ComplexF64},
    Jy::AbstractVector{ComplexF64};
    η::Float64=0.0,
    w::Float64=5e-4
)
    Nm=round(Int,size(Mk0,1)/2)
    w2 = [w,-w]
    Xw = Array{ComplexF64}(undef,2)
    Gw=-1.0.*Mk0
    for iw in 1:2
        ww=w2[iw]+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ]=ww-Mk0[iQ,iQ]
            Gw[iQ+Nm,iQ+Nm]=-ww-Mk0[iQ+Nm,iQ+Nm]
        end
        Xw[iw]=dot(Jx,inv(Gw),Jy)
    end

    return (Xw[1]-Xw[2])/(2im*w-2*η)
end


# 谱分解计算响应

function eigBdG(Mk0)
    Nm=round(Int,size(Mk0,1)/2)
    tz=Diagonal([ones(Nm);fill(-1.0,Nm)])

    ben,bev=eigen(tz*Mk0)
    pt=sortperm(real.(ben))
    pt.=[pt[Nm+1:end];reverse(pt[1:Nm])]
    ben.=ben[pt]
    bev.=bev[:,pt]

    norBev!(bev)
    gaugev!(bev)
    return ben,bev
end

function Xspec1(w,Hx,Hy,ben,bev,ϕG; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

    v0=[ϕG,-1.0.*conj.(ϕG)]
    E0=ben[1]
    for iw in 1:Nw
        ww=w[iw]+1im*η
        abs(ww)<1e-6 && (ww=(w[iw+1]-w[iw])/2+1im*η)
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=dot(v0,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v0)/(ww+ben[nn]-E0)
        end
        Xw[iw]=tmp/ww
    end
    return Xw
end

function Xspec2(w,Hx,Hy,ben,bev; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

    v0=bev[:,1].*√2
    E0=ben[1]
    for iw in 1:Nw
        ww=w[iw]+1im*η
        abs(ww)<1e-6 && (ww=(w[iw+1]-w[iw])/2)
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=dot(v0,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v0)/(ww+ben[nn]-E0)
        end
        Xw[iw]=tmp/ww
    end
    return Xw
end
