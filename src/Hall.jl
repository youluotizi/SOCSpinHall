#---------------------------------------------------------
#           Hall conductivity
#---------------------------------------------------------
export cal_Ju,cal_Jθ,Green1,eigBdG,Xspec1,Xspec2

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
function cal_Jθ(
    ϕ::Vector{ComplexF64},
    kk::Vector{Float64},
    Kvec::Array{Float64,2},
    θ::Float64;
    d::Float64=1e-4, #  中心差分步长
    sp::Int=1 # sp=±1 代表粒子流和自旋流
)
    NQ=size(Kvec,2)
    Nm=NQ*2

    Ju=Vector{ComplexF64}(undef,Nm*2)
    Dhu=similar(Ju)

    k1 = (kk[1]+d*cos(θ),kk[2]+d*sin(θ))
    k2 = (kk[1]-d*cos(θ),kk[2]-d*sin(θ))
    # k3 = (-kk[1]+d*cos(θ),-kk[2]+d*sin(θ))
    # k4 = (-kk[1]-d*cos(θ),-kk[2]-d*sin(θ))

    for iQ in 1:NQ
        tmp = (k1[1]+Kvec[1,iQ])^2+(k1[2]+Kvec[2,iQ])^2
        tmp-= (k2[1]+Kvec[1,iQ])^2+(k2[2]+Kvec[2,iQ])^2
        Dhu[iQ]= tmp/(2*d)
        Dhu[iQ+NQ]=Dhu[iQ]*sp

        tmp = (-k1[1]+Kvec[1,iQ])^2+(-k1[2]+Kvec[2,iQ])^2
        tmp-= (-k2[1]+Kvec[1,iQ])^2+(-k2[2]+Kvec[2,iQ])^2
        Dhu[iQ+Nm]=tmp/(2*d)
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
        ww = abs(w[iw])>1e-4 ? w[iw]+1im*η : 1e-4+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ]=ww-Mk0[iQ,iQ]
            Gw[iQ+Nm,iQ+Nm]=-ww-Mk0[iQ+Nm,iQ+Nm]
        end
        Xw[iw]=dot(Jx,inv(Gw),Jy)*1im/ww
    end
    return Xw
end

function Green1(
    Mk0::Matrix{ComplexF64},
    Jx::AbstractVector{ComplexF64},
    Jy::AbstractVector{ComplexF64};
    η::Float64=0.0,
    w::Float64=1e-4
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

    return (Xw[1]-Xw[2])/(2*η-2im*w)
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

function Xspec1(w::AbstractArray{Float64},Hx,Hy,ben,bev,ϕG; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

    v0=[ϕG; -1.0.*conj.(ϕG)]
    E0=ben[1]
    for iw in 1:Nw
        ww = abs(w[iw])>5e-6 ? w[iw]+1im*η : 5e-6+1im*η
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=dot(v0,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v0)/(ww+ben[nn]-E0)
            # if nn <13
            #     print(nn,", ",expshow(dot(v0,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)))
            #     println(", ",expshow(dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v0)))
            # end
        end
        Xw[iw]=tmp*1im/ww
    end
    return Xw
end

function Xspec2(w::AbstractArray{Float64},Hx,Hy,ben,bev,ϕG; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

    v0=[ϕG; conj.(ϕG)]
    E0=ben[1]

    NK = round(Int,length(ben)/4)
    sz = Diagonal([ones(NK);fill(-1.0,NK);ones(NK);fill(-1.0,NK)])
    mz = dot(v0,sz,bev[:,2])
    for iw in 1:Nw
        ww = abs(w[iw])>5e-5 ? w[iw]+1im*η : 5e-5+1im*η
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=mz*dot(bev[:,2],Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,bev[:,2])*conj(mz)/(ww+ben[nn]-E0)
        end
        Xw[iw]=tmp*1im/ww
    end
    return Xw
end

function Xspec2(Hx,Hy,ben,bev,ϕG,ϕG2)
    Nm=round(Int,length(ben)/2)
    v0=[ϕG; conj.(ϕG)]
    v1=similar(v0)
    if abs(ben[2]-ben[1])<1e-6
        v1.=[ϕG2; zeros(ComplexF64, Nm)]
    else
        v1.=bev[:,2]
    end
    E0=ben[1]

    NK = round(Int,length(ben)/4)
    sz = Diagonal([ones(NK);fill(-1.0,NK);ones(NK);fill(-1.0,NK)])
    mz = dot(v0,sz,v1)
    println(mz)

    tmp=0.0im
    @views for nn in 2:Nm
        w = nn==2 ? 1e-6 : 0.0
        tmp+=-2*imag(mz*dot(v1,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0))/(w+E0-ben[nn])^2
        # tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v1)*conj(mz)/(w+ben[nn]-E0)^2
    end

    return tmp
end
