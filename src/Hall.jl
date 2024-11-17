#---------------------------------------------------------
#           Hall conductivity
#---------------------------------------------------------
export cal_Ju,cal_Jθ,Green1,eigBdG,Xspec1,Xspec2,Hall2ed,FermiHall,Hall_time,cal_Dθ

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

        Dhu[iQ+Nm]=2*(-kk[u]+Kvec[u,iQ])
        Dhu[iQ+Nm+NQ]=Dhu[iQ+Nm]*sp
    end
    for iQ in 1:Nm
        Ju[iQ]=Dhu[iQ]*ϕ[iQ]
        Ju[iQ+Nm]=Dhu[iQ+Nm]*conj(ϕ[iQ])
    end

    return Ju,Diagonal(Dhu)
end

# diag[∇h(k), ∇h^*(-k)]
function cal_Dθ(
    kk::AbstractVector{Float64},
    Kvec::Array{Float64,2},
    θ::Float64;
    d::Float64=1e-4, #  中心差分步长
    sp::Int=1        # sp=±1 代表粒子流和自旋流
)
    NQ=size(Kvec,2)
    Nm=NQ*2
    Dhu=Vector{Float64}(undef,Nm*2)

    k1 = (kk[1]+d*cos(θ),kk[2]+d*sin(θ))
    k2 = (kk[1]-d*cos(θ),kk[2]-d*sin(θ))

    for iQ in 1:NQ
        tmp = (k1[1]+Kvec[1,iQ])^2+(k1[2]+Kvec[2,iQ])^2
        tmp-= (k2[1]+Kvec[1,iQ])^2+(k2[2]+Kvec[2,iQ])^2
        Dhu[iQ]= tmp/(2*d)
        Dhu[iQ+NQ]=Dhu[iQ]*sp

        tmp = (-k1[1]+Kvec[1,iQ])^2+(-k1[2]+Kvec[2,iQ])^2
        tmp-= (-k2[1]+Kvec[1,iQ])^2+(-k2[2]+Kvec[2,iQ])^2
        Dhu[iQ+Nm]=tmp/(-2*d)
        Dhu[iQ+Nm+NQ]=Dhu[iQ+Nm]*sp
    end
    return Diagonal(Dhu)
end

function cal_Jθ(
    ϕ::Vector{ComplexF64},
    kk::Vector{Float64},
    Kvec::Array{Float64,2},
    θ::Float64;
    d::Float64=1e-4, #  中心差分步长
    sp::Int=1        # sp=±1 代表粒子流和自旋流
)
    Nm = size(Kvec,2)*2 
    Dhu = cal_Dθ(kk,Kvec,θ; sp=sp,d=d)
    Ju = Vector{ComplexF64}(undef,Nm*2)
    for iQ in 1:Nm
        Ju[iQ] = Dhu[iQ,iQ]*ϕ[iQ]
        Ju[iQ+Nm] = Dhu[iQ+Nm,iQ+Nm]*conj(ϕ[iQ])
    end
    return Ju
end

#  finite frequency Green function
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

#  zero-frequency Green function
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
function Xspec1(w::AbstractArray{Float64},Hx,Hy,ben,bev,ϕG; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

    v0=[ϕG; conj.(ϕG)]
    E0=ben[1]
    for iw in 1:Nw
        ww = abs(w[iw])>5e-6 ? w[iw]+1im*η : 5e-6+1im*η
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=dot(v0,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v0)/(ww+ben[nn]-E0)
        end
        Xw[iw]=tmp*1im/ww
    end
    return Xw
end

# ⟨ϕ|σ|v₂⟩×⟨v₂|j|vₙ⟩⟨vₙ|j|ϕ⟩，有限频
function Xspec2(w::AbstractArray{Float64},Hx,Hy,ben,bev,ϕG; η::Float64=0.0)
    Nm=round(Int,length(ben)/2)
    Nw=length(w)
    Xw=zeros(ComplexF64,Nw)

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
    for iw in 1:Nw
        ww = abs(w[iw])>5e-5 ? w[iw]+1im*η : 5e-5+1im*η
        tmp=0.0im
        @views for nn in 2:Nm
            tmp+=mz*dot(v1,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0)/(ww-ben[nn]+E0)
            tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v1)*conj(mz)/(ww+ben[nn]-E0)
        end
        Xw[iw]=tmp*1im/ww
    end
    return Xw
end

# ⟨ϕ|σ|v₂⟩×⟨v₂|j|vₙ⟩⟨vₙ|j|ϕ⟩,零频
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
        tmp+=2*imag(mz*dot(v1,Hx,bev[:,nn])*dot(bev[:,nn],Hy,v0))/(w+E0-ben[nn])^2
        # tmp-=dot(v0,Hy,bev[:,nn])*dot(bev[:,nn],Hx,v1)*conj(mz)/(w+ben[nn]-E0)^2
    end

    return tmp
end

# ---------------------------------------------
#     second order Bogoliubov contribution
# ---------------------------------------------
function _Hall2ed(en,ev,j1,j2,nE)
    Nb = round(Int, length(en)/2)
    s = 0.0im
    @views @inbounds for n in 1:2*Nb,m in 1:2*Nb
        a = nE[m]-nE[n]
        a == 0 && continue
        s-=0.5im*a*dot(ev[:,n],j1,ev[:,m])*dot(ev[:,m],j2,ev[:,n])/(en[m]-en[n])^2
    end
    return s
end

function Hall2ed(en,ev,bz,Kvec;θ::Tuple{Float64,Float64}=(0.0,pi/2),sp::Tuple{Int,Int}=(-1,1))
    Nb = round(Int, size(en,1)/2)
    nE = vcat(zeros(Int,Nb),fill(-1,Nb))
    s = Array{ComplexF64}(undef,size(en,2),size(en,3))
    Threads.@threads for iy in axes(en,3)
        @views for ix in axes(en,2)
            Dhx = cal_Dθ(bz[:,ix,iy],Kvec,θ[1]; sp=sp[1])
            Dhy = cal_Dθ(bz[:,ix,iy],Kvec,θ[2]; sp=sp[2])
            s[ix,iy] = _Hall2ed(en[:,ix,iy],ev[:,:,ix,iy],Dhx,Dhy,nE)
        end
    end
    return s
end

function Qudep(ev)
    _,Nb,Nx,Ny = size(ev)
    Nb = round(Int, Nb/2)
    ndep = Array{ComplexF64}(undef,Nx,Ny)
    Threads.@threads for iy in axes(ev,4)
        for ix in axes(ev,3)
            s = -Nb+0.0im
            @views for ib in Nb+1:2*Nb
                s+=dot(ev[:,ib,ix,iy],ev[:,ib,ix,iy])
            end
            ndep[ix,iy] = s/2
        end
    end
    return ndep
end

# ---------------------------------------
#       nointeracting Fermi gas
# ---------------------------------------
function cal_Jθ(
    kk::Vector{Float64},
    Kvec::Array{Float64,2},
    θ::Float64;
    d::Float64=1e-4, #  中心差分步长
    sp::Int=1        # sp=±1 代表粒子流和自旋流
)
    Ju=Vector{Float64}(undef,2*size(Kvec,2))
    k1 = (kk[1]+d*cos(θ),kk[2]+d*sin(θ))
    k2 = (kk[1]-d*cos(θ),kk[2]-d*sin(θ))

    for iQ in axes(Kvec,2)
        tmp = (k1[1]+Kvec[1,iQ])^2+(k1[2]+Kvec[2,iQ])^2
        tmp-= (k2[1]+Kvec[1,iQ])^2+(k2[2]+Kvec[2,iQ])^2
        Ju[iQ]= tmp/(2*d)
        Ju[iQ+NQ]=Ju[iQ]*sp
    end

    return Diagonal(Ju)
end

function _fermiHall(en,ev,hx,hy)
    Nb = length(en)
    s1 = 0.0
    E1 = en[1]
    @views for i in 3:Nb
        abs(en[i]-E1)<1e-6 && continue
        tmp = dot(ev[:,1],hx,ev[:,i])*dot(ev[:,i],hy,ev[:,1])
        s1+=2*imag(tmp)/(en[i]-E1)^2
    end

    s2 = 0.0
    E2 = en[2]
    @views for i in 3:Nb
        abs(en[i]-E2)<1e-6 && continue
        tmp = dot(ev[:,2],hx,ev[:,i])*dot(ev[:,i],hy,ev[:,2])
        s2+=2*imag(tmp)/(en[i]-E2)^2
    end
    return s1,s2
end
function FermiHall(bz,lat,θ)
    (;v0,m0,mz,NK,Kvec,Kcoe) = lat
    mat=zeros(ComplexF64,2*NK,2*NK)
    matoff!(mat,v0,m0,Kcoe)

    _,Nx,Ny = size(bz)
    σs = Array{Float64}(undef,4,Nx,Ny)
    lmz = abs(mz)<1e-9
    for iy in 1:Ny,ix in 1:Nx
        matdiag!(mat,bz[:,ix,iy],Kvec, v0, mz)
        en,ev=eigen(Hermitian(mat))
        lmz && zeeman_split!(ev)

        hsx = cal_Jθ(bz[:,ix,iy],Kvec,θ;sp=-1)
        hy = cal_Jθ(bz[:,ix,iy],Kvec,θ+pi/2,sp=1)
        hsy = cal_Jθ(bz[:,ix,iy],Kvec,θ+pi/2,sp=-1)
        σs[1:2,ix,iy].= _fermiHall(en,ev,hsx,hy)
        σs[3:4,ix,iy].= _fermiHall(en,ev,hsy,hy)
    end
    return σs
end

# time depend Hall
function Hall_time(ϕG,en,ev,J1,J2,t;η::Float64=0.0)
    Nb = length(ϕG)
    Vg = [ϕG; conj.(ϕG)]
    s = 0.0im
    @views for n in 2:Nb
        tmp = dot(Vg,J1,ev[:,n])*dot(ev[:,n],J2,Vg)/(en[n])^2
        tmp*= 1-cis((1im*η-en[n])*t)
        s += tmp - conj(tmp)
    end
    return s*(-1im)
end