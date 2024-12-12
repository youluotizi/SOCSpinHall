module RK

function calmat_t!(H,NK,k,Kvec,v0,mz)
    for i in 1:NK
        tmp = (k[1]+Kvec[1,i])^2+(k[2]+Kvec[2,i])^2+v0
        H[i,i] = tmp+mz
        H[i+NK,i+NK] = tmp-mz
    end
    nothing
end

FF(t::Float64,k,r)=(k[1]+t*r[1],k[2]+t*r[2]) # k + F(t)
function rkw(
    r::Vector{Float64},
    k::Vector{Float64},
    hk0::Array{ComplexF64,2},
    ψ0::Array{ComplexF64,1},
    Kvec::Array{Float64,2},
    dt::Float64,
    step::Int,
    Nsample::Int,
    v0::Float64,
    mz::Float64
)
    hk=conj.(hk0)   # 对Hk取转置，使Hk*ψ由对Hk的行求和变为列求和; 对加速不明显，可以不做
    NK=size(Kvec,2)

    ψ=copy(ψ0)
    K1=similar(ψ)
    K2=similar(ψ)
    K3=similar(ψ)
    K4=similar(ψ)
    tmp=similar(ψ)
    
    ψtmp=Array{ComplexF64,2}(undef,2*NK,Nsample+1)
    ψtmp[:,1].=ψ
    klist = Array{Float64}(undef,2,Nsample+1)
    klist[ :,1].=k
    t=0.0
    @inbounds for ii in 2:Nsample
        # normalize!(ψ)
        for _ in 1:step
            t+=dt
            kk = FF(t,k,r)
            calmat_t!(hk,NK,kk,Kvec,v0,mz)
            func!(K1,ψ,hk,NK)         # 计算 K1
    
            kk = FF(t+0.5*dt,k,r)
            calmat_t!(hk,NK,kk,Kvec,v0,mz)      # 计算 Hk(t+dt/2)
            Threads.@threads for iQ in 1:2*NK
                tmp[iQ]=ψ[iQ]+0.5*dt*K1[iQ]
            end
            func!(K2,tmp,hk,NK)       # 计算 K2
    
            Threads.@threads for iQ in 1:2*NK
                tmp[iQ]=ψ[iQ]+0.5*dt*K2[iQ]
            end
            func!(K3,tmp,hk,NK)       # 计算 K3
    
            kk = FF(t+dt,k,r)
            calmat_t!(hk,NK,kk,Kvec,v0,mz)      # 计算 Hk(t+dt)
            Threads.@threads for iQ in 1:2*NK
                tmp[iQ]=ψ[iQ]+dt*K3[iQ]
            end
            func!(K4,tmp,hk,NK)       # 计算 K4
    
            Threads.@threads for iQ in 1:2*NK
                ψ[iQ]+=(dt/6)*(K1[iQ]+2*K2[iQ]+2*K3[iQ]+K4[iQ])
            end
        end
        klist[:,ii].= FF(t,k,r)
        Threads.@threads for iQ in 1:2*NK
            ψtmp[iQ,ii]=ψ[iQ]
        end
    end

    return ψtmp,klist
end

function func!(
    K::Array{ComplexF64,1},
    ψ::Array{ComplexF64,1},
    hk::Array{ComplexF64,2},
    NK::Int
)
    Threads.@threads for iQ in 1:NK
        ψup=ψdw=0.0im
        @inbounds for ii in 1:2*NK
            ψup+=hk[ii,iQ]*ψ[ii]           # 因为取了转置，所以对Hk第1个指标求和
            ψdw+=hk[ii,iQ+NK]*ψ[ii]
        end
        K[iQ]=-1im*ψup
        K[iQ+NK]=-1im*ψdw
    end
    nothing
end

function momentum(ψ,k,Kvec;θ::Float64=0.0)
    NK = size(Kvec,2)
    px = py = 0.0
    for i in 1:NK
        k1 = k[1]+Kvec[1,i]
        k2 = k[2]+Kvec[2,i]
        px+= (abs2(ψ[i])+abs2(ψ[i+NK]))*k1
        py+= (abs2(ψ[i])+abs2(ψ[i+NK]))*k2
    end
    p1 = cos(θ)*px+sin(θ)py
    p2 = -sin(θ)*px+cos(θ)py
    return p1,p2
end

function momentum_sp(ψ,k,Kvec;θ::Float64=0.0,spu::Bool=true)
    NK = size(Kvec,2)
    px = py = 0.0
    sp = spu ? 0 : NK
    for i in 1:NK
        k1 = k[1]+Kvec[1,i]
        k2 = k[2]+Kvec[2,i]
        px+= abs2(ψ[i+sp])*k1
        py+= abs2(ψ[i+sp])*k2
    end
    p1 = cos(θ)*px+sin(θ)py
    p2 = -sin(θ)*px+cos(θ)py
    return p1,p2
end

end