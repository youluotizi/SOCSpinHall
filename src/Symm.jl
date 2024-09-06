export PTtransform

function PTtransform(ψ::AbstractArray{<:Number})
    ϕ=similar(ψ)
    Nv = round(Int,length(ψ)/2)
    ϕ[1:Nv].= conj.(ψ[Nv+1:end])
    ϕ[Nv+1:end].= conj.(ψ[1:Nv]).*(-1)
    return ϕ
end
#=
function myint(
    v1::Vector{ComplexF64},
    v2::Vector{ComplexF64},
    Vq::Array{Float64,2},
    Foo::Function,
    II::Int
)
    w1=Psix(v1,Vq,symC,0)
    w2=Psix(v2,Vq,Foo,II)#.|>conj
    _myint(w1,w2)
end

function _myint(w1::Array{ComplexF64,3},w2::Array{ComplexF64,3})
    _,ly,lx=size(w1)
    tmp=Array{ComplexF64,2}(undef,ly,lx)
    Threads.@threads for ix in 1:lx
        for iy in 1:ly
            tmp[iy,ix]=conj(w1[1,iy,ix])*w2[1,iy,ix]+conj(w1[2,iy,ix])*w2[2,iy,ix]
        end
    end
    xl=range(0.0,1.0,length=lx)
    yl=range(0.0,1.0,length=ly)
    trapz((yl,xl),tmp)
end
function mymesh(Klist::Vector{Vector{Float64}},nn::Array{Int,1})
    ndim=length(nn)
    p0=Klist[1]

    a=Array{Float64}(undef,ndim,ndim)
    for ii in 1:ndim
        a[:,ii].=(Klist[ii+1].-p0)./(nn[ii]-1)
    end

    mgrid=zeros(Float64,ndim,nn...)
    xrn=[1:nn[ii] for ii in 1:ndim]
    
    Vtmp=Array{Float64}(undef,ndim)
    for idx in Iterators.product(xrn...)
        Vtmp.=p0
        for ii in 1:ndim
            Vtmp.+=(idx[ii]-1).*a[:,ii]
        end
        mgrid[:,idx...].=Vtmp
    end
    mgrid
end

function Psix(v::Vector{ComplexF64},Kvec0::Array{Float64,2},Foo::T,II::Int) where{T}
    a1=[1pi,1pi]
    a2=[-1pi,1pi]
    p0=-0.5.*(a1.+a2)
    xx=mymesh([p0,p0.+a1,p0.+a2],[255,255])
    dd=[0.0,0.0] #((xx[:,2,1].-xx[:,1,1]).+(xx[:,1,2].-xx[:,1,1])).*0.5

    Kvec,sz,d0=Foo(Kvec0,II)
    dd[1]+=d0
    _,lx,ly=size(xx)
    Threads.@threads for ix in 1:lx
        for iy in 1:ly
            xx[1,iy,ix]+=dd[1]
            xx[2,iy,ix]+=dd[2]
        end
    end
    _Psix(v,Kvec,sz,xx)
end

function _Psix(
    v::Vector{ComplexF64},
    Vq::Array{Float64,2},
    sz::Matrix{ComplexF64},
    xx::Array{Float64,3}
)
    NQ=size(Vq,2)
    _,ly,lx=size(xx)
    
    w=Array{ComplexF64,3}(undef,2,ly,lx)
    Threads.@threads for ix in 1:lx
        for iy in 1:ly
            t1=t2=0.0im
            for iQ in 1:NQ
                tmp=cis(Vq[1,iQ]*xx[1,iy,ix]+Vq[2,iQ]*xx[2,iy,ix])
                t1+=v[iQ]*tmp
                t2+=v[iQ+NQ]*tmp
            end
            w[1,iy,ix]=sz[1,1]*t1+sz[1,2]*t2
            w[2,iy,ix]=sz[2,1]*t1+sz[2,2]*t2
        end
    end
    return w
end

function symC(V::Array{Float64,2},II::Int)
    D=[cospi(0.5*II) -sinpi(0.5*II);sinpi(0.5*II) cospi(0.5*II)]
    Vq=similar(V)
    NQ=size(V,2)
    for iQ in 1:NQ
        Vq[1,iQ]=D[1,1]*V[1,iQ]+D[1,2]*V[2,iQ]
        Vq[2,iQ]=D[2,1]*V[1,iQ]+D[2,2]*V[2,iQ]
    end
    sz=[cispi(-II/4) 0.0im;0.0im cispi(II/4)]
    return Vq,sz,0.0
end

function symS(V::Array{Float64,2},II::Int)
    k1=cospi(II/4)
    k2=sinpi(II/4)
    w1=cospi(1.0)
    Sn=[k1^2*(1-w1)+w1 k1*k2*(1-w1);k1*k2*(1-w1) k2^2*(1-w1)+w1]

    Vq=similar(V)
    NQ=size(V,2)
    for iQ in 1:NQ
        Vq[1,iQ]=Sn[1,1]*V[1,iQ]+Sn[1,2]*V[2,iQ]
        Vq[2,iQ]=Sn[2,1]*V[1,iQ]+Sn[2,2]*V[2,iQ]
    end
    sx=[0.0im k1-1im*k2;k1+1im*k2 0.0im]
    sx.=cis(-0.5pi.*sx)
    return Vq,sx,0.0
end

function symTx(V::Array{Float64,2},II::Int)
    Vq=copy(V)
    # sz=[1.0 00im;0 -1.0]
    sz=[-1.0im 0im;0 1.0im]
    d0=1pi
    return Vq,sz,d0
end

function symPT(V::Array{Float64,2},II::Int)
    Vq=similar(V)
    NQ=size(V,2)
    for iQ in 1:NQ
        Vq[1,iQ]=-V[1,iQ]
        Vq[2,iQ]=-V[2,iQ]
    end
    sz = [0.0 -1.0im;1.0im 0.0].*1.0im
    return Vq,sz,0.0
end
function symT(V::Array{Float64,2},II::Int)
    Vq=copy(V)
    sz = [0.0im 1.0;1.0 0.0im]
    return Vq,sz,0.0
end
=#