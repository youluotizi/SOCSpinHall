module Symm
using LinearAlgebra
# using MKL
using Trapz

function myint(
    v1::Vector{ComplexF64},
    v2::Vector{ComplexF64},
    Vq::Array{Float64,2},
    Foo::Function,
    II::Int
)
    w1=Psix(v1,Vq,symC,0)
    @time w2=Psix(v2,Vq,Foo,II)#.|>conj
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

function Psix(v::Vector{ComplexF64},Vq0::Array{Float64,2},Foo::Function,II::Int)
    a1=[1pi,1pi]
    a2=[-1pi,1pi]
    p0=-0.5.*(a1.+a2)
    xx=bz2d([p0,p0.+a1,p0.+a2],[255,255])
    dd=[0.0,0.0] #((xx[:,2,1].-xx[:,1,1]).+(xx[:,1,2].-xx[:,1,1])).*0.5

    Vq,sz,d0=Foo(Vq0,II)
    dd[1]+=d0
    _,lx,ly=size(xx)
    Threads.@threads for ix in 1:lx
        for iy in 1:ly
            xx[1,iy,ix]+=dd[1]
            xx[2,iy,ix]+=dd[2]
        end
    end
    _Psix(v,Vq,sz,xx)
end

function Psix(v::Vector{ComplexF64},Vq::Array{Float64,2})
    a=√2pi*1.4
    p0=[-0.5*a,-0.5*a]
    a1=[a,0.0]
    a2=[0.0,a]
    xx=bz2d([p0,p0.+a1,p0.+a2],[255,255])
    sz=diagm([1.0+0.0im,1.0+0.0im])
    _Psix(v,Vq,sz,xx)
end
function bz2d(plist::Array{Array{Float64,1},1},nn::Array{Int,1})
    b1=(plist[2].-plist[1])./nn[1]
    b2=(plist[3].-plist[1])./nn[2]
    p=plist[1]
    bz=Array{Float64,3}(undef,2,nn[1]+1,nn[2]+1)
    for jj in 0:nn[2],ii in 0:nn[1]
        bz[:,ii+1,jj+1].=p.+jj.*b2.+ii.*b1
    end
    return bz
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
    sz=[-1.0im 00im;0 1.0im]
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

end