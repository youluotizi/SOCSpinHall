# Bloch 算激发谱
function matHu02!(matH::Array{ComplexF64,2},lat::Lattice,nb::Vector{Int})
    lenmat=2*lat.NQ
    Mii=diag(matH)

    mat=calmat(lat,[0.0,0.0])
    matH[1:lenmat,1:lenmat].+=mat
    matH[lenmat+1:end,lenmat+1:end].+=conj.(mat)

    _,v1=eigen(Hermitian(mat))
    Vcut=zeros(ComplexF64,2*lenmat,2*lenmat)
    Vcut[1:lenmat,1:lenmat].=v1
    Vcut[lenmat+1:end,lenmat+1:end].=conj.(v1)
    Mk2=Vcut'*matH*Vcut

    Nb1=nb[1]
    Nb2=nb[2]
    Nben=nb[3]
    Mk3=Array{ComplexF64,2}(undef,2*Nben,2*Nben)
    Mk3[1:Nben,1:Nben].=Mk2[Nb1:Nb2,Nb1:Nb2]
    Mk3[Nben+1:end,1:Nben].=Mk2[lenmat+Nb1:lenmat+Nb2,Nb1:Nb2]
    Mk3[1:Nben,Nben+1:end].=Mk2[Nb1:Nb2,lenmat+Nb1:lenmat+Nb2]
    Mk3[Nben+1:end,Nben+1:end].=Mk2[lenmat+Nb1:lenmat+Nb2,lenmat+Nb1:lenmat+Nb2]

    u0=minimum(eigvals(Hermitian(Mk3)))
    return Mii,u0,mat
end

function Bloch_BdG(lat::Lattice,ban::Band,uu::Float64,ϕ::Vector{ComplexF64})
    matH=zeros(ComplexF64,4*lat.NQ,4*lat.NQ)
    intBdgM!(matH,ϕ,lat)
    Mii,u0,mat=matHu02!(matH,lat,ban.nb)
    println("check_u0: ",uu-u0)
    u0=lat.v0-u0

    Vq=calVq(lat)
    en=Array{Float64,2}(undef,ban.nb[3],ban.Nk)

    lenmat=2*lat.NQ
    Nb1=ban.nb[1]
    Nb2=ban.nb[2]
    Nben=ban.nb[3]
    tz=Diagonal([ones(ComplexF64,Nben); fill(-1.0+0.0im,Nben)])
    for ik=1:ban.Nk
        mtmp=copy(matH)
        mat2=copy(mat)
        for iQ in 1:lat.NQ
            tmp=(ban.kk[1,ik]+Vq[1,iQ])^2+(ban.kk[2,ik]+Vq[2,iQ])^2
            mtmp[iQ,iQ]=Mii[iQ]+tmp+lat.mz+u0
            mtmp[iQ+lat.NQ,iQ+lat.NQ]=Mii[iQ+lat.NQ]+tmp+u0-lat.mz
            mat2[iQ,iQ]=tmp+lat.v0+lat.mz
            mat2[iQ+lat.NQ,iQ+lat.NQ]=tmp+lat.v0-lat.mz

            tmp=(-ban.kk[1,ik]+Vq[1,iQ])^2+(-ban.kk[2,ik]+Vq[2,iQ])^2
            mtmp[iQ+2*lat.NQ,iQ+2*lat.NQ]=Mii[iQ+2*lat.NQ]+tmp+lat.mz+u0
            mtmp[iQ+3*lat.NQ,iQ+3*lat.NQ]=Mii[iQ+3*lat.NQ]+tmp-lat.mz+u0
        end
    
        Vcut=zeros(ComplexF64,2*lenmat,2*lenmat)
        _,Vcut[1:lenmat,1:lenmat]=eigen(Hermitian(mat2))
        for iQ in 1:lat.NQ
            tmp=(-ban.kk[1,ik]+Vq[1,iQ])^2+(-ban.kk[2,ik]+Vq[2,iQ])^2
            mat2[iQ,iQ]=tmp+lat.v0+lat.mz
            mat2[iQ+lat.NQ,iQ+lat.NQ]=tmp+lat.v0-lat.mz
        end
        _,Vcut[lenmat+1:end,lenmat+1:end]=eigen(Hermitian(mat2))

        Vcut[lenmat+1:end,lenmat+1:end].=conj.(Vcut[lenmat+1:end,lenmat+1:end])
 
        mtmp.=Vcut'*mtmp*Vcut
        Mk3=Array{ComplexF64,2}(undef,2*Nben,2*Nben)
        Mk3[1:Nben,1:Nben].=mtmp[Nb1:Nb2,Nb1:Nb2]
        Mk3[Nben+1:end,1:Nben].=mtmp[lenmat+Nb1:lenmat+Nb2,Nb1:Nb2]
        Mk3[1:Nben,Nben+1:end].=mtmp[Nb1:Nb2,lenmat+Nb1:lenmat+Nb2]
        Mk3[Nben+1:end,Nben+1:end].=mtmp[lenmat+Nb1:lenmat+Nb2,lenmat+Nb1:lenmat+Nb2]

        lmul!(tz,Mk3)
        entmp=real.(eigvals!(Mk3))
        pst=partialsortperm(entmp,Nben+1:2*Nben)
        en[:,ik].=entmp[pst]
    end
    return en
end
