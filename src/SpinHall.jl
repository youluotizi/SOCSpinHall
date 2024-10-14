module SpinHall

using LinearAlgebra #,MKL
# BLAS.set_num_threads(Threads.nthreads())

include("utilities.jl")
include("Bloch.jl")

using Trapz
using BlackBoxOptim
using JLD2,FileIO
include("GroundState.jl")
include("pwBdG.jl")
include("Hall.jl")

include("Symm.jl")

end # module SpinHall
