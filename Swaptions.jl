
using Random, Optim, Distributions, DiffFusion, Plots, Interpolations, Flux, LinearAlgebra, Zygote

# Gaussian HJM Model
ch = DiffFusion.correlation_holder("");
δ = DiffFusion.flat_parameter([ 0., ]);
χ = DiffFusion.flat_parameter([ 0.01, ]);
     
times = [  1.,  2.,  5., 10. ]
values = [ 50.,  60.,  70.,  80., ]' * 1.0e-4 
σ = DiffFusion.backward_flat_volatility("", times, values);

model = DiffFusion.gaussian_hjm_model("md/EUR", δ, χ, σ, ch, nothing);

DiffFusion.Sigma_T()

# Simulation
times = 0.0:0.25:10.0
n_paths = 2^10

sim = DiffFusion.simple_simulation(
    model,
    ch,
    times,
    n_paths,
    with_progress_bar = false,
    brownian_increments = DiffFusion.sobol_brownian_increments,
);

DiffFusion.state_alias(model)

yc_estr = DiffFusion.zero_curve(
    "yc/EUR:ESTR",
    [1.0, 3.0, 6.0, 10.0],
    [1.0, 1.0, 1.0,  1.0] .* 1e-2,
)
yc_euribor6m = DiffFusion.zero_curve(
    "yc/EUR:EURIBOR6M",
    [1.0, 3.0, 6.0, 10.0],
    [2.0, 2.0, 2.0,  2.0] .* 1e-2,
)

ts_list = [
    yc_estr,
    yc_euribor6m,
];

_empty_key = DiffFusion._empty_context_key
context = DiffFusion.Context(
    "Std",
    DiffFusion.NumeraireEntry("EUR", "md/EUR", Dict(_empty_key => "yc/EUR:ESTR")),
    Dict{String, DiffFusion.RatesEntry}([
        ("EUR", DiffFusion.RatesEntry("EUR", "md/EUR", Dict(
            _empty_key  => "yc/EUR:ESTR",
            "ESTR"      => "yc/EUR:ESTR",
            "EURIBOR6M" => "yc/EUR:EURIBOR6M",
        ))),
    ]),
    Dict{String, DiffFusion.AssetEntry}(),
    Dict{String, DiffFusion.ForwardIndexEntry}(),
    Dict{String, DiffFusion.FutureIndexEntry}(),
    Dict{String, DiffFusion.FixingEntry}(),
);

path = DiffFusion.path(sim, ts_list, context, DiffFusion.LinearPathInterpolation);

# Cashflows
fixed_flows = [
    DiffFusion.FixedRateCoupon( 1.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 2.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 3.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 4.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 5.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 6.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 7.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 8.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon( 9.0, 0.02, 1.0),
    DiffFusion.FixedRateCoupon(10.0, 0.02, 1.0),
];
libor_flows = [
    DiffFusion.SimpleRateCoupon(0.0, 0.0, 0.5, 0.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(0.5, 0.5, 1.0, 1.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(1.0, 1.0, 1.5, 1.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(1.5, 1.5, 2.0, 2.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(2.0, 2.0, 2.5, 2.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(2.5, 2.5, 3.0, 3.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(3.0, 3.0, 3.5, 3.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(3.5, 3.5, 4.0, 4.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(4.0, 4.0, 4.5, 4.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(4.5, 4.5, 5.0, 5.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(5.0, 5.0, 5.5, 5.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(5.5, 5.5, 6.0, 6.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(6.0, 6.0, 6.5, 6.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(6.5, 6.5, 7.0, 7.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(7.0, 7.0, 7.5, 7.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(7.5, 7.5, 8.0, 8.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(8.0, 8.0, 8.5, 8.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(8.5, 8.5, 9.0, 9.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(9.0, 9.0, 9.5, 9.5, 0.5, "EUR:EURIBOR6M", nothing, nothing),
    DiffFusion.SimpleRateCoupon(9.5, 9.5, 10.0, 10.0, 0.5, "EUR:EURIBOR6M", nothing, nothing),
];

# Swaption Legs
swpt = DiffFusion.SwaptionLeg(
    "leg/swpn/2y",
    2.0, #expiry time
    2.0, #settlement_time
    libor_flows[5:end], #float leg
    fixed_flows[3:end], #fixed leg
    -1.0, #payer_receiver, 
    "EUR:ESTR",
    DiffFusion.SwaptionPhysicalSettlement, #settlement_type,
    10_000.00, #notional,
    "EUR:ESTR", #swpt_disc_curve_key,
    nothing, #swpt_fx_key,
    1.0 #swpt_long_short,
)


#option value to time obs_time over all paths of a swaption leg
function optionValue(
    leg, 
    obs_time
    )

    swaptionPayoff = DiffFusion.discounted_cashflows(leg, obs_time)[1]
    values = DiffFusion.at(swaptionPayoff, path)
    return values

end

optionValue(swpt, 0)


# neural network with input dimension and amount of layer parameters
# push! is used to connect layers to a network
function buildNeuralNetwork(input_dim::Int, layer_dim::Int)
    layers = []
    for _ in 1:layer_dim
        push!(layers, Dense(input_dim, 20, relu; init = Flux.glorot_uniform))
        BatchNorm(20)
        push!(layers, Dense(20, 20, relu; init = Flux.glorot_uniform))
        input_dim = 20 
    end
    push!(layers, Dense(20, 1; init = Flux.glorot_uniform)) 
    return Chain(layers...)
end

# Choose network and Optimizer
NNmodel = buildNeuralNetwork(2,2)
optimizer = ADAM(0.1)


# BSDE solver parameters
S = 1024 # number of simulations
T = 10.0  # maturity
h = 0.25  # step size
N = Int(T / h) # effective time steps

# discretized BSDE starting at final value uN calculated as optionValue()
function backwardBSDEsolver(
    network,
    uN,
    inkr,
    X
    )
 
    # u = zeros(S, N + 1)
    # u[:, N + 1] = uN

    for tindex in N:-1:1
        t = tindex / T
        argument = vcat( [t for t in 1:S]',  X[1, :, tindex]')
        f = vec(network(argument))
        # u[:, tindex] .= u[:, tindex + 1] .- f .* sigmat .* inkr[:, tindex + 1]
        uN = uN .- f .* inkr[:, tindex + 1]
    end

    # return u[:, 1]   
    return uN

end

# penalty function based on the assumption, that the option value to time 0
# should be the same across all sample paths. which means that u(0,X0)=E[û(0,X0)]
# as a result one can try to minimize (u(0,X0)-E[û(0,X0)])^2 which is the basis for
# the loss function.
function penalty(
    network,
    uN,
    inkr,
    X
    )

    u0 = backwardBSDEsolver(network, uN, inkr, X)
    mw = sum(u0) / S
    emp_variance = 1 / S * sum((u0 .- mw).^2)

    return emp_variance

end

# simulatied state variable
X = sim.X


# training data as starting point for optimization, might have to be adjusted
# swaption has expiration time 2.0. Therefore uN is beeing assumed to be time 2.
uN = optionValue(swpt, 1.99999) 
X_train = uN
Y_train = uN
data = [(X_train, Y_train)]

inkr = randn(S, N + 1) * sqrt(h)
u0 = backwardBSDEsolver(NNmodel, uN, inkr, X)
mean(optionValue(swpt, 1.99999))


# training process
print("mean before training:", mean(u0)) 
print("std before training:", std(u0))

epochs = 100
for epoch in 1:epochs
    Flux.train!((x,y) -> penalty(NNmodel, x, inkr, X) , Flux.params(NNmodel), data, optimizer)
end
u0 = backwardBSDEsolver(NNmodel, uN, inkr, X)

print("mean after training:", mean(u0)) 
print("std after training:", std(u0))

