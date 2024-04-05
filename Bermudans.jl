using Random, Optim, Distributions, DiffFusion, Plots, Interpolations, Flux, LinearAlgebra, Zygote

# Gaussian HJM-Modell
function createModel()
    ch = DiffFusion.correlation_holder("");
    δ = DiffFusion.flat_parameter([ 0., ]);
    χ = DiffFusion.flat_parameter([ 0.01, ]);
    timesVol = [  1.,  2.,  5., 10. ]
    valuesVol = [ 50.,  60.,  70.,  80., ]' * 1.0e-4 
    σ = DiffFusion.backward_flat_volatility("", timesVol, valuesVol);

    model = DiffFusion.gaussian_hjm_model("md/EUR", δ, χ, σ, ch, nothing);
    return model
end





# simulation: state variable sim.X[1,:,:] and integrated state variable sim.X[2,:,:]
# second argument for number of paths (number_of_paths), third argument for simulation 
# time steps of the simulation grid (simulation_grid) additionally create brownian 
# increments which are created by the same brownian motion for all following methods 
# (assumption of the method)
function sobol_brownian_increments(
    n_states::Int,
    n_paths::Int,
    n_times::Int,  # without zero
    )


    seq = DiffFusion.SobolSeq(n_states * n_times)
    seq = skip(seq, n_paths)
    U = zeros(n_states, n_paths, n_times)
    for p in 1:n_paths
        U[:,p,:] = reshape(DiffFusion.next!(seq), (n_states,n_times))
    end
    m = mean(U, dims=2)
    U = U .- reshape(m, (n_states,1,n_times)) .+ 0.5
    return quantile.(Normal(), U)
end



function createSimulation()
    model = createModel()
    n_times = 0.0:0.25:10.0
    n_paths = 2^10

    sim = DiffFusion.simple_simulation(
        model,
        model.correlation_holder,
        n_times,
        n_paths,
        with_progress_bar = true,
        brownian_increments = sobol_brownian_increments,
    );

    return sim, sim.X
end

sim, X = createSimulation()




function createPath()
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


    return DiffFusion.path(sim, ts_list, context, DiffFusion.LinearPathInterpolation)
end

path = createPath()



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
    

fixed_notionals = 10_000.00 * ones(length(fixed_flows))
fixed_leg = DiffFusion.cashflow_leg(
    "leg/1", fixed_flows, fixed_notionals, "EUR:ESTR", nothing,  1.0,
)

libor_notionals = 10_000.00 * ones(length(libor_flows))
libor_leg = DiffFusion.cashflow_leg(
    "leg/2", libor_flows, libor_notionals, "EUR:ESTR", nothing,  -1.0
);


make_regression_variables(t) = [ DiffFusion.LiborRate(t, t, 10.0, "EUR:EURIBOR6M"), ]

swap_2y_10y = [
    DiffFusion.cashflow_leg("leg/fixed/2y-10y",fixed_flows[3:end], fixed_notionals[3:end], "EUR:ESTR", nothing,  1.0),  # receiver
    DiffFusion.cashflow_leg("leg/libor/2y-10y",libor_flows[5:end], libor_notionals[5:end], "EUR:ESTR", nothing, -1.0),  # payer
]


swap_4y_10y = [
    DiffFusion.cashflow_leg("leg/fixed/4y-10y",fixed_flows[5:end], fixed_notionals[5:end], "EUR:ESTR", nothing,  1.0),  # receiver
    DiffFusion.cashflow_leg("leg/libor/4y-10y",libor_flows[9:end], libor_notionals[9:end], "EUR:ESTR", nothing, -1.0),  # payer
]

swap_6y_10y = [
    DiffFusion.cashflow_leg("leg/fixed/6y-10y",fixed_flows[7:end], fixed_notionals[7:end], "EUR:ESTR", nothing,  1.0),  # receiver
    DiffFusion.cashflow_leg("leg/libor/6y-10y",libor_flows[13:end], libor_notionals[13:end], "EUR:ESTR", nothing, -1.0),  # payer
]

swap_8y_10y = [
    DiffFusion.cashflow_leg("leg/fixed/6y-10y",fixed_flows[9:end], fixed_notionals[9:end], "EUR:ESTR", nothing,  1.0),  # receiver
    DiffFusion.cashflow_leg("leg/libor/6y-10y",libor_flows[17:end], libor_notionals[17:end], "EUR:ESTR", nothing, -1.0),  # payer
];
    

exercise_2y = DiffFusion.bermudan_exercise(2.0, swap_2y_10y, make_regression_variables)
exercise_4y = DiffFusion.bermudan_exercise(4.0, swap_4y_10y, make_regression_variables)
exercise_6y = DiffFusion.bermudan_exercise(6.0, swap_6y_10y, make_regression_variables)
exercise_8y = DiffFusion.bermudan_exercise(8.0, swap_8y_10y, make_regression_variables);

bermudanExercises = [ exercise_2y, exercise_4y, exercise_6y, exercise_8y, ]



berm = DiffFusion.bermudan_swaption_leg(
    "berm/10-nc-2",
    bermudanExercises,
    # [ exercise_2y, ],
    1.0, # long option
    "", # default discounting (curve key)
    make_regression_variables,
    nothing, # path
    nothing, # make_regression
);




#option value to time obs_time over all paths of a swaption leg
function optionValue(
    leg, 
    obs_time
    )

    swaptionPayoff = DiffFusion.discounted_cashflows(leg, obs_time)[1]
    values = DiffFusion.at(swaptionPayoff, path)
    return values

end

optionValue(berm, 0)






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







# discretized BSDE starting at final value uN calculated as optionValue()
function backwardBSDEsolver(
    network,
    uN,
    brownian_increments,
    X
    )

    # X = sim.X                              # Simulated state and integrated state variable
    # n_paths = size(X[1,:,1])[1]            # number of simulations
    # T = sim.times[end]                     # maturity
    # h = sim.times[2] -sim.times[1]         # step size (if equidistant)
    # N = Int(T / h)                         # effective time steps
    # inkr = randn(n_paths, N + 1) * sqrt(h) # Increments for simulations

    
    for tindex in N:-1:1
        t = tindex / T
        argument = vcat( [t for t in 1:n_paths]',  X[1, :, tindex]')
        f = vec(network(argument))
        uN = uN .- f .* brownian_increments[1,:, tindex + 1]
    end

    return uN

end


# penalty function based on the assumption, that the option value to time 0
# should be the same across all sample paths. which means that u(0,X0)=E[û(0,X0)]
# as a result one can try to minimize (u(0,X0)-E[û(0,X0)])^2 which is the basis for
# the loss function.
function penalty(
    network,
    uN,
    brownian_increments,
    X
    )

    n_paths = size(X[1,:,1])[1]
    u0 = backwardBSDEsolver(network, uN, brownian_increments, X)

    empirical_variance = sum((u0 .- mean(u0)).^2) / n_paths

    return empirical_variance

end


model = createModel()
n_paths = size(X[1,:,1])[1]
n_times = sim.times  
# h = sim.times[2] -sim.times[1] # step size (if equidistant)
# N = Int(T / h) # eff

inkr = sobol_brownian_increments(
    length(DiffFusion.state_alias(model)),
    n_paths,
    length(n_times),
)

   

    X = sim.X                              # Simulated state and integrated state variable
    n_paths = size(X[1,:,1])[1]            # number of simulations
    T = sim.times[end]                     # maturity
    h = sim.times[2] -sim.times[1]         # step size (if equidistant)
    N = Int(T / h)                         # effective time steps
    inkr = randn(n_paths, N + 1) * sqrt(h) # Increments for simulations


# training data as starting point for optimization, might have to be adjusted
# swaption has expiration time 2.0. Therefore uN is beeing assumed to be time 2.
function trainNetwork(
    )
    
    model = createModel()
    n_paths = size(X[1,:,1])[1]
    n_times = sim.times  
    # h = sim.times[2] -sim.times[1] # step size (if equidistant)
    # N = Int(T / h) # eff

    inkr = sobol_brownian_increments(
        length(DiffFusion.state_alias(model)),
        n_paths,
        length(n_times),
    )

    uN = optionValue(berm, 9.99999) 
    X_train = uN
    Y_train = uN
    data = [(X_train, Y_train)]
    NNmodel = buildNeuralNetwork(2,2)
    optimizer = ADAM(0.1)

    u0_beforeTraining = backwardBSDEsolver(NNmodel, uN, inkr, X)

    epochs = 100
    for epoch in 1:epochs
        Flux.train!((x,y) -> penalty(NNmodel, x, inkr, X) , Flux.params(NNmodel), data, optimizer)
    end

    u0_afterTraining = backwardBSDEsolver(NNmodel, uN, inkr, X)
    
    return u0_beforeTraining, u0_afterTraining
end

function Result()
    u0_beforeTraining, u0_afterTraining = trainNetwork()
    println("mean DiffFusion price: ", mean(optionValue(berm, 0)))
    println("mean before training: ", mean(u0_beforeTraining))
    println("std before training: ", std(u0_beforeTraining))
    println("mean after training: ", mean(u0_afterTraining))
    println("std after training: ", std(u0_afterTraining))
end


Resulting = Result()





# some data types used in the DiffFusion package
"""
ModelTime: A type alias for variables representing time.
ModelValue: A type alias for variables representing modelled quantities.
BermudanExercise:
CashFlowLeg:
alias(leg::CashFlowLeg): Return the leg alias
BermudanSwaptionLeg: of type CashFlowLeg, creates Leg objects for Bermudans which are inputs
for the next method BSDE_cashflows, where the Legs are combined to a Pay(berm, obs_time) object,
This object represents the Value process of the Bermudan and can by evaluated 

"""
# ModelTime = Number

# ModelValue = Number

# struct BermudanExercise
#     exercise_time::ModelTime
#     cashflow_legs::AbstractVector 
#     # make_regression_variables::Function
# end

# abstract type CashFlowLeg end

# alias(leg::CashFlowLeg) = leg.alias

# struct BermudanSwaptionLeg <: CashFlowLeg
#     alias::String
#     bermudan_exercises::AbstractVector
#     option_long_short::Int 
#     numeraire_curve_key::String
#     hold_values::AbstractVector 
#     exercise_triggers::AbstractVector  
# end



function bermudan_swaption_leg(
    alias::String,
    bermudan_exercises::AbstractVector,
    option_long_short, #::ModelValue
    numeraire_curve_key::String,
    # path, ::Union{AbstractPath, Nothing}
    # make_regression::Union{Function, Nothing},
    regression_on_exercise_trigger = true,
    )
    #
    @assert length(bermudan_exercises) > 0
    exercise_times = [e.exercise_time for e in bermudan_exercises]
    if length(exercise_times) > 1
        @assert exercise_times[begin+1:end] > exercise_times[begin:end-1]
    end

    # Initialize BSDE solver
    NNmodel, optimizer = initialize_bsde_solver()

    # Backward induction algorithm
    #
    # Handle last exercise specifically
    Hk = Fixed(0.0)  # Assume hold value after last exercise is 0
    Uk = sum([discounted_cashflows(leg, bermudan_exercises[end].exercise_time)
              for leg in bermudan_exercises[end].cashflow_legs])  # Using the adjusted discounted_cashflows
    hold_values = [Max(Hk, Uk)]  # Initial hold value
    exercise_triggers = [Hk > Uk]  # Initial exercise trigger

    # Backward sweep for each exercise date
    for ex in reverse(bermudan_exercises[begin:end-1])
        Uk = sum([discounted_cashflows(leg, ex.exercise_time)
                  for leg in ex.cashflow_legs])  # Adjusted to use discounted_cashflows
        
        # Calculate Hk using the backwardBSDEsolver
        uN = optionValue(ex, ex.exercise_time)  
        Hk = backwardBSDEsolver(NNmodel, uN, inkr, X)  
        
        # Determine if exercising is beneficial
        Ik = Hk > Uk
        
        push!(hold_values, Hk)
        push!(exercise_triggers, Ik)
    end

    #
    return BermudanSwaptionLeg(
        alias,
        bermudan_exercises,
        option_long_short,
        numeraire_curve_key,
        reverse(hold_values),  # in ascending order again
        reverse(exercise_triggers),
        # make_regression_variables,
        # path,
        # make_regression,
    )
end




function BSDE_cashflows(
    leg, #::BermudanSwaptionLeg, 
    obs_time #::ModelTime
    )

    # Initialization of BSDE solver parameters and model
    NNmodel = buildNeuralNetwork(2, 2) # Example neural network initialization
    optimizer = ADAM(0.1) # Optimizer
    X = sim.X # Simulated state and integrated state variable
    S = size(X[1,:,1])[1] # number of simulations
    T = sim.times[end]  # maturity
    h = sim.times[2] -sim.times[1] # step size (if equidistant)
    N = Int(T / h) # effective time steps
    inkr = randn(S, N + 1) * sqrt(h) # Increments for simulations


    if obs_time < leg.bermudan_exercises[begin].exercise_time
        uN = optionValue(leg, obs_time)
        Ht = backwardBSDEsolver(NNmodel, uN, inkr, X) # Ht now is characterized by the BSDE method
        return [ Ht ]
    end

    exercise_times = [ e.exercise_time for e in leg.bermudan_exercises ]
    last_exercise_idx = searchsortedlast(exercise_times, obs_time)
    @assert last_exercise_idx ≥ 1  # otherwise, we should have returned earlier

    Ht = Fixed(0.0)  # Option value after the last exercise is assumed to be 0
    if last_exercise_idx < length(exercise_times)
        uN = optionValue(leg, obs_time) # Compute option value just before the observation time
        Ht = backwardBSDEsolver(NNmodel, uN, inkr, X) # Solve BSDE backwards
    end

    # Process for computing the underlying if exercised
    Ut = vcat([
        discounted_cashflows(leg, obs_time)
        for leg in leg.bermudan_exercises[last_exercise_idx].cashflow_legs
    ]...)
    if length(Ut) > 0
        Ut = sum(Ut)
    else
        Ut = nothing
    end

    # Check for earlier exercise
    not_exercised_trigger = leg.exercise_triggers[begin]
    for trigger in leg.exercise_triggers[begin+1:last_exercise_idx]
        not_exercised_trigger = not_exercised_trigger * trigger
    end

    # Final payoff calculation considering the exercise decision
    if last_exercise_idx < length(exercise_times)
        @assert !isnothing(Ut)  # Avoid degenerated cases
        berm = (not_exercised_trigger * Ht + (1.0 - not_exercised_trigger) * Ut)
    else
        if !isnothing(Ut)
            berm = (1.0 - not_exercised_trigger) * Ut
        else
            return Payoff[]
        end
    end
    return [ Pay(berm, obs_time) ]
end



#Comparison
mean(optionValue(berm, 0))
mean(BSDE_cashflows(berm, 0)[1])

