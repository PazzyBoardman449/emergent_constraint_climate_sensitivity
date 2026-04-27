## Load in Packages
using CSV
using DataFrames
using Plots
using LaTeXStrings
using Smoothing
using RollingFunctions
using Statistics
using GLM

cd(@__DIR__)

# Define a function which extracts the model name and simulation conditions from the filename:
function extract_model_and_simulation(filename)

    split_filename = split(basename(filename), "_")

    # Extract Model:
    model = join(split_filename[3:4], "_")

    # Remove the date range of format YYYYMM-YYYYMM from model
    model = replace(model, r"[0-9]{6}-[0-9]{6}" => "")

    # Extract Simulation:
    simulation = split(split_filename[end], ".")[1]

    return model, simulation
end

# Filter out Model runs which don't have a TOA Flux column
function remove_dfs_without_column!(dfs, column)
    for i in length(dfs):-1:1
        if !(column in Symbol.(names(dfs[i])))
            #println("Deleting DataFrame $i")
            deleteat!(dfs, i)
        end
    end
end

#Filter out Model Runs with anomalously TOA Flux values
function remove_dfs_with_anomalously_high_TOA_Flux!(dfs)
    for i in length(dfs):-1:1
        if maximum(dfs[i][!, :Net_TOA_Flux]) > 10 || minimum(dfs[i][!, :Net_TOA_Flux]) < -10
            println(maximum(dfs[i][!, :Net_TOA_Flux]))
            println("Deleting DataFrame $i")
            deleteat!(dfs, i)
        end
    end
end

#Filter out Model runs which are too short:
function remove_dfs_with_too_short_a_length!(dfs, min_length)
    for i in length(dfs):-1:1
        if size(dfs[i])[1] < min_length
            println("Deleting DataFrame $i")
            deleteat!(dfs, i)
        end
    end
end

# Define a function that gets the model and simulation conditions for a given DataFrame
function get_model_and_simulation_key(df)
    model = df[1, :Model]
    simulation_conditions = df[1, :Simulation]
    return model, simulation_conditions
end

# Define a function that can calculate the ECS Values for a given DataFrame
function calculate_ECS(control_data, abrupt4xCO2_data, start_year, end_year)

    # Find common dates
    common_dates = intersect(control_data[!, :Year], abrupt4xCO2_data[!, :Year])

    # Filter both DataFrames to only include common dates
    control_filtered = filter(row -> row[:Year] in common_dates, control_data)
    abrupt4xCO2_filtered = filter(row -> row[:Year] in common_dates, abrupt4xCO2_data)

    # Ensure that the filtered DataFrames have the same length
    if nrow(control_filtered) != nrow(abrupt4xCO2_filtered)
        println("Filtered DataFrames have different lengths")
        return NaN, NaN, NaN
    end

    # Calculate N_TOA anomaly (4xCO2 minus control)
    ntoa_anomaly = abrupt4xCO2_filtered[!, :Net_TOA_Flux] .- control_filtered[!, :Net_TOA_Flux]

    # Restrict to the time period defined by start_year and end_year
    valid_indices = (control_filtered[!, :Year] .>= start_year) .& (control_filtered[!, :Year] .<= end_year)
    ntoa_anomaly = ntoa_anomaly[valid_indices]
    delta_t = abrupt4xCO2_filtered[valid_indices, :Surface_Temperature] .- control_filtered[valid_indices, :Surface_Temperature]

    # Create a DataFrame for the regression
    regression_df = DataFrame(ntoa_anomaly = ntoa_anomaly, delta_t = delta_t)

    # Perform linear regression (Gregory method) on Delta T vs. N_TOA anomaly
    reg = lm(@formula(ntoa_anomaly ~ delta_t), regression_df)

    # Calculate the feedback parameter λ = -slope
    λ = -coef(reg)[2]

    # Calculate the forcing Q_2x = -intercept
    q_2x = 0.5*coef(reg)[1]

    #Calculate ECS based on Gregory method ECS = -0.5 * intercept / slope
    ecs = -0.5 * coef(reg)[1] / coef(reg)[2]

    # Return the ECS value, slope, intercept, and R^2 value
    return λ, q_2x, ecs
end

# Define a function that loops through all the DataFrames and calculates the ECS values
function calculate_ECS_for_all_dfs(control_data, abrupt4xCO2_data, start_year, end_year)
    
    # Create dictionaries to match DataFrames by (Model, Simulation Conditions) key
    control_dict = Dict(get_model_and_simulation_key(df) => df for df in control_data)
    abrupt4xCO2_dict = Dict(get_model_and_simulation_key(df) => df for df in abrupt4xCO2_data)

    # Ensure that we only calculate ECS for matching pairs of DataFrames
    common_keys = intersect(keys(control_dict), keys(abrupt4xCO2_dict))

    # Create a DataFrame to store the ECS and λ values
    ecs_df = DataFrame(Model = String[], Simulation = String[], λ = Float64[], λ_inv = Float64[], Q_2x = Float64[], ECS = Float64[])

    for key in common_keys
        λ, Q_2x, ecs = calculate_ECS(control_dict[key], abrupt4xCO2_dict[key], start_year, end_year)
        push!(ecs_df, (Model = key[1], Simulation = key[2], λ = λ, λ_inv = λ^-1, Q_2x = Q_2x, ECS = ecs))
    end

    #Filter out values which are NaN
    ecs_df = ecs_df[.!isnan.(ecs_df[!, :ECS]), :]

    return ecs_df

end

# Define a function to count the number of '1's in the Simulation_Conditions column
function count_ones(simulation_conditions)
    return count(x -> x == '1', simulation_conditions)
end

#Load in Data
piControl_directory = "data/ESM_piControl_runs"
abrupt4xCO2_directory = "data/ESM_abrupt4xCO2_runs"

#Initialise Series of DataFrames
piControldf = []
abrupt4xCO2df = []

#Store each file in a DataFrame, and create a 3 dimensional array of DataFrames
for file in readdir(piControl_directory)
    if file != ".DS_Store"
        csv_file = CSV.File("$piControl_directory/$file")
        df = DataFrame(csv_file)

        #Add a new column called 'Model, which is populated with the model name extracted from the filename
        model = extract_model_and_simulation(file)[1]
        simulation = extract_model_and_simulation(file)[2]

        df[!, :Model] .= model
        df[!, :Simulation] .= simulation

        push!(piControldf, df)
    end
end

#Store each file in a DataFrame, and create a 3 dimensional array of DataFrames
for file in readdir(abrupt4xCO2_directory)
    if file != ".DS_Store"
        csv_file = CSV.File("$abrupt4xCO2_directory/$file")
        df = DataFrame(csv_file)

        #Add a new column called 'Model, which is populated with the model name extracted from the filename
        model = extract_model_and_simulation(file)[1]
        simulation = extract_model_and_simulation(file)[2]

        df[!, :Model] .= model
        df[!, :Simulation] .= simulation

        push!(abrupt4xCO2df, df)
    end
end

remove_dfs_without_column!(piControldf, :Net_TOA_Flux)
remove_dfs_with_anomalously_high_TOA_Flux!(piControldf)
remove_dfs_with_too_short_a_length!(piControldf, 100)

remove_dfs_without_column!(abrupt4xCO2df, :Net_TOA_Flux)
remove_dfs_with_anomalously_high_TOA_Flux!(abrupt4xCO2df)
remove_dfs_with_too_short_a_length!(abrupt4xCO2df, 100)

# Create a DataFrame to store the λ, q_2x and ECS values
ecs_results = calculate_ECS_for_all_dfs(piControldf, abrupt4xCO2df, 1854, 2000)

# Add a new column to tcr_results with the count of '1's in the Simulation
ecs_results[!, :OnesCount] = count_ones.(ecs_results[:, :Simulation])

# Group by Model and select the row with the maximum count of '1's in the Simulation
ecs_results_grouped = combine(groupby(ecs_results, [:Model]), :OnesCount => maximum => :OnesCount)

# Join the grouped results back with the original tcr_results to get the corresponding TCR values
ecs_results_grouped = innerjoin(ecs_results_grouped, ecs_results, on = [:Model, :OnesCount])

#Neaten the DataFram up a lil:
select!(ecs_results_grouped, Not(:OnesCount))
select!(ecs_results_grouped, Not(:Simulation))
sort!(ecs_results_grouped, :Model)

CSV.write("output/ECS_Own_Results.csv", ecs_results_grouped)