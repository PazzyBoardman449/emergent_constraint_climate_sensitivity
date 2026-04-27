using CSV
using DataFrames
using Plots
using LaTeXStrings
using Smoothing
using RollingFunctions
using Statistics

#Initialisation
cd(@__DIR__)

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

# Define a function which extracts the model name and simulation conditions from the filename:
function extract_model_and_simulation(filename)
    split_filename = split(basename(filename), "_")

    # Extract Model:
    model = split_filename[4]

    # Remove the date range of format YYYYMM-YYYYMM from model
    model = replace(model, r"[0-9]{6}-[0-9]{6}" => "")

    # Extract Simulation:
    simulation = split(split_filename[end], ".")[1]

    return model, simulation
end

# Define a function that gets the model and simulation conditions for a given DataFrame
function get_model_and_simulation_key(df)
    model = df[1, :Model]
    simulation_conditions = df[1, :Simulation]
    return model, simulation_conditions
end

# Filter models to select r1i1p1f1 if available, otherwise select the first available one
function filter_models(dataframes)
    model_dict = Dict{String, DataFrame}()

    for df in dataframes
        model = df.Model[1]

        simulation = df.Simulation[1]

        if simulation == "r1i1p1f1"
            model_dict[model] = df
        elseif !haskey(model_dict, model)
            model_dict[model] = df
        else 
            println("Model-Simulation $(model) $(simulation) does not have matching key")
        end
    end

    return collect(values(model_dict))
end

# Define a function that can calculate the mean temperature over a given time period
function calculate_mean_temperature(df, start_year, end_year, window_size)
    # Calculate the mean temperature over the given time period using the smoothed temperatures
    mean_temperature = mean(df[(df[:, :Year] .>= start_year) .& (df[:, :Year] .<= end_year), :Surface_Temperature])
    return mean_temperature
end

# Define a function that can calculate the Transient Climate Response (TCR) between the piControl and onepctCO2 data.
function calculate_tcr(control_data, one_pct_data, start_year, end_year)

    # Create dictionaries to match DataFrames by (Model, Simulation Conditions) key
    control_dict = Dict(get_model_and_simulation_key(df) => df for df in control_data)
    one_pct_dict = Dict(get_model_and_simulation_key(df) => df for df in one_pct_data)
 
    # Ensure that we only calculate TCR for matching pairs of DataFrames
    common_keys = intersect(keys(control_dict), keys(one_pct_dict))
 
    tcr_results = DataFrame(Model = String[], Simulation_Conditions = String[], TCR = Float64[])
 
    for key in common_keys
        
        # Retrieve the control and one percent DataFrames for the matching key
        control_df = control_dict[key]
        one_pct_df = one_pct_dict[key]
 
        # Calculate the 11-year mean for the control and one percent CO₂ data between years 60-80
        control_mean_temp = calculate_mean_temperature(control_df, start_year, end_year, 1)
        one_pct_mean_temp = calculate_mean_temperature(one_pct_df, start_year, end_year, 1)
 
        # Calculate the TCR by subtracting the control mean temperature from the one percent CO₂ mean temperature
        tcr_value = one_pct_mean_temp - control_mean_temp
 
        # Filter out anomalously large TCR values
        if abs(tcr_value) < 10
            # Append results to the DataFrame
            push!(tcr_results, (Model = key[1], Simulation_Conditions = key[2], TCR = tcr_value))
        else
            println("Anomalous TCR Value Detected for $(key[1]) $(key[2]): $tcr_value")
        end
    end
 
    return tcr_results
end

# Load Data: 
piControl_directory = "data/ESM_piControl_runs"
onepctCO2_directory = "data/ESM_1pctCO2_runs"

# Initialise Series of DataFrames
piControldf = []
onepctCO2df = []

# Store each file in a DataFrame, and create a 3 dimensional array of DataFrames
for file in readdir(piControl_directory)
    if file != ".DS_Store"
        csv_file = CSV.File("$piControl_directory/$file")
        df = DataFrame(csv_file)

        # Add a new column called 'Model', which is populated with the model name extracted from the filename
        model = extract_model_and_simulation(file)[1]
        simulation = extract_model_and_simulation(file)[2]

        df[!, :Model] .= model
        df[!, :Simulation] .= simulation

        push!(piControldf, df)
    end
end

# Store each file in a DataFrame, and create a 3 dimensional array of DataFrames
for file in readdir(onepctCO2_directory)
    if file != ".DS_Store"
        csv_file = CSV.File("$onepctCO2_directory/$file")
        df = DataFrame(csv_file)

        # Add a new column called 'Model', which is populated with the model name extracted from the filename
        model = extract_model_and_simulation(file)[1]
        simulation = extract_model_and_simulation(file)[2]

        df[!, :Model] .= model
        df[!, :Simulation] .= simulation

        push!(onepctCO2df, df)
    end
end

# Filter the piControl and onepctCO2 dataframes
piControldf = filter_models(piControldf)
onepctCO2df = filter_models(onepctCO2df)

# Initialise a DataFrame to Store the Results
tcr_results = calculate_tcr(piControldf, onepctCO2df, 1910, 1930)

# Neaten the DataFrame
sort!(tcr_results, :Model)

# Plot the TCR results as a histogram: (Presentation Plot)
histogram(tcr_results.TCR, title="TCR Values from ESM", bins = 10, alpha =:0.4, grid =:false, xlabel="TCR [°C]", ylabel="Frequency", label="", legend=:topright, color=:blue, marker=:o, markersize=5)

# Export the TCR Results to a CSV File
CSV.write("output/TCR_Own_Results.csv", tcr_results)