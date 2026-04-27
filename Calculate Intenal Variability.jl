using CSV
using DataFrames
using CairoMakie
using LaTeXStrings
using Smoothing
using RollingFunctions
using Statistics
using ColorSchemes

#Initialisation
cd(@__DIR__)

function movingaverage(X::Vector, window_size::Int)
    half_window = floor(Int, window_size / 2)
    len = length(X)
    Y = similar(X)
    σ_Y = similar(X)
    
    # Pad the vector to handle the edges
    padded_X = vcat(fill(X[1], half_window), X, fill(X[end], half_window))
    
    for n in 1:len
        lo = n
        hi = n + window_size - 1
        window = padded_X[lo:hi]
        Y[n] = sum(skipmissing(window))/window_size
    end
    
    return Y
end

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

        if maximum(df.Surface_Temperature) > 100 && minimum(df.Year) > 1860 && issubset(1851:2014, unique(df.Year))
            println("Dropping $(model) $(simulation) since its piControl run has a maximum value of $(maximum(df.Surface_Temperature))")
            continue
        end

        if simulation == "r1i1p1f1"
            model_dict[model] = df
        elseif !haskey(model_dict, model)
            model_dict[model] = df
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

# Load Data: 
piControl_directory = "data/ESM_piControl_runs"
onepctCO2_directory = "data/ESM_1pctCO2_runs"
historical_directory = "data/ESM_historical_runs/ssp245"

# Initialise Series of DataFrames
piControldf = []
onepctCO2df = []
historicaldf = []

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
for file in readdir(historical_directory)
    if file != ".DS_Store"
        csv_file = CSV.File("$historical_directory/$file")
        df = DataFrame(csv_file)

        # Add a new column called 'Model', which is populated with the model name extracted from the filename
        model = extract_model_and_simulation(file)[1]
        simulation = extract_model_and_simulation(file)[2]

        df[!, :Model] .= model
        df[!, :Simulation] .= simulation

        push!(historicaldf, df)
    end
end

# Filter the piControl and onepctCO2 dataframes
piControldf = filter_models(piControldf)
onepctCO2df = filter_models(onepctCO2df)
historicaldf = filter_models(historicaldf)

window_size = 11

pi_control_variabilities = DataFrame(
    Model = String[],
    Variability = Float64[],
    Detrended_Variability = Float64[]
)

historical_variabilities = DataFrame(
    Model = String[],
    Variability = Float64[],
    Detrended_Variability = Float64[]
)

piControldf_filtered = []
historicaldf_filtered = []

for df in piControldf
    df_filtered = filter(row -> 1850 <= row.Year <= 2350, df)
    push!(piControldf_filtered, df_filtered)
end

for df in historicaldf
    df_filtered = filter(row -> 1850 <= row.Year <= 2350, df)
    push!(historicaldf_filtered, df_filtered)
end

#Calculate Internal Variability of the piControl Runs:
for df in piControldf_filtered

    df[:, :Smoothed_Surface_Temperature] = movingaverage(df.Surface_Temperature, window_size)
    df[:, :Smoothed_Residuals] = df[:, :Surface_Temperature] - df[:, :Smoothed_Surface_Temperature]
    
    variability = std(df[:, :Surface_Temperature], corrected = false) #/ sqrt(window_size)
    det_variability = std(df[:, :Smoothed_Residuals], corrected = false) #/ sqrt(window_size) #Non-Detrended Variability

    push!(pi_control_variabilities, ("$(df[:, :Model][1])", variability, det_variability))
end

#Calculate Internal Variability of the piControl Runs:
for df in historicaldf_filtered

    df[:, :Smoothed_Surface_Temperature] = movingaverage(df.Surface_Temperature, window_size)

    df[:, :Smoothed_Residuals] = df[:, :Surface_Temperature] - df[:, :Smoothed_Surface_Temperature]
    
    variability = std(df[:, :Surface_Temperature], corrected = false) #/ sqrt(window_size)
    det_variability = std(df[:, :Smoothed_Residuals], corrected = false) #/ sqrt(window_size) #Non-Detrended Variability

    push!(historical_variabilities, ("$(df[:, :Model][1])", variability, det_variability))
end

# Find models present in both piControl and historical datasets
common_models = intersect(pi_control_variabilities.Model, historical_variabilities.Model)

# Filter both DataFrames to keep only the common models
pi_control_variabilities = filter(row -> row.Model in common_models, pi_control_variabilities)
historical_variabilities = filter(row -> row.Model in common_models, historical_variabilities)

# Join the two DataFrames by Model
variability_df = outerjoin(
    pi_control_variabilities, 
    historical_variabilities,
    on = :Model, 
    makeunique=true
)

# Optionally, rename the columns for clarity
rename!(variability_df, Dict(
    :Variability => :PiControl_Variability, 
    :Detrended_Variability => :PiControl_Det_Variability, 
    :Variability_1 => :Historical_Variability,
    :Detrended_Variability_1 => :Historical_Det_Variability
))

println(maximum(variability_df[!, :PiControl_Variability]./ variability_df[!, :Historical_Det_Variability]))


f = Figure(size= (1000,1350))
ax1 = Axis(f[1:2, 1:3], xlabel = "Year", ylabel = "Surfacte Temp [˚C]", title = "piControl Time Series", limits = (1850, 2100, 11, 16),)
ax2 = Axis(f[3:4, 1:3], xlabel = "Year", ylabel = "Surfacte Temp [˚C]", title = "Historical + SSP245 Time Series", limits = (1850, 2100, 12, 18),)

for (i, df) in enumerate(piControldf_filtered)
    i == 41 && continue
    model_name = df[:, :Model][1]
    lines!(ax1, df[:, :Year], df[:, :Surface_Temperature], label = model_name)
end



for df in historicaldf_filtered
    model_name = df[:, :Model][1] 
    lines!(ax2, df[:, :Year], df[:, :Surface_Temperature], label = model_name  
    )
end

limits = (0.04, 0.25, 0, 20)
scatter_limits = (0.0, 0.20, 0.0, 0.20)


# Combine all data that go into the histograms
all_vals = vcat(
    variability_df.PiControl_Variability,
    variability_df.PiControl_Det_Variability,
    variability_df.Historical_Variability,
    variability_df.Historical_Det_Variability
)

# Choose a fixed bin width
binwidth = 0.025  # adjust as needed

# Compute global min/max
xmin, xmax = extrema(all_vals)

# Construct fixed bin edges
bin_edges = collect(xmin:binwidth:(xmax + binwidth))  # +binwidth ensures last bin closes
bin_edges_big = 5*collect(xmin:binwidth:(xmax + binwidth))  # +binwidth ensures last bin closes

ax3 = Axis(f[1, 4], xlabel = "σ",  title = "piControl Variability (Raw)", limits = limits)
hist!(ax3, variability_df.PiControl_Variability, bins = bin_edges)

ax3b = Axis(f[2, 4], xlabel = "σ",  title = "piControl Variability (Detrended)", limits = limits)
hist!(ax3b, variability_df.PiControl_Det_Variability, bins = bin_edges)

ax4 = Axis(f[3, 4], xlabel = "σ",  title = "Historical Variability (Raw)", limits = (0.5, 1.7, 0, 10))
hist!(ax4, variability_df.Historical_Variability, bins = bin_edges_big)

ax4 = Axis(f[4, 4], xlabel = "σ",  title = "Historical Variability (Detrended)", limits = limits)
hist!(ax4, variability_df.Historical_Det_Variability, bins = bin_edges)
#Add 1:1 Line!!

ax5 = Axis(f[5:6, 1:2], xlabel = "piControl Variability (Detrended)", ylabel = "Historical Variability (Detrended)",  title = "Comparison of piControl Variability (Non-Detrended) \n and Historical Variability (Detrended)", aspect = 1, limits = scatter_limits)
scatter!(ax5, variability_df.PiControl_Variability, variability_df.Historical_Det_Variability)
lines!(ax5, 0.01:0.01:0.20, 0.01:0.01:0.20, linestyle = :dash, label ="1:1 Line")
axislegend(ax5)

ax6 = Axis(f[5:6, 3:4], xlabel = "piControl Variability (Detrended)", ylabel = "Historical Variability (Detrended)",  title = "Comparison of piControl Variability (Detrended) \n and Historical Variability (Detrended)", aspect = 1, limits = scatter_limits)
scatter!(ax6, variability_df.PiControl_Det_Variability, variability_df.Historical_Det_Variability)
lines!(ax6, 0.01:0.01:0.20, 0.01:0.01:0.20, linestyle = :dash, label ="1:1 Line")
axislegend(ax6)

f

save("output/Internal_Variability.pdf", f)