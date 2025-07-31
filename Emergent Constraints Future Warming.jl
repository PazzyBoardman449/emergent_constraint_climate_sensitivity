## Load in Packages
using NCDatasets
using DataFrames
using Dates
using CFTime
using Glob
using CSV
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using Smoothing
using RollingFunctions
using Statistics
using StatsBase
using GLM
using Distributions
using MTH229
using SymPy
using LsqFit

cd(@__DIR__)

### Plot Settings:
default(fontfamily="Computer Modern", xlabelfontsize = 8, ylabelfontsize = 8, titlefontsize=12, legendfontsize = 8)

num_steps = 1000


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

# Function which extracts the model name and simulation conditions from the filename:
function extract_model_and_simulation(filename)
    split_filename = split(basename(filename), "_")

    # Extract Centre: 
    centre = split_filename[3]

    # Extract Model:
    model = split_filename[4]

    # Remove the date range of format YYYYMM-YYYYMM from model
    model = replace(model, r"[0-9]{6}-[0-9]{6}" => "")

    # Extract Simulation:
    simulation = split_filename[5]

    return model, simulation, centre
end

# Function to translate confidence interval to a standard deviation σ
function confidence_to_zscore(confidence_level)
    # Calculate the z-score for the given confidence level
    zscore = quantile(Normal(0, 1), 1 - (1 - confidence_level) / 2)
    return zscore
end

function calculate_distribution(Data, Obs_Data, params, s)
    
    # Parameters and input values
    nfitx = 1000  # Number of x values
    mfity = 1000  # Number of y values

    # Pre-allocate arrays
    Pxy = zeros(nfitx, mfity)
    Pyx = zeros(mfity, nfitx)
    Py = zeros(mfity)

    # Extract data and parameters
    ΔT = Data.ΔT
    ΔT_SSP = Data.ΔT_SSP
    N = length(ΔT)
 
    ΔT_Obs_Mean = Obs_Data[1]
    ΔT_Obs_Upper = Obs_Data[2]
    ΔT_Obs_Lower = Obs_Data[3]

    σ_ΔT_Obs = (ΔT_Obs_Upper - ΔT_Obs_Mean)

    # Define P_x(x)
    P_x(x) = pdf(Normal(ΔT_Obs_Mean, σ_ΔT_Obs), x)

    local f, σ_f
    f = x -> params[1] + params[2] * x
    σ_f = x ->  s * sqrt(1 + 1/N + ((x - mean(ΔT))^2 / (N*(std(ΔT))^2)))

    x_min, x_max = 0, 2 * mean(ΔT)
    y_min, y_max = 0, 2 * mean(ΔT_SSP)

    # Calculate dx and dy
    dx = (x_max - x_min) / nfitx
    dy = (y_max - y_min) / mfity

    # Generate x and y values
    x_values = range(x_min, stop=x_max, length=nfitx)
    y_values = range(y_min, stop=y_max, length=mfity)

    # Calculate Pxy, and Py
    for m in 1:mfity
        y = y_values[m]
        for n in 1:nfitx
            x = x_values[n]
            f_x = f(x)
            σ_f_x = σ_f(x)
            #P_y_given_x = pdf(Normal(f(x), σ_f_x), y)
            P_y_given_x = (1/(sqrt(2*pi*σ_f_x^2)))*exp(-(y - f_x)^2/(2*σ_f_x^2))
            Pxy[n, m] = P_x(x) * P_y_given_x
        end
        # Integrate over x to get Py
        Py[m] = sum(Pxy[:, m] * dx)
    end

    # Normalize Py
    Py_norm = sum(Py) * dy
    Py ./= Py_norm

    return y_values, Py
end

#= Stage 1. Function to produce a list of time series data for a given SSP run, smoothed by a given smoothing window. 
Each dataframe represents a model-simulation run combination. =#
function extract_data_for_ssp_run(directory_path, ssp_run)
    directory = "$directory_path/$(ssp_run)"

    # Initialise Series of DataFrames
    dataframes = []

    # Store each file in a DataFrame, and create a 3 dimensional array of DataFrames
    for file in readdir(directory)
        if file != ".DS_Store"
            csv_file = CSV.File("$directory/$file")
            df = DataFrame(csv_file)

            # Add a new column called 'Model', which is populated with the model name extracted from the filename
            model = extract_model_and_simulation(file)[1]
            simulation = extract_model_and_simulation(file)[2]
            centre = extract_model_and_simulation(file)[3]

            df[!, :Centre] .= centre
            df[!, :Model] .= model
            df[!, :Simulation] .= simulation

            push!(dataframes, df)
        end
    end

    # Create a dictionary to store the DataFrame with the required simulation conditions for each model
    model_dict = Dict{String, DataFrame}()

    # Add each DataFrame to the dictionary with the lowest alphanumeric Simulation value
    for df in dataframes
        model = df.Model[1]
        simulation = df.Simulation[1]
        
        if simulation == "r1i1p1f1"
            model_dict[model] = df
        elseif !haskey(model_dict, model)
            model_dict[model] = df
        end
    end
    
    # Convert the dictionary values back to a list of DataFrames
    dataframes = collect(values(model_dict))

    return dataframes
end

# Stage 2. Function to calculate the ΔT value from the list of dataframes, over a given start and end year. 
function calculate_ESM_warming_trend(dataframes, period, window_size, confidence)
    # Initialise a DataFrame to store the results
    ΔT_df= DataFrame(
    Centre = String[], 
    Model = String[], 
    Simulation_Run = String[], 
    T_Initial = Float64[], 
    T_Final = Float64[], 
    ΔT = Float64[],
    ΔT_per_decade = Float64[])

    start_index = findfirst(dataframes[1][:, :Year] .== period[1])
    end_index = findfirst(dataframes[1][:, :Year] .== period[2])

    # Check if the start and end indices are valid
    for i in 1:length(dataframes)
        dataframes[i][:, :Smoothed_Surface_Temperature] = movingaverage(dataframes[i][:, :Surface_Temperature], window_size)

        dataframes[i][:, :Smoothed_Residuals] = dataframes[i][:, :Surface_Temperature] - dataframes[i][:, :Smoothed_Surface_Temperature]
        dataframes[i][:, :Smoothed_Surface_Temperature_err] .= confidence_to_zscore(confidence)*std(dataframes[i][:, :Smoothed_Residuals], corrected = false) / sqrt(window_size)
    end 

    # Loop over each DataFrame and extract the change in temperature between the two years
    for i in 1:length(dataframes)
        if nrow(dataframes[i]) >= end_index 
            ΔT = dataframes[i].Smoothed_Surface_Temperature[end_index] - dataframes[i].Smoothed_Surface_Temperature[start_index]
            push!(ΔT_df, (
                    Centre = dataframes[i].Centre[1], 
                    Model = dataframes[i].Model[1], 
                    Simulation_Run = dataframes[i].Simulation[1], 
                    T_Initial = dataframes[i].Smoothed_Surface_Temperature[start_index], 
                    T_Final = dataframes[i].Smoothed_Surface_Temperature[end_index], 
                    ΔT = ΔT,
                    ΔT_per_decade = 10*(dataframes[i].Smoothed_Surface_Temperature[end_index] - dataframes[i].Smoothed_Surface_Temperature[start_index]) / (period[2] - period[1])
                    ))
        else
            println("Delta T Cannot be calculated for the ESM")
        end
    end

    #Sort the dataframe by alphabetical order of the model name:
    ΔT_df = sort(ΔT_df, [:Centre])

    return ΔT_df
end

# Stage 3. Function that calculates the observed ΔT using the Met Office HadCRUT5 data for a given start and end year, and smoothing period. 
function calculate_observed_warming_trend(obs_dataframe, period, window_size)

    #Smooth the Observed Data and the upper and lower limits
    obs_dataframe[!, :Smoothed_Temperature_Anomaly] = movingaverage(obs_dataframe[!, :Temperature_Anomaly], window_size)

    #Calculate the Internal variability
    obs_dataframe[!, :Resids] = obs_dataframe[!, :Temperature_Anomaly] - obs_dataframe[!, :Smoothed_Temperature_Anomaly]
    obs_dataframe[!, :Internal_Variability] .= std(obs_dataframe[!, :Resids], corrected = false) / sqrt(window_size)

    #Calculate the Measurement Error
    obs_dataframe[!, :Smoothed_Upper_Temperature_Anomaly] = movingaverage(obs_dataframe[!, :Upper_Temperature_Anomaly_Limit], window_size)
    obs_dataframe[!, :Smoothed_Lower_Temperature_Anomaly] = movingaverage(obs_dataframe[!, :Lower_Temperature_Anomaly_Limit], window_size)
    obs_dataframe[!, :Meas_Err] = (obs_dataframe[!, :Smoothed_Upper_Temperature_Anomaly] - obs_dataframe[!, :Smoothed_Lower_Temperature_Anomaly])/(2* confidence_to_zscore(0.95))

    #Calculate the Correction Due to Autocorrelation
    rhodum = cor(obs_dataframe[!, :Resids][1:end-1], obs_dataframe[!, :Resids][2:end])
    k = (1 + rhodum) / (1 - rhodum)
    N_eff = window_size / k

    obs_dataframe[!, :Meas_Err] = obs_dataframe[!, :Meas_Err]/sqrt(N_eff)

    #Calculate the Total Error
    obs_dataframe[!, :Tot_Err] = sqrt.(obs_dataframe[!, :Internal_Variability].^2 + obs_dataframe[!, :Meas_Err].^2)

    #Calculate the ΔT between 1980 and 2014
    start_index = findfirst(obs_dataframe[!, :Year] .== period[1])
    end_index = findfirst(obs_dataframe[!, :Year] .== period[2])

    #Calculate Central Value
    ΔT_Obs = obs_dataframe.Smoothed_Temperature_Anomaly[end_index] - obs_dataframe.Smoothed_Temperature_Anomaly[start_index]

    # Propagate errors correctly
    ΔT_Obs_Uncertainty = obs_dataframe.Tot_Err[start_index] .+ obs_dataframe.Tot_Err[end_index]

    # Adjust uncertainty limits
    ΔT_Obs_Upper = ΔT_Obs + ΔT_Obs_Uncertainty 
    ΔT_Obs_Lower = ΔT_Obs - ΔT_Obs_Uncertainty 

    #Create a dataframe to store the result
    Calculated_ΔT_Obs = DataFrame(Initial_Year = period[1],
                                    Final_Year = period[2],
                                    ΔT_Obs = ΔT_Obs,
                                    ΔT_Obs_Lower = ΔT_Obs_Lower,
                                    ΔT_Obs_Upper = ΔT_Obs_Upper,
                                    ΔT_Obs_Per_Decade = 10 * ΔT_Obs / (period[2] - period[1]),
                                    ΔT_Obs_Per_Decade_Lower = 10 * ΔT_Obs_Lower / (period[2] - period[1]),
                                    ΔT_Obs_Per_Decade_Upper = 10 * ΔT_Obs_Upper / (period[2] - period[1]))
    return Calculated_ΔT_Obs
end

# Stage 4. Function that performs linear regression on ΔT and the variable to be constrained, and uses the observed data to create a PDF of the constrained variable.
function calculate_emergent_constraint_GLM(Delta_T_ESM_period_1_df, Delta_T_ESM_period_2_df, Delta_T_Obs_df, confidence)
    
    # Rename the ΔT column for the second period to ΔT_SSP:
    Delta_T_ESM_period_2_df = rename(Delta_T_ESM_period_2_df, :ΔT => :ΔT_SSP)
    Delta_T_ESM_period_2_df = rename(Delta_T_ESM_period_2_df, :ΔT_per_decade => :ΔT_per_decade_SSP)

    # Combine data
    ESM_Data = innerjoin(Delta_T_ESM_period_1_df, Delta_T_ESM_period_2_df, on = :Model, makeunique=true)

    # Extract the observed ΔT and its bounds
    ΔT_Obs_Mean = Delta_T_Obs_df[1, :ΔT_Obs]
    ΔT_Obs_Upper = Delta_T_Obs_df[1, :ΔT_Obs_Upper]
    ΔT_Obs_Lower = Delta_T_Obs_df[1, :ΔT_Obs_Lower]

    ΔT_Obs_Per_Decade = Delta_T_Obs_df[1, :ΔT_Obs_Per_Decade]
    ΔT_Obs_Per_Decade_Upper = Delta_T_Obs_df[1, :ΔT_Obs_Per_Decade_Upper]
    ΔT_Obs_Per_Decade_Lower = Delta_T_Obs_df[1, :ΔT_Obs_Per_Decade_Lower]

    Obs_Stats = [ΔT_Obs_Mean, ΔT_Obs_Upper, ΔT_Obs_Lower]
    ΔT_Obs_STD = (ΔT_Obs_Upper - ΔT_Obs_Mean) / confidence_to_zscore(0.95)

    Obs_Stats_Per_Decade = [ΔT_Obs_Per_Decade, ΔT_Obs_Per_Decade_Upper, ΔT_Obs_Per_Decade_Lower]
    ΔT_Obs_Per_Decade_STD = (ΔT_Obs_Per_Decade_Upper - ΔT_Obs_Per_Decade) / confidence_to_zscore(0.95)
    
    ΔT_Mean = mean(ESM_Data.ΔT)
    ΔT_range = range(0, 2 * ΔT_Mean, length = num_steps)

    ΔT_per_decade_Mean = mean(ESM_Data.ΔT_per_decade)
    ΔT_per_decade_range = range(0, 2 * ΔT_per_decade_Mean, length = num_steps)

    formula = @eval @formula($(Symbol(:ΔT_SSP)) ~ ΔT_per_decade)

    ΔT_SSP_Reg = lm(formula, ESM_Data)
    
    #Extract the length:
    N = length(ESM_Data.ΔT)
    params = coef(ΔT_SSP_Reg)
    errors = stderror(ΔT_SSP_Reg)
    r_value = cor(ESM_Data[!, :ΔT_SSP], predict(ΔT_SSP_Reg))

    #Extract the Least Squares Sum:
    s = sqrt((1/(N-2))*sum(abs2, residuals(ΔT_SSP_Reg)))

    # Assuming ΔT_SSP_Reg is the fitted model and ΔT_range is the range of ΔT values
    ΔT_SSP_CI = predict(ΔT_SSP_Reg, DataFrame(ΔT_per_decade = ΔT_per_decade_range), interval = :confidence, level = confidence)
    ΔT_SSP_PI = predict(ΔT_SSP_Reg, DataFrame(ΔT_per_decade = ΔT_per_decade_range), interval = :prediction, level = confidence)

    # Calculate the metric distribution
    y_vals, Py_vals = calculate_distribution(ESM_Data, Obs_Stats_Per_Decade, params, s)

    # Normalize the PDF values to a fixed height (optional)
    Py_vals_norm = Py_vals / maximum(Py_vals) * ΔT_Obs_Per_Decade

    return Dict(
        :params => params,
        :errors => errors,
        :r_value => r_value,
        :ΔT_SSP_Reg => ΔT_SSP_Reg,
        :ΔT_SSP_CI => ΔT_SSP_CI,
        :ΔT_SSP_PI => ΔT_SSP_PI,
        :ΔT_range => ΔT_range,
        :ΔT_per_decade_range => ΔT_per_decade_range,
        :ΔT_Obs => ΔT_Obs_Mean,
        :ΔT_Obs_Upper => ΔT_Obs_Upper,
        :ΔT_Obs_Lower => ΔT_Obs_Lower,
        :ΔT_Obs_Per_Decade => ΔT_Obs_Per_Decade,
        :ΔT_Obs_Per_Decade_Upper => ΔT_Obs_Per_Decade_Upper,
        :ΔT_Obs_Per_Decade_Lower => ΔT_Obs_Per_Decade_Lower,
        :ESM_Data => ESM_Data,
        :ΔT_SSP_Range => y_vals,
        :P_ΔT_SSP_vals => Py_vals,
        :P_ΔT_SSP_vals_norm => Py_vals_norm
    )
end

# Stage 5: Function that calculate the cumulative distribution of the emergent contsraiint, and returns the median value, and the confidence interval
function calculate_emergent_constraint_cumulative(y_vals, Py_vals_norm, confidence)

    #Perform an integration of Py_vals_norm to obtain the cumulative distribution:
    Cy_vals = cumsum(Py_vals_norm) * (y_vals[2] - y_vals[1])

    #Print the value of Cy_vals which is closest to 0.5:
    Median_Index = findmin(x->abs(x-0.5), Cy_vals)[2]
    Metric_Median = y_vals[Median_Index]

    #For a given confidence, extract the lower and upper estimates:
    Upper = 0.5 + confidence/2
    Lower = 0.5 - confidence/2

    Metric_Upper_Index = findmin(x->abs(x-Upper), Cy_vals)[2]
    Metric_Lower_Index = findmin(x->abs(x-Lower), Cy_vals)[2]

    Metric_Upper_Bound = y_vals[Metric_Upper_Index]
    Metric_Lower_Bound = y_vals[Metric_Lower_Index]

    return Cy_vals, Metric_Median, Metric_Upper_Bound, Metric_Lower_Bound

end


### ========  Auxiliary Plotting Functions: ======== ###
function plot_emergent_constraint_with_distribution(results, ssp_run, period_2, confidence)

    # Define the color scheme for each SSP run
    color_scheme = Dict(
        "ssp126" => "#02D977",
        "ssp245" => "#BFEB64",
        "ssp370" => "#FFAE65",
        "ssp585" => "#FF7764"
    )

    # Extract results
    r_value = results[:r_value]
    ΔT_SSP_CI = results[:ΔT_SSP_CI]
    ΔT_SSP_PI = results[:ΔT_SSP_PI]
    ΔT_range = results[:ΔT_range]
    ΔT_per_decade_range = results[:ΔT_per_decade_range]
    ΔT_Obs = results[:ΔT_Obs]
    ΔT_Obs_Upper = results[:ΔT_Obs_Upper]
    ΔT_Obs_Lower = results[:ΔT_Obs_Lower]
    ΔT_Obs_Per_Decade = results[:ΔT_Obs_Per_Decade]
    ΔT_Obs_Per_Decade_Upper = results[:ΔT_Obs_Per_Decade_Upper]
    ΔT_Obs_Per_Decade_Lower = results[:ΔT_Obs_Per_Decade_Lower]
    ESM_Data = results[:ESM_Data]
    y_vals = results[:ΔT_SSP_Range]
    Py_vals = results[:P_ΔT_SSP_vals]
    Py_vals_norm = results[:P_ΔT_SSP_vals_norm]

    # Get the color for the current SSP run
    color = color_scheme[ssp_run]

    _, ΔT_SSP_Median, ΔT_SSP_Upper, ΔT_SSP_Lower = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

    # Create or overlay the plot
    plot = scatter(
        ESM_Data.ΔT_per_decade,
        ESM_Data.ΔT_SSP,
        xlims=(minimum(ΔT_per_decade_range), maximum(ΔT_per_decade_range)),
        xlabel=L"$\ \Delta T \$ /10yr  [K yr$^{-1}\$]",
        ylabel=" ΔT [K]",
        grid = false,
        #label = "n = $(size(ESM_Data, 1))",
        label = "Individual ESMs", 
        color = color,
        title = "$(ssp_run), $(period_2[2]) ",
        #aspect_ratio = .1,
        markerstrokewidth = 0
    )

    # Plot the curve of best fit
    plot!(
        ΔT_per_decade_range,
        ΔT_SSP_PI.prediction,
        ribbon = (ΔT_SSP_PI.upper - ΔT_SSP_PI.prediction, ΔT_SSP_PI.prediction - ΔT_SSP_PI.lower),
        label = "OLS Linear Fit", 
        color = color,
        #label = "r = $(round(r_value, digits = 2))",
        fillalpha = 0.2, 
        fillcolor = color
    )

    #=

    plot!(
        ΔT_per_decade_range,
        ΔT_SSP_CI.prediction,
        ribbon = (ΔT_SSP_CI.upper - ΔT_SSP_CI.prediction, ΔT_SSP_CI.prediction - ΔT_SSP_CI.lower),
        label = false,
        fillalpha = 0.4, 
        fillcolor = color
    )
    =#

    plot!(
        ΔT_per_decade_range,
        ΔT_SSP_CI.prediction,
        label = "",
        color = color
    )
    
    # Add vertical dashed line for observed ΔT
    vline!(
        [ΔT_Obs_Per_Decade],
        label = "Observed Warming",
        color = :black,
        linestyle = :dash,
        linewidth = 0.8
    )

    # Add shaded region for observed ΔT range
    vspan!(
        [ΔT_Obs_Per_Decade_Upper, ΔT_Obs_Per_Decade_Lower],
        fillalpha = 0.1,
        label = "",
        #label = "\$ ΔT_{Obs}\$/10yr : $(round(ΔT_Obs_Per_Decade, digits = 2))",
        fillcolor = :gray
    )

    hline!([ΔT_SSP_Median], color = color, linewidth = 1, label = "ΔT Central Prediction",
            #label = "\$ΔT($(period_2[2])) \$: $(round(ΔT_SSP_Median, digits = 2))")
    )
    #=

    hspan!([ΔT_SSP_Lower, ΔT_SSP_Upper], alpha = .1 , color = color, label = "",
        #label = "90% CI: $(round(ΔT_SSP_Lower, digits = 2)) - $(round(ΔT_SSP_Upper, digits = 2))")
        )
        =#
    
    plot!(Py_vals_norm, y_vals, label = false, color = color)

    return plot
end

function plot_ΔT_distribution(results, ssp_run, period_2, confidence)

    # Define the color scheme for each SSP run
    color_scheme = Dict(
        "ssp126" => "#02D977",
        "ssp245" => "#BFEB64",
        "ssp370" => "#FFAE65",
        "ssp585" => "#FF7764"
    )


    # Extract results
    r_value = results[:r_value]
    ΔT_SSP_CI = results[:ΔT_SSP_CI]
    ΔT_SSP_PI = results[:ΔT_SSP_PI]
    ΔT_range = results[:ΔT_range]
    ΔT_per_decade_range = results[:ΔT_per_decade_range]
    ΔT_Obs = results[:ΔT_Obs]
    ΔT_Obs_Upper = results[:ΔT_Obs_Upper]
    ΔT_Obs_Lower = results[:ΔT_Obs_Lower]
    ΔT_Obs_Per_Decade = results[:ΔT_Obs_Per_Decade]
    ΔT_Obs_Per_Decade_Upper = results[:ΔT_Obs_Per_Decade_Upper]
    ΔT_Obs_Per_Decade_Lower = results[:ΔT_Obs_Per_Decade_Lower]
    ESM_Data = results[:ESM_Data]
    y_vals = results[:ΔT_SSP_Range]
    Py_vals = results[:P_ΔT_SSP_vals]
    Py_vals_norm = results[:P_ΔT_SSP_vals_norm]

    Cy_vals, ΔT_SSP_Median, ΔT_SSP_Upper, ΔT_SSP_Lower = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

    # Get the color for the current SSP run
    color = color_scheme[ssp_run]

    # Plot the distribution
    plots = plot(y_vals, Py_vals, grid = false, label = false, title = "$(ssp_run) ΔT($(period_2[2]))")  
    plot!(y_vals, Cy_vals, label = false)

    #Annotate with the median and confidence interval:
    vline!([ΔT_SSP_Median], label = "ΔT ($(period_2[2])) = $(round(ΔT_SSP_Median, digits = 2))", color = :black)
    vspan!([ΔT_SSP_Lower, ΔT_SSP_Upper], fillalpha = 0.1, label = " $(Int(round(100*confidence)))% CI: $(round(ΔT_SSP_Lower, digits = 2)) - $(round(ΔT_SSP_Upper, digits = 2))", fillcolor = :gray)

    return plots
end

### ========  Main Functions for Plots for Paper: ======== ###

function produce_single_constraint_future_warming(directory, period_1, end_year, ssp_run, window_size, confidence)
    plots = []

    period_2 = (1850 + floor(window_size / 2), end_year)

    All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

    Delta_T_ESM_period_1_df = calculate_ESM_warming_trend(All_Models_1850_2100, period_1, window_size, confidence)
    Delta_T_ESM_period_2_df = calculate_ESM_warming_trend(All_Models_1850_2100, period_2, window_size, confidence)

    Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period_1, window_size)

    SSP_ΔT_Results = calculate_emergent_constraint_GLM(Delta_T_ESM_period_1_df, Delta_T_ESM_period_2_df, Delta_T_Obs_df, confidence)

    #SSP_ΔT_Plot = plot_ΔT_distribution(SSP_ΔT_Results, ssp_run, period_1, period_2, confidence)
    SSP_ΔT_Plot = plot_emergent_constraint_with_distribution(SSP_ΔT_Results, ssp_run, period_2, confidence)

    savefig(SSP_ΔT_Plot, "output/Single_Future_Warming_EC_SSP245.pdf")

    return SSP_ΔT_Plot
end


function produce_emergent_constraint_future_warming_end_year_and_ssp_comparsion(directory, period_1, end_years, ssp_runs, window_size, confidence)

    plots = []

    for end_year in end_years
        period_2 = (1850 + floor(window_size / 2), end_year)

        for ssp_run in ssp_runs
            All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

            Delta_T_ESM_period_1_df = calculate_ESM_warming_trend(All_Models_1850_2100, period_1, window_size, confidence)

            Delta_T_ESM_period_2_df = calculate_ESM_warming_trend(All_Models_1850_2100, period_2, window_size, confidence)

            Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period_1, window_size)

            SSP_ΔT_Results = calculate_emergent_constraint_GLM(Delta_T_ESM_period_1_df, Delta_T_ESM_period_2_df, Delta_T_Obs_df, confidence)

            #SSP_ΔT_Plot = plot_ΔT_distribution(SSP_ΔT_Results, ssp_run, period_1, period_2, confidence)
            SSP_ΔT_Plot = plot_emergent_constraint_with_distribution(SSP_ΔT_Results, ssp_run, period_2, confidence)

            push!(plots, SSP_ΔT_Plot)
        end
    end

    # Arrange the plots in a 3x4 grid
    plot_grid = plot(plots..., layout = (3, 4), size = (1200, 1000), legend = false)

    separate_plot = plot(plots[1], 
        title = "",
        xlabel = "",
        xlims = (1, 2),
        ylabel = "",
        ylims = (10, 15),
        ticks = false,
        framestyle = :none,
        legend = :top
    )

    combined_plot = plot(plot_grid, separate_plot, layout = @layout([a{0.85h}; b]))

    savefig(combined_plot, "output/Future_Warming_EC.pdf")

    return combined_plot
end

#Load in the Directory Path of the ESM Temperature Time Series Runs
ESM_Directory = "data/ESM_historical_runs/"

#Load in the Met Office HadCRUT5 Temperature
Delta_T_Obs_raw_df = CSV.read("data/observational_data/All_Data_Temperature_Anom.csv", DataFrame);

end_year = 2050
ssp_run = "ssp245"

period_1 = (1980, 2019)
ssp_runs = ["ssp126", "ssp245", "ssp370", "ssp585"] # List of SSP runs
end_years = [2030, 2050, 2090] # List of end years
window_size = 11 # Smoothing window size
confidence = 0.90 # Confidence level

produce_single_constraint_future_warming(ESM_Directory, period_1, end_year, ssp_run, window_size, confidence)
#produce_emergent_constraint_future_warming_end_year_and_ssp_comparsion(ESM_Directory, period_1, end_years, ssp_runs, window_size, confidence)