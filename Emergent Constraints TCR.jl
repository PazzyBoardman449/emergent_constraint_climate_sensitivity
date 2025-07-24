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
using Printf

cd(@__DIR__)

### Plot Settings:
default(fontfamily="Computer Modern", xlabelfontsize = 10, ylabelfontsize = 10, titlefontsize=10, legendfontsize = 8, left_margin = 20px, bottom_margin = 20px)

# ==== Auxuliary Functions === #

function save_as_latex_table(df::DataFrame, output_path::String)
    # Define the models that require an asterisk
    models_with_asterisk = Set(["INM-CM5-0", "GFDL-CM4", "GFDL-ESM4"])

    # Open the file for writing
    open(output_path, "w") do io
        # Write the LaTeX table header
        write(io, """
        \\begin{table}[t]
            \\centering
            \\begin{tabular}{c  c  c  c  c  c}
                \\hline
                 Centre & Model & Ensemble & \$\\Delta T\$ (1975 to 2019) & \$\\Delta T\$ (1975 to 2024) & TCR \\\\
                 \\hline
        """)

        # Write the rows of the DataFrame
        for row in eachrow(df)
            # Format numerical values to two decimal places
            ΔT_1975_2019 = @sprintf("%.2f", row.ΔT)
            ΔT_1975_2024 = @sprintf("%.2f", row.ΔT2)
            tcr_value = @sprintf("%.2f", row.TCR)

            # Check if the model requires an asterisk
            if row.Model in models_with_asterisk
                tcr_value = "$(tcr_value)*"  # Add an asterisk to the TCR value
            end

            # Write the row to the LaTeX table
            write(io, "$(row.Centre) & $(row.Model) & $(row.Simulation_Conditions) & $(ΔT_1975_2019) & $(ΔT_1975_2024) & $(tcr_value) \\\\\n")
        end

        # Write the LaTeX table footer
        write(io, """
                \\hline
            \\end{tabular}
            \\caption{List of CMIP6 ESMs used in this study. Models with an asterisk (*) still met the criteria but did not have TCR values which were not listed in the IPCC AR6 Chapter 7 appendix, and were therefore calculated directly using the piControl and 1pctCO2 experimental simulations.}
            \\label{tab:CMIP6 Models}
        \\end{table}
        """) 
    end
end

function gaussian_weights(window_size::Int, sigma::Float64)
    half_window = floor(Int, window_size / 2)
    x = -half_window:half_window
    weights = exp.(-0.5 .* (x ./ sigma).^2)
    return weights ./ sum(weights)  # Normalize the weights
end

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
    #simulation = split(split_filename[end], ".")[1]
    simulation = split_filename[5]

    return centre, model, simulation
end

# Function to translate confidence interval to a standard deviation σ
function confidence_to_zscore(confidence_level)
    # Calculate the z-score for the given confidence level
    zscore = quantile(Normal(0, 1), 1 - (1 - confidence_level) / 2)
    return zscore
end

# Function that performs integration to produce the probability distribution of the vairable to be constrained (e.g ECS)
function calculate_metric_distribution(Data, Obs_Data, params, s, metric_column)
    
    # Parameters and input values
    nfitx = 1000  # Number of x values
    mfity = 1000  # Number of y values

    # Pre-allocate arrays
    Pxy = zeros(nfitx, mfity)
    Py = zeros(mfity)

    # Extract data and parameters
    ΔT_data = Data.ΔT
    N = length(ΔT_data)

    ΔT_Obs_Mean = Obs_Data[1]
    ΔT_Obs_Upper = Obs_Data[2]
    ΔT_Obs_Lower = Obs_Data[3]

    σ_ΔT_Obs = (ΔT_Obs_Upper - ΔT_Obs_Mean) #/ (confidence_to_zscore(0.95))

    # Define P_x(x)
    P_x(x) = pdf(Normal(ΔT_Obs_Mean, σ_ΔT_Obs), x)

    # Initialize variables
    local f, σ_f, x_min, x_max, y_min, y_max

    # Define the model and ranges based on the metric_column
    f = x -> params[1] + params[2] * x
    #σ_f = x -> errors[2] * std(ΔT_data) * sqrt(1 + N + ((x - mean(ΔT_data))^2 / (std(ΔT_data))^2))
    σ_f = x ->  s * sqrt(1 + 1/N + ((x - mean(ΔT_data))^2 / (N*std(ΔT_data, corrected=false)^2)))
    x_min, x_max = 0, 2 * mean(ΔT_data)
    y_min, y_max = 0, 2 * mean(Data[:, metric_column])

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
            #σ_f_x = sqrt(σ_f(x)^2 + params[2]^2*σ_x^2)
            #P_y_given_x = pdf(Normal(f(x), σ_f_x), y)
            P_y_given_x = (1/(sqrt(2*pi*σ_f_x^2)))*exp(-(y - f_x)^2/(2*σ_f_x^2))
            Pxy[n, m] = P_x(x) * P_y_given_x
        end
        # Integrate over x to get Py
        Py[m] = sum(Pxy[:, m] * dx)
    end

    # Intgrate the curve again to obtain the cumulative distributions:

    # Normalize Py
    Py_norm = sum(Py) * dy
    Py ./= Py_norm

    return y_values, Py

end

#= Main Procedure Here!!: =#

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
            centre = extract_model_and_simulation(file)[1]
            model = extract_model_and_simulation(file)[2]
            simulation = extract_model_and_simulation(file)[3]

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
    ΔT_df= DataFrame(Centre = String[], 
    Model = String[], 
    Simulation_Run = String[], 
    T_Initial = Float64[], 
    T_Final = Float64[], 
    ΔT = Float64[],
    ΔT_err = Float64[])

    start_index = findfirst(dataframes[1][:, :Year] .== period[1])
    end_index = findfirst(dataframes[1][:, :Year] .== period[2])

    for i in 1:length(dataframes)
        dataframes[i][:, :Smoothed_Surface_Temperature] = movingaverage(dataframes[i][:, :Surface_Temperature], window_size)

        dataframes[i][:, :Smoothed_Residuals] = dataframes[i][:, :Surface_Temperature] - dataframes[i][:, :Smoothed_Surface_Temperature]
        dataframes[i][:, :Smoothed_Surface_Temperature_err] .= confidence_to_zscore(confidence)*std(dataframes[i][:, :Smoothed_Residuals], corrected = false) / sqrt(window_size)
    end 

    # Loop over each DataFrame and extract the change in temperature between the two years
    for i in 1:length(dataframes)
        if nrow(dataframes[i]) >= end_index
            push!(ΔT_df, (Centre = dataframes[i].Centre[1], Model = dataframes[i].Model[1], 
                Simulation_Run = dataframes[i].Simulation[1], 
                T_Initial = dataframes[i].Smoothed_Surface_Temperature[start_index], 
                T_Final = dataframes[i].Smoothed_Surface_Temperature[end_index], 
                ΔT = dataframes[i].Smoothed_Surface_Temperature[end_index] - dataframes[i].Smoothed_Surface_Temperature[start_index], 
                ΔT_err = sqrt((dataframes[i].Smoothed_Surface_Temperature_err[end_index])^2 + (dataframes[i].Smoothed_Surface_Temperature_err[start_index])^2)))
        else
            println("Delta T Cannot be calculated for the ESM")
        end
    end    

    #Sort the dataframe by alphabetical order of the model name:
    ΔT_df = sort(ΔT_df, [:Centre])

    #println(ΔT_df)

    return ΔT_df
end

# Stage 3. Function to calculate the observed warming trend over a given period, and return the ΔT value and its uncertainty
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
                                    ΔT_Obs_Upper = ΔT_Obs_Upper)
    return Calculated_ΔT_Obs
end

# Stage 4. Function that performs linear regression on ΔT and the variable to be constrained, and uses the observed data to create a PDF of the constrained variable. =#
function calculate_emergent_constraint_GLM(Delta_T_ESM_df, Delta_T_Obs_df, Metric_ESM_df, confidence)
    
    # Combine data
    ESM_Data = innerjoin(Delta_T_ESM_df, Metric_ESM_df, on = :Model)

    # Dynamically identify the Metric column name (assumes the second column is Metric)
    metric_column = names(Metric_ESM_df)[end]
    Metric = ESM_Data[!, metric_column]

    # Create a DataFrame for regression with dynamic column name
    regression_df = DataFrame(ΔT = ESM_Data.ΔT)
    regression_df[!, metric_column] = Metric

    # Extract the observed ΔT and its bounds
    ΔT_Obs_Mean = Delta_T_Obs_df[1, :ΔT_Obs]
    ΔT_Obs_Upper = Delta_T_Obs_df[1, :ΔT_Obs_Upper]
    ΔT_Obs_Lower = Delta_T_Obs_df[1, :ΔT_Obs_Lower]

    Obs_Stats = [ΔT_Obs_Mean, ΔT_Obs_Upper, ΔT_Obs_Lower]
    
    ΔT_Mean = mean(ESM_Data.ΔT)
    # Generate a range of ΔT values
    ΔT_range = range(0, 2 * ΔT_Mean, length = 1000)

    local formula

    if metric_column == "TCR"
        formula = @eval @formula($(Symbol(metric_column)) ~ ΔT)
    elseif metric_column == "λ"
        formula = @eval @formula($(Symbol(metric_column)) ~ 1/ΔT)
    elseif metric_column == "ECS"
        formula = @eval @formula($(Symbol(metric_column)) ~ ΔT)
    else
        error("Unsupported metric column: $metric_column")
    end

    Metric_Reg = lm(formula, regression_df)
    
    #Extract coefficients
    N = length(Metric)
    params = coef(Metric_Reg)
    errors = stderror(Metric_Reg)
    r_value = cor(Metric, predict(Metric_Reg))

    #Extract the Least Squares Sum:
    s = sqrt((1/(N-2))*sum(abs2, residuals(Metric_Reg)))

    #println(N)

    #Print the Equation of the Regression Line:
    #println("$(metric_column) = $(round(params[1], digits=4)) + $(round(params[2], digits=4)) * ΔT")

    # Assuming Metric_Reg is the fitted model and ΔT_range is the range of ΔT values
    Metric_CI = predict(Metric_Reg, DataFrame(ΔT = ΔT_range), interval = :confidence, level = confidence)
    Metric_PI = predict(Metric_Reg, DataFrame(ΔT = ΔT_range), interval = :prediction, level = confidence)

    # Calculate the metric distribution
    y_vals, Py_vals = calculate_metric_distribution(regression_df, Obs_Stats, params, s, metric_column)

    # Normalize the PDF values to a fixed height (optional)
    Py_vals_norm = Py_vals / maximum(Py_vals) * ΔT_Obs_Mean

    return Dict(
        :params => params,
        :errors => errors,
        :r_value => r_value,
        :Metric_Reg => Metric_Reg,
        :Metric_CI => Metric_CI,
        :Metric_PI => Metric_PI,
        :ΔT_range => ΔT_range,
        :ΔT_Obs => ΔT_Obs_Mean,
        :ΔT_Obs_Upper => ΔT_Obs_Upper,
        :ΔT_Obs_Lower => ΔT_Obs_Lower,
        :Metric_Column => metric_column,
        :ESM_Data => ESM_Data,
        :Metric_Range => y_vals,
        :Metric_vals => Py_vals,
        :Metric_vals_norm => Py_vals_norm
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

### ========  Auxiliary Functions: ======== ###

#Figure 3a: Dependence of Running Mean on Emergent Constraint
function produce_emergent_constraint_dependence_on_running_mean(directory, period, ssp_run, window_size_var, confidence)

    metric_values = Float64[]
    metric_lower_bounds = Float64[]
    metric_upper_bounds = Float64[]

    for window_size in window_size_var
        All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

        Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
        Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period, window_size)

        results_df = calculate_emergent_constraint_GLM(Delta_T_ESM_df, Delta_T_Obs_df, TCR_ESMs_df, confidence)

        y_vals = results_df[:Metric_Range]
        Py_vals = results_df[:Metric_vals]

        _, Metric_Median, Metric_Upper_Bound, Metric_Lower_Bound = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

        #Add the results to the results table
        push!(metric_values, Metric_Median)
        push!(metric_lower_bounds, Metric_Lower_Bound)
        push!(metric_upper_bounds, Metric_Upper_Bound)   
    end

     # Plot the results with confidence interval
     p = plot(
        window_size_var,
        metric_values,
        xlabel = "Window Size",
        ylabel = "TCR [K]",
        xlims = (minimum(window_size_var), maximum(window_size_var)),
        ylims = (0, maximum(metric_upper_bounds) * 1.1),  # Adjust y-limits dynamically
        label = "TCR Central Estimate",
        grid = false,
        marker = :circle,
        legend = :bottomright
    )

    # Add shaded region for confidence interval
    plot!(
        p,
        window_size_var,
        metric_values,
        fillbetween = (metric_upper_bounds, metric_lower_bounds),
        fillalpha = 0.2,
        label = "$(Int(100 * confidence))% Confidence Range"
    )
end

#Figure 3b: Dependence of End Year on Emergent Constraint
function produce_emergent_constraint_dependence_on_end_year(directory, period_var, ssp_run, window_size, confidence)

    end_year_vals = Float64[]
    metric_values = Float64[]
    metric_lower_bounds = Float64[]
    metric_upper_bounds = Float64[]

    for period in period_var
        All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

        Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
        Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period, window_size)

        results_df = calculate_emergent_constraint_GLM(Delta_T_ESM_df, Delta_T_Obs_df, TCR_ESMs_df, confidence)

        y_vals = results_df[:Metric_Range]
        Py_vals = results_df[:Metric_vals]

        _, Metric_Median, Metric_Upper_Bound, Metric_Lower_Bound = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

        #Add the results to the results table
        push!(end_year_vals, period[2]) 
        push!(metric_values, Metric_Median)
        push!(metric_lower_bounds, Metric_Lower_Bound)
        push!(metric_upper_bounds, Metric_Upper_Bound)   
    end

     # Plot the results with confidence interval
     p = plot(
        end_year_vals,
        metric_values,
        xlabel = "End Year",
        ylabel = "TCR [K]",
        xlims = (minimum(end_year_vals), maximum(end_year_vals)),
        ylims = (0, maximum(metric_upper_bounds) * 1.1),  # Adjust y-limits dynamically
        label = "TCR Central Estimate",
        grid = false,
        marker = :circle,
        legend = :bottomright
    )

    # Add shaded region for confidence interval
    plot!(
        p,
        end_year_vals,
        metric_values,
        fillbetween = (metric_upper_bounds, metric_lower_bounds),
        fillalpha = 0.2,
        label = "$(Int(100 * confidence))% Confidence Range"
    )
end

#Figure 3c: Dependence of Start Year on Emergent Constraint
function produce_emergent_constraint_dependence_on_start_year(directory, period_var, ssp_run, window_size, confidence)

    start_year_vals = Float64[]
    metric_values = Float64[]
    metric_lower_bounds = Float64[]
    metric_upper_bounds = Float64[]

    for period in period_var
        All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

        Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
        Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period, window_size)

        results_df = calculate_emergent_constraint_GLM(Delta_T_ESM_df, Delta_T_Obs_df, TCR_ESMs_df, confidence)

        y_vals = results_df[:Metric_Range]
        Py_vals = results_df[:Metric_vals]

        _, Metric_Median, Metric_Upper_Bound, Metric_Lower_Bound = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

        #Add the results to the results table
        push!(start_year_vals, period[1]) 
        push!(metric_values, Metric_Median)
        push!(metric_lower_bounds, Metric_Lower_Bound)
        push!(metric_upper_bounds, Metric_Upper_Bound)   
    end

     # Plot the results with confidence interval
     p = plot(
        start_year_vals,
        metric_values,
        xlabel = "Start Year",
        ylabel = "TCR [K]",
        xlims = (minimum(start_year_vals), maximum(start_year_vals)),
        ylims = (0, maximum(metric_upper_bounds) * 1.1),  # Adjust y-limits dynamically
        label = "TCR Central Estimate",
        grid = false,
        marker = :circle,
        legend = :bottomright
    )

    # Add shaded region for confidence interval
    plot!(
        p,
        start_year_vals,
        metric_values,
        fillbetween = (metric_upper_bounds, metric_lower_bounds),
        fillalpha = 0.2,
        label = "$(Int(100 * confidence))% Confidence Range"
    )
end

function export_data_to_csv(directory, ssp_run, period, confidence, window_size)
    # Initialize a DataFrame to store the results
    #results = DataFrame(Model = String[], Simulation = String[], TCR = Float64[], ΔT = Float64[], ssp_run = String[], Start_Year = Int[], End_Year = Int[], Window_Size = Int[])

    All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)
    Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)

    #Inner Join the TCR_ESMs_df with the Delta_T_ESM_df
    results_df = innerjoin(Delta_T_ESM_df, TCR_ESMs_df, on = :Model)

    #Select the relevant columns
    results_df = select(results_df, [:Model, :Simulation_Conditions, :TCR, :ΔT])

    #Add the ssp_run, start_year, end_year, and window_size to the DataFrame
    results_df[!, :ssp_run] .= ssp_run
    results_df[!, :Start_Year] .= period[1]
    results_df[!, :End_Year] .= period[2]
    results_df[!, :Window_Size] .= window_size

    #Export the DataFrame to a CSV file
    CSV.write("/Users/pb662/PhD Local FIles/Modelling (Julia)/PAPER - The Effects of Recent Warming/Data/Emergent_Constraint_Results_$(ssp_run)_$(period[1])_$(period[2])_$(window_size)_$(Int(100*confidence)).csv", results_df)

end

### ========  Main Functions for Plots for Paper: ======== ###

# Table 1: Export a text file which has the TCR values for each ESM.
function produce_TCR_table(directory, ssp_run, period_1, period_2, window_size, confidence)
    All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

    #Calculate the warming trends in the ESMs between both pairs of years
    Delta_T_ESM_df_1 = calculate_ESM_warming_trend(All_Models_1850_2100, period_1, window_size, confidence)
    Delta_T_ESM_df_2 = calculate_ESM_warming_trend(All_Models_1850_2100, period_2, window_size, confidence)

    Delta_T_ESM_df_2_Filtered = select(Delta_T_ESM_df_2, [:Model, :ΔT])
    rename!(Delta_T_ESM_df_2_Filtered, :ΔT => :ΔT2)

    #Inner Join the TCR_ESMs_df with the Delta_T_ESM_df
    results_df = innerjoin(Delta_T_ESM_df_1, TCR_ESMs_df, on = :Model)
    results_df = innerjoin(results_df, Delta_T_ESM_df_2_Filtered, on = :Model)

    #Select the relevant columns
    results_df = select(results_df, [:Centre, :Model, :Simulation_Conditions,:ΔT, :ΔT2, :TCR])

    #Rounnd all values in ΔT and ΔT2 to two decimal places
    results_df[!, :ΔT] = round.(results_df[!, :ΔT], digits = 2)
    results_df[!, :ΔT2] = round.(results_df[!, :ΔT2], digits = 2)
    
    #Convert the results_df dataframe into a Latex table:
    println(results_df)

    save_as_latex_table(results_df, "output/TCR_Table.txt")
end

# Figure 1a: Time Series Plot of ESMS colored by TCR Value
function produce_ΔT_time_series_plot_for_SSP_Run_One_Period(directory, Delta_T_Obs_raw_df, ssp_run, window_size, period_var, confidence, TCR_ESMs_df)

    plot_font = "Computer Modern"
    default(fontfamily=plot_font, linewidth=2, framestyle=:box, label=nothing, grid=false)

    # Extract the Data
    All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

    # Initialize a dictionary to store ΔT values for each model
    model_ΔT = Dict{String, Vector{Float64}}()
    end_years = []

    for period in period_var
        end_year = period[2]
        push!(end_years, end_year)
        Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
        
        for row in eachrow(Delta_T_ESM_df)
            model = row[:Model]
            ΔT = row[:ΔT]
            if !haskey(model_ΔT, model)
                model_ΔT[model] = Float64[]
            end
            push!(model_ΔT[model], ΔT)
        end
    end

    # Created a truncated version of the period_var_obs, where the end years only extend to the present day:
    period_var_obs = [(period_var[1][1], year) for year in period_var[1][2]:period_var[end][2]]

    # Calculate the Observed ΔT
    ΔT_Obs = Float64[]
    ΔT_Obs_Upper = Float64[]
    ΔT_Obs_Lower = Float64[]

    for period in period_var_obs
        Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period, window_size)
        push!(ΔT_Obs, Delta_T_Obs_df[1, :ΔT_Obs])
        push!(ΔT_Obs_Upper, Delta_T_Obs_df[1, :ΔT_Obs_Upper])
        push!(ΔT_Obs_Lower, Delta_T_Obs_df[1, :ΔT_Obs_Lower])
    end

    # Create a plot for the time series data
    plt = plot(
        xlabel = "Year",
        ylabel = "ΔT [K]",
        grid = :false,

        #background_color=:transparent,
        linecolor=:white,
        linewidth = 1,

        #Set the size of the plot
        size = (400, 300)
    )


    # Extract TCR values for each model
    model_TCR = Dict{String, Float64}()
    for row in eachrow(TCR_ESMs_df)
        model_TCR[row[:Model]] = row[:TCR]
    end

    # Calculate the average ΔT_vals for each model
    model_avg_ΔT = [(model, mean(ΔT_vals)) for (model, ΔT_vals) in model_ΔT if haskey(model_TCR, model)]

    # Sort the models based on the TCR values
    sorted_models = sort(model_avg_ΔT, by = x -> model_TCR[x[1]])

    # Extract the sorted model names
    sorted_model_names = [x[1] for x in sorted_models]

    # Get the color palette
    palette1 = reverse(palette(:RdYlBu_9, length(sorted_model_names)))
    palette2 = reverse(palette(:RdYlBu_9, 100))

    # Plot ΔT values for each model with colors based on the sorted order
    for (i, model) in enumerate(sorted_model_names)
        ΔT_vals = model_ΔT[model]
        plot!(
            plt,
            end_years[1:length(ΔT_vals)],  # Ensure x-values match the length of y-values
            ΔT_vals,
            seriestype = :line,
            label = i == 1 ? "ESM Ensemble Runs" : "",  # Only show label on the first iteration
            linewidth = 2,
            color = palette1[i]
        )
    end

    # Plot ΔT_Obs values over the top
    plot!(
        plt,
        end_years[1:length(ΔT_Obs)],
        ΔT_Obs,
        fillbetween = (ΔT_Obs_Lower, ΔT_Obs_Upper),
        fillcolor = :red,
        label = "Observed ΔT",
        seriestype = :line,
        #color = :transparent,
        linecolor = :black,
        linewidth = 2,
        alpha = 0.3,
        linealpha = 1
    )

    # Create a horizontal colorbar for TCR values
    tcr_values = [model_TCR[model] for model in sorted_model_names]
    min_tcr = minimum(tcr_values)
    max_tcr = maximum(tcr_values)

    # Use scatter to simulate a horizontal colorbar
    colorbar_plot = heatmap(rand(2,2), clims=(min_tcr, max_tcr), 
        framestyle=:none, c=palette2, 
        cbar=true, 
        title="TCR [K]", 
        lims=(-1, 0)
    )
    
    # Plot them side by side
    TCR_plots = plot(plt, colorbar_plot, layout = @layout([a{0.95w} b{0.05w}]))  # Set the legend to have two column)

    # Save the combined plot
    savefig(TCR_plots, "output/ΔT_time_series_plot_one_period.pdf")

    return TCR_plots
end

# Figure 1b: Time Series Plot of ESMS colored by TCR Value for two Periods
function produce_ΔT_time_series_plot_for_SSP_Run_Two_Periods(directory, Delta_T_Obs_raw_df, ssp_run, window_size, period_var_1, period_var_2, confidence, TCR_ESMs_df)

    All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

    # Initialize plots for both periods
    plots = []
    all_sorted_model_names = []  # Store sorted model names for both periods
    all_model_TCR = Dict{String, Float64}()  # Store TCR values for all models

    # Extract TCR values for each model (outside the loop)
    for row in eachrow(TCR_ESMs_df)
        all_model_TCR[row[:Model]] = row[:TCR]
    end

    for period_var in [period_var_1, period_var_2]
        end_years = []
        model_ΔT = Dict{String, Vector{Float64}}()

        # Process ESM data for the current period
        for period in period_var
            end_year = period[2]
            push!(end_years, end_year)
            Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)

            for row in eachrow(Delta_T_ESM_df)
                model = row[:Model]
                ΔT = row[:ΔT]
                if !haskey(model_ΔT, model)
                    model_ΔT[model] = Float64[]
                end
                push!(model_ΔT[model], ΔT)
            end
        end

        # Create a truncated version of the period_var_obs, where the end years only extend to the present day
        period_var_obs = [(period_var[1][1], year) for year in period_var[1][2]:period_var[end][2]]

        # Calculate the Observed ΔT
        ΔT_Obs = Float64[]
        ΔT_Obs_Upper = Float64[]
        ΔT_Obs_Lower = Float64[]

        for period in period_var_obs
            Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period, window_size)
            push!(ΔT_Obs, Delta_T_Obs_df[1, :ΔT_Obs])
            push!(ΔT_Obs_Upper, Delta_T_Obs_df[1, :ΔT_Obs_Upper])
            push!(ΔT_Obs_Lower, Delta_T_Obs_df[1, :ΔT_Obs_Lower])
        end

        # Create a plot for the time series data
        plt = plot(
            xlabel = "Year",
            ylabel = "ΔT Since $(period_var[1][1]) [K]",
            grid = :false,
            linewidth = 1,
            size = (400, 300),
            legend =:false
        )

        # Calculate the average ΔT_vals for each model
        model_avg_ΔT = [(model, mean(ΔT_vals)) for (model, ΔT_vals) in model_ΔT if haskey(all_model_TCR, model)]

        # Sort the models based on the TCR values
        sorted_models = sort(model_avg_ΔT, by = x -> all_model_TCR[x[1]])

        # Extract the sorted model names
        sorted_model_names = [x[1] for x in sorted_models]
        push!(all_sorted_model_names, sorted_model_names)  # Store sorted model names for this period

        # Get the color palette
        palette1 = reverse(palette(:RdYlBu_9, length(sorted_model_names)))

        # Plot ΔT values for each model with colors based on the sorted order
        for (i, model) in enumerate(sorted_model_names)
            ΔT_vals = model_ΔT[model]
            plot!(
                plt,
                end_years[1:length(ΔT_vals)],  # Ensure x-values match the length of y-values
                ΔT_vals,
                seriestype = :line,
                label = i == 1 ? "ESM Ensemble Runs" : "",  # Only show label on the first iteration
                linewidth = 2,
                color = palette1[i]
            )
        end

        # Plot ΔT_Obs values over the top
        plot!(
            plt,
            end_years[1:length(ΔT_Obs)],
            ΔT_Obs,
            fillbetween = (ΔT_Obs_Lower, ΔT_Obs_Upper),
            fillcolor = :red,
            label = "Observed ΔT",
            seriestype = :line,
            linecolor = :black,
            linewidth = 2,
            alpha = 0.3,
            linealpha = 1
        )

        # Append the plot to the list of plots
        push!(plots, plt)
    end


    period_var = period_var_1
        end_years = []
        model_ΔT = Dict{String, Vector{Float64}}()

        # Process ESM data for the current period
        for period in period_var
            end_year = period[2]
            push!(end_years, end_year)
            Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)

            for row in eachrow(Delta_T_ESM_df)
                model = row[:Model]
                ΔT = row[:ΔT]
                if !haskey(model_ΔT, model)
                    model_ΔT[model] = Float64[]
                end
                push!(model_ΔT[model], ΔT)
            end
        end

        # Calculate the average ΔT_vals for each model
        model_avg_ΔT = [(model, mean(ΔT_vals)) for (model, ΔT_vals) in model_ΔT if haskey(all_model_TCR, model)]

        # Sort the models based on the TCR values
        sorted_models = sort(model_avg_ΔT, by = x -> all_model_TCR[x[1]])

        # Extract the sorted model names
        sorted_model_names = [x[1] for x in sorted_models]
        push!(all_sorted_model_names, sorted_model_names)  

        # Get the color palette
        palette1 = reverse(palette(:RdYlBu_9, length(sorted_model_names)))

        # Create a truncated version of the period_var_obs, where the end years only extend to the present day
        period_var_obs = [(period_var[1][1], year) for year in period_var[1][2]:period_var[end][2]]

        # Calculate the Observed ΔT
        ΔT_Obs = Float64[]
        ΔT_Obs_Upper = Float64[]
        ΔT_Obs_Lower = Float64[]

        for period in period_var_obs
            Delta_T_Obs_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period, window_size)
            push!(ΔT_Obs, Delta_T_Obs_df[1, :ΔT_Obs])
            push!(ΔT_Obs_Upper, Delta_T_Obs_df[1, :ΔT_Obs_Upper])
            push!(ΔT_Obs_Lower, Delta_T_Obs_df[1, :ΔT_Obs_Lower])
        end

        # Create a plot for the time series data
        legend_plot = plot(
            xlabel = "",
            ylabel = "",
            grid = :false,
            framestyle = :none,
            legend =:top,
            linewidth = 1,
            xlims = (0,1),
            ylime = (0,1)
        )

        # Plot ΔT values for each model with colors based on the sorted order
        for (i, model) in enumerate(sorted_model_names)
            ΔT_vals = model_ΔT[model]
            plot!(
                legend_plot,
                end_years[1:length(ΔT_vals)],  # Ensure x-values match the length of y-values
                ΔT_vals,
                seriestype = :line,
                label = i == 1 ? "ESM Ensemble Runs" : "",  # Only show label on the first iteration
                linewidth = 2,
                color = palette1[i]
            )
        end

        # Plot ΔT_Obs values over the top
        plot!(
            legend_plot,
            end_years[1:length(ΔT_Obs)],
            ΔT_Obs,
            fillbetween = (ΔT_Obs_Lower, ΔT_Obs_Upper),
            fillcolor = :red,
            label = "Observed ΔT",
            seriestype = :line,
            linecolor = :black,
            linewidth = 2,
            alpha = 0.3,
            linealpha = 1
        )

    # Create a horizontal colorbar for TCR values
    tcr_values = [all_model_TCR[model] for model in all_sorted_model_names[1]]  # Use the first period's sorted models
    min_tcr = minimum(tcr_values)
    max_tcr = maximum(tcr_values)

    # Use heatmap to simulate a horizontal colorbar
    palette2 = reverse(palette(:RdYlBu_9, 100))
    colorbar_plot = heatmap(rand(2, 2), clims=(min_tcr, max_tcr), 
        framestyle=:none, c=palette2, 
        cbar=true, 
        title="TCR [K]", 
        lims=(-1, 0)
    )

    # Combine the two plots, the colorbar, and the legend-only plot
    combined_plot = plot(
        plot(plots[1], plots[2], colorbar_plot, layout = @layout([a b c{0.05w}])),
        legend_plot,
        layout = @layout([d; e{0.05h}]),  # Add the legend-only plot below the combined plot
        size = (1100, 400)  # Adjust the size to accommodate the legend-only plot
    )

    # Save the combined plot
    savefig(combined_plot, "output/ΔT_time_series_plot_two_periods.pdf")

    return combined_plot
end

# Figure 2a: Produce a single emergent constraint on TCR
function produce_emergent_constraint_plot(directory, period_1, period_2, ssp_run, window_size, confidence, metric)
    All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

    #Calculate the warming trends in the ESMs between both pairs of years
    Delta_T_ESM_period1_df = calculate_ESM_warming_trend(All_Models_1850_2100, period_1, window_size, confidence)

    #Calculate the warming trends in the ESMs between both pairs of years
    Delta_T_Obs_period1_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period_1, window_size)

    #Calculate the emergent constraint on the chosen variable

    # Determine the appropriate ESM dataframe and plot settings
    metric_df = TCR_ESMs_df

    results_period1_df = calculate_emergent_constraint_GLM(Delta_T_ESM_period1_df, Delta_T_Obs_period1_df, metric_df, confidence)

    # Extract results
    Metric_Column = results_period1_df[:Metric_Column]

    # Extract results for the first peridd
    y_vals_1 = results_period1_df[:Metric_Range]
    ΔT_range_1 = results_period1_df[:ΔT_range]
    Metric_PI_1 = results_period1_df[:Metric_PI]
    ΔT_Obs_1 = results_period1_df[:ΔT_Obs]
    ΔT_Obs_Upper_1 = results_period1_df[:ΔT_Obs_Upper]
    ΔT_Obs_Lower_1 = results_period1_df[:ΔT_Obs_Lower]
    ESM_Data_1 = results_period1_df[:ESM_Data]
    Py_vals_1 = results_period1_df[:Metric_vals]
    Py_vals_1_norm = results_period1_df[:Metric_vals_norm]

    _, Metric_Median_1, Metric_Upper_Bound_1, Metric_Lower_Bound_1 = calculate_emergent_constraint_cumulative(y_vals_1, Py_vals_1, confidence)

    color1 = "#00D8A2"

    plot = scatter(
        xlims=(minimum(ΔT_range_1), maximum(ΔT_range_1)),
        ylims=(minimum(y_vals_1), maximum(y_vals_1)),
        xlabel=L"$\ \Delta T \$",
        ylabel="TCR",
        title="Emergent Constraint on TCR",
        aspect_ratio = .4,
        legend =:bottomright,
        grid=false,
        size = (400, 400)
    )


    scatter!(
        ESM_Data_1.ΔT,
        ESM_Data_1[!, Metric_Column],
        #xlabel=L"$\\Delta T$ [K]",
        #ylabel="\$ y \$",
        color =color1,
        #label="ESMs $(period_1[1]) - $(period_1[2]) (\$n=\$ $(size(ESM_Data_1, 1)))",
        label = "ESMs"
    )

    

    # Plot the curve of best fit
    plot!(
        ΔT_range_1,
        Metric_PI_1.prediction,
        ribbon = (Metric_PI_1.upper - Metric_PI_1.prediction, Metric_PI_1.prediction - Metric_PI_1.lower),
        fillbetween = (Metric_PI_1.upper - Metric_PI_1.prediction, Metric_PI_1.prediction - Metric_PI_1.lower),
        fillalpha = 0.2, 
        fillcolor = color1,
        #label = "Regression $(period_1[1]) - $(period_1[2]) (\$r=\$ $(round(r_value_1, digits=2)))",
        #label = "OLS Linear Fit",
        color=color1
    )

    # Plot the curve of best fit
    plot!(
        ΔT_range_1,
        Metric_PI_1.prediction,
        #ribbon = (Metric_PI_1.upper - Metric_PI_1.prediction, Metric_PI_1.prediction - Metric_PI_1.lower),
        fillbetween = (Metric_PI_1.upper - Metric_PI_1.prediction, Metric_PI_1.prediction - Metric_PI_1.lower),
        fillalpha = 0.2, 
        fillcolor = color1,
        #label = "Regression $(period_1[1]) - $(period_1[2]) (\$r=\$ $(round(r_value_1, digits=2)))",
        label = "OLS Linear Fit",
        color=color1
    )

    
    # Add vertical dashed line for observed ΔT
    vline!(
        [ΔT_Obs_1],
        #label="\$Δ T_{Obs} \$ $(period_1[1]) - $(period_1[2]): $(round(ΔT_Obs_1, digits=2))K",
        label="Observed ΔT",
        color=:black,
        linestyle=:dash
    )


    vspan!(
        [ΔT_Obs_Lower_1, ΔT_Obs_Upper_1],
        fillalpha=0.2,
        fillcolor=color1
        )

    


    hline!(
        [Metric_Median_1],
        #label="TCR $(period_1[1]) - $(period_1[2]): $(round(Metric_Median_1, digits=2)) $(Int(100*confidence))% Confidence: ($(round(Metric_Lower_Bound_1, sigdigits = 3)) - $(round(Metric_Upper_Bound_1, sigdigits = 3)))",
        label = "TCR Estimate",
        linestyle=:dashdot,
        color = color1
    )

    plot!(Py_vals_1_norm, y_vals_1, label="", color=color1)
    #plot!(Py_vals_2_norm, y_vals_2, label="$(Metric_Column) Prediction $(period_2[1]) - $(period_2[2])", color=:purple)
    
    
    # Save the figure as PDF
    savefig(plot, "output/EC_TCR.pdf")
    
    return plot
    
end

# Figure 2b: Emergent constraint plots for two periods, with distrubution.
function produce_combined_emergent_constraint_comparison_plots(directory, period_1, period_2, ssp_run, window_size, confidence)

    All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

    #Calculate the warming trends in the ESMs between both pairs of years
    Delta_T_ESM_period1_df = calculate_ESM_warming_trend(All_Models_1850_2100, period_1, window_size, confidence)
    Delta_T_ESM_period2_df = calculate_ESM_warming_trend(All_Models_1850_2100, period_2, window_size, confidence)


    #Calculate the warming trends in the ESMs between both pairs of years
    Delta_T_Obs_period1_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period_1, window_size)
    Delta_T_Obs_period2_df = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period_2, window_size)


    # Determine the appropriate ESM dataframe and plot settings
    metric_df = TCR_ESMs_df

    results_period1_df = calculate_emergent_constraint_GLM(Delta_T_ESM_period1_df, Delta_T_Obs_period1_df, metric_df, confidence)
    results_period2_df = calculate_emergent_constraint_GLM(Delta_T_ESM_period2_df, Delta_T_Obs_period2_df, metric_df, confidence)


    # Extract results
    Metric_Column = results_period1_df[:Metric_Column]

    # Extract results for the first peridd
    y_vals_1 = results_period1_df[:Metric_Range]
    ΔT_range_1 = results_period1_df[:ΔT_range]
    Metric_PI_1 = results_period1_df[:Metric_PI]
    ΔT_Obs_1 = results_period1_df[:ΔT_Obs]
    ΔT_Obs_Upper_1 = results_period1_df[:ΔT_Obs_Upper]
    ΔT_Obs_Lower_1 = results_period1_df[:ΔT_Obs_Lower]
    ESM_Data_1 = results_period1_df[:ESM_Data]
    Py_vals_1 = results_period1_df[:Metric_vals]

    # Extract results for the second period
    y_vals_2 = results_period2_df[:Metric_Range]
    ΔT_range_2 = results_period2_df[:ΔT_range]
    Metric_PI_2 = results_period2_df[:Metric_PI]
    ΔT_Obs_2 = results_period2_df[:ΔT_Obs]
    ΔT_Obs_Upper_2 = results_period2_df[:ΔT_Obs_Upper]
    ΔT_Obs_Lower_2 = results_period2_df[:ΔT_Obs_Lower]
    ESM_Data_2 = results_period2_df[:ESM_Data]
    Py_vals_2 = results_period2_df[:Metric_vals]

    _, Metric_Median_1, Metric_Upper_Bound_1, Metric_Lower_Bound_1 = calculate_emergent_constraint_cumulative(y_vals_1, Py_vals_1, confidence)
    _, Metric_Median_2, Metric_Upper_Bound_2, Metric_Lower_Bound_2 = calculate_emergent_constraint_cumulative(y_vals_2, Py_vals_2, confidence)

    color1 = "#00DCA5"
    color2 = "#902CB7"

    #Print the number of models:
    println("Number of ESMs in period 1: ", size(ESM_Data_1, 1))

    # Create or overlay the plot
    default(label=false)

    plot_comparison = scatter(
        xlims=(minimum(ΔT_range_1), maximum(ΔT_range_1)),
        ylims=(minimum(y_vals_1), maximum(y_vals_1)),
        xlabel="ΔT [K]",
        ylabel="$(Metric_Column) [K]",
        title="",
        #title="Emergent Constraint on $(Metric_Column)",
        aspect_ratio = .4,
        grid=false)

    scatter!(
        ESM_Data_1.ΔT,
        ESM_Data_1[!, Metric_Column],
        color = color1,
        label = "CMIP6 ESMs"
    )

    scatter!(
        ESM_Data_2.ΔT,
        ESM_Data_2[!, Metric_Column],
        color = color2,
    )

    # Plot the curve of best fit
    plot!(
        ΔT_range_1,
        Metric_PI_1.prediction,
        ribbon = (Metric_PI_1.upper - Metric_PI_1.prediction, Metric_PI_1.prediction - Metric_PI_1.lower),
        fillalpha = 0.2, 
        fillcolor = "#00C896",
        #label = "Regression $(period_1[1]) - $(period_1[2]) (\$r=\$ $(round(r_value_1, digits=2)))",
        label = "OLS Linear Fit",
        color=color1,
        linewidth = 1
    )

    # Plot the curve of best fit
    plot!(
        ΔT_range_2,
        Metric_PI_2.prediction,
        ribbon = (Metric_PI_2.upper - Metric_PI_2.prediction, Metric_PI_2.prediction - Metric_PI_2.lower),
        fillalpha = 0.2,
        fillcolor = color2,
        #label = "Regression $(period_1[1]) - $(period_1[2]) (\$r=\$ $(round(r_value_1, digits=2)))",
        color=color2,
        linewidth = 1
    )

    # Add vertical dashed line for observed ΔT
    vline!(
        [ΔT_Obs_1],
        #label="\$Δ T_{Obs} \$ $(period_1[1]) - $(period_1[2]): $(round(ΔT_Obs_1, digits=2))K",
        label="Observed ΔT",
        color=color1,
        linewidth = 1,
        linestyle=:dash
    )

    # Add vertical dashed line for observed ΔT
    vline!(
        [ΔT_Obs_2],
        #label="\$Δ T_{Obs} \$ $(period_1[1]) - $(period_1[2]): $(round(ΔT_Obs_1, digits=2))K",
        color=color2,
        linewidth = 1,
        linestyle=:dash
    )

    vspan!(
        [ΔT_Obs_Lower_1, ΔT_Obs_Upper_1],
        fillalpha=0.2,
        fillcolor=color1,
        strokealpha=0
        )

    vspan!(
        [ΔT_Obs_Lower_2, ΔT_Obs_Upper_2],
        fillalpha=0.2,
        fillcolor=color2,
        strokealpha=0
        )
    
    hline!(
        [Metric_Median_1],
        #label="TCR $(period_1[1]) - $(period_1[2]): $(round(Metric_Median_1, digits=2)) $(Int(100*confidence))% Confidence: ($(round(Metric_Lower_Bound_1, sigdigits = 3)) - $(round(Metric_Upper_Bound_1, sigdigits = 3)))",
        label = "TCR Estimate",
        linestyle=:dashdot,
        color = color1,
        linewidth = 1
    )

    hline!(
        [Metric_Median_2],
        #label="TCR $(period_1[1]) - $(period_1[2]): $(round(Metric_Median_1, digits=2)) $(Int(100*confidence))% Confidence: ($(round(Metric_Lower_Bound_1, sigdigits = 3)) - $(round(Metric_Upper_Bound_1, sigdigits = 3)))",
        linestyle=:dashdot,
        color = color2,
        linewidth = 1
    )

    # Function to generate plots for a given metric
    function generate_plot()

        # Determine the appropriate ESM dataframe and plot settings
        metric_df = TCR_ESMs_df

        results_period1_df = calculate_emergent_constraint_GLM(Delta_T_ESM_period1_df, Delta_T_Obs_period1_df, metric_df, confidence)
        results_period2_df = calculate_emergent_constraint_GLM(Delta_T_ESM_period2_df, Delta_T_Obs_period2_df, metric_df, confidence)

        # Extract results
        Metric_Column = results_period1_df[:Metric_Column]

        y_vals_1 = results_period1_df[:Metric_Range]
        y_vals_2 = results_period2_df[:Metric_Range]


        Py_vals_1 = results_period1_df[:Metric_vals]
        Py_vals_2 = results_period2_df[:Metric_vals]

        AR5_TCR_Median = 1.8
        AR5_TCR_Upper = 2.5
        AR5_TCR_Lower = 1.1

        AR6_TCR_Median = 1.8
        AR6_TCR_Upper = 2.4
        AR6_TCR_Lower = 1.2

        AR5_TCR_Std = (AR5_TCR_Upper - AR5_TCR_Lower)/(2*confidence_to_zscore(0.66))
        AR6_TCR_Std = (AR6_TCR_Upper - AR6_TCR_Lower)/(2*confidence_to_zscore(0.90))

        cumulative_results_df_1, Metric_Median_1, Metric_Upper_Bound_1, Metric_Lower_Bound_1 = calculate_emergent_constraint_cumulative(y_vals_1, Py_vals_1, confidence)
        cumulative_results_df_2, Metric_Median_2, Metric_Upper_Bound_2, Metric_Lower_Bound_2 = calculate_emergent_constraint_cumulative(y_vals_2, Py_vals_2, confidence)

        color1 = "#00DCA5"
        color2 = "#902CB7"
        hist_color = "#63C2A8"

        default(label=false)

        # Create the plot
        p = histogram(
            metric_df[!, Metric_Column],
            bins = 10,            
            normalize = true,
            #markersize=2,
            #alpha = 0.3,
            color = hist_color,
            alpha = 0.5,
            linecolor =:white,
            linewidth = 1,
            label = "CMIP6 ESMs",
        )
        
        
        plot!(y_vals_1, Py_vals_1, 
                label="EC $(period_1[1]) - $(period_1[2])", 
                color=color1,
                xlabel="$(Metric_Column) [K]",
                ylabel= "PDF($(Metric_Column)) [K\$^{-1}\$]",  # Correct LaTeX syntax for superscript
                linewidth=1,
                grid=:false,
                aspect_ratio=3,
                xlim=(0, 4),
                ylim=(0, 1.33)
            )

        plot!(y_vals_2, Py_vals_2, 
        label="EC $(period_2[1]) - $(period_2[2])", 
        color= color2,
        linewidth = 1,
        )

        # Print the likely range of TCR values
        println("TCR = $(Metric_Median_1) [$(Metric_Lower_Bound_1) - $(Metric_Upper_Bound_1)]")
        println("TCR = $(Metric_Median_2) [$(Metric_Lower_Bound_2) - $(Metric_Upper_Bound_2)]")


        #Plot AR6 TCR Range
        plot!(y_vals_1, pdf(Normal(AR6_TCR_Median, AR6_TCR_Std), y_vals_1),
            linewidth = 1,
        label = "AR6 Likely Range", color = "#999", linestyle = :dash)
    end

    # Generate plots for each metric
    plot_TCR =generate_plot()
    
    # Combine the two plots side by side
    combined_plot = plot(plot_comparison, plot_TCR, layout = (1,2), 
        #background_color=:transparent,
    
    # Remove the legend border
    foreground_color_legend = nothing,

    size = (900, 405)
    )

    savefig(combined_plot, "output/EC_TCR_Comparison.pdf")

    return combined_plot
end

#Figure S1: Emergent Constraint Dependence on Start Year, End Year, and Running Mean
function plot_emergent_constraints_side_by_side(directory, period, ssp_run, window_size, confidence, period_var_start, period_var_end, window_size_var)
    # Generate the individual plots
    plot1 = produce_emergent_constraint_dependence_on_start_year(directory, period_var_start, ssp_run, window_size, confidence)
    plot2 = produce_emergent_constraint_dependence_on_running_mean(directory, period, ssp_run, window_size_var, confidence)
    plot3 = produce_emergent_constraint_dependence_on_end_year(directory, period_var_end, ssp_run, window_size, confidence)

    # Combine the plots side by side
    combined_plot = plot(
        plot1, plot2, plot3,
        layout = @layout([a b c]),  # Arrange plots horizontally
        size = (1500, 400)          # Adjust the size of the combined plot
    )

    savefig(combined_plot, "output/TCR_Robustness.pdf")

    return combined_plot
end

# All Custom Parameters Here!: 

#Load in the Directory Path of the ESM Temperature Time Series Runs
ESM_Directory = "data/ESM_historical_runs/"

#Load in Observational Data
Delta_T_Obs_raw_df = CSV.read("data/observational_data/All_Data_Temperature_Anom.csv", DataFrame);

#Load in the TCR Results
TCR_ESMs_df = CSV.read("data/TCR_Results_IPCC.csv", DataFrame);

window_size = 11
end_year_1 = 2014
end_year_2 = 2019 #Int(2024 - floor(window_size/2))

period_1 = (1980, end_year_1)
period_2 = (1980, end_year_2)

end_year_var_1 = (1860:1:end_year_2)
end_year_var_2 = (1980:1:end_year_2)

mean_window_var = (3:1:21)

period_var_1 = [(1860, end_year) for end_year in end_year_var_1]
period_var_2 = [(1980, end_year) for end_year in end_year_var_2]

period_var_start = [(start_year, 2019) for start_year in (1970:1:2008)]
period_var_end = [(1980, end_year) for end_year in (1990:1:end_year_2)]

ssp_run = "ssp245" # Chosen SSP run 
confidence = 0.90 # Confidence Level

# CALL FUNCTIONS HERE!!

#produce_TCR_table(ESM_Directory, ssp_run, period_1, period_2, window_size, confidence)
#produce_ΔT_time_series_plot_for_SSP_Run_One_Period(ESM_Directory, Delta_T_Obs_raw_df, ssp_run, window_size, period_var_2, confidence, TCR_ESMs_df)
#produce_ΔT_time_series_plot_for_SSP_Run_Two_Periods(ESM_Directory, Delta_T_Obs_raw_df, ssp_run, window_size, period_var_1, period_var_2, confidence, TCR_ESMs_df)
#produce_emergent_constraint_plot(ESM_Directory, period_1, period_2, ssp_run, window_size, confidence, "TCR")
#produce_combined_emergent_constraint_comparison_plots(ESM_Directory, period_1, period_2, ssp_run, window_size, confidence)
plot_emergent_constraints_side_by_side(ESM_Directory, period_2, ssp_run, window_size, confidence, period_var_start, period_var_end, mean_window_var)
