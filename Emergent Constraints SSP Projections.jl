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
using StatsPlots


cd(@__DIR__)

num_steps = 1000

### Plot Settings:
default(fontfamily="Computer Modern", xlabelfontsize = 12, ylabelfontsize = 12, titlefontsize=12, legendfontsize = 10, left_margin = 20px, bottom_margin = 40px)

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

# Function that performs integration to produce the probability distribution of ΔT in the future
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
    #σ_f = x -> errors[2] * std(ΔT) * sqrt(1+N + ((x - mean(ΔT))^2 / (std(ΔT))^2))
    σ_f = x ->  s * sqrt(1 + 1/N + ((x - mean(ΔT))^2 / (N*(std(ΔT))^2)))

    x_min, x_max = 0, 2 * mean(ΔT)
    y_min, y_max = 0, 2 * mean(ΔT_SSP)

    # Calculate dx and dy
    dx = (x_max - x_min) / nfitx
    dy = (y_max - y_min) / mfity

    # Generate x and y values
    x_values = range(x_min, stop=x_max, length=nfitx)
    y_values = range(y_min, stop=y_max, length=mfity)

    #=

    # Calculate Pxy, Pyx, and Py
    for m in 1:mfity
        y = y_values[m]
        for n in 1:nfitx
            x = x_values[n]
            σ_f_x = σ_f(x)
            P_y_given_x = pdf(Normal(f(x), σ_f_x), y)
            Pxy[n, m] = P_x(x) * P_y_given_x
            Pyx[m, n] = Pxy[n, m]
        end
        # Integrate over x to get Py
        Py[m] = sum(Pxy[:, m]) * dx
    end

    =#

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


    # Intgrate the curve again to obtain the cumulative distributions:

    # Normalize Py
    Py_norm = sum(Py) * dy
    Py ./= Py_norm


    return y_values, Py

end

function get_min_max_values(model_ΔT::Dict{String, Vector{Float64}})
    # Find the minimum length of all vectors
    num_steps = minimum(length.(values(model_ΔT)))

    min_vals = [Inf for _ in 1:num_steps]
    max_vals = [-Inf for _ in 1:num_steps]

    for ΔT_vals in values(model_ΔT)
        # Ensure the vector is truncated to the minimum length
        ΔT_vals = ΔT_vals[1:min(num_steps, length(ΔT_vals))]
        for i in 1:num_steps
            if i <= length(ΔT_vals)
                if ΔT_vals[i] < min_vals[i]
                    min_vals[i] = ΔT_vals[i]
                end
                if ΔT_vals[i] > max_vals[i]
                    max_vals[i] = ΔT_vals[i]
                end
            end
        end
    end

    return min_vals, max_vals
end

#Function to calculate the temperature offset between start of the period used for warming trend and the start of the model run:
function calculate_temperature_anomaly_offset(obs_dataframe, period, window_size)

    reference_year = 1850 + floor(window_size / 2)
    start_year = period[1]

    #Apply the smoothing window to the observed data
    obs_dataframe[!, :Smoothed_Temperature_Anomaly] = movingaverage(obs_dataframe[!, :Temperature_Anomaly], window_size)

    #Calculate the offset, that is the difference between the temperature at the reference year and the start year:
    offset = obs_dataframe.Smoothed_Temperature_Anomaly[findfirst(obs_dataframe[!, :Year] .== start_year)] - obs_dataframe.Smoothed_Temperature_Anomaly[findfirst(obs_dataframe[!, :Year] .== reference_year)]

    return offset
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
            push!(ΔT_df, (
                Centre = dataframes[i].Centre[1],
                Model = dataframes[i].Model[1], 
                Simulation_Run = dataframes[i].Simulation[1], 
                T_Initial = dataframes[i].Smoothed_Surface_Temperature[start_index], 
                T_Final = dataframes[i].Smoothed_Surface_Temperature[end_index], 
                ΔT = dataframes[i].Smoothed_Surface_Temperature[end_index] - dataframes[i].Smoothed_Surface_Temperature[start_index], 
                ΔT_err = sqrt((dataframes[i].Smoothed_Surface_Temperature_err[end_index])^2 + (dataframes[i].Smoothed_Surface_Temperature_err[start_index])^2)))
        else
            println("Delta T Cannot be calculated for the ESM")
        end
    end

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
                                    ΔT_Obs_Upper = ΔT_Obs_Upper)
    return Calculated_ΔT_Obs
end

# Stage 4. Function that performs linear regression on ΔT and the variable to be constrained, and uses the observed data to create a PDF of the constrained variable.
function calculate_emergent_constraint_GLM(Delta_T_ESM_period_1_df, Delta_T_ESM_period_2_df, Delta_T_Obs_df, confidence)

    # Rename the ΔT column for the second period to ΔT_SSP:
    Delta_T_ESM_period_2_df = rename(Delta_T_ESM_period_2_df, :ΔT => :ΔT_SSP)

    # Combine data
    ESM_Data = innerjoin(Delta_T_ESM_period_1_df, Delta_T_ESM_period_2_df, on = :Model, makeunique=true)

    # Extract the observed ΔT and its bounds
    ΔT_Obs_Mean = Delta_T_Obs_df[1, :ΔT_Obs]
    ΔT_Obs_Upper = Delta_T_Obs_df[1, :ΔT_Obs_Upper]
    ΔT_Obs_Lower = Delta_T_Obs_df[1, :ΔT_Obs_Lower]

    ΔT_Obs_STD = (ΔT_Obs_Upper - ΔT_Obs_Mean)
    Obs_Stats = [ΔT_Obs_Mean, ΔT_Obs_Upper, ΔT_Obs_Lower]
    
    ΔT_Mean = mean(ESM_Data.ΔT)
    ΔT_range = range(0, 2 * ΔT_Mean, length = num_steps)

    formula = @eval @formula($(Symbol(:ΔT_SSP)) ~ ΔT)

    ΔT_SSP_Reg = lm(formula, ESM_Data)

    #Extract coefficients
    N = length(ESM_Data.ΔT)
    params = coef(ΔT_SSP_Reg)
    errors = stderror(ΔT_SSP_Reg)
    r_value = cor(ESM_Data[!, :ΔT_SSP], predict(ΔT_SSP_Reg))


    #Extract the Least Squares Sum:
    s = sqrt((1/(N-2))*sum(abs2, residuals(ΔT_SSP_Reg)))

    # Assuming ΔT_SSP_Reg is the fitted model and ΔT_range is the range of ΔT values
    ΔT_SSP_CI = predict(ΔT_SSP_Reg, DataFrame(ΔT = ΔT_range), interval = :confidence, level = confidence)
    ΔT_SSP_PI = predict(ΔT_SSP_Reg, DataFrame(ΔT = ΔT_range), interval = :prediction, level = confidence)

    # Calculate the metric distribution
    y_vals, Py_vals = calculate_distribution(ESM_Data, Obs_Stats, params, s)

    # Normalize the PDF values to a fixed height (optional)
    Py_vals_norm = Py_vals / maximum(Py_vals) * ΔT_Obs_Mean

    return Dict(
        :params => params,
        :errors => errors,
        :r_value => r_value,
        :ΔT_SSP_Reg => ΔT_SSP_Reg,
        :ΔT_SSP_CI => ΔT_SSP_CI,
        :ΔT_SSP_PI => ΔT_SSP_PI,
        :ΔT_range => ΔT_range,
        :ΔT_Obs => ΔT_Obs_Mean,
        :ΔT_Obs_Upper => ΔT_Obs_Upper,
        :ΔT_Obs_Lower => ΔT_Obs_Lower,
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

### ========  Main Functions for Plots for Paper: ======== ###

#Figure 1a: Time Series Plot of ΔT for all ESMS, as well as constraint, stratified by SSP Run
function produce_constrained_ΔT_time_series_for_SSP_Runs(directory, Delta_T_Obs_raw_df, ssp_runs, window_size, period_var, confidence)
    
    color_scheme = Dict(
        "ssp126" => "#02D977",
        "ssp245" => "#BFEB64",
        "ssp370" => "#FFAE65",
        "ssp585" => "#FF7764"
    )

    plots = []
    ssp126_plot = nothing  # Placeholder for the ssp126 plot

    for ssp_run in ssp_runs
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

        filter!(model -> length(model[2]) >= length(period_var), model_ΔT)

        # Get the minimum and maximum values at each time step across all models
        min_vals, max_vals = get_min_max_values(model_ΔT)

        if end_years[end] < Delta_T_Obs_raw_df.Year[end]
            final_year = end_years[end]
        else
            final_year = Delta_T_Obs_raw_df.Year[end] - floor(Int, window_size/2)
        end

        # Created a truncated version of the period_var_obs, where the end years only extend to the present day:
        period_var_obs = [(period_var[1][1], year) for year in period_var[1][2]:final_year]

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

        Delta_T_ESM_df_known = calculate_ESM_warming_trend(All_Models_1850_2100, period_var_obs[end], window_size, confidence)
        Delta_T_Obs_df_known = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period_var_obs[end], window_size)

        # Create a dataframe to store the end year with the central ΔT prediction
        ΔT_Predictions = DataFrame(End_Year = Int64[], ΔT_Prediction = Float64[], ΔT_Upper = Float64[], ΔT_Lower = Float64[])

        for i in 1:length(period_var)
            period = period_var[i]
    
            Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
            results = calculate_emergent_constraint_GLM(Delta_T_ESM_df_known, Delta_T_ESM_df, Delta_T_Obs_df_known, confidence)
        
            y_vals, Py_vals = results[:ΔT_SSP_Range], results[:P_ΔT_SSP_vals]
            _, median, upper, lower = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

            # Check whether median value is appropriate, if not, replace with the last value
            if abs(median) < 1E-6 && i != 1
                println("Had to be replaced at year $(period[2])")
                median = ΔT_Predictions[!, :ΔT_Prediction][i-1]
                upper = ΔT_Predictions[!, :ΔT_Upper][i-1]
                lower = ΔT_Predictions[!, :ΔT_Lower][i-1]
            end

            push!(ΔT_Predictions, (End_Year = period[2], ΔT_Prediction = median, ΔT_Upper = upper, ΔT_Lower = lower))
        end

        color = color_scheme[ssp_run]

        offset = calculate_temperature_anomaly_offset(Delta_T_Obs_raw_df, period_var[1][1], window_size)

        # Create a plot for the time series data
        plt = plot(
            xlabel = "Year",
            ylabel = "ΔT [K]",
            title = "$(ssp_run)",
            grid = :false,
            label = "",  # Disable legend for individual plots
            xlims = (1980, 2099),
            ylime = (0, 8),
        )

        #=

        # Plot the model range
        plot!(
            plt,
            end_years,
            min_vals .+ offset,
            color =:gray,
            linewidth = 0.5 ,
            label = "Individual Models"
        )

        =#


        plot!(
            plt,
            end_years,
            max_vals .+ offset,
            fillbetween = (min_vals .+ offset, max_vals .+ offset),
            label = "CMIP6 ESM Range",
            linewidth = 0,
            fillalpha = 0.1,
            color =:gray,
            fillcolor = color
        )

        
        plot!(
            plt,
            end_years,
            min_vals .+ offset,
            color =:gray,
            linewidth = 0,
            label = ""
        )


        # Plot all the models
        for (model, ΔT_vals) in model_ΔT
            plot!(
                plt,
                end_years[1:length(ΔT_vals)],
                ΔT_vals .+ offset,
                label = "",
                seriestype = :line,
                linewidth = 1,
                color = :black,
                alpha = 0.1
            )
        end

        # Plot ΔT_Obs values over the top
        plot!(
            plt,
            end_years[1:length(ΔT_Obs)],
            ΔT_Obs .+ offset,
            label = "Observed ΔT",
            seriestype = :line,
            color = :gray,
            alpha = 0.6,
            linecolor = :black,
            linewidth = 2,
            linestyle = :dash,
            linealpha = 1,
        )

        # Plot ΔT Predictions
        plot!(
            plt,
            ΔT_Predictions[!, :End_Year],
            ΔT_Predictions[!, :ΔT_Prediction] .+ offset,
            fillbetween = (ΔT_Predictions[!, :ΔT_Lower] .+ offset, ΔT_Predictions[!, :ΔT_Upper] .+ offset),
            label = "Constrained ΔT ($(Int(round(confidence*100)))% Confidence)",
            seriestype = :line,
            color = color,
            linecolor = :black,
            linewidth = 2,
            linealpha = 1,
            alpha = 0.4
        )

        # Save the ssp126 plot separately
        if ssp_run == "ssp126"
            ssp126_plot = plot(
                plt,
                title = "",
                xlabel = "",
                xlims = (0, 1),
                ylabel = "",
                ylims = (10, 15),
                ticks = false,
                framestyle = :none,
                legend = :top
            )
        end

        # Add the plot to the array of plots
        push!(plots, plt)
    end

    # Combine all individual plots into a 2x2 layout
    combined_plot = plot(plots..., layout = (2, 2), legend = false)

    # Add the ssp126 plot underneath the 2x2 layout
    final_plot = plot(
        combined_plot, 
        ssp126_plot, 
        #layout = (2,1),
        layout = @layout([a{0.9h} ; b]),
        size = (1000, 750)    
        )

    savefig(final_plot, "output/Delta_T_SSP_Scenarios.pdf")

    return final_plot
end

#Figure 1b: Time Series Plot of ΔT for all ESMS, as well as constraint on a single plot
function produce_constrained_ΔT_time_series_for_all_SSP_Runs_color(directory, Delta_T_Obs_raw_df, ssp_runs, window_size, period_var, confidence)
    # Create the main plot for the time series data
    plt_main = plot(
        xlabel = "Year",
        ylabel = "ΔT($(period_var[1][1])) [K]",
        title = "ΔT for all SSP Runs with $(window_size)yr Smoothing",
        grid = :false,
        xlims = (1980, 2099),
        ylims = (-0.5, 8),
        legend =:false
    )

    # Create the subplot for the bar plot
    plt_bar = plot(
        grid = :false,
        ylims = (-0.5, 8),
        framestyle = :none
    )

    legend_shown = false
    #colors = reverse(palette(:coolwarm, length(ssp_runs)))
    colors = ["#ABDDA4", "#E6F598", "#FDAE61", "#fd7361"]

    for (index, ssp_run) in enumerate(ssp_runs)
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

        filter!(model -> length(model[2]) >= length(period_var), model_ΔT)

        # Get the minimum and maximum values at each time step across all models
        min_vals, max_vals = get_min_max_values(model_ΔT)

        if end_years[end] < Delta_T_Obs_raw_df.Year[end]
            final_year = end_years[end]
        else
            final_year = Delta_T_Obs_raw_df.Year[end] - floor(Int, window_size/2)
        end

        # Created a truncated version of the period_var_obs, where the end years only extend to the present day:
        period_var_obs = [(period_var[1][1], year) for year in period_var[1][2]:final_year]

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

        Delta_T_ESM_df_known = calculate_ESM_warming_trend(All_Models_1850_2100, period_var_obs[end], window_size, confidence)
        Delta_T_Obs_df_known = calculate_observed_warming_trend(Delta_T_Obs_raw_df, period_var_obs[end], window_size)

        # Create a dataframe to store the end year with the central ΔT prediction
        ΔT_Predictions = DataFrame(End_Year = Int64[], ΔT_Prediction = Float64[], ΔT_Upper = Float64[], ΔT_Lower = Float64[])

        for i in 1:length(period_var)
            period = period_var[i]
            
            Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
            results = calculate_emergent_constraint_GLM(Delta_T_ESM_df_known, Delta_T_ESM_df, Delta_T_Obs_df_known, confidence)
        
            y_vals, Py_vals = results[:ΔT_SSP_Range], results[:P_ΔT_SSP_vals]
            _, median, upper, lower = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

            # Check whether median value is appropriate, if not, replace with the last value
            if abs(median) < 1E-6 && i != 1
                println("Had to be replaced at year $(period[2])")
                median = ΔT_Predictions[!, :ΔT_Prediction][i-1]
                upper = ΔT_Predictions[!, :ΔT_Upper][i-1]
                lower = ΔT_Predictions[!, :ΔT_Lower][i-1]
            end

            push!(ΔT_Predictions, (End_Year = period[2], ΔT_Prediction = median, ΔT_Upper = upper, ΔT_Lower = lower))
        end

        color = colors[index]

        offset = calculate_temperature_anomaly_offset(Delta_T_Obs_raw_df, period_var[1][1], window_size)


        # Plot ΔT_Obs values over the top
        plot!(
            plt_main,
            end_years[1:length(ΔT_Obs)],
            ΔT_Obs .+ offset,
            fillbetween = (ΔT_Obs_Lower .+ offset, ΔT_Obs_Upper .+ offset),
            label = legend_shown ? "" : "Observed ΔT",
            seriestype = :line,
            color = :red,
            linecolor = :black,
            linewidth = 1,
            alpha = 0.2
        )

        legend_shown = true

        # Plot the model range
        plot!(
            plt_main,
            end_years,
            max_vals .+ offset,
            fillbetween = (min_vals .+ offset, max_vals .+ offset),
            label = "$(ssp_run) Model Range",
            linewidth = 0,
            fillalpha = 0.1,
            fillcolor = color 
        )


        # Plot ΔT Predictions
        plot!(
            plt_main,
            ΔT_Predictions[!, :End_Year],
            fillbetween = (ΔT_Predictions[!, :ΔT_Lower] .+ offset, ΔT_Predictions[!, :ΔT_Upper] .+ offset),
            ΔT_Predictions[!, :ΔT_Prediction] .+ offset,
            seriestype = :line,
            label = "$(ssp_run) $(Int(round(100*confidence)))% Constraint",
            linecolor = color,
            linealpha = 0.9,
            color = color,
            linewidth = 3,
            alpha = 0.3
        )

        # Plot the ranges of ΔT_Predictions as filled solid bars to the right
        bar_full_range = index + 1  # Adjust the x position to avoid overlap
        bar_constrained = index + 1  # Adjust the x position to avoid overlap

        bar_width = 0.8  # Width of the bar


        plot!(
            plt_bar,
            [bar_full_range - bar_width / 2, bar_full_range + bar_width / 2, bar_full_range + bar_width / 2, bar_full_range - bar_width / 2, bar_full_range - bar_width / 2],  # x positions for the bar
            [min_vals[end], min_vals[end], max_vals[end], max_vals[end], min_vals[end]] .+ offset,  # y positions for the bar
            seriestype = :shape,
            fillcolor = color,
            linecolor = color,  # Remove the outline
            label = "",
            alpha = 0.2
        )

        plot!(
            plt_bar,
            [bar_constrained - bar_width / 2, bar_constrained + bar_width / 2, bar_constrained + bar_width / 2, bar_constrained - bar_width / 2, bar_constrained - bar_width / 2],  # x positions for the bar
            [ΔT_Predictions[end, :ΔT_Lower], ΔT_Predictions[end, :ΔT_Lower], ΔT_Predictions[end, :ΔT_Upper], ΔT_Predictions[end, :ΔT_Upper], ΔT_Predictions[end, :ΔT_Lower]] .+ offset,  # y positions for the bar
            seriestype = :shape,
            fillcolor = color,
            linecolor = color,  # Remove the outline
            label = "",
            alpha = 0.2
        )

        # Add ΔT_Prediction as a line marker in the same color
        plot!(
            plt_bar,
            [bar_constrained - bar_width / 2, bar_constrained + bar_width / 2],  # x positions for the line
            [ΔT_Predictions[end, :ΔT_Prediction], ΔT_Predictions[end, :ΔT_Prediction]] .+ offset,  # y positions for the line
            seriestype = :line,
            linecolor = color,
            linewidth = bar_width,
            label = ""
        )

        legend_shown = true
    end

    # Combine the main plot and the bar plot into a single layout
    combined_plot = plot(plt_main, plt_bar, layout = @layout([a{0.8w} b{0.2w}]))

    savefig(combined_plot, "output/Delta_T_SSP_Scenarios_Combined.pdf")

    return combined_plot
end

#Figure 2: Time Series plot of ΔT for all ESMs
function produce_ΔT_estimates_for_SSP_Runs(directory, Delta_T_Obs_raw_df, ssp_runs, window_size, target_var, confidence)
    # Create the bar plot layout
    plt_bar = plot(
        xlabel = "",
        ylabel = "ΔT [K]",
        grid = :false,
        ylims = (1, 8),
        framestyle = :box,
        layout = (1, length(target_var))
    )

    colors = ["#02D977", "#BFEB64", "#FFAE65", "#FF7764"]

    for (year_index, period) in enumerate(target_var)
        for (index, ssp_run) in enumerate(ssp_runs)
            All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

            # Initialize a dictionary to store ΔT values for each model
            model_ΔT = Dict{String, Vector{Float64}}()

            Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
            for row in eachrow(Delta_T_ESM_df)
                model = row[:Model]
                ΔT = row[:ΔT]
                if !haskey(model_ΔT, model)
                    model_ΔT[model] = Float64[]
                end
                push!(model_ΔT[model], ΔT)
            end

            filter!(model -> length(model[2]) >= 1, model_ΔT)

            # Get the minimum and maximum values at the target year across all models
            min_vals, max_vals = get_min_max_values(model_ΔT)

            final_year = Delta_T_Obs_raw_df.Year[end] - floor(Int, window_size/2)

            Delta_T_ESM_df_known = calculate_ESM_warming_trend(All_Models_1850_2100, (target_var[1][1], final_year), window_size, confidence)
            Delta_T_Obs_df_known = calculate_observed_warming_trend(Delta_T_Obs_raw_df, (target_var[1][1], final_year), window_size)

            # Calculate the emergent constraint for the target year
            results = calculate_emergent_constraint_GLM(Delta_T_ESM_df_known, Delta_T_ESM_df, Delta_T_Obs_df_known, confidence)
            y_vals, Py_vals = results[:ΔT_SSP_Range], results[:P_ΔT_SSP_vals]
            _, median, upper, lower = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)

            color = colors[index]

            # Plot the ranges of ΔT_Predictions as filled solid bars
            bar_full_range = index   # Adjust the x position to avoid overlap
            bar_constrained = index   # Adjust the x position to avoid overlap

            bar_width = 0.8  # Width of the bar

            offset = calculate_temperature_anomaly_offset(Delta_T_Obs_raw_df, period_var[1][1], window_size)

            plot!(
                plt_bar[year_index],
                [bar_full_range - bar_width / 2, bar_full_range + bar_width / 2, bar_full_range + bar_width / 2, bar_full_range - bar_width / 2, bar_full_range - bar_width / 2],  # x positions for the bar
                [min_vals[end], min_vals[end], max_vals[end], max_vals[end], min_vals[end]] .+ offset,  # y positions for the bar
                seriestype = :shape,
                fillcolor = color,
                linecolor = color,  # Remove the outline
                label = "",
                alpha = 0.3
            )

            plot!(
                plt_bar[year_index],
                title = "$(period[2])",
                [bar_constrained - bar_width / 2, bar_constrained + bar_width / 2, bar_constrained + bar_width / 2, bar_constrained - bar_width / 2, bar_constrained - bar_width / 2],  # x positions for the bar
                [lower, lower, upper, upper, lower] .+ offset,  # y positions for the bar
                seriestype = :shape,
                fillcolor = color,
                linecolor = color,  # Remove the outline
                label = "",
                alpha = 0.3
            )

            # Add ΔT_Prediction as a line marker in the same color
            plot!(
                plt_bar[year_index],
                [bar_constrained - bar_width / 2, bar_constrained + bar_width / 2],  # x positions for the line
                [median, median] .+ offset,  # y positions for the line
                seriestype = :line,
                linecolor =:black,
                linewidth = 1,
                label = "",
            )

        end

        # Set the x-ticks to align with the bars
        xticks!(plt_bar[year_index], 1:length(ssp_runs), ssp_runs)
    end

    separate_plot = plot(
        title = "",
        xlabel = "",
        xlims = (0, 1),
        ylabel = "",
        ylims = (10, 15),
        ticks = false,
        framestyle = :none,
        legend = :top
    )

    function produce_separate_plot()
        period = target_var[1]
        ssp_run = ssp_runs[1]

        All_Models_1850_2100 = extract_data_for_ssp_run(directory, ssp_run)

        # Initialize a dictionary to store ΔT values for each model
        model_ΔT = Dict{String, Vector{Float64}}()
    
        Delta_T_ESM_df = calculate_ESM_warming_trend(All_Models_1850_2100, period, window_size, confidence)
        for row in eachrow(Delta_T_ESM_df)
            model = row[:Model]
            ΔT = row[:ΔT]
            if !haskey(model_ΔT, model)
                model_ΔT[model] = Float64[]
            end
            push!(model_ΔT[model], ΔT)
        end
    
        filter!(model -> length(model[2]) >= 1, model_ΔT)
    
        # Get the minimum and maximum values at the target year across all models
        min_vals, max_vals = get_min_max_values(model_ΔT)
    
        final_year = Delta_T_Obs_raw_df.Year[end] - floor(Int, window_size/2)
    
        Delta_T_ESM_df_known = calculate_ESM_warming_trend(All_Models_1850_2100, (target_var[1][1], final_year), window_size, confidence)
        Delta_T_Obs_df_known = calculate_observed_warming_trend(Delta_T_Obs_raw_df, (target_var[1][1], final_year), window_size)
    
        # Calculate the emergent constraint for the target year
        results = calculate_emergent_constraint_GLM(Delta_T_ESM_df_known, Delta_T_ESM_df, Delta_T_Obs_df_known, confidence)
        y_vals, Py_vals = results[:ΔT_SSP_Range], results[:P_ΔT_SSP_vals]
        _, median, upper, lower = calculate_emergent_constraint_cumulative(y_vals, Py_vals, confidence)
    
        color = colors[1]
    
        # Plot the ranges of ΔT_Predictions as filled solid bars
        bar_full_range = 1   # Adjust the x position to avoid overlap
        bar_constrained = 1   # Adjust the x position to avoid overlap
    
        bar_width = 0.8  # Width of the bar
    
        offset = calculate_temperature_anomaly_offset(Delta_T_Obs_raw_df, period_var[1][1], window_size)
    
        plot!(
            separate_plot[1],
            [bar_full_range - bar_width / 2, bar_full_range + bar_width / 2, bar_full_range + bar_width / 2, bar_full_range - bar_width / 2, bar_full_range - bar_width / 2],  # x positions for the bar
            [min_vals[end], min_vals[end], max_vals[end], max_vals[end], min_vals[end]] .+ offset,  # y positions for the bar
            seriestype = :shape,
            fillcolor = color,
            linecolor =:transparent,  # Remove the outline
            label = "CMIP6 ESM Range",
            alpha = 0.3
        )

        # Add ΔT_Prediction as a line marker in the same color
        plot!(
            separate_plot[1],
            [bar_constrained - bar_width / 2, bar_constrained + bar_width / 2],  # x positions for the line
            [median, median] .+ offset,  # y positions for the line
            seriestype = :line,
            linecolor =:black,
            linewidth = 1,
            label = "Central Prediction",
        )
    
        plot!(
            separate_plot[1],
            [bar_constrained - bar_width / 2, bar_constrained + bar_width / 2, bar_constrained + bar_width / 2, bar_constrained - bar_width / 2, bar_constrained - bar_width / 2],  # x positions for the bar
            [lower, lower, upper, upper, lower] .+ offset,  # y positions for the bar
            seriestype = :shape,
            fillcolor = color,
            linecolor =:transparent,  # Remove the outline
            label = "Constrained ΔT ($(Int(round(100*confidence)))% Confidence)",
            alpha = 0.6
        )
    

        return separate_plot
    end

    separate_plot = produce_separate_plot()

    # Combine plot_bar and separate_plot into a single layout
    combined_plot = plot(plt_bar, separate_plot, layout = @layout([a{0.85h} ; b]), size = (1000, 450))

    # Add a legend at the bottom of the bar plot
    savefig(combined_plot, "output/Delta_T_SSP_Scenarios_Barplot.pdf")

    return combined_plot
end

#Load in the Directory Path of the ESM Temperature Time Series Runs
ESM_Directory = "data/ESM_historical_runs/"

#Load in the Observational Temperature Anomaly Dataset
Delta_T_Obs_raw_df = CSV.read("data/observational_data/All_Data_Temperature_Anom.csv", DataFrame);

window_size = 11 # Window Size
confidence = 0.90 # Confidence Level

end_year_var = 1980:1:2099  # End years for the time series
target_years = (2030, 2050, 2090)

period_var = [(1980, end_year) for end_year in end_year_var]
target_var = [(1980, end_year) for end_year in target_years]

ssp_runs = ["ssp126", "ssp245", "ssp370", "ssp585"]

#produce_constrained_ΔT_time_series_for_SSP_Runs(ESM_Directory, Delta_T_Obs_raw_df, ssp_runs, window_size, period_var, confidence)
#produce_constrained_ΔT_time_series_for_all_SSP_Runs_color(ESM_Directory, Delta_T_Obs_raw_df, ssp_runs, window_size, period_var, confidence)
produce_ΔT_estimates_for_SSP_Runs(ESM_Directory, Delta_T_Obs_raw_df, ssp_runs, window_size, target_var, confidence)
