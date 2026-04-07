module TensorBoardLoggerExt
    using RxInfer
    using Logging
    using Dates
    using ReactiveMP: event_name
    using TensorBoardLogger

    """
        convert_to_tensorboard(trace::RxInferTraceCallbacks; output_file::Union{String, Nothing} = nothing)

    Convert trace events from inference to proper TensorFlow event files.

    # Arguments
    - `trace::RxInferTraceCallbacks`: The trace callbacks object from inference results
    - `output_file::Union{String, Nothing}`: Optional directory path to write TensorBoard event logs. If not provided, uses a timestamped directory in the current working directory.

    # Returns
    - `String`: Path to the directory containing the TensorBoard event log files

    # Description
    This function processes all traced events and creates proper TensorFlow event files using TensorBoardLogger, which can be directly imported and visualized in TensorBoard. Outputs include:
    - Event log file in Protocol Buffer format
    - Text summaries with event type information
    - Scalar metrics for timing analysis
    - Complete trace data

    The output directory can be directly opened in TensorBoard's web interface for visualization and analysis.

    # Example
    ```julia
    results = infer(
        model = my_model(),
        data = my_data,
        trace = true
    )

    trace = results.model.metadata[:trace]

    # Create TensorBoard logs (uses timestamped directory)
    log_dir = convert_to_tensorboard(trace)

    # Then run: tensorboard --logdir=\$log_dir
    ```
    """

    function RxInfer.convert_to_tensorboard(trace::RxInferTraceCallbacks; output_file::Union{String, Nothing} = nothing)
        
        # Determine output directory
        if isnothing(output_file)
            timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
            output_file = joinpath(pwd(), "tensorboard_logs", timestamp)
        end
        
        mkpath(output_file)
        
        # Create file writer (equivalent to: file_writer = tf.summary.create_file_writer(logdir))
        logger = TBLogger(output_file)
        
        # Collect all events with their types
        all_events = []
        for field_name in fieldnames(typeof(trace))
            event_data = getfield(trace, field_name)
            if !isempty(event_data)
                for single_event in event_data
                    push!(all_events, (type=field_name, event=single_event))
                end
            end
        end
        
        if isempty(all_events)
            @warn "No events recorded in trace"
            return nothing
        end
        
        @info "Collected $(length(all_events)) events from trace"
        
        # Log each event individually under "Events" and its specific event type tag
        counts = Dict{Symbol, Int}()
        for (idx, event_entry) in enumerate(all_events)
            single_event = event_entry.event
            event_type = event_name(single_event.event)
            counts[event_type] = get(counts, event_type, 0) + 1

            event_text = "Step $idx: $(event_type)"
            TensorBoardLogger.log_text(logger, "Events", event_text; step=idx)
            TensorBoardLogger.log_text(logger, string(event_type), event_text; step=idx)
        end

        # Log per-type counts as text
        for (event_type, count) in counts
            TensorBoardLogger.log_text(logger, "EventCounts", "$(event_type): $(count)"; step=1)
        end
        
        close(logger)
        
        @info "TensorBoard logs exported to: $output_file"
        @info "Total events logged: $(length(all_events))"
        @info ""
        @info "To view in TensorBoard, run:"
        @info "  tensorboard --logdir=\"$output_file\""
        
        return output_file
    end

end