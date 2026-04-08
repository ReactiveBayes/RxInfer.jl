module TensorBoardLoggerExt
    using RxInfer
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
    - Text summaries with event type information and counts

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

        events = RxInfer.tracedevents(trace)

        if isempty(events)
            @warn "No events recorded in trace"
            return nothing
        end

        @info "Collected $(length(events)) events from trace"

        # Create file writer
        logger = TBLogger(output_file)

        counts = Dict{Symbol, Int}()

        # Pre-compute iteration durations from matched before/after pairs via span_id
        iteration_durations = Dict{Int, Float64}()  # iteration number => elapsed ms
        before_times = Dict{Any, Tuple{Int, UInt64}}()  # span_id => (iteration, time_ns)
        for traced_event in events
            ev = traced_event.event
            et = event_name(typeof(ev))
            if et === :before_iteration
                before_times[ev.span_id] = (ev.iteration, traced_event.time_ns)
            elseif et === :after_iteration
                if haskey(before_times, ev.span_id)
                    (iter, t0) = before_times[ev.span_id]
                    iteration_durations[iter] = (traced_event.time_ns - t0) / 1e6
                end
            end
        end

        for (idx, traced_event) in enumerate(events)
            ev = traced_event.event
            event_type = event_name(typeof(ev))
            counts[event_type] = get(counts, event_type, 0) + 1

            # Log all events together in a single "Events" tag
            TensorBoardLogger.log_text(logger, "Events", "Step $idx: $(event_type)"; step=idx)

            if event_type === :before_model_creation
                TensorBoardLogger.log_text(logger, "before_model_creation",
                    "span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_model_creation
                TensorBoardLogger.log_text(logger, "after_model_creation",
                    "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :before_inference
                TensorBoardLogger.log_text(logger, "before_inference",
                    "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_inference
                TensorBoardLogger.log_text(logger, "after_inference",
                    "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :before_iteration
                TensorBoardLogger.log_text(logger, "before_iteration",
                    "model: $(ev.model) | iteration: $(ev.iteration) | stop_iteration: $(ev.stop_iteration) | span_id: $(ev.span_id)"; step=ev.iteration)

            elseif event_type === :after_iteration
                if haskey(iteration_durations, ev.iteration)
                    TensorBoardLogger.log_value(logger, "iteration_time_ms", iteration_durations[ev.iteration]; step=ev.iteration)
                end
                TensorBoardLogger.log_text(logger, "after_iteration",
                    "model: $(ev.model) | iteration: $(ev.iteration) | stop_iteration: $(ev.stop_iteration) | span_id: $(ev.span_id)"; step=ev.iteration)

            elseif event_type === :before_data_update
                TensorBoardLogger.log_text(logger, "before_data_update",
                    "model: $(ev.model) | data: $(ev.data) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_data_update
                TensorBoardLogger.log_text(logger, "after_data_update",
                    "model: $(ev.model) | data: $(ev.data) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :on_marginal_update
                TensorBoardLogger.log_text(logger, "on_marginal_update/$(ev.variable_name)",
                    "model: $(ev.model) | variable: $(ev.variable_name) | update: $(ev.update)"; step=idx)

            elseif event_type === :before_autostart
                TensorBoardLogger.log_text(logger, "before_autostart",
                    "engine: $(ev.engine) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_autostart
                TensorBoardLogger.log_text(logger, "after_autostart",
                    "engine: $(ev.engine) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :before_message_rule_call
                TensorBoardLogger.log_text(logger, "before_message_rule_call",
                    "mapping: $(ev.mapping) | messages: $(ev.messages) | marginals: $(ev.marginals) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_message_rule_call
                TensorBoardLogger.log_text(logger, "after_message_rule_call",
                    "mapping: $(ev.mapping) | messages: $(ev.messages) | marginals: $(ev.marginals) | result: $(ev.result) | annotations: $(ev.annotations) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :before_product_of_messages
                TensorBoardLogger.log_text(logger, "before_product_of_messages",
                    "variable: $(ev.variable) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_product_of_messages
                TensorBoardLogger.log_text(logger, "after_product_of_messages",
                    "variable: $(ev.variable) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :before_product_of_two_messages
                TensorBoardLogger.log_text(logger, "before_product_of_two_messages",
                    "variable: $(ev.variable) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_product_of_two_messages
                TensorBoardLogger.log_text(logger, "after_product_of_two_messages",
                    "variable: $(ev.variable) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | result: $(ev.result) | annotations: $(ev.annotations) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :before_marginal_computation
                TensorBoardLogger.log_text(logger, "before_marginal_computation",
                    "variable: $(ev.variable) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_marginal_computation
                TensorBoardLogger.log_text(logger, "after_marginal_computation",
                    "variable: $(ev.variable) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :before_form_constraint_applied
                TensorBoardLogger.log_text(logger, "before_form_constraint_applied",
                    "variable: $(ev.variable) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | span_id: $(ev.span_id)"; step=idx)

            elseif event_type === :after_form_constraint_applied
                TensorBoardLogger.log_text(logger, "after_form_constraint_applied",
                    "variable: $(ev.variable) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)

            else
                TensorBoardLogger.log_text(logger, "unknown_events",
                    "event_type: $(event_type)"; step=idx)
            end
        end

        # Log per-type counts as a single-column list
        sorted_counts = sort(collect(counts), by=first)
        counts_table = reshape(
            vcat(["$(k): $(v)" for (k, v) in sorted_counts], ["total: $(sum(values(counts)))"]),
            :, 1
        )
        TensorBoardLogger.log_text(logger, "EventCounts", counts_table; step=1)

        close(logger)

        @info "TensorBoard logs exported to: $output_file"
        @info "Total events logged: $(length(events))"
        @info ""
        @info "To view in TensorBoard, run:"
        @info "  tensorboard --logdir=\"$output_file\""

        return output_file
    end

end