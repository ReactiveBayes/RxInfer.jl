import JuliaFormatter;
using ArgParse;

s = ArgParseSettings()

@add_arg_table s begin
    "--overwrite"
    help = "overwrite your files with JuliaFormatter"
    action = :store_true
end

commandline_args = parse_args(s)
folders_to_format = ["scripts", "src", "test"]

overwrite = commandline_args["overwrite"]
formatted = all(
    map(
        folder -> JuliaFormatter.format(folder, overwrite = overwrite, verbose = true),
        folders_to_format
    )
)

if !formatted && !overwrite
    @error "JuliaFormatter lint has failed. Run `make format` from `ReactiveMP.jl` main directory and commit your changes to fix code style."
    exit(1)
elseif !formatted && overwrite
    @info "JuliaFormatter has overwritten files according to style guidelines"
elseif formatted
    @info "Codestyle from JuliaFormatted checks have passed"
end
