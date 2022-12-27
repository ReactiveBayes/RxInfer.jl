# julia -g 0 -O3 --min-optlevel=3 --check-bounds=no --startup-file=no example.jl
docker run --rm \
    --volume $PWD:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
