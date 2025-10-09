#!/usr/bin/env julia
#
# Deploy built VitePress documentation using DocumenterVitepress.jl
# This script is called after VitePress build is complete and handles:
# - Regular deployments to gh-pages
# - PR preview deployments to gh-pages/previews/PR##/
# - Dual deployment to secondary repository (ai.damtp.cam.ac.uk)
#

using DocumenterVitepress

# Get deployment target from environment (for dual deployment)
deployment_target = get(ENV, "DEPLOYMENT_TARGET", "primary")

println("Starting DocumenterVitepress deployment...")
println("Deployment target: $deployment_target")
println("Event: $(get(ENV, "GITHUB_EVENT_NAME", "unknown"))")
println("Ref: $(get(ENV, "GITHUB_REF", "unknown"))")

# DocumenterVitepress expects files in dist/1/ (versioned subdirectory)
# But VitePress builds directly to dist/, so we need to restructure
dist_root = joinpath(@__DIR__, "dist")
dist_versioned = joinpath(dist_root, "1")

if !isdir(dist_versioned) && isdir(dist_root)
    println("Restructuring dist/ for DocumenterVitepress...")
    # Move all files from dist/ to dist/1/
    temp_dir = joinpath(@__DIR__, "dist_temp")
    mv(dist_root, temp_dir)
    mkpath(dist_root)
    mv(temp_dir, dist_versioned)
end

# Create bases.txt (required by DocumenterVitepress)
bases_file = joinpath(dist_root, "bases.txt")
println("Creating bases.txt...")
write(bases_file, "1\n")

if deployment_target == "secondary"
    # Secondary deployment to ai.damtp.cam.ac.uk
    println("Deploying to secondary repository (ai.damtp.cam.ac.uk)")
    ENV["DOCUMENTER_KEY"] = get(ENV, "DAMTP_DEPLOY_KEY", "")
    DocumenterVitepress.deploydocs(;
        repo="github.com/ai-damtp-cam-ac-uk/pysr.git",
        push_preview=true,
        target=joinpath(@__DIR__, "dist"),
        devbranch="master",
    )
else
    # Primary deployment to MilesCranmer/PySR
    println("Deploying to primary repository (MilesCranmer/PySR)")
    DocumenterVitepress.deploydocs(;
        repo="github.com/MilesCranmer/PySR.git",
        push_preview=true,
        target=joinpath(@__DIR__, "dist"),
        devbranch="master",
    )
end

println("Deployment complete!")
