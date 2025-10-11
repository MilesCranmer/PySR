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

# Get deployment decision from Documenter to determine correct subfolder
using Documenter

# Custom DeployConfig that bypasses PR origin check for cross-repo deployments
# This allows deploying PR previews to ai.damtp.cam.ac.uk/pysr even though
# PRs exist in MilesCranmer/PySR (Documenter's security check would normally block this)
struct BypassPRCheckConfig <: Documenter.DeployConfig end

function Documenter.deploy_folder(
    ::BypassPRCheckConfig;
    repo,
    devbranch,
    devurl,
    push_preview,
    branch = "gh-pages",
    branch_previews = branch,
    kwargs...
)
    # Manually determine deployment subfolder from GitHub Actions environment
    github_event = get(ENV, "GITHUB_EVENT_NAME", "")
    github_ref = get(ENV, "GITHUB_REF", "")

    # Check for pull request
    if github_event == "pull_request" && push_preview
        m = match(r"refs/pull/(\d+)/merge", github_ref)
        if m !== nothing
            pr_number = m.captures[1]
            subfolder = "previews/PR$(pr_number)"
            println("BypassPRCheckConfig: Detected PR preview deployment to $(subfolder)")
            return Documenter.DeployDecision(;
                all_ok = true,
                branch = branch_previews,
                is_preview = true,
                repo = repo,
                subfolder = subfolder
            )
        end
    end

    # Check for master/main branch push
    if github_event in ["push", "workflow_dispatch", "schedule"]
        m = match(r"^refs/heads/(.*)$", github_ref)
        if m !== nothing && String(m.captures[1]) == devbranch
            println("BypassPRCheckConfig: Detected $(devbranch) branch deployment to $(devurl)")
            return Documenter.DeployDecision(;
                all_ok = true,
                branch = branch,
                is_preview = false,
                repo = repo,
                subfolder = devurl
            )
        end
    end

    # Check for tag deployment
    if occursin(r"^refs/tags/", github_ref)
        m = match(r"^refs/tags/(.*)$", github_ref)
        if m !== nothing
            tag = m.captures[1]
            println("BypassPRCheckConfig: Detected tag deployment to $(tag)")
            return Documenter.DeployDecision(;
                all_ok = true,
                branch = branch,
                is_preview = false,
                repo = repo,
                subfolder = tag
            )
        end
    end

    # No deployment
    println("BypassPRCheckConfig: No deployment criteria met")
    return Documenter.DeployDecision(; all_ok = false)
end

Documenter.authentication_method(::BypassPRCheckConfig) = Documenter.SSH
Documenter.documenter_key(::BypassPRCheckConfig) = ENV["DOCUMENTER_KEY"]

# Configure deployment based on target
if deployment_target == "secondary"
    # Secondary: Use custom config to bypass PR origin check
    deploy_config = BypassPRCheckConfig()
    ENV["DOCUMENTER_KEY"] = get(ENV, "DAMTP_DEPLOY_KEY", "")

    deploy_decision = Documenter.deploy_folder(
        deploy_config;
        repo="github.com/ai-damtp-cam-ac-uk/pysr",
        devbranch="master",
        devurl="dev",
        push_preview=true,
    )
else
    # Primary: Use normal Documenter flow with security checks
    deploy_config = Documenter.auto_detect_deploy_system()

    deploy_decision = Documenter.deploy_folder(
        deploy_config;
        repo="github.com/MilesCranmer/PySR",
        devbranch="master",
        devurl="dev",
        push_preview=true,
    )
end

println("Deploy decision: all_ok=$(deploy_decision.all_ok), is_preview=$(deploy_decision.is_preview), subfolder=$(deploy_decision.subfolder)")

# Build VitePress with the correct base path for this deployment
# VitePress needs the base path set at build time (it's hardcoded into assets)
# Primary uses /PySR/ (capital P), secondary uses /pysr/ (lowercase p)
base_prefix = deployment_target == "secondary" ? "/pysr/" : "/PySR/"
full_base = "$(base_prefix)$(deploy_decision.subfolder)$(isempty(deploy_decision.subfolder) ? "" : "/")"
println("Building VitePress with base: $full_base")

config_path = joinpath(@__DIR__, "src", ".vitepress", "config.mts")
original_config = read(config_path, String)
# Match either /pysr/ or /PySR/ in the config
modified_config = replace(original_config, r"base:\s*'/[Pp]y[Ss][Rr]/'" => "base: '$full_base'")
write(config_path, modified_config)

try
    # Build VitePress (outputs to docs/dist/)
    cd(@__DIR__) do
        run(`npm run build:vitepress`)
    end
    println("VitePress build complete")
finally
    # Restore original config (don't commit the modified version)
    write(config_path, original_config)
    println("Restored original config.mts")
end

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

# Create bases.txt with the correct subfolder from deploy_decision
# This tells DocumenterVitepress where to deploy (e.g., "dev", "previews/PR1056", "v1.2.3")
bases_file = joinpath(dist_root, "bases.txt")
println("Creating bases.txt with subfolder: $(deploy_decision.subfolder)")
write(bases_file, "$(deploy_decision.subfolder)\n")

if deployment_target == "secondary"
    # Secondary deployment to ai.damtp.cam.ac.uk
    println("Deploying to secondary repository (ai.damtp.cam.ac.uk)")
    DocumenterVitepress.deploydocs(;
        root=@__DIR__,
        repo="github.com/ai-damtp-cam-ac-uk/pysr.git",
        deploy_config=deploy_config,  # Use custom config with bypassed PR check
        push_preview=true,
        target="dist",
        devbranch="master",
    )
else
    # Primary deployment to MilesCranmer/PySR
    println("Deploying to primary repository (MilesCranmer/PySR)")
    DocumenterVitepress.deploydocs(;
        root=@__DIR__,
        repo="github.com/MilesCranmer/PySR.git",
        deploy_config=deploy_config,  # Use normal GitHubActions config
        push_preview=true,
        target="dist",
        devbranch="master",
    )
end

println("Deployment complete!")
