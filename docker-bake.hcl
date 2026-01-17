group "default" {
  targets = ["pysr"]
}

group "slurm" {
  targets = ["pysr-slurm"]
}

target "pysr" {
  dockerfile = "Dockerfile"
  target = "pysr"
  tags = ["pysr:local"]
}

target "pysr-runtime" {
  dockerfile = "Dockerfile"
  target = "pysr-runtime"
  tags = ["pysr-runtime:local"]
}

target "pysr-slurm" {
  dockerfile = "Dockerfile"
  target = "pysr-slurm"
  tags = ["pysr-slurm:local"]
}
