using Distributed
using ClusterManagers

addprocs(SlurmManager(np))
addprocs(cpus_per)

hosts = []
pids = []
println("Workers are connected and ready.")
for i in workers()
	host, pid = fetch(@spawnat i (gethostname(), getpid()))
	println(host, pid)
	push!(hosts, host)
	push!(pids, pid)
end
