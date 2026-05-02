param(
    [string]$Registry = "",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$shortSha = (git rev-parse --short HEAD).Trim()
$branch = (git rev-parse --abbrev-ref HEAD).Replace("/", "-")
$safeBranch = if ([string]::IsNullOrWhiteSpace($branch)) { "unknown" } else { $branch }

$localName = "mlops-lab3-api"
$tag = "${safeBranch}-${shortSha}"
$fullLocal = "${localName}:${tag}"

Write-Host "Building $fullLocal ..."
docker build -t $fullLocal -t "${localName}:latest" $PSScriptRoot

if ($Registry) {
    $remote = "${Registry}/${localName}:${tag}"
    docker tag $fullLocal $remote
    docker tag "${localName}:latest" "${Registry}/${localName}:latest"
    Write-Host "Tagged $remote"
    if ($Push) {
        docker push $remote
        docker push "${Registry}/${localName}:latest"
    }
} elseif ($env:DOCKERHUB_USER) {
    $user = $env:DOCKERHUB_USER.Trim()
    $remote = "${user}/${localName}:${tag}"
    docker tag $fullLocal $remote
    docker tag "${localName}:latest" "${user}/${localName}:latest"
    Write-Host "Tagged $remote (set DOCKERHUB_USER)"
    if ($Push) {
        docker push $remote
        docker push "${user}/${localName}:latest"
    }
}
