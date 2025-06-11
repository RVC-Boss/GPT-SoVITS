$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Not Found"
    exit 1
}

$Lite = $false
$CudaVersion = "12.6"

function Write-Help {
    Write-Host @"
Usage: powershell -File docker_build.ps1 [OPTIONS]

Options:
  --cuda 12.6|12.8    Specify the CUDA VERSION (REQUIRED)
  --lite              Build a Lite Image
  -h, --help          Show this help message and exit

Examples:
  powershell -File docker_build.ps1 --cuda 12.6 --lite
"@
}

if ($args.Count -eq 0) {
    Write-Help
    exit 0
}

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        '--cuda' {
            $i++
            $val = $args[$i]
            if ($val -ne "12.6" -and $val -ne "12.8") {
                Write-Host "Error: Invalid CUDA_VERSION: $val"
                Write-Host "Choose From: [12.6, 12.8]"
                exit 1
            }
            $CudaVersion = $val
        }
        '--lite' {
            $Lite = $true
        }
        '-h' { Write-Help; exit 0 }
        '--help' { Write-Help; exit 0 }
        default {
            Write-Host "Unknown Argument: $($args[$i])"
            Write-Host "Use -h or --help to see available options."
            exit 1
        }
    }
}

$arch = (Get-CimInstance Win32_Processor).Architecture
$TargetPlatform = if ($arch -eq 9) { "linux/amd64" } else { "linux/arm64" }

if ($Lite) {
    $TorchBase = "lite"
} else {
    $TorchBase = "full"
}

docker build `
    --build-arg CUDA_VERSION=$CudaVersion `
    --build-arg LITE=$Lite `
    --build-arg TARGETPLATFORM=$TargetPlatform `
    --build-arg TORCH_BASE=$TorchBase `
    -t "$env:USERNAME/gpt-sovits:local" `
    .