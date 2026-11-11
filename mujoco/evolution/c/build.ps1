# Builds the C port against the mujoco.dll that the Python environment already
# installed, so nothing Python runs at build or run time.
#
#   .\build.ps1
#   .\build.ps1 -MujocoDir "C:\path\to\mujoco"
#
[CmdletBinding()]
param(
    [string]$MujocoDir = "",
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here

try {
    $gcc = Get-Command gcc -ErrorAction SilentlyContinue
    if (-not $gcc) {
        throw "gcc not found on PATH. Install MinGW-w64 (winget install BrechtSanders.WinLibs.POSIX.UCRT) and open a new shell."
    }

    if (-not $MujocoDir) {
        $candidates = @(
            (Join-Path $here "../../../.venv-mujoco/Lib/site-packages/mujoco"),
            (Join-Path $here "../../../.venv/Lib/site-packages/mujoco")
        )
        foreach ($candidate in $candidates) {
            if (Test-Path (Join-Path $candidate "mujoco.dll")) {
                $MujocoDir = (Resolve-Path $candidate).Path
                break
            }
        }
    }
    if (-not $MujocoDir -or -not (Test-Path (Join-Path $MujocoDir "mujoco.dll"))) {
        throw "mujoco.dll not found. Pass -MujocoDir with the folder that holds mujoco.dll and include\."
    }

    $MujocoDir = (Resolve-Path $MujocoDir).Path
    $dll = Join-Path $MujocoDir "mujoco.dll"
    $include = Join-Path $MujocoDir "include"

    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    $common = @("src/rng.c", "src/brain.c", "src/npz.c", "src/dogenv.c")
    $flags = @("-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic",
        "-I", $include, "-static-libgcc")

    foreach ($app in @("play_best", "ga_train")) {
        $out = Join-Path $BuildDir "$app.exe"
        & $gcc @flags -o $out "$app.c" @common $dll
        if ($LASTEXITCODE -ne 0) {
            throw "$app.c failed to build"
        }
        Write-Host "built $out"
    }

    # The exe loads mujoco.dll from its own folder before PATH.
    Copy-Item -Force $dll (Join-Path $BuildDir "mujoco.dll")
    Write-Host ""
    Write-Host "Next, from the repo root:"
    Write-Host "  .\mujoco\evolution\c\build\play_best.exe --episodes 1"
    Write-Host "  .\mujoco\evolution\c\build\ga_train.exe --generations 30 --pop 20"
}
finally {
    Pop-Location
}
