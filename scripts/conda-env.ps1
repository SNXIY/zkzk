param(
    [string]$EnvName = "nanobot-dev",
    [string]$PythonVersion = "3.11",
    [string]$Extras = "dev,api",
    [switch]$WithWebUI,
    [switch]$UseBun,
    [ValidateSet("setup", "status", "test", "lint", "gateway", "agent", "webui-dev", "webui-build", "remove")]
    [string]$Action = "setup",
    [string]$Message = "Hello!"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$WebUiDir = Join-Path $RepoRoot "webui"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Command '$Name' was not found. Install it first, then rerun this script."
    }
}

function Conda-Env-Exists {
    param([string]$Name)
    $envs = conda env list --json | ConvertFrom-Json
    foreach ($path in $envs.envs) {
        if ((Split-Path $path -Leaf) -eq $Name) {
            return $true
        }
    }
    return $false
}

function Conda-Run {
    $commandArgs = $args
    & conda run -n $EnvName @commandArgs
}

function Ensure-Env {
    Require-Command "conda"

    if (-not (Conda-Env-Exists $EnvName)) {
        Write-Host "Creating conda environment '$EnvName' with Python $PythonVersion..."
        & conda create -y -n $EnvName "python=$PythonVersion" pip
    }
    else {
        Write-Host "Using existing conda environment '$EnvName'."
    }

    if ($WithWebUI) {
        Write-Host "Ensuring Node.js is available in '$EnvName' for WebUI development..."
        & conda install -y -n $EnvName -c conda-forge nodejs
    }

    Write-Host "Upgrading pip tooling..."
    Conda-Run python -m pip install --upgrade pip setuptools wheel

    $editableSpec = "."
    if ($Extras.Trim().Length -gt 0) {
        $editableSpec = ".[${Extras}]"
    }

    Write-Host "Installing nanobot from source: pip install -e `"$editableSpec`""
    Push-Location $RepoRoot
    try {
        Conda-Run python -m pip install -e $editableSpec
    }
    finally {
        Pop-Location
    }

    if ($WithWebUI) {
        Push-Location $WebUiDir
        try {
            if ($UseBun) {
                Require-Command "bun"
                Write-Host "Installing WebUI dependencies with bun..."
                & bun install
            }
            else {
                Write-Host "Installing WebUI dependencies with npm ci..."
                Conda-Run npm ci
            }
        }
        finally {
            Pop-Location
        }
    }
}

switch ($Action) {
    "setup" {
        Ensure-Env
        Write-Host ""
        Write-Host "Done."
        Write-Host "Activate: conda activate $EnvName"
        Write-Host "Initialize config: nanobot onboard --wizard"
        Write-Host "Test backend: nanobot status; nanobot agent -m `"$Message`""
        Write-Host "Start WebUI gateway: nanobot gateway"
    }
    "status" {
        Conda-Run nanobot status
    }
    "test" {
        Push-Location $RepoRoot
        try { Conda-Run pytest }
        finally { Pop-Location }
    }
    "lint" {
        Push-Location $RepoRoot
        try { Conda-Run ruff check nanobot }
        finally { Pop-Location }
    }
    "gateway" {
        Conda-Run nanobot gateway
    }
    "agent" {
        Conda-Run nanobot agent -m $Message
    }
    "webui-dev" {
        Push-Location $WebUiDir
        try {
            if ($UseBun) {
                Require-Command "bun"
                & bun run dev
            }
            else {
                Conda-Run npm run dev
            }
        }
        finally {
            Pop-Location
        }
    }
    "webui-build" {
        Push-Location $WebUiDir
        try {
            if ($UseBun) {
                Require-Command "bun"
                & bun run build
            }
            else {
                Conda-Run npm run build
            }
        }
        finally {
            Pop-Location
        }
    }
    "remove" {
        Require-Command "conda"
        & conda env remove -y -n $EnvName
    }
}
