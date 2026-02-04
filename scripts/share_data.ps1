param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$ShareName = "market_data",
    [string[]]$ReadAccess = @("Everyone")
)

$resolved = Resolve-Path -Path $Path -ErrorAction Stop

if (-not (Test-Path -Path $resolved -PathType Container)) {
    Write-Error "Path is not a folder: $resolved"
    exit 1
}

try {
    $existing = Get-SmbShare -Name $ShareName -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Host "Share '$ShareName' already exists."
    } else {
        New-SmbShare -Name $ShareName -Path $resolved -ReadAccess $ReadAccess | Out-Null
        Write-Host "Created share '$ShareName' for $resolved"
    }

    $hostName = $env:COMPUTERNAME
    Write-Host "Share path: \\\\${hostName}\\${ShareName}"
    Write-Host "Use on PC: python scripts/get_data.py --src \"\\\\${hostName}\\${ShareName}\" --out \"data\""
} catch {
    Write-Error $_.Exception.Message
    Write-Host "Note: You may need to run this PowerShell session as Administrator."
    exit 1
}