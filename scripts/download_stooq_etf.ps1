param(
  [string]$Symbols = "IAUM,PDBC,PDBA,DBB",
  [string]$OutDir = "data/etf"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$syms = $Symbols.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne "" }
foreach ($s in $syms) {
  $url = "https://stooq.com/q/d/l/?s=$($s.ToLower()).us&i=d"
  $out = Join-Path $OutDir "$s.csv"
  try {
    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing | Out-Null
    $head = Get-Content $out -TotalCount 1
    if (-not $head -or $head -notmatch "Date,Open,High,Low,Close,Volume") {
      Write-Warning "Downloaded $s but header unexpected. Check $out."
    } else {
      Write-Host "wrote $out"
    }
  } catch {
    Write-Warning "failed $s from $url : $($_.Exception.Message)"
  }
}
