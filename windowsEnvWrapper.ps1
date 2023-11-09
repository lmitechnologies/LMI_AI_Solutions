#For occasions where a user wants to use windows for data processing
#They should run this ps1 file to load the linux vars into windows

$envFile = "lmi_ai.env"

if (Test-Path $envFile) {
    $envLines = Get-Content $envFile
    foreach ($line in $envLines) {
        if ($line -match '^([^#=]+)=([^#]+)') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value)
        }
    }
    Write-Host "Environment variables from $envFile loaded."
} else {
    Write-Host "$envFile not found."
}