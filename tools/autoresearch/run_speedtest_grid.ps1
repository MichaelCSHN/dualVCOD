# Grid speedtest: 9 configs (batch_size=32/48/64 × num_workers=0/2/4)
$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..").Path

$configs = @(
    @{id="_speedtest_bs32_nw0"; cfg="configs\autoresearch\_speedtest_bs32_nw0.json"},
    @{id="_speedtest_bs32_nw2"; cfg="configs\autoresearch\_speedtest_bs32_nw2.json"},
    @{id="_speedtest_bs32_nw4"; cfg="configs\autoresearch\_speedtest_bs32_nw4.json"},
    @{id="_speedtest_bs48_nw0"; cfg="configs\autoresearch\_speedtest_bs48_nw0.json"},
    @{id="_speedtest_bs48_nw2"; cfg="configs\autoresearch\_speedtest_bs48_nw2.json"},
    @{id="_speedtest_bs48_nw4"; cfg="configs\autoresearch\_speedtest_bs48_nw4.json"},
    @{id="_speedtest_bs64_nw0"; cfg="configs\autoresearch\_speedtest_bs64_nw0.json"},
    @{id="_speedtest_bs64_nw2"; cfg="configs\autoresearch\_speedtest_bs64_nw2.json"},
    @{id="_speedtest_bs64_nw4"; cfg="configs\autoresearch\_speedtest_bs64_nw4.json"}
)

$total = $configs.Count
$idx = 0
$startAll = Get-Date

Write-Host "============================================================"
Write-Host "  GRID SPEEDTEST: $total configs (batch_size x num_workers)"
Write-Host "  Start: $($startAll.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "============================================================"

foreach ($c in $configs) {
    $idx++
    $elapsed = [math]::Round(((Get-Date) - $startAll).TotalMinutes, 1)
    Write-Host ""
    Write-Host "--- [$idx/$total] $($c.id) (elapsed: ${elapsed}m) ---"

    $startRun = Get-Date
    $exitCode = 0

    python "$ProjectRoot\tools\autoresearch\run_trial_minimal.py" `
        --trial_id $c.id `
        --config "$ProjectRoot\$($c.cfg)"

    $exitCode = $LASTEXITCODE
    $runTime = [math]::Round(((Get-Date) - $startRun).TotalSeconds, 0)

    if ($exitCode -ne 0) {
        Write-Host "  WARNING: $($c.id) exited with code $exitCode after ${runTime}s"
    } else {
        Write-Host "  OK: $($c.id) completed in ${runTime}s"
    }
}

$totalTime = [math]::Round(((Get-Date) - $startAll).TotalMinutes, 1)
Write-Host ""
Write-Host "============================================================"
Write-Host "  GRID COMPLETE in ${totalTime}m"
Write-Host "============================================================"
