#!/usr/bin/env bash
# Start two ArduCopter SITL instances for the formation (TCP 5760 and 5770).
# Usage: ARDUPILOT=~/ardupilot scripts/start_sitl.sh [speedup]
# Build first:  cd $ARDUPILOT && ./waf configure --board sitl && ./waf copter
set -euo pipefail
AP="${ARDUPILOT:?set ARDUPILOT to your ardupilot checkout}"
SPEEDUP="${1:-1}"
BIN="$AP/build/sitl/bin/arducopter"
PARM="$AP/Tools/autotest/default_params/copter.parm"
# Common origin (CMAC, ArduPilot's default test field). Launch points must match
# SimConfig.launch_positions: 5 m west and 5 m east of the origin.
LAT=-35.363261; LON=149.165230
DLON=$(python3 -c "import math;print(5/(6378137*math.cos(math.radians($LAT)))*180/math.pi)")
LON_L=$(python3 -c "print($LON-$DLON)"); LON_R=$(python3 -c "print($LON+$DLON)")
mkdir -p sitl/0 sitl/1
(cd sitl/0 && "$BIN" -S --model + --speedup "$SPEEDUP" -I0 --defaults "$PARM" --home "$LAT,$LON_L,584,0" > log.txt 2>&1 &)
(cd sitl/1 && "$BIN" -S --model + --speedup "$SPEEDUP" -I1 --defaults "$PARM" --home "$LAT,$LON_R,584,0" > log.txt 2>&1 &)
echo "SITL instances starting on tcp:127.0.0.1:5760 (left) and tcp:127.0.0.1:5770 (right)"
echo "Stop with: pkill -f build/sitl/bin/arducopter"
