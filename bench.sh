#!/bin/bash

for mode in Debug ReleaseSmall ReleaseSafe ReleaseFast; do
    ~/dev/zig/zig/build/stage3/bin/zig build -Dcpu=znver5 -Doptimize=$mode & >/dev/null
done
wait
for mode in Debug ReleaseSmall ReleaseSafe ReleaseFast; do
    ~/dev/zig/zig/build/stage3/bin/zig build -Dcpu=znver5 -Doptimize=$mode run -- $1
done