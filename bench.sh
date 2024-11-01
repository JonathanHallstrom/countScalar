#!/bin/bash

for mode in Debug ReleaseSmall ReleaseSafe ReleaseFast; do
    zig build -Doptimize=$mode & >/dev/null
done
wait
for mode in Debug ReleaseSmall ReleaseSafe ReleaseFast; do
    zig build -Doptimize=$mode run -- $1
done