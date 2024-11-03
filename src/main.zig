const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const assert = std.debug.assert;

const backend_supports_vectors = switch (@import("builtin").zig_backend) {
    .stage2_llvm, .stage2_c => true,
    else => false,
};

pub fn countScalarNaive(comptime T: type, haystack: []const T, needle: T) usize {
    var found: usize = 0;

    for (haystack) |elem| {
        found += @intFromBool(std.meta.eql(elem, needle));
    }

    return found;
}

test countScalarNaive {
    try testCountScalar(countScalarNaive);
}

pub fn countScalarProtty(comptime T: type, haystack: []const T, needle: T) usize {
    var found: usize = 0;

    if (haystack.len == 0) return found;
    if (haystack.len < 4) {
        @branchHint(.unlikely);
        found += (haystack.len >> 1) & (haystack.len & 1) & @intFromBool(haystack[haystack.len - 1] == needle);
        found += (haystack.len >> 1) & @intFromBool(haystack[haystack.len >> 1] == needle);
        found += @intFromBool(haystack[0] == needle);
        return found;
    }

    if (backend_supports_vectors and
        !std.debug.inValgrind() and // https://github.com/ziglang/zig/issues/17717
        !@inComptime() and
        (@typeInfo(T) == .int or @typeInfo(T) == .float) and
        std.math.isPowerOfTwo(@bitSizeOf(T)) and
        std.meta.hasUniqueRepresentation(T))
    {
        const Count = std.meta.Int(.unsigned, @bitSizeOf(T));
        const V = struct {
            fn count(n: comptime_int, vec: @Vector(n, T), scalar: T) @Vector(n, Count) {
                const value: @Vector(n, T) = @splat(scalar);
                const ones: @Vector(n, Count) = @splat(1);
                return @select(Count, vec == value, ones, ones - ones);
            }

            fn countLast(n: comptime_int, vec: @Vector(n, T), scalar: T, len: usize) usize {
                const value: @Vector(n, T) = @splat(scalar);
                var mask: std.meta.Int(.unsigned, n) = @bitCast(vec == value);
                mask >>= @truncate(n - len);
                return @popCount(mask);
            }
        };

        if (std.simd.suggestVectorLength(T)) |vec_size| {
            inline for (2..@max(2, @ctz(@as(usize, vec_size)))) |n| {
                const min_vec: comptime_int = 2 << n;
                if (haystack.len <= min_vec) {
                    @branchHint(.unlikely);
                    const vec: @Vector(min_vec, T) = @bitCast([_][min_vec / 2]T{
                        haystack[haystack.len - (min_vec / 2) ..][0 .. min_vec / 2].*,
                        haystack[0 .. min_vec / 2].*,
                    });
                    return V.countLast(min_vec, vec, needle, haystack.len);
                }
            }

            var i: usize = 0;
            while (haystack[i..].len > vec_size) {
                @branchHint(.likely);
                var acc: @Vector(vec_size, Count) = @splat(0);
                for (0..@min((haystack[i..].len - 1) / vec_size, std.math.maxInt(Count), std.math.maxInt(usize))) |_| {
                    acc += V.count(vec_size, haystack[i..][0..vec_size].*, needle);
                    i += vec_size;
                }
                found += @reduce(.Add, @as(@Vector(vec_size, usize), @intCast(acc)));
            }
            found += V.countLast(vec_size, haystack[haystack.len - vec_size ..][0..vec_size].*, needle, haystack.len % vec_size);
            return found;
        }
    }

    for (haystack) |elem| found += @intFromBool(elem == needle);
    return found;
}

test countScalarProtty {
    try testCountScalar(countScalarProtty);
}

pub fn countScalarMultiAccum(comptime T: type, haystack: []const T, needle: T) usize {
    var found: usize = 0;

    if (haystack.len == 0) return found;
    if (haystack.len < 4) {
        @branchHint(.unlikely);
        found += (haystack.len >> 1) & (haystack.len & 1) & @intFromBool(haystack[haystack.len - 1] == needle);
        found += (haystack.len >> 1) & @intFromBool(haystack[haystack.len >> 1] == needle);
        found += @intFromBool(haystack[0] == needle);
        return found;
    }

    if (backend_supports_vectors and
        !std.debug.inValgrind() and // https://github.com/ziglang/zig/issues/17717
        !@inComptime() and
        (@typeInfo(T) == .int or @typeInfo(T) == .float) and
        std.math.isPowerOfTwo(@bitSizeOf(T)) and
        std.meta.hasUniqueRepresentation(T))
    {
        const Count = std.meta.Int(.unsigned, @bitSizeOf(T));
        const V = struct {
            fn count(n: comptime_int, vec: @Vector(n, T), scalar: T) @Vector(n, Count) {
                const value: @Vector(n, T) = @splat(scalar);
                const ones: @Vector(n, Count) = @splat(1);
                return @select(Count, vec == value, ones, ones - ones);
            }

            fn countLast(n: comptime_int, vec: @Vector(n, T), scalar: T, len: usize) usize {
                const value: @Vector(n, T) = @splat(scalar);
                var mask: std.meta.Int(.unsigned, n) = @bitCast(vec == value);
                mask >>= @truncate(n - len);
                return @popCount(mask);
            }
        };

        if (std.simd.suggestVectorLength(T)) |vec_size| {
            inline for (2..@max(2, @ctz(@as(usize, vec_size)))) |n| {
                const min_vec = 2 << n;
                if (haystack.len <= min_vec) {
                    @branchHint(.unlikely);
                    const vec: @Vector(min_vec, T) = @bitCast([_][min_vec / 2]T{
                        haystack[haystack.len - (min_vec / 2) ..][0 .. min_vec / 2].*,
                        haystack[0 .. min_vec / 2].*,
                    });
                    return V.countLast(min_vec, vec, needle, haystack.len);
                }
            }

            var i: usize = 0;
            while (haystack[i..].len > vec_size) {
                @branchHint(.unlikely);
                const num_accumulators = 4;
                var accs: [num_accumulators]@Vector(vec_size, Count) = .{@as(@Vector(vec_size, Count), @splat(0))} ** num_accumulators;
                const iters = @min((haystack[i..].len - 1) / vec_size, std.math.maxInt(Count), std.math.maxInt(usize));
                var iter: usize = 0;
                while (iter + num_accumulators - 1 < iters) : (iter += num_accumulators) {
                    for (0..num_accumulators) |acc_idx| {
                        accs[acc_idx] += V.count(vec_size, haystack[i..][0..vec_size].*, needle);
                        i += vec_size;
                    }
                }
                for (1..num_accumulators) |acc_idx| {
                    if (iter < iters) {
                        accs[0] += V.count(vec_size, haystack[i..][0..vec_size].*, needle);
                        i += vec_size;
                        iter += 1;
                    }
                    accs[0] += accs[acc_idx];
                }
                found += @reduce(.Add, @as(@Vector(vec_size, usize), @intCast(accs[0])));
            }
            found += V.countLast(vec_size, haystack[haystack.len - vec_size ..][0..vec_size].*, needle, haystack.len % vec_size);
            return found;
        }
    }

    for (haystack) |elem| found += @intFromBool(elem == needle);
    return found;
}

test countScalarMultiAccum {
    try testCountScalar(countScalarMultiAccum);
}

pub fn countScalarStreaming(comptime T: type, haystack: []const T, needle: T) usize {
    var found: usize = 0;

    if (haystack.len == 0) return found;
    if (haystack.len < 4) {
        @branchHint(.unlikely);
        found += (haystack.len >> 1) & (haystack.len & 1) & @intFromBool(haystack[haystack.len - 1] == needle);
        found += (haystack.len >> 1) & @intFromBool(haystack[haystack.len >> 1] == needle);
        found += @intFromBool(haystack[0] == needle);
        return found;
    }

    if (backend_supports_vectors and
        !std.debug.inValgrind() and // https://github.com/ziglang/zig/issues/17717
        !@inComptime() and
        (@typeInfo(T) == .int or @typeInfo(T) == .float) and
        std.math.isPowerOfTwo(@bitSizeOf(T)) and
        std.meta.hasUniqueRepresentation(T))
    {
        const Count = std.meta.Int(.unsigned, @bitSizeOf(T));
        const V = struct {
            fn count(n: comptime_int, vec: @Vector(n, T), scalar: T) @Vector(n, Count) {
                const value: @Vector(n, T) = @splat(scalar);
                const ones: @Vector(n, Count) = @splat(1);
                return @select(Count, vec == value, ones, ones - ones);
            }

            fn countLast(n: comptime_int, vec: @Vector(n, T), scalar: T, len: usize) usize {
                const value: @Vector(n, T) = @splat(scalar);
                var mask: std.meta.Int(.unsigned, n) = @bitCast(vec == value);
                mask >>= @truncate(n - len);
                return @popCount(mask);
            }
        };

        if (std.simd.suggestVectorLength(T)) |vec_size| {
            inline for (2..@max(2, @ctz(@as(usize, vec_size)))) |n| {
                const min_vec = 2 << n;
                if (haystack.len <= min_vec) {
                    @branchHint(.unlikely);
                    const vec: @Vector(min_vec, T) = @bitCast([_][min_vec / 2]T{
                        haystack[haystack.len - (min_vec / 2) ..][0 .. min_vec / 2].*,
                        haystack[0 .. min_vec / 2].*,
                    });
                    return V.countLast(min_vec, vec, needle, haystack.len);
                }
            }

            var i: usize = 0;
            while (haystack[i..].len > vec_size) {
                @branchHint(.unlikely);
                const num_accumulators = 4;
                var accs: [num_accumulators]@Vector(vec_size, Count) = .{@as(@Vector(vec_size, Count), @splat(0))} ** num_accumulators;
                const iters = @min((haystack[i..].len - 1) / vec_size, std.math.maxInt(Count), std.math.maxInt(usize));
                var iter: usize = 0;

                const page_size_bytes = std.mem.page_size;
                const page_size = page_size_bytes / @sizeOf(T);
                const vecs_per_page = page_size / vec_size;
                const cache_line_size = std.atomic.cache_line / @sizeOf(T);

                while (iter + vecs_per_page * num_accumulators - 1 < iters) : ({
                    iter += vecs_per_page * num_accumulators;
                    i += page_size * num_accumulators;
                }) {
                    @branchHint(.unlikely);
                    assert(vecs_per_page % 2 == 0);
                    inline for (0..vecs_per_page / 2) |half_vec_idx| {
                        const vec_idx = half_vec_idx * 2;
                        inline for (0..num_accumulators) |acc_idx| {
                            @prefetch(haystack[i + vec_idx * vec_size + acc_idx * page_size ..].ptr + 2 * cache_line_size, .{ .locality = 0 });
                            accs[acc_idx] += V.count(vec_size, haystack[i + vec_idx * vec_size + acc_idx * page_size ..][0..vec_size].*, needle);
                            accs[acc_idx] += V.count(vec_size, haystack[i + (vec_idx + 1) * vec_size + acc_idx * page_size ..][0..vec_size].*, needle);
                        }
                    }
                }

                while (iter + num_accumulators - 1 < iters) : (iter += num_accumulators) {
                    inline for (0..num_accumulators) |acc_idx| {
                        accs[acc_idx] += V.count(vec_size, haystack[i..][0..vec_size].*, needle);
                        i += vec_size;
                    }
                }
                for (1..num_accumulators) |acc_idx| {
                    if (iter < iters) {
                        accs[0] += V.count(vec_size, haystack[i..][0..vec_size].*, needle);
                        i += vec_size;
                        iter += 1;
                    }
                    accs[0] += accs[acc_idx];
                }
                found += @reduce(.Add, @as(@Vector(vec_size, usize), @intCast(accs[0])));
            }
            found += V.countLast(vec_size, haystack[haystack.len - vec_size ..][0..vec_size].*, needle, haystack.len % vec_size);
            return found;
        }
    }

    for (haystack) |elem| found += @intFromBool(elem == needle);
    return found;
}

test countScalarStreaming {
    try testCountScalar(countScalarStreaming);
}

fn testCountScalar(countScalar: anytype) !void {
    try testing.expectEqual(0, countScalar(u8, &.{0}, 1));
    try testing.expectEqual(1, countScalar(u8, &.{1}, 1));
    try testing.expectEqual(0, countScalar(u8, &.{ 0, 0 }, 1));
    try testing.expectEqual(1, countScalar(u8, &.{ 0, 1 }, 1));
    try testing.expectEqual(1, countScalar(u8, &.{ 1, 0 }, 1));
    try testing.expectEqual(2, countScalar(u8, &.{ 1, 1 }, 1));
    try testing.expectEqual(0, countScalar(u8, "", 'h'));
    try testing.expectEqual(1, countScalar(u8, "h", 'h'));
    try testing.expectEqual(2, countScalar(u8, "hh", 'h'));
    try testing.expectEqual(0, countScalar(u8, "world!", 'g'));
    try testing.expectEqual(1, countScalar(u8, "hello world!", 'h'));
    try testing.expectEqual(3, countScalar(u8, "   abcabc   abc", 'a'));
    try testing.expectEqual(2, countScalar(u8, "udexdcbvbruhasdrw", 'b'));
    try testing.expectEqual(1, countScalar(u8, "foo bar", 'b'));
    try testing.expectEqual(3, countScalar(u8, "foofoofoo", 'f'));
    try testing.expectEqual(7, countScalar(u8, "fffffff", 'f'));
    try testing.expectEqual(700, countScalar(u8, "fffffff" ** 100, 'f'));
    try testing.expectEqual(3, countScalar(u8, "owowowu", 'o'));
    try testing.expectEqual(300, countScalar(u8, "owowowu" ** 100, 'o'));
    try testing.expectEqual(3, countScalar(u64, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(3, countScalar(f32, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(3, countScalar(f64, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(300, countScalar(u64, &(.{ 0, 0, 1, 2, 0, 3 } ** 100), 0));
    try testing.expectEqual(3, countScalar(u128, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(300, countScalar(u128, &(.{ 0, 0, 1, 2, 0, 3 } ** 100), 0));
}

fn fuzzOne(bytes: []const u8) !void {
    inline for (.{ u8, u16, u24, u32, u64, f32, f64 }) |T| {
        const bytes_discard_for_alignment = (@alignOf(T) - @intFromPtr(bytes.ptr) % @alignOf(T)) % @alignOf(T);
        const array_len = (bytes.len -% bytes_discard_for_alignment) / @sizeOf(T);
        if (array_len >= 1 and bytes.len >= bytes_discard_for_alignment) {
            const array: []const T = @as([*]const T, @ptrCast(@alignCast(bytes.ptr[bytes_discard_for_alignment..])))[0..array_len];

            const len = array_len - 1;
            const needle = array[0];
            const haystack = array[1..];

            inline for (.{ "protty", "multi", "streaming" }, .{ countScalarProtty, countScalarMultiAccum, countScalarStreaming }) |name, countScalar| {
                testing.expectEqual(
                    countScalarNaive(T, haystack, needle),
                    countScalar(T, haystack, needle),
                ) catch |e| {
                    std.debug.print("{d} {any} {}\n", .{ needle, haystack, len });
                    const new_outp = try std.heap.page_allocator.dupe(T, haystack);
                    defer std.heap.page_allocator.free(new_outp);
                    for (new_outp) |*elem| {
                        if (elem.* == needle) {
                            elem.* = 1;
                        } else {
                            elem.* = 0;
                        }
                    }
                    std.debug.print("{s} failed on: {d} {any} {}\n", .{ name, 1, new_outp, len });
                    return e;
                };
            }
        }
    }
}

test "fuzzCountScalar" {
    try testing.fuzz(fuzzOne, .{});
}

pub fn sched_setaffinity(pid: std.os.linux.pid_t, set: *const std.os.linux.cpu_set_t) !void {
    const size = @sizeOf(std.os.linux.cpu_set_t);
    const rc = std.os.linux.syscall3(.sched_setaffinity, @as(usize, @bitCast(@as(isize, pid))), size, @intFromPtr(set));

    switch (std.posix.errno(rc)) {
        .SUCCESS => return,
        else => |err| return std.posix.unexpectedErrno(err),
    }
}

inline fn flushFromCache(comptime T: type, slice: []const T) void {
    var offs: usize = 0;
    while (offs < slice.len) : (offs += std.atomic.cache_line / @sizeOf(T)) {
        asm volatile ("clflush %[ptr]"
            :
            : [ptr] "m" (slice[offs..]),
            : "memory"
        );
    }
    // for (0..slice.len / @sizeOf(T)) |chunk| {
    //     const offset = slice.ptr + (chunk * @sizeOf(T));
    //     asm volatile ("clflush %[ptr]"
    //         :
    //         : [ptr] "m" (offset),
    //         : "memory"
    //     );
    // }
}

inline fn rdtsc() u64 {
    var a: u32 = undefined;
    var b: u32 = undefined;
    asm volatile ("rdtscp"
        : [a] "={edx}" (a),
          [b] "={eax}" (b),
        :
        : "ecx"
    );
    return (@as(u64, a) << 32) | b;
}

fn measureCycles(comptime func: anytype, args: anytype, comptime flush_func: anytype, flush_args: anytype) f64 {
    const invoke_n = struct {
        fn impl(n: usize, args_: anytype, flush_args_: anytype) usize {
            _ = @call(.auto, flush_func, flush_args_);
            const start = rdtsc();
            for (0..n) |_| {
                std.mem.doNotOptimizeAway(@call(.never_inline, func, args_));
            }
            const end = rdtsc();
            return end - start;
        }
    }.impl;

    const max_samples = 1 << 14;

    // minimum number of cycles we want each invocation to take
    // 640Ki should be enough for anyone
    const cycle_per_run_thresh = 640 << 10;
    const total_cycle_min_thresh = cycle_per_run_thresh << 2;
    const total_cycle_max_thresh = cycle_per_run_thresh << 10;

    var sum_cycles: u64 = invoke_n(1, args, flush_args);
    var run_iters: u64 = 1;
    while (sum_cycles < cycle_per_run_thresh) {
        run_iters *= 2;
        sum_cycles = invoke_n(run_iters, args, flush_args);
    }
    // one invocation is enough
    if (sum_cycles >= total_cycle_max_thresh) {
        return @floatFromInt(sum_cycles);
    }

    const min_sample_count = 2;
    var buf: [max_samples]u64 = undefined;
    var sample_count: usize = 1;
    buf[0] = sum_cycles;
    var sum_sqr: u64 = sum_cycles * sum_cycles;
    var avg = @as(f64, @floatFromInt(sum_cycles)) / @as(f64, @floatFromInt(sample_count));
    var std_dev = (@as(f64, @floatFromInt(sum_sqr)) - 2 * avg * @as(f64, @floatFromInt(sum_cycles))) / @as(f64, @floatFromInt(sample_count)) + avg * avg;
    var cv = (1 + 1 / @as(f64, @floatFromInt(4 * sample_count))) * std_dev / avg;
    const cv_thresh = 0.1;
    while ((cv > cv_thresh or sum_cycles < total_cycle_min_thresh or sample_count < min_sample_count) and sample_count < max_samples and sum_cycles < total_cycle_max_thresh) {
        const sample = invoke_n(run_iters, args, flush_args);
        sum_cycles += sample;
        sum_sqr += sample * sample;
        sample_count += 1;

        if (sample_count > 16 and cv > cv_thresh) {
            std.sort.pdq(u64, buf[0..sample_count], void{}, std.sort.asc(u64));

            const lo: f64 = @floatFromInt(buf[0]);
            const hi: f64 = @floatFromInt(buf[sample_count - 1]);
            if (@abs(lo - avg) > @abs(hi - avg)) {
                std.mem.copyForwards(u64, buf[0 .. sample_count - 1], buf[1..sample_count]);
            }
            sample_count -= 1;
        }

        avg = @as(f64, @floatFromInt(sum_cycles)) / @as(f64, @floatFromInt(sample_count));
        std_dev = @sqrt((@as(f64, @floatFromInt(sum_sqr)) - 2 * avg * @as(f64, @floatFromInt(sum_cycles))) / @as(f64, @floatFromInt(sample_count)) + avg * avg);
        cv = (1 + 1 / @as(f64, @floatFromInt(4 * sample_count))) * std_dev / avg;
    }

    return avg / @as(f64, @floatFromInt(run_iters));
}

pub fn main() !void {
    if (builtin.os.tag == .linux) {
        const cpu0001: std.os.linux.cpu_set_t = [1]usize{0b0001} ++ ([_]usize{0} ** (16 - 1));
        try sched_setaffinity(0, &cpu0001);
    }
    const mode = builtin.mode;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var arena = std.heap.ArenaAllocator.init(if (builtin.link_libc) std.heap.raw_c_allocator else gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    _ = args.next().?;

    const max_buffer_size_bytes = try std.fmt.parseInt(usize, args.next() orelse blk: {
        std.debug.print("Pass a max buffer size as the first argument to customize it, defaulting to 100_000\n", .{});
        break :blk "100_000";
    }, 10);

    std.debug.print("build mode: {s}\n", .{@tagName(mode)});
    std.debug.print("max buffer size (bytes): {d}\n", .{max_buffer_size_bytes});

    inline for (.{
        u8,
        u16,
        u32,
        u64,
    }) |ElemType| {
        const buf = try allocator.alloc(ElemType, max_buffer_size_bytes / @sizeOf(ElemType));

        var rng = std.Random.DefaultPrng.init(0);

        // u8 so the counts arent just zero every time
        for (buf) |*e| e.* = rng.random().int(u8);

        const num_values_to_test_correctness = 1024;
        var buffer_size: usize = 1;

        var out_file = try std.fs.cwd().createFile(@typeName(ElemType) ++ ".csv", .{});
        defer out_file.close();
        const output = out_file.writer();
        try output.print("size,current,naive,protty,multi,streaming\n", .{});
        const values: []ElemType = try allocator.alloc(ElemType, num_values_to_test_correctness);
        const max_buffer_size = max_buffer_size_bytes / @sizeOf(ElemType);
        while (buffer_size <= max_buffer_size) : (buffer_size = @min(buffer_size * 100 / 99 + 1, max_buffer_size) + @intFromBool(buffer_size == max_buffer_size)) {
            for (values) |*e| e.* = rng.random().int(u8);

            const value_to_look_for = rng.random().int(u8);
            const current_stdlib = measureCycles(std.mem.count, .{ ElemType, buf[0..buffer_size], &.{value_to_look_for} }, flushFromCache, .{ ElemType, buf[0..buffer_size] });
            const naive = measureCycles(countScalarNaive, .{ ElemType, buf[0..buffer_size], value_to_look_for }, flushFromCache, .{ ElemType, buf[0..buffer_size] });
            const protty = measureCycles(countScalarProtty, .{ ElemType, buf[0..buffer_size], value_to_look_for }, flushFromCache, .{ ElemType, buf[0..buffer_size] });
            const multi = measureCycles(countScalarMultiAccum, .{ ElemType, buf[0..buffer_size], value_to_look_for }, flushFromCache, .{ ElemType, buf[0..buffer_size] });
            const streaming = measureCycles(countScalarStreaming, .{ ElemType, buf[0..buffer_size], value_to_look_for }, flushFromCache, .{ ElemType, buf[0..buffer_size] });

            for (values) |v| {
                const naive_cnt = countScalarNaive(ElemType, buf[0..buffer_size], v);
                const protty_cnt = countScalarProtty(ElemType, buf[0..buffer_size], v);
                const multi_cnt = countScalarMultiAccum(ElemType, buf[0..buffer_size], v);
                const streaming_cnt = countScalarStreaming(ElemType, buf[0..buffer_size], v);
                try std.testing.expectEqual(naive_cnt, protty_cnt);
                try std.testing.expectEqual(naive_cnt, multi_cnt);
                try std.testing.expectEqual(naive_cnt, streaming_cnt);
            }

            try output.print("{},{d},{d},{d},{d},{d}\n", .{
                buffer_size * @sizeOf(ElemType),
                current_stdlib,
                naive,
                protty,
                multi,
                streaming,
            });
        }
    }
}
