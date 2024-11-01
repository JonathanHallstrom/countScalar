const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const assert = std.debug.assert;

const backend_supports_vectors = switch (@import("builtin").zig_backend) {
    .stage2_llvm, .stage2_c => true,
    else => false,
};

fn bits(x: anytype) [@sizeOf(@TypeOf(x)) * 9 - 1]u8 {
    var out: [@sizeOf(@TypeOf(x)) * 9]u8 = undefined;
    for (0.., std.mem.toBytes(x)) |i, byte| {
        _ = std.fmt.bufPrint(out[9 * i ..], "{b:0>8} ", .{byte}) catch unreachable;
    }
    return out[0 .. @sizeOf(@TypeOf(x)) * 9 - 1].*;
}

pub fn countScalar(comptime T: type, haystack: []const T, needle: T) usize {
    var found: usize = 0;
    var i: usize = 0;

    if (backend_supports_vectors and
        !std.debug.inValgrind() and // https://github.com/ziglang/zig/issues/17717
        !@inComptime() and
        (@typeInfo(T) == .int or @typeInfo(T) == .float) and std.math.isPowerOfTwo(@bitSizeOf(T)))
    {
        const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
        if (std.simd.suggestVectorLength(T)) |main_block_len| {
            const max_iters_per_reset = 32;

            if (main_block_len * max_iters_per_reset < std.math.maxInt(usize)) {
                while (i + main_block_len * max_iters_per_reset <= haystack.len) {
                    var accs: @Vector(main_block_len, UT) = @splat(0);

                    for (0..max_iters_per_reset) |_| {
                        const zeros: @Vector(main_block_len, UT) = @splat(0);
                        const ones: @Vector(main_block_len, UT) = @splat(1);
                        const needles: @Vector(main_block_len, T) = @splat(needle);
                        const vals: @Vector(main_block_len, T) = haystack[i..][0..main_block_len].*;
                        accs += @select(UT, vals == needles, ones, zeros);
                        i += main_block_len;
                    }

                    for (@as([main_block_len]UT, accs)) |acc| found += @intCast(acc);
                }
            }

            if (i + main_block_len <= haystack.len) {
                var accs: @Vector(main_block_len, UT) = @splat(0);

                while (i + main_block_len <= haystack.len) {
                    const zeros: @Vector(main_block_len, UT) = @splat(0);
                    const ones: @Vector(main_block_len, UT) = @splat(1);
                    const needles: @Vector(main_block_len, T) = @splat(needle);
                    const vals: @Vector(main_block_len, T) = haystack[i..][0..main_block_len].*;
                    accs += @select(UT, vals == needles, ones, zeros);
                    i += main_block_len;
                }

                for (@as([main_block_len]UT, accs)) |acc| found += @intCast(acc);
            }

            const reg_size = @sizeOf(usize);
            const elem_bytes = @sizeOf(T);
            // swar
            if (reg_size > elem_bytes and @typeInfo(T) != .float) {
                const register_block_len = reg_size / elem_bytes;
                const lowest_bits: usize = std.math.maxInt(usize) / ((1 << @bitSizeOf(T)) - 1) * 0x01;
                const highest_bits = lowest_bits * (1 << @bitSizeOf(T) - 1);
                const needles = lowest_bits * needle;

                const native_endian = builtin.cpu.arch.endian();

                {
                    // handle [0, floor((main_block_len - 1) / block_len) * block_len] elements
                    // eg if main_block_len=32 and block_len=8 then this handles up to 24 elements
                    var is_needle: usize = 0;
                    comptime var num_bits_used = 0;
                    comptime var num_iters = (main_block_len / register_block_len) / 2;
                    inline while (num_iters > 0) : (num_iters /= 2) {
                        const run = i + num_iters * register_block_len <= haystack.len;
                        inline for (0..num_iters) |_| {
                            if (run) {
                                const vals = std.mem.readInt(usize, @ptrCast(haystack[i..][0..register_block_len]), native_endian);

                                const check = vals ^ needles;
                                const equal_lane = check | (check | highest_bits) - lowest_bits;
                                const res = ~equal_lane & highest_bits;

                                is_needle |= res >> num_bits_used;
                                i += register_block_len;
                            }
                            num_bits_used += 1;
                            if (num_bits_used == @bitSizeOf(T)) {
                                found += @popCount(is_needle);
                                num_bits_used = 0;
                                is_needle = 0;
                            }
                        }
                    }

                    if (num_bits_used > 0) {
                        found += @popCount(is_needle);
                    }
                }
                if (i < haystack.len) {
                    // handle last [0, block_len) elements
                    var vals: usize = ~needles;
                    const remaining = haystack.len - i;
                    comptime var block_len = register_block_len / 2;
                    assert(block_len & block_len - 1 == 0);
                    inline while (block_len > 0) : (block_len /= 2) {
                        if (remaining & block_len != 0) {
                            const bits_to_read = block_len * @bitSizeOf(T);
                            const IntToRead = std.meta.Int(.unsigned, bits_to_read);
                            vals <<= bits_to_read;
                            const read_int = std.mem.readInt(IntToRead, @ptrCast(haystack[i..][0..block_len]), native_endian);
                            vals |= read_int;
                            i += block_len;
                        }
                    }
                    const check = vals ^ needles;
                    const equal_lane = check | (check | highest_bits) - lowest_bits;
                    const res = ~equal_lane & highest_bits;
                    found += @popCount(res);
                    return found;
                }
            }
        }
    }

    for (haystack[i..]) |elem| {
        found += @intFromBool(std.meta.eql(elem, needle));
    }

    return found;
}


test countScalar {
    for (0..256) |i| {
        for (0..256) |j| {
            try testing.expectEqual(@intFromBool(i == j), countScalar(u8, &.{@intCast(i)}, @intCast(j)));
        }
    }
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
    try testing.expectEqual(1023, countScalar(u64, &(.{0} ** 1023), 0));
    try testing.expectEqual(4, countScalar(struct { u32, u32 }, &.{ .{ 0, 0 }, .{ 1, 1 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } }, .{ 0, 0 }));
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
        (@typeInfo(T) == .int or @typeInfo(T) == .float) and std.math.isPowerOfTwo(@bitSizeOf(T)))
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
                        haystack[haystack.len - (min_vec / 2)..][0..min_vec / 2].*,
                        haystack[0..min_vec / 2].*,
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
            found += V.countLast(vec_size, haystack[haystack.len - vec_size..][0..vec_size].*, needle, haystack.len % vec_size);
            return found;
        }
    }

    for (haystack) |elem| found += @intFromBool(elem == needle);
    return found;
}


test countScalarProtty {
    try testing.expectEqual(0, countScalarProtty(u8, &.{0}, 1));
    try testing.expectEqual(1, countScalarProtty(u8, &.{1}, 1));
    try testing.expectEqual(0, countScalarProtty(u8, &.{ 0, 0 }, 1));
    try testing.expectEqual(1, countScalarProtty(u8, &.{ 0, 1 }, 1));
    try testing.expectEqual(1, countScalarProtty(u8, &.{ 1, 0 }, 1));
    try testing.expectEqual(2, countScalarProtty(u8, &.{ 1, 1 }, 1));
    try testing.expectEqual(0, countScalarProtty(u8, "", 'h'));
    try testing.expectEqual(1, countScalarProtty(u8, "h", 'h'));
    try testing.expectEqual(2, countScalarProtty(u8, "hh", 'h'));
    try testing.expectEqual(0, countScalarProtty(u8, "world!", 'g'));
    try testing.expectEqual(1, countScalarProtty(u8, "hello world!", 'h'));
    try testing.expectEqual(3, countScalarProtty(u8, "   abcabc   abc", 'a'));
    try testing.expectEqual(2, countScalarProtty(u8, "udexdcbvbruhasdrw", 'b'));
    try testing.expectEqual(1, countScalarProtty(u8, "foo bar", 'b'));
    try testing.expectEqual(3, countScalarProtty(u8, "foofoofoo", 'f'));
    try testing.expectEqual(7, countScalarProtty(u8, "fffffff", 'f'));
    try testing.expectEqual(700, countScalarProtty(u8, "fffffff" ** 100, 'f'));
    try testing.expectEqual(3, countScalarProtty(u8, "owowowu", 'o'));
    try testing.expectEqual(300, countScalarProtty(u8, "owowowu" ** 100, 'o'));
    try testing.expectEqual(3, countScalarProtty(u64, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(3, countScalarProtty(f32, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(3, countScalarProtty(f64, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(300, countScalarProtty(u64, &(.{ 0, 0, 1, 2, 0, 3 } ** 100), 0));
    try testing.expectEqual(3, countScalarProtty(u128, &.{ 0, 0, 1, 2, 0, 3 }, 0));
    try testing.expectEqual(300, countScalarProtty(u128, &(.{ 0, 0, 1, 2, 0, 3 } ** 100), 0));
    try testing.expectEqual(4, countScalarProtty(struct { u32, u32 }, &.{ .{ 0, 0 }, .{ 1, 1 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } }, .{ 0, 0 }));
}

pub fn countScalarNaive(comptime T: type, haystack: []const T, needle: T) usize {
    var found: usize = 0;

    for (haystack) |elem| {
        found += @intFromBool(std.meta.eql(elem, needle));
    }

    return found;
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

            testing.expectEqual(
                countScalarNaive(T, haystack, needle),
                countScalar(T, haystack, needle),
            ) catch |e| {
                std.debug.print("{d} {any} {}\n", .{ needle, haystack, len });
                return e;
            };
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
pub fn main() !void {
    if (@import("builtin").os.tag == .linux) {
        const cpu0001: std.os.linux.cpu_set_t = [1]usize{0b0001} ++ ([_]usize{0} ** (16 - 1));
        try sched_setaffinity(0, &cpu0001);
    }
    const mode = @import("builtin").mode;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    _ = args.next().?;

    const ElemType = u8; // try different types!
    const buffer_size = try std.fmt.parseInt(usize, args.next() orelse blk: {
        std.debug.print("Pass a buffer size as the first argument to customize it, defaulting to 10000\n", .{});
        break :blk "10000";
    }, 10);
    std.debug.print("build mode: {s}\n", .{@tagName(mode)});
    std.debug.print("buffer size: {d}\n", .{buffer_size});
    std.debug.print("element type: {s}\n", .{@typeName(ElemType)});
    std.debug.print("buffer size (bytes): {d}\n", .{buffer_size * @sizeOf(ElemType)});

    const buf = try allocator.alloc(ElemType, buffer_size);

    var rng = std.Random.DefaultPrng.init(0);

    // u8 so the counts arent just zero every time
    for (buf) |*e| e.* = rng.random().int(u8);

    const iters = switch (mode) {
        .Debug => ((1 << 27) + buffer_size - 1) / buffer_size, // made debug mode actually complete in a reasonable amount of time
        else => ((1 << 30) + buffer_size - 1) / buffer_size,
    };

    const values: []ElemType = try allocator.alloc(ElemType, iters);
    for (values) |*e| e.* = rng.random().int(u8);

    for (buf) |*e| @prefetch(e, .{});

    var timer = try std.time.Timer.start();

    for (values) |v| std.mem.doNotOptimizeAway(std.mem.count(ElemType, buf, &.{v}));
    _ = timer.lap();
    for (values) |v| std.mem.doNotOptimizeAway(std.mem.count(ElemType, buf, &.{v}));
    const current_stdlib = timer.lap();

    for (values) |v| std.mem.doNotOptimizeAway(countScalarNaive(ElemType, buf, v));
    _ = timer.lap();
    for (values) |v| std.mem.doNotOptimizeAway(countScalarNaive(ElemType, buf, v));
    const naive = timer.lap();

    for (values) |v| std.mem.doNotOptimizeAway(countScalar(ElemType, buf, v));
    _ = timer.lap();
    for (values) |v| std.mem.doNotOptimizeAway(countScalar(ElemType, buf, v));
    const manual_simd = timer.lap();

    for (values) |v| std.mem.doNotOptimizeAway(countScalarProtty(ElemType, buf, v));
    _ = timer.lap();
    for (values) |v| std.mem.doNotOptimizeAway(countScalarProtty(ElemType, buf, v));
    const protty = timer.lap();

    std.debug.print("current std lib: {d:>6.2}GB/s (runtime: {d:>9})\n", .{ @as(f64, @floatFromInt(buffer_size * iters)) / @as(f64, @floatFromInt(current_stdlib)), std.fmt.fmtDuration(current_stdlib) });
    std.debug.print("naive:           {d:>6.2}GB/s (runtime: {d:>9})\n", .{ @as(f64, @floatFromInt(buffer_size * iters)) / @as(f64, @floatFromInt(naive)), std.fmt.fmtDuration(naive) });
    std.debug.print("manual simd:     {d:>6.2}GB/s (runtime: {d:>9})\n", .{ @as(f64, @floatFromInt(buffer_size * iters)) / @as(f64, @floatFromInt(manual_simd)), std.fmt.fmtDuration(manual_simd) });
    std.debug.print("protty:          {d:>6.2}GB/s (runtime: {d:>9})\n", .{ @as(f64, @floatFromInt(buffer_size * iters)) / @as(f64, @floatFromInt(protty)), std.fmt.fmtDuration(protty) });
}
