const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const assert = std.debug.assert;

const backend_supports_vectors = switch (@import("builtin").zig_backend) {
    .stage2_llvm, .stage2_c => true,
    else => false,
};

pub fn countScalar(comptime T: type, haystack: []const T, needle: T) usize {
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

test countScalar {
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

    const max_buffer_size = try std.fmt.parseInt(usize, args.next() orelse blk: {
        std.debug.print("Pass a max buffer size as the first argument to customize it, defaulting to 100_000\n", .{});
        break :blk "100_000";
    }, 10);

    std.debug.print("build mode: {s}\n", .{@tagName(mode)});
    std.debug.print("max buffer size (bytes): {d}\n", .{max_buffer_size});

    inline for (.{ u8, u16, u32, u64 }) |ElemType| {
        const buf = try allocator.alloc(ElemType, max_buffer_size / @sizeOf(ElemType));

        var rng = std.Random.DefaultPrng.init(0);

        // u8 so the counts arent just zero every time
        for (buf) |*e| e.* = rng.random().int(u8);

        const iters = 1024;
        var buffer_size: usize = 1;

        var out_file = try std.fs.cwd().createFile(@typeName(ElemType) ++ ".csv", .{});
        defer out_file.close();
        const output = out_file.writer();
        try output.print("size,current,naive,protty\n", .{});
        const values: []ElemType = try allocator.alloc(ElemType, iters);
        while (buffer_size * @sizeOf(ElemType) <= max_buffer_size) : (buffer_size = @min(buffer_size * 100 / 99 + 1, max_buffer_size) + @intFromBool(buffer_size == max_buffer_size)) {
            for (values) |*e| e.* = rng.random().int(u8);


            var timer = try std.time.Timer.start();

            for (buf[0..buffer_size]) |*e| @prefetch(e, .{});
            for (values) |v| std.mem.doNotOptimizeAway(std.mem.count(ElemType, buf[0..buffer_size], &.{v}));
            _ = timer.lap();
            for (values) |v| std.mem.doNotOptimizeAway(std.mem.count(ElemType, buf[0..buffer_size], &.{v}));
            const current_stdlib = timer.lap();

            for (buf[0..buffer_size]) |*e| @prefetch(e, .{});
            for (values) |v| std.mem.doNotOptimizeAway(countScalarNaive(ElemType, buf[0..buffer_size], v));
            _ = timer.lap();
            for (values) |v| std.mem.doNotOptimizeAway(countScalarNaive(ElemType, buf[0..buffer_size], v));
            const naive = timer.lap();

            for (buf[0..buffer_size]) |*e| @prefetch(e, .{});
            for (values) |v| std.mem.doNotOptimizeAway(countScalar(ElemType, buf[0..buffer_size], v));
            _ = timer.lap();
            for (values) |v| std.mem.doNotOptimizeAway(countScalar(ElemType, buf[0..buffer_size], v));
            const protty = timer.lap();

            // std.debug.print("current std lib: {d:>6.2}GB/s (runtime: {d:>9})\n", .{ @as(f64, @floatFromInt(buffer_size * iters)) / @as(f64, @floatFromInt(current_stdlib)), std.fmt.fmtDuration(current_stdlib) });
            // std.debug.print("naive:           {d:>6.2}GB/s (runtime: {d:>9})\n", .{ @as(f64, @floatFromInt(buffer_size * iters)) / @as(f64, @floatFromInt(naive)), std.fmt.fmtDuration(naive) });
            // std.debug.print("protty:          {d:>6.2}GB/s (runtime: {d:>9})\n", .{ @as(f64, @floatFromInt(buffer_size * iters)) / @as(f64, @floatFromInt(protty)), std.fmt.fmtDuration(protty) });

            try output.print("{},{d},{d},{d}\n", .{
                buffer_size * @sizeOf(ElemType),
                @as(f64, @floatFromInt(current_stdlib)) / iters,
                @as(f64, @floatFromInt(naive)) / iters,
                @as(f64, @floatFromInt(protty)) / iters,
            });
        }
    }
}
