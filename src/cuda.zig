const std = @import("std");
const slimy = @import("slimy.zig");

const CudaResult = extern struct {
    x: i32,
    z: i32,
    count: u32,
};
const CudaContext = opaque {};

extern fn slimy_cuda_get_device_count() c_int;
extern fn slimy_cuda_get_device_score(device_index: c_int, score: *f32, free_mem: *u64, total_mem: *u64) c_int;
extern fn slimy_cuda_context_init(
    device_index: c_int,
    max_width: u32,
    max_height: u32,
    max_out_capacity: u32,
    out_ctx: *?*CudaContext,
) c_int;
extern fn slimy_cuda_context_deinit(ctx: *CudaContext) c_int;
extern fn slimy_cuda_search_batch_ctx(
    ctx: *CudaContext,
    world_seed: i64,
    threshold: u8,
    x0: i32,
    z0: i32,
    width: u32,
    height: u32,
    out_results: [*]CudaResult,
    out_capacity: u32,
    out_count: *u32,
) c_int;
extern fn slimy_cuda_search_batch(
    device_index: c_int,
    world_seed: i64,
    threshold: u8,
    x0: i32,
    z0: i32,
    width: u32,
    height: u32,
    out_results: [*]CudaResult,
    out_capacity: u32,
    out_count: *u32,
) c_int;

pub fn search(
    params: slimy.SearchParams,
    callback_context: anytype,
    comptime resultCallback: fn (@TypeOf(callback_context), slimy.Result) void,
    comptime progressCallback: ?fn (@TypeOf(callback_context), completed: u64, total: u64) void,
) !void {
    std.debug.assert(params.method == .cuda);

    var arena_impl = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const cfg = params.method.cuda;
    const devices = try discoverDevices(arena, cfg.max_cards);
    if (devices.len == 0) return error.NoCudaDevice;

    const width: u32 = @intCast(@as(i33, params.x1) - params.x0);
    const total_rows: u32 = @intCast(@as(i33, params.z1) - params.z0);

    const tasks = try buildStaticTasks(arena, devices, total_rows);
    var active_count: usize = 0;
    for (tasks) |task| {
        if (task.end_row > task.start_row) active_count += 1;
    }
    if (active_count == 0) return;

    var shared: Shared = .{
        .completed_rows = .init(0),
        .total_rows = total_rows,
        .width = width,
    };

    const threads = try arena.alloc(std.Thread, active_count);
    var t_idx: usize = 0;
    for (devices, tasks) |device, task| {
        if (task.end_row <= task.start_row) continue;
        const chunk_rows = try chooseChunkRows(width, task.end_row - task.start_row);
        threads[t_idx] = try std.Thread.spawn(.{}, worker, .{
            params,
            device,
            task,
            chunk_rows,
            &shared,
            callback_context,
            resultCallback,
            progressCallback,
        });
        t_idx += 1;
    }

    for (threads[0..t_idx]) |thread| {
        thread.join();
    }

    if (shared.first_error) |err| return err;
}

const Shared = struct {
    completed_rows: std.atomic.Value(u64),
    total_rows: u32,
    width: u32,

    err_lock: std.Thread.Mutex = .{},
    first_error: ?anyerror = null,

    fn noteError(self: *Shared, err: anyerror) void {
        self.err_lock.lock();
        defer self.err_lock.unlock();
        if (self.first_error == null) self.first_error = err;
    }
};

const Device = struct {
    index: i32,
    score: f64,
};

const Task = struct {
    start_row: u32,
    end_row: u32,
};

fn worker(
    params: slimy.SearchParams,
    device: Device,
    task: Task,
    chunk_rows: u32,
    shared: *Shared,
    callback_context: anytype,
    comptime resultCallback: fn (@TypeOf(callback_context), slimy.Result) void,
    comptime progressCallback: ?fn (@TypeOf(callback_context), completed: u64, total: u64) void,
) void {
    const max_height = chunk_rows;
    const capacity_u64 = @as(u64, shared.width) * max_height;
    const capacity = std.math.cast(u32, capacity_u64) orelse {
        shared.noteError(error.CudaBatchTooLarge);
        return;
    };

    const buf = std.heap.page_allocator.alloc(CudaResult, capacity) catch {
        shared.noteError(error.OutOfMemory);
        return;
    };
    defer std.heap.page_allocator.free(buf);

    var ctx: ?*CudaContext = null;
    if (slimy_cuda_context_init(device.index, shared.width, max_height, capacity, &ctx) != 0 or ctx == null) {
        shared.noteError(error.CudaRuntime);
        return;
    }
    defer _ = slimy_cuda_context_deinit(ctx.?);

    var cursor = task.start_row;
    while (cursor < task.end_row) {
        if (shared.first_error != null) return;

        const start = cursor;
        const end = @min(start + chunk_rows, task.end_row);
        const height = end - start;
        cursor = end;

        var out_count: u32 = 0;
        const batch_capacity = std.math.cast(u32, @as(u64, shared.width) * height) orelse {
            shared.noteError(error.CudaBatchTooLarge);
            return;
        };
        const code = slimy_cuda_search_batch_ctx(
            ctx.?,
            params.world_seed,
            params.threshold,
            params.x0,
            params.z0 + @as(i32, @intCast(start)),
            shared.width,
            height,
            buf.ptr,
            batch_capacity,
            &out_count,
        );
        switch (code) {
            0 => {},
            2 => {
                shared.noteError(error.CudaOutputOverflow);
                return;
            },
            else => {
                shared.noteError(error.CudaRuntime);
                return;
            },
        }

        for (buf[0..out_count]) |res| {
            resultCallback(callback_context, .{
                .x = res.x,
                .z = res.z,
                .count = res.count,
            });
        }

        if (progressCallback) |cb| {
            const completed_rows = shared.completed_rows.fetchAdd(height, .monotonic) + height;
            cb(callback_context, completed_rows * shared.width, @as(u64, shared.total_rows) * shared.width);
        }
    }
}

fn discoverDevices(allocator: std.mem.Allocator, max_cards: u8) ![]Device {
    const count = slimy_cuda_get_device_count();
    if (count <= 0) return try allocator.dupe(Device, &.{});

    var list = std.ArrayList(Device).empty;
    errdefer list.deinit(allocator);

    const hard_limit = if (max_cards == 0)
        @as(usize, @intCast(count))
    else
        @min(@as(usize, max_cards), @as(usize, @intCast(count)));

    for (0..hard_limit) |i| {
        var score: f32 = 0;
        var free_mem: u64 = 0;
        var total_mem: u64 = 0;

        const code = slimy_cuda_get_device_score(@intCast(i), &score, &free_mem, &total_mem);
        if (code != 0) continue;
        if (total_mem == 0) continue;

        try list.append(allocator, .{
            .index = @intCast(i),
            .score = @max(1.0, score),
        });
    }

    std.mem.sort(Device, list.items, {}, struct {
        fn lessThan(_: void, a: Device, b: Device) bool {
            return a.score > b.score;
        }
    }.lessThan);

    return list.toOwnedSlice(allocator);
}

fn buildStaticTasks(allocator: std.mem.Allocator, devices: []const Device, total_rows: u32) ![]Task {
    const tasks = try allocator.alloc(Task, devices.len);
    if (devices.len == 0 or total_rows == 0) {
        for (tasks) |*task| task.* = .{ .start_row = 0, .end_row = 0 };
        return tasks;
    }

    var total_score: f64 = 0;
    for (devices) |dev| total_score += dev.score;
    total_score = @max(1.0, total_score);

    var assigned: u32 = 0;
    var score_prefix: f64 = 0;
    for (devices, tasks, 0..) |dev, *task, i| {
        task.start_row = assigned;
        if (i + 1 == devices.len) {
            task.end_row = total_rows;
            assigned = total_rows;
            continue;
        }

        score_prefix += dev.score;
        const target_f = @as(f64, @floatFromInt(total_rows)) * (score_prefix / total_score);
        var target = std.math.lossyCast(u32, target_f);
        target = @max(target, assigned);
        target = @min(target, total_rows);
        task.end_row = target;
        assigned = target;
    }

    return tasks;
}

fn chooseChunkRows(width: u32, task_rows: u32) !u32 {
    if (width == 0 or task_rows == 0) return 1;

    // Aggressive by default, but bounded by per-batch output buffer bytes.
    const target_bytes_per_batch: u64 = 512 * 1024 * 1024;
    const bytes_per_row = @as(u64, width) * @sizeOf(CudaResult);
    if (bytes_per_row == 0 or bytes_per_row > target_bytes_per_batch) {
        return error.CudaBatchTooLarge;
    }

    const max_rows_by_capacity = @as(u32, @intCast(target_bytes_per_batch / bytes_per_row));
    if (max_rows_by_capacity == 0) return error.CudaBatchTooLarge;

    var rows = @min(task_rows, max_rows_by_capacity);
    rows = @min(rows, 16384);
    rows = @max(rows, 1);
    return rows;
}
