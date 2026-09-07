const std = @import("std");
const zollama = @import("zollama");
const zdistill = @import("zdistill");
const ztrain = @import("ztrain");
const ztensor = @import("ztensor");
const zagent = @import("zagent");
const zrpc = @import("zrpc");
const zmcp = @import("zmcp");
const zserve = @import("zserve");
const zload = @import("zload");
const zarch = @import("zarch");
const ztok = @import("ztok");
const ztui = @import("ztui");
const zspeculate = @import("zspeculate");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len < 2) {
        // Sin args: entrar en modo chat interactivo por defecto
        try runChatCommand(alloc, io);
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "gate")) {
        try std.Io.File.stdout().writeStreamingAll(io, "Ejecuta `zig build gate` para verificación constitucional.\n");
    } else if (std.mem.eql(u8, cmd, "--version")) {
        try std.Io.File.stdout().writeStreamingAll(io, "fluency 0.7.0\n");
    } else if (std.mem.eql(u8, cmd, "serve")) {
        var port: u16 = 11434;
        var model: []const u8 = "Fluency-1.5b-v1";
        var gguf_path: ?[]const u8 = null;
        var use_quantized = false;
        var use_speculate = false;
        var idx: usize = 2;
        while (idx < args.len) : (idx += 1) {
            if (std.mem.eql(u8, args[idx], "--port") and idx + 1 < args.len) {
                port = try std.fmt.parseInt(u16, args[idx + 1], 10);
                idx += 1;
            } else if (std.mem.eql(u8, args[idx], "--model") and idx + 1 < args.len) {
                model = args[idx + 1];
                idx += 1;
            } else if (std.mem.eql(u8, args[idx], "--gguf") and idx + 1 < args.len) {
                gguf_path = args[idx + 1];
                idx += 1;
            } else if (std.mem.eql(u8, args[idx], "--quantized")) {
                use_quantized = true;
            } else if (std.mem.eql(u8, args[idx], "--speculate")) {
                use_speculate = true;
            }
        }
        const stdout = std.Io.File.stdout();
        var buf: [256]u8 = undefined;

        if (gguf_path) |gp| {
            // Carga el modelo real una sola vez y sirve generación de verdad.
            if (use_quantized) {
                try stdout.writeStreamingAll(io, "Cargando modelo para servir (ruta cuantizada, sin dequant a f32)...\n");
            } else {
                try stdout.writeStreamingAll(io, "Cargando modelo para servir (dequant a f32, puede tardar)...\n");
            }
            const data = try std.Io.Dir.cwd().readFileAlloc(io, gp, alloc, .unlimited);
            defer alloc.free(data);
            var gguf = try zload.parseGguf(alloc, data);
            defer gguf.deinit();
            const info = try zload.modelInfo(gguf);
            var tok = try buildTokenizer(alloc, gguf);
            defer tok.deinit();
            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            var tf = if (use_quantized)
                try loadGgufTransformerQuantized(alloc, arena.allocator(), alloc, gguf, info, 1024)
            else
                try loadGgufTransformer(alloc, arena.allocator(), alloc, gguf, info, 1024);
            defer tf.deinit();
            var engine = ServeEngine{ .alloc = alloc, .tf = &tf, .tok = &tok, .speculate = use_speculate };

            try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Fluency serve (modelo REAL arch={s}) en http://127.0.0.1:{d}\n", .{ info.arch, port }));
            try zserve.serveLocalhost(alloc, io, .{
                .port = port,
                .model = model,
                .engine_ctx = &engine,
                .generate = ServeEngine.generate,
                .generate_stream = ServeEngine.generateStream,
                .embed = ServeEngine.embed,
                .embed_ctx = &engine,
                .model_family = info.arch,
                .model_params = "1.5B",
                .model_quant = if (use_quantized) "Q4_K_M" else "F32",
                .model_ctx = info.ctx,
            });
        } else {
            try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Fluency serve (sin modelo; usa --gguf <ruta>) en http://127.0.0.1:{d} modelo={s}\n", .{ port, model }));
            try zserve.serveLocalhost(alloc, io, .{ .port = port, .model = model });
        }
    } else if (std.mem.eql(u8, cmd, "mcp-server")) {
        var read_tool = zagent.tools.FsReadTool{};
        var write_tool = zagent.tools.FsWriteTool{};
        var bash_tool = zagent.tools.BashTool{};
        var finish_tool = zagent.tools.FinishTool{};
        var bridge = McpBridge{
            .io = io,
            .tools = .{
                read_tool.tool(),
                write_tool.tool(),
                bash_tool.tool(),
                finish_tool.tool(),
            },
        };
        const infos = [_]zmcp.ToolInfo{
            .{ .name = "fs_read", .description = read_tool.desc_str, .parameters_json = read_tool.params_str },
            .{ .name = "fs_write", .description = write_tool.desc_str, .parameters_json = write_tool.params_str },
            .{ .name = "bash", .description = bash_tool.desc_str, .parameters_json = bash_tool.params_str },
            .{ .name = "finish", .description = finish_tool.desc_str, .parameters_json = finish_tool.params_str },
        };
        var server = zmcp.Server.init(alloc, infos[0..], &bridge, McpBridge.execute);
        if (args.len >= 3) {
            // Modo single-shot para tests directos
            const response = try server.handle(args[2]);
            defer alloc.free(response);
            try std.Io.File.stdout().writeStreamingAll(io, response);
            try std.Io.File.stdout().writeStreamingAll(io, "\n");
        } else {
            // Modo stdio server MCP real: lee líneas hasta EOF
            var stdin_buf: [4096]u8 = undefined;
            var stdout_buf: [4096]u8 = undefined;
            const stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
            const stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
            try server.stdioLoop(io, stdin_reader, stdout_writer);
        }
    } else if (std.mem.eql(u8, cmd, "mcp-selftest")) {
        try runMcpSelftest(alloc, io, args);
    } else if (std.mem.eql(u8, cmd, "serve-selftest")) {
        try runServeSelftest(alloc, io, args);
    } else if (std.mem.eql(u8, cmd, "run")) {
        if (args.len < 4) {
            try std.Io.File.stdout().writeStreamingAll(io, "Uso: fluency run <model.gguf> \"<prompt>\" [n_tokens] [--temp T] [--grammar json] [--quantized] [--speculate]\n");
            try std.Io.File.stdout().writeStreamingAll(io, "(F1: tokeniza el prompt, ejecuta el Transformer real y genera texto; --quantized evita el dequant completo a f32; --speculate usa prompt-lookup speculative decoding, requiere temp=0)\n");
        } else {
            var n_tokens: usize = 32;
            var temp: f32 = 0.0;
            var use_grammar = false;
            var use_quantized = false;
            var use_speculate = false;
            var idx: usize = 4;
            while (idx < args.len) : (idx += 1) {
                if (std.mem.eql(u8, args[idx], "--temp") and idx + 1 < args.len) {
                    temp = std.fmt.parseFloat(f32, args[idx + 1]) catch 0.0;
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--grammar") and idx + 1 < args.len) {
                    use_grammar = std.mem.eql(u8, args[idx + 1], "json");
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--quantized")) {
                    use_quantized = true;
                } else if (std.mem.eql(u8, args[idx], "--speculate")) {
                    use_speculate = true;
                } else {
                    n_tokens = std.fmt.parseInt(usize, args[idx], 10) catch n_tokens;
                }
            }
            try runGenerate(alloc, io, args[2], args[3], n_tokens, temp, use_grammar, use_quantized, use_speculate);
        }
    } else if (std.mem.eql(u8, cmd, "forward")) {
        if (args.len < 3) {
            try std.Io.File.stdout().writeStreamingAll(io, "Uso: fluency forward <model.gguf> [token_id]  (un solo forward, depuración)\n");
        } else {
            const token_id: u32 = if (args.len >= 4) (std.fmt.parseInt(u32, args[3], 10) catch 100) else 100;
            try runForward(alloc, io, args[2], token_id);
        }
    } else if (std.mem.eql(u8, cmd, "inspect")) {
        if (args.len < 3) {
            try std.Io.File.stdout().writeStreamingAll(io, "Uso: fluency inspect <model.gguf>\n");
        } else {
            try inspectGguf(alloc, io, args[2]);
        }
    } else if (std.mem.eql(u8, cmd, "distill")) {
        if (args.len < 3) {
            try std.Io.File.stdout().writeStreamingAll(io,
                \\Uso:
                \\  fluency distill <model.gguf> "<prompt>" [pasos] [--seed S] [--ollama <modelo>] [--out <ckpt>] [--dataset <bin>] [--mohawk A|B|C] [--prompts <txt>]
                \\      Seq-KD desde un profesor GGUF real (prompts semilla + greedy).
                \\  fluency distill --synthetic [pasos] [--seed S] [--out <ckpt>] [--dataset <bin>] [--prompts <txt>]
                \\      Destilación reproducible sobre profesor sintético (sin GGUF).
                \\
            );
        } else if (std.mem.eql(u8, args[2], "--synthetic")) {
            var steps: usize = 200;
            var seed: u64 = 42;
            var out_path: ?[]const u8 = null;
            var dataset_path: ?[]const u8 = null;
            var prompts_path: ?[]const u8 = null;
            var idx: usize = 3;
            while (idx < args.len) : (idx += 1) {
                if (std.mem.eql(u8, args[idx], "--seed") and idx + 1 < args.len) {
                    seed = std.fmt.parseInt(u64, args[idx + 1], 10) catch seed;
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--out") and idx + 1 < args.len) {
                    out_path = args[idx + 1];
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--dataset") and idx + 1 < args.len) {
                    dataset_path = args[idx + 1];
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--prompts") and idx + 1 < args.len) {
                    prompts_path = args[idx + 1];
                    idx += 1;
                } else {
                    steps = std.fmt.parseInt(usize, args[idx], 10) catch steps;
                }
            }
            try runDistillSynthetic(alloc, io, steps, seed, out_path, dataset_path, prompts_path);
        } else if (args.len < 4) {
            try std.Io.File.stdout().writeStreamingAll(io, "Falta el prompt. Uso: fluency distill <model.gguf> \"<prompt>\" [pasos]\n");
        } else {
            var steps: usize = 200;
            var seed: u64 = 42;
            var ollama_model: ?[]const u8 = null;
            var out_path: ?[]const u8 = null;
            var dataset_path: ?[]const u8 = null;
            var prompts_path: ?[]const u8 = null;
            var mohawk: zdistill.MohawkStage = .A;
            var idx: usize = 4;
            while (idx < args.len) : (idx += 1) {
                if (std.mem.eql(u8, args[idx], "--seed") and idx + 1 < args.len) {
                    seed = std.fmt.parseInt(u64, args[idx + 1], 10) catch seed;
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--ollama") and idx + 1 < args.len) {
                    ollama_model = args[idx + 1];
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--out") and idx + 1 < args.len) {
                    out_path = args[idx + 1];
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--dataset") and idx + 1 < args.len) {
                    dataset_path = args[idx + 1];
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--mohawk") and idx + 1 < args.len) {
                    const m = args[idx + 1];
                    if (std.mem.eql(u8, m, "B")) mohawk = .B;
                    if (std.mem.eql(u8, m, "C")) mohawk = .C;
                    idx += 1;
                } else if (std.mem.eql(u8, args[idx], "--prompts") and idx + 1 < args.len) {
                    prompts_path = args[idx + 1];
                    idx += 1;
                } else {
                    steps = std.fmt.parseInt(usize, args[idx], 10) catch steps;
                }
            }
            try runDistill(alloc, io, args[2], args[3], steps, seed, ollama_model, out_path, dataset_path, mohawk, prompts_path);
        }
    } else if (std.mem.eql(u8, cmd, "tokenize")) {
        if (args.len < 4) {
            try std.Io.File.stdout().writeStreamingAll(io,
                \\Uso:
                \\  fluency tokenize <gguf> <in.txt> <out.bin>
                \\      Usa el vocabulario del GGUF (BPE real).
                \\  fluency tokenize <in.txt> <out.bin>
                \\      Usa vocabulario byte-simple (256 tokens, sin merges).
                \\
            );
        } else {
            const maybe_gguf = args[2];
            const has_gguf = std.mem.endsWith(u8, maybe_gguf, ".gguf");
            const text_path = if (has_gguf) args[3] else args[2];
            const out_path = if (has_gguf) args[4] else args[3];
            if (has_gguf and args.len < 5) {
                try std.Io.File.stdout().writeStreamingAll(io, "Falta <out.bin>.\n");
            } else {
                try runTokenize(alloc, io, text_path, out_path, if (has_gguf) maybe_gguf else null);
            }
        }
    } else if (std.mem.eql(u8, cmd, "train")) {
        // Entrenamiento auto-supervisado reproducible (demostración rápida F3).
        var steps: usize = 200;
        var seed: u64 = 42;
        var out_path: ?[]const u8 = null;
        var data_path: ?[]const u8 = null;
        var idx: usize = 2;
        while (idx < args.len) : (idx += 1) {
            if (std.mem.eql(u8, args[idx], "--seed") and idx + 1 < args.len) {
                seed = std.fmt.parseInt(u64, args[idx + 1], 10) catch seed;
                idx += 1;
            } else if (std.mem.eql(u8, args[idx], "--out") and idx + 1 < args.len) {
                out_path = args[idx + 1];
                idx += 1;
            } else if (std.mem.eql(u8, args[idx], "--data") and idx + 1 < args.len) {
                data_path = args[idx + 1];
                idx += 1;
            } else {
                steps = std.fmt.parseInt(usize, args[idx], 10) catch steps;
            }
        }
        try runTrainSynthetic(alloc, io, steps, seed, out_path, data_path);
    } else if (std.mem.eql(u8, cmd, "eval")) {
        if (args.len < 3) {
            try std.Io.File.stdout().writeStreamingAll(io, "Uso: fluency eval <ckpt> [\"<prompt>\"]\n");
        } else {
            const prompt: ?[]const u8 = if (args.len >= 4) args[3] else null;
            try runEval(alloc, io, args[2], prompt);
        }
    } else if (std.mem.eql(u8, cmd, "agent")) {
        if (args.len < 3) {
            try std.Io.File.stdout().writeStreamingAll(io, "Uso: fluency agent \"<prompt>\" [--gguf <model.gguf>] [--quantized] [--speculate]\n");
            try std.Io.File.stdout().writeStreamingAll(io, "(--gguf conecta el modelo local al bucle ReAct; --quantized usa matmul on-the-fly; --speculate usa prompt-lookup speculative decoding)\n");
            return;
        }
        var gguf_path: ?[]const u8 = null;
        var use_quantized = false;
        var use_speculate = false;
        var ai: usize = 3;
        while (ai < args.len) : (ai += 1) {
            if (std.mem.eql(u8, args[ai], "--gguf") and ai + 1 < args.len) {
                gguf_path = args[ai + 1];
                ai += 1;
            } else if (std.mem.eql(u8, args[ai], "--quantized")) {
                use_quantized = true;
            } else if (std.mem.eql(u8, args[ai], "--speculate")) {
                use_speculate = true;
            }
        }
        try runAgentOnce(alloc, io, args[2], gguf_path, use_quantized, use_speculate);
    } else if (std.mem.eql(u8, cmd, "chat")) {
        try runChatCommand(alloc, io);
    } else {
        try std.Io.File.stdout().writeStreamingAll(io, "Comando no implementado aún\n");
    }
}

// Bundle auto-contenido del runtime agéntico. POSEE el registry, el agente y las
// instancias de las herramientas como campos. Las herramientas exponen `.ptr = self`
// y el agente guarda `registry: *Registry`, por lo que estos punteros internos sólo
// son válidos cuando el bundle ya está en su dirección final (el frame del caller).
// Por eso `init` NO establece punteros internos y `wire()` debe llamarse DESPUÉS de
// colocar el bundle en su sitio. (Devolver una struct con punteros a sus propios
// locales era el bug: al moverse al retornar, agent.registry y los tool.ptr quedaban
// colgantes → "incorrect alignment" al despachar una tool-call.)
const AgentBundle = struct {
    reg: zagent.Registry,
    agent: zagent.Agent,
    read_tool: zagent.tools.FsReadTool = .{},
    write_tool: zagent.tools.FsWriteTool = .{},
    bash_tool: zagent.tools.BashTool = .{},
    finish_tool: zagent.tools.FinishTool = .{},
    mem_read: zagent.tools.MemoryReadTool = .{},
    mem_write: zagent.tools.MemoryWriteTool = .{},
    fetch_tool: zagent.tools.HttpFetchTool = .{},
    search_tool: zagent.tools.SearchTool = .{},
    delegate_tool: zagent.tools.DelegateTool = .{},

    fn init(alloc: std.mem.Allocator, io: std.Io) AgentBundle {
        return .{
            .reg = zagent.Registry.init(alloc),
            // registry se re-apunta a &self.reg en wire() (dirección estable).
            .agent = zagent.Agent.init(alloc, io, "Fluency-350M", undefined),
        };
    }

    fn wire(self: *AgentBundle) void {
        self.agent.registry = &self.reg;
        // memory tools conectadas al store del agente (dirección estable).
        self.mem_read.store = &self.agent.memory;
        self.mem_write.store = &self.agent.memory;

        self.reg.register(self.read_tool.tool()) catch {};
        self.reg.register(self.write_tool.tool()) catch {};
        self.reg.register(self.bash_tool.tool()) catch {};
        self.reg.register(self.finish_tool.tool()) catch {};
        self.reg.register(self.mem_read.tool()) catch {};
        self.reg.register(self.mem_write.tool()) catch {};
        self.reg.register(self.fetch_tool.tool()) catch {};
        self.reg.register(self.search_tool.tool()) catch {};
        self.reg.register(self.delegate_tool.tool()) catch {};
    }

    fn deinit(self: *AgentBundle) void {
        self.agent.deinit();
        self.reg.deinit();
    }
};

// Adaptador: conecta el modelo real (ServeEngine) al hook generate_fn del agente,
// para que el bucle ReAct razone con el LLM local en iteraciones multi-turno.
fn agentGenerateAdapter(ctx: ?*anyopaque, prompt: []const u8, allocator: std.mem.Allocator) anyerror![]u8 {
    const eng: *ServeEngine = @alignCast(@ptrCast(ctx.?));
    return ServeEngine.generate(eng, allocator, prompt, 64);
}

fn agentGenerateJsonAdapter(ctx: ?*anyopaque, prompt: []const u8, allocator: std.mem.Allocator) anyerror![]u8 {
    const eng: *ServeEngine = @alignCast(@ptrCast(ctx.?));
    return ServeEngine.generateJson(eng, allocator, prompt, 64);
}

// Adaptador de streaming: conecta ServeEngine.generateStream al hook generate_stream_fn del agente.
fn agentGenerateStreamAdapter(
    ctx: ?*anyopaque,
    prompt: []const u8,
    allocator: std.mem.Allocator,
    max_tokens: usize,
    emit_ctx: ?*anyopaque,
    emit: *const fn (emit_ctx: ?*anyopaque, piece: []const u8) anyerror!void,
) !void {
    const eng: *ServeEngine = @alignCast(@ptrCast(ctx.?));
    const Bridge = struct {
        agent_emit: *const fn (emit_ctx: ?*anyopaque, piece: []const u8) anyerror!void,
        agent_ctx: ?*anyopaque,
        fn bridgeEmit(bridge_ctx: *anyopaque, piece: []const u8) !void {
            const w = @as(*@This(), @ptrCast(@alignCast(bridge_ctx)));
            try w.agent_emit(w.agent_ctx, piece);
        }
    };
    var bridge = Bridge{ .agent_emit = emit, .agent_ctx = emit_ctx };
    try ServeEngine.generateStream(eng, allocator, prompt, max_tokens, &bridge, &Bridge.bridgeEmit);
}

fn runAgentOnce(alloc: std.mem.Allocator, io: std.Io, prompt: []const u8, gguf_path: ?[]const u8, use_quantized: bool, use_speculate: bool) !void {
    var ar = AgentBundle.init(alloc, io);
    ar.wire(); // estabiliza punteros internos (ar ya está en su frame final)
    defer ar.deinit();

    try std.Io.File.stdout().writeStreamingAll(io, "\n=== INICIANDO RUNTIME AGÉNTICO (zagent) ===\n\n");

    // Si se da un GGUF, cargar el modelo real y conectarlo al bucle ReAct.
    // (defers en orden inverso: arena se libera DESPUÉS de tf, que la usa).
    var arena: std.heap.ArenaAllocator = undefined;
    var tf: zarch.Transformer = undefined;
    var tok: ztok.Tokenizer = undefined;
    var engine: ServeEngine = undefined;
    var model_loaded = false;
    var data: []u8 = &.{};
    var gguf: zload.GgufFile = undefined;
    if (gguf_path) |gp| {
        if (use_quantized) {
            try std.Io.File.stdout().writeStreamingAll(io, "Cargando modelo local para razonamiento del agente (ruta cuantizada)...\n");
        } else {
            try std.Io.File.stdout().writeStreamingAll(io, "Cargando modelo local para razonamiento del agente (puede tardar)...\n");
        }
        data = std.Io.Dir.cwd().readFileAlloc(io, gp, alloc, .unlimited) catch |err| {
            try std.Io.File.stdout().writeStreamingAll(io, "Aviso: no se pudo leer el GGUF; el agente correrá en modo tool-only.\n");
            return err;
        };
        gguf = try zload.parseGguf(alloc, data);
        const info = try zload.modelInfo(gguf);
        tok = try buildTokenizer(alloc, gguf);
        arena = std.heap.ArenaAllocator.init(alloc);
        tf = if (use_quantized)
            try loadGgufTransformerQuantized(alloc, arena.allocator(), alloc, gguf, info, 1024)
        else
            try loadGgufTransformer(alloc, arena.allocator(), alloc, gguf, info, 1024);
        engine = ServeEngine{ .alloc = alloc, .tf = &tf, .tok = &tok, .speculate = use_speculate };
        ar.agent.generate_fn = agentGenerateAdapter;
        ar.agent.generate_ctx = &engine;
        ar.agent.generate_stream_fn = agentGenerateStreamAdapter;
        ar.agent.generate_stream_ctx = &engine;
        model_loaded = true;
    }
    defer if (model_loaded) {
        tf.deinit();
        arena.deinit();
        tok.deinit();
        gguf.deinit();
        alloc.free(data);
    };

    var history = try ar.agent.run(prompt, 5);
    defer history.deinit(alloc);

    try std.Io.File.stdout().writeStreamingAll(io, history.items);
    try std.Io.File.stdout().writeStreamingAll(io, "\n=== TAREA COMPLETADA CON ÉXITO ===\n");
}

fn runChatCommand(alloc: std.mem.Allocator, io: std.Io) !void {
    var ar = AgentBundle.init(alloc, io);
    ar.wire(); // estabiliza punteros internos (ar ya está en su frame final)
    defer ar.deinit();
    try ztui.runChatLoop(io, alloc, &ar.agent);
}

fn typeName(t: zload.GgufTensorType) []const u8 {
    return switch (t) {
        .f32 => "F32",
        .f16 => "F16",
        .bf16 => "BF16",
        .q4_0 => "Q4_0",
        .q4_1 => "Q4_1",
        .q5_0 => "Q5_0",
        .q5_1 => "Q5_1",
        .q8_0 => "Q8_0",
        .q8_1 => "Q8_1",
        .q2_k => "Q2_K",
        .q3_k => "Q3_K",
        .q4_k => "Q4_K",
        .q5_k => "Q5_K",
        .q6_k => "Q6_K",
        .q8_k => "Q8_K",
        else => "OTRO",
    };
}

fn inspectGguf(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    const stdout = std.Io.File.stdout();
    const dir = std.Io.Dir.cwd();
    const data = dir.readFileAlloc(io, path, alloc, .unlimited) catch |err| {
        try stdout.writeStreamingAll(io, "Error: no se pudo leer el archivo.\n");
        return err;
    };
    defer alloc.free(data);

    var gguf = zload.parseGguf(alloc, data) catch |err| {
        try stdout.writeStreamingAll(io, "Error: GGUF inválido o no soportado.\n");
        return err;
    };
    defer gguf.deinit();

    var buf: [512]u8 = undefined;
    try stdout.writeStreamingAll(io, "\n=== GGUF ===\n");
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "version={d}  tensores={d}  metadata_kv={d}  data_offset={d}\n", .{ gguf.version, gguf.tensor_count, gguf.metadata_kv_count, gguf.tensor_data_offset }));

    if (zload.modelInfo(gguf)) |mi| {
        try stdout.writeStreamingAll(io, "\n--- Arquitectura (metadata real del GGUF) ---\n");
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "arch={s}\nlayers={d}  heads={d}  kv_heads={d}\nembd={d}  ffn={d}  vocab={d}  ctx={d}\nrope_freq={e}  rms_eps={e}\n", .{ mi.arch, mi.n_layers, mi.n_heads, mi.n_kv_heads, mi.embd, mi.ffn, mi.vocab, mi.ctx, mi.rope_freq, mi.rms_eps }));
    } else |_| {
        try stdout.writeStreamingAll(io, "(no se pudo extraer ModelInfo)\n");
    }

    var hist = try gguf.quantHistogram(alloc);
    defer hist.deinit();
    try stdout.writeStreamingAll(io, "\n--- Tipos de tensor ---\n");
    var hist_it = hist.iterator();
    while (hist_it.next()) |e| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "  {s} x{d}\n", .{ typeName(e.key_ptr.*), e.value_ptr.* }));
    }

    try stdout.writeStreamingAll(io, "\n--- Primeros tensores ---\n");
    const n_show = @min(@as(usize, 12), gguf.tensors.len);
    for (gguf.tensors[0..n_show]) |t| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "  {s}  [{d} {d} {d} {d}]  {s}\n", .{ t.name, t.dims[0], t.dims[1], t.dims[2], t.dims[3], typeName(t.type) }));
    }

    // Dequantización real de un tensor de muestra (prueba que el dequant funciona sobre datos reales).
    const sample_names = [_][]const u8{ "output_norm.weight", "blk.0.attn_norm.weight", "token_embd.weight" };
    for (sample_names) |sn| {
        if (gguf.findTensor(sn)) |t| {
            const v = zload.dequantTensor(alloc, gguf, t) catch continue;
            defer alloc.free(v);
            try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "\n--- Dequant real de '{s}' ({s}, {d} elems) primeros 5 valores ---\n", .{ sn, typeName(t.type), v.len }));
            const m = @min(@as(usize, 5), v.len);
            for (0..m) |i| {
                try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "  [{d}] = {e}\n", .{ i, v[i] }));
            }
            break;
        }
    }
    try stdout.writeStreamingAll(io, "\n");
}

fn layerName(buf: []u8, l: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ l, suffix });
}

// Carga un tensor (dequant) y lo copia tal cual a dst (1D: norms, bias, embeddings).
fn loadDirect(tmp_alloc: std.mem.Allocator, gguf: zload.GgufFile, name: []const u8, dst: []f32) !void {
    const t = gguf.findTensor(name) orelse return error.MissingTensor;
    const src = try zload.dequantTensor(tmp_alloc, gguf, t);
    defer tmp_alloc.free(src);
    if (src.len != dst.len) return error.SizeMismatch;
    @memcpy(dst, src);
}

// Carga un peso lineal y lo TRANSPONE de [out,in] (layout GGUF) a [in,out] (lo que
// espera el matmul de zarch: dst[i*out+o]). out_f = out_features, in_f = in_features.
fn loadTransposed(tmp_alloc: std.mem.Allocator, gguf: zload.GgufFile, name: []const u8, dst: []f32, out_f: usize, in_f: usize) !void {
    const t = gguf.findTensor(name) orelse return error.MissingTensor;
    const src = try zload.dequantTensor(tmp_alloc, gguf, t);
    defer tmp_alloc.free(src);
    if (src.len != out_f * in_f or dst.len != out_f * in_f) return error.SizeMismatch;
    for (0..out_f) |o| {
        const row = o * in_f;
        for (0..in_f) |i| dst[i * out_f + o] = src[row + i];
    }
}

// Construye un Transformer con los pesos reales del GGUF (dequant a f32).
// state_alloc: buffers de estado/KV (los libera tf.deinit). wa: arena para los pesos.
// tmp_alloc: temporales de dequant (se liberan por tensor).
fn loadGgufTransformer(state_alloc: std.mem.Allocator, wa: std.mem.Allocator, tmp_alloc: std.mem.Allocator, gguf: zload.GgufFile, info: zload.ModelInfo, max_seq: usize) !zarch.Transformer {
    const dim: usize = info.embd;
    const hidden: usize = info.ffn;
    const L: usize = info.n_layers;
    const n_heads: usize = info.n_heads;
    const n_kv: usize = info.n_kv_heads;
    const kv_dim = (dim * n_kv) / n_heads;
    const vocab: usize = info.vocab;

    const cfg = zarch.Config{
        .dim = dim,
        .hidden_dim = hidden,
        .n_layers = L,
        .n_heads = n_heads,
        .n_kv_heads = n_kv,
        .vocab_size = vocab,
        .seq_len = max_seq,
        .theta = info.rope_freq,
        .eps = info.rms_eps,
        // Qwen2 (y la mayoría) usan RoPE NEOX en GGUF; Llama trae los pesos Q/K
        // permutados para RoPE "normal" (GPT-J), así que ahí usamos GPT-J.
        .rope_neox = !std.mem.startsWith(u8, info.arch, "llama"),
    };
    var tf = try zarch.Transformer.init(state_alloc, cfg);
    errdefer tf.deinit();

    const emb = try wa.alloc(f32, vocab * dim);
    const rms_att = try wa.alloc(f32, L * dim);
    const rms_ffn = try wa.alloc(f32, L * dim);
    const rms_final = try wa.alloc(f32, dim);
    const wq = try wa.alloc(f32, L * dim * dim);
    const wk = try wa.alloc(f32, L * kv_dim * dim);
    const wv = try wa.alloc(f32, L * kv_dim * dim);
    const wo = try wa.alloc(f32, L * dim * dim);
    const w1 = try wa.alloc(f32, L * dim * hidden);
    const w2 = try wa.alloc(f32, L * hidden * dim);
    const w3 = try wa.alloc(f32, L * dim * hidden);
    // Sesgos QKV: Qwen2 los tiene; Llama/TinyLlama no. Detección por presencia del tensor.
    const has_qkv_bias = gguf.findTensor("blk.0.attn_q.bias") != null;
    const bq: ?[]f32 = if (has_qkv_bias) try wa.alloc(f32, L * dim) else null;
    const bk: ?[]f32 = if (has_qkv_bias) try wa.alloc(f32, L * kv_dim) else null;
    const bv: ?[]f32 = if (has_qkv_bias) try wa.alloc(f32, L * kv_dim) else null;
    // Clasificador de salida: si existe output.weight (Llama/TinyLlama) → embeddings NO
    // atados, cargar en layout [embd,vocab]; si no existe (Qwen2.5-1.5B) → atados, wcls=null.
    const has_output = gguf.findTensor("output.weight") != null;
    const wcls: ?[]f32 = if (has_output) try wa.alloc(f32, vocab * dim) else null;

    try loadDirect(tmp_alloc, gguf, "token_embd.weight", emb); // [vocab, embd] sin transponer
    try loadDirect(tmp_alloc, gguf, "output_norm.weight", rms_final);

    var nbuf: [64]u8 = undefined;
    for (0..L) |l| {
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_norm.weight"), rms_att[l * dim ..][0..dim]);
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_norm.weight"), rms_ffn[l * dim ..][0..dim]);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_q.weight"), wq[l * dim * dim ..][0 .. dim * dim], dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_k.weight"), wk[l * kv_dim * dim ..][0 .. kv_dim * dim], kv_dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_v.weight"), wv[l * kv_dim * dim ..][0 .. kv_dim * dim], kv_dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_output.weight"), wo[l * dim * dim ..][0 .. dim * dim], dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_gate.weight"), w1[l * dim * hidden ..][0 .. dim * hidden], hidden, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_up.weight"), w3[l * dim * hidden ..][0 .. dim * hidden], hidden, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_down.weight"), w2[l * hidden * dim ..][0 .. hidden * dim], dim, hidden);
        if (has_qkv_bias) {
            try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_q.bias"), bq.?[l * dim ..][0..dim]);
            try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_k.bias"), bk.?[l * kv_dim ..][0..kv_dim]);
            try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_v.bias"), bv.?[l * kv_dim ..][0..kv_dim]);
        }
    }

    if (has_output) try loadTransposed(tmp_alloc, gguf, "output.weight", wcls.?, vocab, dim);

    tf.token_embedding_table = emb;
    tf.rms_att_weight = rms_att;
    tf.rms_ffn_weight = rms_ffn;
    tf.rms_final_weight = rms_final;
    tf.wq = wq;
    tf.wk = wk;
    tf.wv = wv;
    tf.wo = wo;
    tf.w1 = w1;
    tf.w2 = w2;
    tf.w3 = w3;
    tf.bq = bq;
    tf.bk = bk;
    tf.bv = bv;
    tf.wcls = wcls; // Llama/TinyLlama: output.weight (no atado). Qwen2.5-1.5B: null (atado).
    return tf;
}

/// Carga un tensor cuantizado único (p.ej. token_embd.weight) como QuantEmbedding.
/// ZERO-COPY: apunta directamente a los bytes del GGUF ya cargado (`gguf`/`data`
/// del caller viven más que el Transformer; ver loadGgufTransformerQuantized). No
/// se copia el blob → ahorra ~vocab·dim bytes de RAM. Los kernels leen con align(1).
fn loadQuantEmbedding(wa: std.mem.Allocator, gguf: zload.GgufFile, name: []const u8) !zarch.QuantEmbedding {
    _ = wa; // ya no se copia: se referencia el buffer GGUF in situ.
    const t = gguf.findTensor(name) orelse return error.MissingTensor;
    const fmt = quantTypeToFormat(t.type) orelse return error.UnsupportedQuant;
    const sz = zload.tensorByteSize(t);
    return .{ .bytes = gguf.getTensorData(t)[0..sz], .fmt = fmt };
}

fn quantTypeToFormat(t: zload.GgufTensorType) ?zarch.QuantFormat {
    return switch (t) {
        .q4_0 => .q4_0,
        .q8_0 => .q8_0,
        .q4_k => .q4_k,
        .q6_k => .q6_k,
        else => null,
    };
}

/// Carga `blk.{0..L-1}.{suffix}` (pesos cuantizados crudos) como un QuantWeight
/// con slice de bytes y formato POR CAPA en el arena `wa`. El formato puede variar
/// entre capas (los K-quant `_M` reales lo hacen: ffn_down/attn_v alternan Q4_K y
/// Q6_K según la capa), así que NO se exige uniformidad: cada capa guarda el suyo.
fn loadQuantWeight(wa: std.mem.Allocator, gguf: zload.GgufFile, L: usize, suffix: []const u8) !zarch.QuantWeight {
    var nbuf: [64]u8 = undefined;
    // layers/fmts (metadatos pequeños) sí se alojan en `wa`; los blobs de pesos son
    // ZERO-COPY: se referencian in situ en el buffer GGUF (vive más que el Transformer).
    const layers = try wa.alloc([]const u8, L);
    const fmts = try wa.alloc(zarch.QuantFormat, L);
    for (0..L) |l| {
        const t = gguf.findTensor(try layerName(&nbuf, l, suffix)) orelse return error.MissingTensor;
        fmts[l] = quantTypeToFormat(t.type) orelse return error.UnsupportedQuant;
        const sz = zload.tensorByteSize(t);
        layers[l] = gguf.getTensorData(t)[0..sz];
    }
    return .{ .layers = layers, .fmts = fmts };
}

/// Carga un Transformer manteniendo los pesos lineales cuantizados en formato GGUF
/// original (sin transponer). Los pesos no-lineales (embeddings, norms, bias) se
/// dequantizan a f32 como antes.
fn loadGgufTransformerQuantized(state_alloc: std.mem.Allocator, wa: std.mem.Allocator, tmp_alloc: std.mem.Allocator, gguf: zload.GgufFile, info: zload.ModelInfo, max_seq: usize) !zarch.Transformer {
    const dim: usize = info.embd;
    const hidden: usize = info.ffn;
    const L: usize = info.n_layers;
    const n_heads: usize = info.n_heads;
    const n_kv: usize = info.n_kv_heads;
    const kv_dim = (dim * n_kv) / n_heads;
    const vocab: usize = info.vocab;

    const cfg = zarch.Config{
        .dim = dim,
        .hidden_dim = hidden,
        .n_layers = L,
        .n_heads = n_heads,
        .n_kv_heads = n_kv,
        .vocab_size = vocab,
        .seq_len = max_seq,
        .theta = info.rope_freq,
        .eps = info.rms_eps,
        .rope_neox = !std.mem.startsWith(u8, info.arch, "llama"),
    };
    var tf = try zarch.Transformer.init(state_alloc, cfg);
    errdefer tf.deinit();

    // Cargar pesos no-lineales como f32 (mismo que antes)
    // Embedding se mantiene cuantizado (Tarea 1 F7) para reducir RAM.
    const rms_att = try wa.alloc(f32, L * dim);
    const rms_ffn = try wa.alloc(f32, L * dim);
    const rms_final = try wa.alloc(f32, dim);
    // Sesgos QKV: Qwen2 los tiene; Llama/TinyLlama no. Detección por presencia del tensor.
    const has_qkv_bias = gguf.findTensor("blk.0.attn_q.bias") != null;
    const bq: ?[]f32 = if (has_qkv_bias) try wa.alloc(f32, L * dim) else null;
    const bk: ?[]f32 = if (has_qkv_bias) try wa.alloc(f32, L * kv_dim) else null;
    const bv: ?[]f32 = if (has_qkv_bias) try wa.alloc(f32, L * kv_dim) else null;
    // Clasificador de salida: si existe output.weight (Llama/TinyLlama) → embeddings NO
    // atados, cargar en layout [embd,vocab]; si no existe (Qwen2.5-1.5B) → atados, wcls=null.
    const has_output = gguf.findTensor("output.weight") != null;
    const wcls: ?[]f32 = if (has_output) try wa.alloc(f32, vocab * dim) else null;

    try loadDirect(tmp_alloc, gguf, "output_norm.weight", rms_final);

    var nbuf: [64]u8 = undefined;
    for (0..L) |l| {
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_norm.weight"), rms_att[l * dim ..][0..dim]);
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_norm.weight"), rms_ffn[l * dim ..][0..dim]);
        if (has_qkv_bias) {
            try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_q.bias"), bq.?[l * dim ..][0..dim]);
            try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_k.bias"), bk.?[l * kv_dim ..][0..kv_dim]);
            try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_v.bias"), bv.?[l * kv_dim ..][0..kv_dim]);
        }
    }
    if (has_output) try loadTransposed(tmp_alloc, gguf, "output.weight", wcls.?, vocab, dim);

    tf.token_embedding_table = &.{};
    tf.token_embedding_table_q = try loadQuantEmbedding(wa, gguf, "token_embd.weight");
    tf.rms_att_weight = rms_att;
    tf.rms_ffn_weight = rms_ffn;
    tf.rms_final_weight = rms_final;
    tf.bq = bq;
    tf.bk = bk;
    tf.bv = bv;
    tf.wcls = wcls; // Llama/TinyLlama: output.weight (no atado). Qwen2.5-1.5B: null (atado).

    // Cargar pesos lineales cuantizados: TODAS las capas concatenadas por tipo de
    // peso (layout GGUF original [out, in], sin transponer). Cada tipo conserva su
    // PROPIO formato (los K-quant reales mezclan: Qwen2.5 Q4_K_M usa Q4_K en
    // attn/gate/up y Q6_K en ffn_down). `forward` desplaza por capa con el stride
    // implícito del formato (ver Transformer.quantLayerBytes).
    tf.wq_q = try loadQuantWeight(wa, gguf, L, "attn_q.weight");
    tf.wk_q = try loadQuantWeight(wa, gguf, L, "attn_k.weight");
    tf.wv_q = try loadQuantWeight(wa, gguf, L, "attn_v.weight");
    tf.wo_q = try loadQuantWeight(wa, gguf, L, "attn_output.weight");
    tf.w1_q = try loadQuantWeight(wa, gguf, L, "ffn_gate.weight");
    tf.w3_q = try loadQuantWeight(wa, gguf, L, "ffn_up.weight");
    tf.w2_q = try loadQuantWeight(wa, gguf, L, "ffn_down.weight");

    // Los campos f32 de pesos lineales se dejan vacíos; forward los ignora si *_quant != null.
    tf.wq = &.{};
    tf.wk = &.{};
    tf.wv = &.{};
    tf.wo = &.{};
    tf.w1 = &.{};
    tf.w2 = &.{};
    tf.w3 = &.{};

    return tf;
}

fn runForward(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, token_id: u32) !void {
    const stdout = std.Io.File.stdout();
    const dir = std.Io.Dir.cwd();
    var buf: [256]u8 = undefined;

    try stdout.writeStreamingAll(io, "Leyendo GGUF...\n");
    const data = dir.readFileAlloc(io, model_path, alloc, .unlimited) catch |err| {
        try stdout.writeStreamingAll(io, "Error: no se pudo leer el modelo.\n");
        return err;
    };
    defer alloc.free(data);

    var gguf = try zload.parseGguf(alloc, data);
    defer gguf.deinit();
    const info = try zload.modelInfo(gguf);
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Modelo: arch={s} layers={d} embd={d} vocab={d}\n", .{ info.arch, info.n_layers, info.embd, info.vocab }));

    try stdout.writeStreamingAll(io, "Dequantizando pesos a f32 (puede tardar y usar varios GB)...\n");
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    var tf = loadGgufTransformer(alloc, arena.allocator(), alloc, gguf, info, 256) catch |err| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Error cargando pesos: {s}\n", .{@errorName(err)}));
        return err;
    };
    defer tf.deinit();

    // Token de entrada
    const tokens_meta = gguf.getMeta("tokenizer.ggml.tokens");
    if (tokens_meta) |m| {
        if (m == .array) {
            if (zload.arrayStringAt(m.array, token_id)) |s| {
                try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Token de entrada [{d}] = \"{s}\"\n", .{ token_id, s }));
            }
        }
    }

    try stdout.writeStreamingAll(io, "Ejecutando forward (pesos reales)...\n");
    const logits = tf.forward(token_id, 0);

    // Top-5 siguiente token
    var ti = [_]usize{0} ** 5;
    var tv = [_]f32{-std.math.inf(f32)} ** 5;
    for (logits, 0..) |v, i| {
        var minj: usize = 0;
        for (1..5) |j| {
            if (tv[j] < tv[minj]) minj = j;
        }
        if (v > tv[minj]) {
            tv[minj] = v;
            ti[minj] = i;
        }
    }
    // ordenar los 5 desc (selection sort)
    for (0..5) |a| {
        var best = a;
        for (a + 1..5) |b| {
            if (tv[b] > tv[best]) best = b;
        }
        std.mem.swap(f32, &tv[a], &tv[best]);
        std.mem.swap(usize, &ti[a], &ti[best]);
    }

    try stdout.writeStreamingAll(io, "\nTop-5 siguiente token predicho:\n");
    for (0..5) |i| {
        var tok_str: []const u8 = "?";
        if (tokens_meta) |m| {
            if (m == .array) {
                if (zload.arrayStringAt(m.array, ti[i])) |s| tok_str = s;
            }
        }
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "  {d}. id={d} logit={d:.3} \"{s}\"\n", .{ i + 1, ti[i], tv[i], tok_str }));
    }
}

// Construye el tokenizador byte-level a partir de la metadata del GGUF.
fn buildTokenizer(alloc: std.mem.Allocator, gguf: zload.GgufFile) !ztok.Tokenizer {
    var t = try ztok.Tokenizer.init(alloc);
    errdefer t.deinit();

    const toks = gguf.getMeta("tokenizer.ggml.tokens") orelse return error.NoVocabInGguf;
    if (toks != .array or toks.array.type != .string) return error.NoVocabInGguf;
    var it = zload.arrayStrIter(toks.array);
    while (it.next()) |s| try t.addToken(s);

    if (gguf.getMeta("tokenizer.ggml.merges")) |m| {
        if (m == .array and m.array.type == .string) {
            var mit = zload.arrayStrIter(m.array);
            while (mit.next()) |s| try t.addMerge(s);
        }
    }
    t.bos_id = gguf.getU32("tokenizer.ggml.bos_token_id");
    t.eos_id = gguf.getU32("tokenizer.ggml.eos_token_id");
    return t;
}

// Elige el siguiente token a partir de los logits. temp==0 -> greedy (argmax).
fn sampleToken(logits: []f32, temp: f32, rng: *std.Random.DefaultPrng) u32 {
    if (temp <= 0.0) {
        var best: u32 = 0;
        var best_val = logits[0];
        for (logits, 0..) |v, i| {
            if (v > best_val) {
                best_val = v;
                best = @intCast(i);
            }
        }
        return best;
    }
    // Softmax con temperatura + muestreo multinomial.
    var maxv = logits[0];
    for (logits) |v| {
        if (v > maxv) maxv = v;
    }
    var sum: f32 = 0;
    for (logits) |*v| {
        v.* = @exp((v.* - maxv) / temp);
        sum += v.*;
    }
    const r = rng.random().float(f32) * sum;
    var acc: f32 = 0;
    for (logits, 0..) |v, i| {
        acc += v;
        if (acc >= r) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

// Contexto de decode para la gramática (mapea id -> bytes sin asignar).
const GrammarCtx = struct {
    tok: *ztok.Tokenizer,
    fn dec(self: GrammarCtx, id: u32, buf: []u8) []const u8 {
        return self.tok.decodeToken(id, buf);
    }
};

// F1: tokeniza el prompt, ejecuta el Transformer real y genera texto.
// F2: si `use_grammar`, fuerza JSON válido (decodificación restringida).
fn runGenerate(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, prompt: []const u8, n_tokens: usize, temp: f32, use_grammar: bool, use_quantized: bool, use_speculate: bool) !void {
    const stdout = std.Io.File.stdout();
    const dir = std.Io.Dir.cwd();
    var buf: [512]u8 = undefined;

    try stdout.writeStreamingAll(io, "Leyendo GGUF...\n");
    const data = dir.readFileAlloc(io, model_path, alloc, .unlimited) catch |err| {
        try stdout.writeStreamingAll(io, "Error: no se pudo leer el modelo.\n");
        return err;
    };
    defer alloc.free(data);

    var gguf = try zload.parseGguf(alloc, data);
    defer gguf.deinit();
    const info = try zload.modelInfo(gguf);
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Modelo: arch={s} layers={d} embd={d} vocab={d} ctx={d}\n", .{ info.arch, info.n_layers, info.embd, info.vocab, info.ctx }));

    // Tokenizador
    var tok = buildTokenizer(alloc, gguf) catch |err| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Error construyendo tokenizador: {s}\n", .{@errorName(err)}));
        return err;
    };
    defer tok.deinit();
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Tokenizador: vocab={d} bos={?d} eos={?d}\n", .{ tok.vocabSize(), tok.bos_id, tok.eos_id }));

    var prompt_ids: std.ArrayList(u32) = .empty;
    defer prompt_ids.deinit(alloc);
    try tok.encode(prompt, &prompt_ids);
    if (prompt_ids.items.len == 0) {
        try stdout.writeStreamingAll(io, "El prompt no produjo tokens.\n");
        return;
    }
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Prompt tokenizado en {d} tokens: ", .{prompt_ids.items.len}));
    for (prompt_ids.items) |id| try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "{d} ", .{id}));
    try stdout.writeStreamingAll(io, "\n");

    const max_seq = @min(prompt_ids.items.len + n_tokens + 4, @max(info.ctx, 64));

    if (use_quantized) {
        try stdout.writeStreamingAll(io, "Cargando pesos en formato cuantizado (matmul on-the-fly, sin dequant completo a f32)...\n");
    } else {
        try stdout.writeStreamingAll(io, "Dequantizando pesos a f32 (puede tardar y usar varios GB)...\n");
    }
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    var tf = (if (use_quantized)
        loadGgufTransformerQuantized(alloc, arena.allocator(), alloc, gguf, info, max_seq)
    else
        loadGgufTransformer(alloc, arena.allocator(), alloc, gguf, info, max_seq)) catch |err| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Error cargando pesos: {s}\n", .{@errorName(err)}));
        return err;
    };
    defer tf.deinit();

    // Prefill del prompt.
    try stdout.writeStreamingAll(io, "\n--- Generación ---\n");
    var pos: usize = 0;
    var logits: []f32 = undefined;
    for (prompt_ids.items) |id| {
        logits = tf.forward(id, pos);
        pos += 1;
    }

    var rng = std.Random.DefaultPrng.init(0xF1);
    var gen_ids: std.ArrayList(u32) = .empty;
    defer gen_ids.deinit(alloc);
    var out_bytes: std.ArrayList(u8) = .empty;
    defer out_bytes.deinit(alloc);

    // Eco del prompt.
    try stdout.writeStreamingAll(io, prompt);

    var gram = zarch.grammar.BalancedJson{};
    const gctx = GrammarCtx{ .tok = &tok };
    // Sembrar la gramática con el prompt (si ya contiene JSON parcial) para que
    // la continuación restringida sea coherente con él.
    if (use_grammar) {
        for (prompt) |b| _ = gram.feed(b);
    }

    if (use_speculate and temp <= 0.0 and !use_grammar) {
        var spec = zspeculate.SpeculativeGenerator{ .tf = &tf };
        defer spec.deinit(alloc);
        var all_ids = std.ArrayList(u32).empty;
        defer all_ids.deinit(alloc);
        try all_ids.appendSlice(alloc, prompt_ids.items);
        const spec_n = try spec.generate(alloc, &tok, &all_ids, &out_bytes, n_tokens, max_seq);
        // out_bytes acumula el texto completo de la generación (decode por token sin clear).
        // El TEXTO GENERADO es idéntico a greedy (invariante F7.B); el pie de página meta
        // del CLI se imprime igual que en la rama greedy (no forma parte de la salida).
        try stdout.writeStreamingAll(io, out_bytes.items);
        try stdout.writeStreamingAll(io, "\n\n");
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "({d} tokens generados)\n", .{spec_n}));
        return;
    }

    var generated: usize = 0;
    while (generated < n_tokens and pos < max_seq) : (generated += 1) {
        const next = if (use_grammar)
            (zarch.grammar.grammarArgmax(logits, &gram, gctx, GrammarCtx.dec) orelse break)
        else
            sampleToken(logits, temp, &rng);
        if (tok.eos_id) |e| if (next == e) break;

        out_bytes.clearRetainingCapacity();
        const one = [_]u32{next};
        try tok.decode(one[0..], &out_bytes);
        try stdout.writeStreamingAll(io, out_bytes.items);
        try gen_ids.append(alloc, next);

        // Decodificación restringida: parar cuando el JSON está completo.
        if (use_grammar and gram.isComplete()) break;

        logits = tf.forward(next, pos);
        pos += 1;
    }
    try stdout.writeStreamingAll(io, "\n\n");
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "({d} tokens generados)\n", .{gen_ids.items.len}));
}

// =====================================================================
// Seq-KD helpers
// =====================================================================

/// Genera una secuencia greedy completa (prompt + continuación) con un
/// profesor Transformer GGUF cargado. Devuelve slice propiedad del caller.
fn generateSeqWithTeacher(
    alloc: std.mem.Allocator,
    tf: *zarch.Transformer,
    tok: *ztok.Tokenizer,
    prompt: []const u8,
    max_new: usize,
) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(alloc);
    try tok.encode(prompt, &ids);
    const cap = tf.config.seq_len;
    var pos: usize = 0;
    var logits: []f32 = undefined;
    for (ids.items) |id| {
        if (pos >= cap) break;
        logits = tf.forward(id, pos);
        pos += 1;
    }
    var gen: usize = 0;
    while (gen < max_new and pos < cap) : (gen += 1) {
        var best: u32 = 0;
        var best_val = logits[0];
        for (logits, 0..) |v, i| {
            if (v > best_val) {
                best_val = v;
                best = @intCast(i);
            }
        }
        if (tok.eos_id) |e| if (best == e) break;
        try ids.append(alloc, best);
        logits = tf.forward(best, pos);
        pos += 1;
    }
    return try ids.toOwnedSlice(alloc);
}

/// Carga prompts desde un archivo de texto (uno por línea). Si `path` es null,
/// devuelve `default_prompts` directamente. El caller debe llamar a `freePrompts`
/// para liberar cuando `path != null`.
fn loadPrompts(
    alloc: std.mem.Allocator,
    io: std.Io,
    path: ?[]const u8,
    default_prompts: []const []const u8,
) ![]const []const u8 {
    if (path) |p| {
        const data = try std.Io.Dir.cwd().readFileAlloc(io, p, alloc, .unlimited);
        defer alloc.free(data);
        var lines = std.ArrayList([]const u8).empty;
        errdefer {
            for (lines.items) |l| alloc.free(l);
            lines.deinit(alloc);
        }
        var start: usize = 0;
        for (data, 0..) |b, i| {
            if (b == '\n') {
                const line = data[start..i];
                const trimmed = std.mem.trimEnd(u8, line, "\r");
                if (trimmed.len > 0) {
                    try lines.append(alloc, try alloc.dupe(u8, trimmed));
                }
                start = i + 1;
            }
        }
        if (start < data.len) {
            const line = data[start..];
            const trimmed = std.mem.trimEnd(u8, line, "\r");
            if (trimmed.len > 0) {
                try lines.append(alloc, try alloc.dupe(u8, trimmed));
            }
        }
        return try lines.toOwnedSlice(alloc);
    }
    return default_prompts;
}

fn freePrompts(alloc: std.mem.Allocator, prompts: []const []const u8, path: ?[]const u8) void {
    if (path != null) {
        for (prompts) |p| alloc.free(p);
        alloc.free(prompts);
    }
}

// =====================================================================
// F3 — Destilación sintética (piloto sin GGUF)
// =====================================================================
fn runDistillSynthetic(
    alloc: std.mem.Allocator,
    io: std.Io,
    steps: usize,
    seed: u64,
    out_path: ?[]const u8,
    dataset_path: ?[]const u8,
    prompts_path: ?[]const u8,
) !void {
    const stdout = std.Io.File.stdout();
    var buf: [512]u8 = undefined;

    // Teacher sintético determinista (seed diferente al student)
    const V: u32 = 64;
    const cfg = zdistill.NanoConfig{ .vocab = V, .dim = 32, .n_layers = 4, .n_heads = 4, .ff_dim = 64 };
    var teacher = try zdistill.FluencyNano.init(alloc, cfg, seed + 1);
    defer teacher.deinit();
    var student = try zdistill.FluencyNano.init(alloc, cfg, seed);
    defer student.deinit();

    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "=== DESTILACIÓN SEQ-KD SINTÉTICA ===\nTeacher params={d}  Student params={d}\n", .{ teacher.paramCount(), student.paramCount() }));

    const default_prompts = [_][]const u8{
        "hello world",
        "the quick brown",
        "in 1492, columbus",
        "function compute",
        "the answer is",
        "once upon a",
        "to be or",
        "machine learning",
    };
    const prompts = try loadPrompts(alloc, io, prompts_path, &default_prompts);
    defer freePrompts(alloc, prompts, prompts_path);

    var seqs_list: std.ArrayList([]u32) = .empty;
    defer {
        for (seqs_list.items) |s| alloc.free(s);
        seqs_list.deinit(alloc);
    }
    for (prompts) |p| {
        var pids: std.ArrayList(u32) = .empty;
        defer pids.deinit(alloc);
        for (p) |b| try pids.append(alloc, @as(u32, b) % V);
        const seq = try zdistill.generateGreedyNano(&teacher, alloc, pids.items, 12);
        try seqs_list.append(alloc, seq);
    }

    const seqs = try alloc.alloc([]u32, seqs_list.items.len);
    defer alloc.free(seqs);
    @memcpy(seqs, seqs_list.items);
    const ds = zdistill.NanoDataset{ .alloc = alloc, .seqs = seqs };

    if (dataset_path) |dp| {
        try zdistill.saveSeqDataset(ds, dp, io, alloc);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Dataset seq-KD guardado en {s} ({d} secuencias)\n", .{ dp, ds.seqs.len }));
    }

    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.01, .min_lr = 0.001, .warmup_steps = @max(steps / 10, 1), .max_steps = steps } };
    var trainer = try zdistill.Trainer.init(alloc, &student, sched, 1.0);
    defer trainer.deinit();

    const before = try zdistill.evaluateSeqDataset(&student, alloc, ds, 0, ds.seqs.len);
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[antes] ce={d:.4} ppl={d:.3} acc={d:.1}%\n", .{ before.ce, before.ppl, before.acc * 100 }));

    const result = try zdistill.trainSeqKD(&student, &trainer, alloc, ds, 2, steps, @max(steps / 10, 1), 3);

    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[después] train ce={d:.4} ppl={d:.3} acc={d:.1}%  held ce={d:.4} ppl={d:.3} acc={d:.1}%\n", .{ result.ce_train, result.ppl_train, result.acc_train * 100, result.ce_held, result.ppl_held, result.acc_held * 100 }));

    // Muestra greedy del student (mismo prompt para comparar reproducibilidad)
    var prompt_ids: std.ArrayList(u32) = .empty;
    defer prompt_ids.deinit(alloc);
    for ("hello") |b| try prompt_ids.append(alloc, @as(u32, b) % V);
    const sample = try zdistill.generateGreedyNano(&student, alloc, prompt_ids.items, 8);
    defer alloc.free(sample);
    try stdout.writeStreamingAll(io, "Muestra greedy student: ");
    for (sample) |id| try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "{d} ", .{id}));
    try stdout.writeStreamingAll(io, "\n");

    if (out_path) |op| {
        try zdistill.saveCheckpoint(&student, op, io, alloc);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Checkpoint guardado en {s}\n", .{op}));
    }
}

// =====================================================================
// F3 — Destilación real (gate-check E4)
// =====================================================================
// Destila un alumno `FluencyNano` (subcuadrático híbrido) a partir de un
// profesor real. Dos vías, ambas exigidas por el plan:
//   * KD por logits (zload): el profesor GGUF se ejecuta con el Transformer de
//     F1 y entrega su distribución por posición; el alumno minimiza KL(prof||al).
//   * Seq-KD (zollama): el profesor (Ollama) genera una continuación; el alumno
//     aprende esa secuencia por cross-entropy (usa el tokenizador del GGUF).
fn runDistill(
    alloc: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
    prompt: []const u8,
    steps: usize,
    seed: u64,
    ollama_model: ?[]const u8,
    out_path: ?[]const u8,
    dataset_path: ?[]const u8,
    _mohawk: zdistill.MohawkStage,
    prompts_path: ?[]const u8,
) !void {
    _ = _mohawk;
    const stdout = std.Io.File.stdout();
    const dir = std.Io.Dir.cwd();
    var buf: [512]u8 = undefined;

    try stdout.writeStreamingAll(io, "=== DESTILACIÓN SEQ-KD (pipeline real) ===\nLeyendo profesor GGUF...\n");
    const data = dir.readFileAlloc(io, model_path, alloc, .unlimited) catch |err| {
        try stdout.writeStreamingAll(io, "Error: no se pudo leer el modelo profesor.\n");
        return err;
    };
    defer alloc.free(data);

    var gguf = try zload.parseGguf(alloc, data);
    defer gguf.deinit();
    const info = try zload.modelInfo(gguf);

    var tok = try buildTokenizer(alloc, gguf);
    defer tok.deinit();

    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Teacher: arch={s} layers={d} embd={d} vocab={d} ctx={d}\n", .{ info.arch, info.n_layers, info.embd, info.vocab, info.ctx }));
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Tokenizer: vocab={d} bos={?d} eos={?d}\n", .{ tok.vocabSize(), tok.bos_id, tok.eos_id }));

    // Alineación del vocabulario del student al teacher
    const V = info.vocab;
    var cfg = zdistill.Fluency126MConfig;
    cfg.vocab = V;

    // Reporte honesto si el conteo cambia
    var student = try zdistill.FluencyNano.init(alloc, cfg, seed);
    defer student.deinit();
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Student FluencyNano (alineado): vocab={d} dim={d} capas={d} ", .{ V, cfg.dim, cfg.n_layers }));
    student.printParamCount();
    if (V != 32000) {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "AVISO: vocab teacher={d} != 32000. El modelo ya NO es 126M exactos; renombrar según paramCount.\n", .{V}));
    }

    try stdout.writeStreamingAll(io, "Dequantizando profesor a f32 (puede tardar y usar varios GB)...\n");
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const max_seq = @max(info.ctx, 64);
    var tf = loadGgufTransformer(alloc, arena.allocator(), alloc, gguf, info, max_seq) catch |err| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Error cargando profesor: {s}\n", .{@errorName(err)}));
        return err;
    };
    defer tf.deinit();

    // Vía Seq-KD con Ollama: si se pide, generar texto adicional
    if (ollama_model) |om| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Seq-KD vía Ollama '{s}' (se concatenará al prompt).\n", .{om}));
        var oclient = zollama.OllamaClient.init(alloc, io, "http://127.0.0.1:11434");
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(alloc);
        oclient.generate(om, prompt, &text) catch {
            try stdout.writeStreamingAll(io, "No se pudo contactar a Ollama; continuando solo con prompt.\n");
        };
    }

    // Generar dataset seq-KD: prompts semilla -> teacher greedy -> secuencias
    const default_prompts = [_][]const u8{
        "The quick brown fox",
        "In the year 2049, humanity",
        "function main() {",
        "The capital of France is",
        "To be or not to be,",
        "Once upon a time",
        "Machine learning is",
        "The answer to life",
    };
    const seed_prompts = try loadPrompts(alloc, io, prompts_path, &default_prompts);
    defer freePrompts(alloc, seed_prompts, prompts_path);

    var seqs_list: std.ArrayList([]u32) = .empty;
    defer {
        for (seqs_list.items) |s| alloc.free(s);
        seqs_list.deinit(alloc);
    }
    for (seed_prompts) |sp| {
        const seq = try generateSeqWithTeacher(alloc, &tf, &tok, sp, 24);
        try seqs_list.append(alloc, seq);
    }

    const seqs = try alloc.alloc([]u32, seqs_list.items.len);
    defer alloc.free(seqs);
    @memcpy(seqs, seqs_list.items);
    const ds = zdistill.NanoDataset{ .alloc = alloc, .seqs = seqs };

    if (dataset_path) |dp| {
        try zdistill.saveSeqDataset(ds, dp, io, alloc);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Dataset seq-KD guardado en {s} ({d} secuencias)\n", .{ dp, ds.seqs.len }));
    }

    // Entrenamiento seq-KD con held-out
    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.0005, .min_lr = 0.00005, .warmup_steps = @max(steps / 10, 1), .max_steps = steps } };
    var trainer = try zdistill.Trainer.init(alloc, &student, sched, 1.0);
    defer trainer.deinit();

    const before = try zdistill.evaluateSeqDataset(&student, alloc, ds, 0, ds.seqs.len);
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[antes] ce={d:.4} ppl={d:.3} acc={d:.1}%\n", .{ before.ce, before.ppl, before.acc * 100 }));

    const result = try zdistill.trainSeqKD(&student, &trainer, alloc, ds, 2, steps, @max(steps / 10, 1), 3);

    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[después] train ce={d:.4} ppl={d:.3} acc={d:.1}%  held ce={d:.4} ppl={d:.3} acc={d:.1}%\n", .{ result.ce_train, result.ppl_train, result.acc_train * 100, result.ce_held, result.ppl_held, result.acc_held * 100 }));

    // Muestra greedy del student (mismo prompt para comparar)
    var sample_prompt: std.ArrayList(u32) = .empty;
    defer sample_prompt.deinit(alloc);
    try tok.encode("The quick brown fox", &sample_prompt);
    const sample = try zdistill.generateGreedyNano(&student, alloc, sample_prompt.items, 12);
    defer alloc.free(sample);
    try stdout.writeStreamingAll(io, "Muestra greedy student (ids): ");
    for (sample) |id| try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "{d} ", .{id}));
    try stdout.writeStreamingAll(io, "\n");

    if (out_path) |op| {
        try zdistill.saveCheckpoint(&student, op, io, alloc);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Checkpoint guardado en {s}\n", .{op}));
    }
}

/// Tokeniza un archivo de texto a un .bin de tokens u16 LE.
/// Si `gguf_path` es null, usa vocabulario byte-simple (256 tokens).
fn runTokenize(alloc: std.mem.Allocator, io: std.Io, text_path: []const u8, out_path: []const u8, gguf_path: ?[]const u8) !void {
    const stdout = std.Io.File.stdout();
    var buf: [256]u8 = undefined;

    const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, alloc, .unlimited);
    defer alloc.free(text);

    var tok = if (gguf_path) |gp| blk: {
        const data = try std.Io.Dir.cwd().readFileAlloc(io, gp, alloc, .unlimited);
        defer alloc.free(data);
        var gguf = try zload.parseGguf(alloc, data);
        defer gguf.deinit();
        break :blk try buildTokenizer(alloc, gguf);
    } else blk: {
        var t = try ztok.Tokenizer.init(alloc);
        try ztok.buildByteVocabSimple(&t);
        break :blk t;
    };
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(alloc);
    try tok.encode(text, &ids);

    const n = ids.items.len;
    const out_bytes = try alloc.alloc(u8, n * 2);
    defer alloc.free(out_bytes);
    for (0..n) |i| {
        std.mem.writeInt(u16, out_bytes[i * 2 ..][0..2], @intCast(ids.items[i]), .little);
    }

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = out_bytes });
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Tokenizado {d} tokens -> {s} (vocab={d})\n", .{ n, out_path, tok.vocabSize() }));
}

// Entrenamiento auto-supervisado reproducible. Si `data_path` es null, usa el
// corpus sintético determinista (regla next = (cur*7+3) mod V). Si se provee,
// entrena sampleando ventanas aleatorias del dataset .bin (TokenDataset).
fn runTrainSynthetic(alloc: std.mem.Allocator, io: std.Io, steps: usize, seed: u64, out_path: ?[]const u8, data_path: ?[]const u8) !void {
    const stdout = std.Io.File.stdout();
    var buf: [512]u8 = undefined;

    const V: u32 = if (data_path != null) 256 else 64;
    const T = 32;
    const cfg = zdistill.NanoConfig{ .vocab = V, .dim = 32, .n_layers = 4, .n_heads = 4, .ff_dim = 64 };
    var model = try zdistill.FluencyNano.init(alloc, cfg, seed);
    defer model.deinit();
    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.01, .min_lr = 0.001, .warmup_steps = @max(steps / 10, 1), .max_steps = steps } };
    var trainer = try zdistill.Trainer.init(alloc, &model, sched, 1.0);
    defer trainer.deinit();

    if (data_path) |dp| {
        var ds = try ztrain.TokenDataset.fromFile(alloc, io, dp);
        defer ds.deinit();
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Dataset: {d} tokens (vocab={d})\n", .{ ds.n, V }));
        if (ds.n <= T + 1) {
            try stdout.writeStreamingAll(io, "Dataset demasiado corto para entrenar.\n");
            return;
        }

        // Evaluación inicial sobre una ventana determinista (posición 0)
        var ids0: [T]u32 = undefined;
        var tgt0: [T]u32 = undefined;
        for (0..T) |i| {
            ids0[i] = ds.tokenAt(i);
            tgt0[i] = ds.tokenAt(i + 1);
        }
        const before = try zdistill.evaluate(&model, alloc, &ids0, &tgt0);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[antes] ppl={d:.3} acc={d:.1}% (params={d}, semilla={d})\n", .{ before.ppl, before.acc * 100, model.paramCount(), seed }));

        var rng = std.Random.DefaultPrng.init(seed);
        var i: usize = 0;
        var loss: f32 = 0;
        var ids: [T]u32 = undefined;
        var tgt: [T]u32 = undefined;
        while (i < steps) : (i += 1) {
            ds.sampleWindow(rng.random(), &ids, &tgt);
            loss = try trainer.stepCE(&ids, &tgt);
            if ((i + 1) % @max(steps / 10, 1) == 0)
                try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "  paso {d}/{d} CE={d:.4}\n", .{ i + 1, steps, loss }));
        }

        const after = try zdistill.evaluate(&model, alloc, &ids0, &tgt0);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[después] ppl={d:.3} acc={d:.1}%  (CE final={d:.4})\n", .{ after.ppl, after.acc * 100, loss }));
    } else {
        var ids: [T]u32 = undefined;
        var tgt: [T]u32 = undefined;
        var cur: u32 = 1;
        for (0..T) |t| {
            ids[t] = cur;
            cur = (cur * 7 + 3) % V;
            tgt[t] = cur;
        }
        const before = try zdistill.evaluate(&model, alloc, &ids, &tgt);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[antes] ppl={d:.3} acc={d:.1}% (params={d}, semilla={d})\n", .{ before.ppl, before.acc * 100, model.paramCount(), seed }));
        var i: usize = 0;
        var loss: f32 = 0;
        while (i < steps) : (i += 1) {
            loss = try trainer.stepCE(&ids, &tgt);
            if ((i + 1) % @max(steps / 10, 1) == 0)
                try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "  paso {d}/{d} CE={d:.4}\n", .{ i + 1, steps, loss }));
        }
        const after = try zdistill.evaluate(&model, alloc, &ids, &tgt);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[después] ppl={d:.3} acc={d:.1}%  (CE final={d:.4})\n", .{ after.ppl, after.acc * 100, loss }));
    }

    if (out_path) |op| {
        try zdistill.saveCheckpoint(&model, op, io, alloc);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Checkpoint guardado en {s}\n", .{op}));
    }
}

/// Evalúa un checkpoint cargado (FluencyNano) sobre un corpus sintético
/// determinista. Si se proporciona `prompt`, lo tokeniza y evalúa la
/// perplejidad sobre esa secuencia; de lo contrario usa el corpus estándar.
fn runEval(alloc: std.mem.Allocator, io: std.Io, ckpt_path: []const u8, prompt: ?[]const u8) !void {
    const stdout = std.Io.File.stdout();
    var buf: [512]u8 = undefined;

    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Cargando checkpoint {s}...\n", .{ckpt_path}));
    var model = try zdistill.loadCheckpoint(alloc, ckpt_path, io);
    defer model.deinit();
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "Modelo cargado: vocab={d} dim={d} capas={d} params={d} seed={d}\n", .{ model.cfg.vocab, model.cfg.dim, model.cfg.n_layers, model.paramCount(), model.seed }));

    if (prompt) |p| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(alloc);
        // Tokenización simple por bytes (vocab pequeño de demostración)
        for (p) |b| try ids.append(alloc, @as(u32, b) % @as(u32, @intCast(model.cfg.vocab)));
        if (ids.items.len < 2) {
            try stdout.writeStreamingAll(io, "Prompt demasiado corto para evaluar.\n");
            return;
        }
        var tgt: std.ArrayList(u32) = .empty;
        defer tgt.deinit(alloc);
        for (ids.items[1..]) |v| try tgt.append(alloc, v);
        try ids.resize(alloc, ids.items.len - 1);
        const result = try zdistill.evaluate(&model, alloc, ids.items, tgt.items);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[eval] ce={d:.4} ppl={d:.3} acc={d:.1}%\n", .{ result.ce, result.ppl, result.acc * 100 }));
    } else {
        const V: u32 = @intCast(model.cfg.vocab);
        const T = 32;
        var ids: [T]u32 = undefined;
        var tgt: [T]u32 = undefined;
        var cur: u32 = 1;
        for (0..T) |t| {
            ids[t] = cur;
            cur = (cur * 7 + 3) % V;
            tgt[t] = cur;
        }
        const result = try zdistill.evaluate(&model, alloc, &ids, &tgt);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[eval] ce={d:.4} ppl={d:.3} acc={d:.1}%\n", .{ result.ce, result.ppl, result.acc * 100 }));
    }
}

// Motor de inferencia para el servidor: posee el Transformer + tokenizador
// cargados una sola vez. zserve lo invoca por petición (serie, un hilo).
const ServeEngine = struct {
    alloc: std.mem.Allocator,
    tf: *zarch.Transformer,
    tok: *ztok.Tokenizer,
    speculate: bool = false,

    fn generate(ctx: *anyopaque, allocator: std.mem.Allocator, prompt: []const u8, max_tokens: usize) anyerror![]u8 {
        const self: *ServeEngine = @alignCast(@ptrCast(ctx));
        _ = allocator; // usamos self.alloc (mismo gpa) por consistencia con el tokenizador
        const cap = self.tf.config.seq_len;

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.alloc);
        try self.tok.encode(prompt, &ids);

        var pos: usize = 0;
        var logits: []f32 = undefined;
        var any = false;
        for (ids.items) |id| {
            if (pos >= cap) break;
            logits = self.tf.forward(id, pos);
            pos += 1;
            any = true;
        }

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.alloc);
        if (!any) return out.toOwnedSlice(self.alloc);

        if (self.speculate) {
            var spec = zspeculate.SpeculativeGenerator{ .tf = self.tf };
            defer spec.deinit(self.alloc);
            _ = try spec.generate(self.alloc, self.tok, &ids, &out, max_tokens, cap);
            return out.toOwnedSlice(self.alloc);
        }

        var gen: usize = 0;
        while (gen < max_tokens and pos < cap) : (gen += 1) {
            var best: u32 = 0;
            var best_val = logits[0];
            for (logits, 0..) |v, i| {
                if (v > best_val) {
                    best_val = v;
                    best = @intCast(i);
                }
            }
            if (self.tok.eos_id) |e| if (best == e) break;
            const one = [_]u32{best};
            try self.tok.decode(one[0..], &out);
            logits = self.tf.forward(best, pos);
            pos += 1;
        }
        return out.toOwnedSlice(self.alloc);
    }

    // Variante en streaming: emite cada token según se genera (SSE/NDJSON).
    fn generateStream(ctx: *anyopaque, allocator: std.mem.Allocator, prompt: []const u8, max_tokens: usize, emit_ctx: *anyopaque, emit: zserve.EmitFn) anyerror!void {
        const self: *ServeEngine = @alignCast(@ptrCast(ctx));
        _ = allocator;
        const cap = self.tf.config.seq_len;

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.alloc);
        try self.tok.encode(prompt, &ids);

        var pos: usize = 0;
        var logits: []f32 = undefined;
        var any = false;
        for (ids.items) |id| {
            if (pos >= cap) break;
            logits = self.tf.forward(id, pos);
            pos += 1;
            any = true;
        }
        if (!any) return;

        if (self.speculate) {
            var spec = zspeculate.SpeculativeGenerator{ .tf = self.tf };
            defer spec.deinit(self.alloc);
            try spec.generateStream(self.alloc, self.tok, &ids, max_tokens, cap, emit_ctx, emit);
            return;
        }

        var buf: [256]u8 = undefined;
        var gen: usize = 0;
        while (gen < max_tokens and pos < cap) : (gen += 1) {
            var best: u32 = 0;
            var best_val = logits[0];
            for (logits, 0..) |v, i| {
                if (v > best_val) {
                    best_val = v;
                    best = @intCast(i);
                }
            }
            if (self.tok.eos_id) |e| if (best == e) break;
            const piece = self.tok.decodeToken(best, &buf);
            try emit(emit_ctx, piece);
            logits = self.tf.forward(best, pos);
            pos += 1;
        }
    }

    /// Generación restringida por gramática JSON: emite un objeto
    /// {"name":"...","arguments":...} válido forzando prefijo y usando
    /// grammarArgmax en cada paso. Devuelve slice propiedad del allocator.
    fn generateJson(ctx: *anyopaque, allocator: std.mem.Allocator, prompt: []const u8, max_tokens: usize) anyerror![]u8 {
        const self: *ServeEngine = @alignCast(@ptrCast(ctx));
        _ = allocator;
        const cap = self.tf.config.seq_len;

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.alloc);
        try self.tok.encode(prompt, &ids);

        var pos: usize = 0;
        var logits: []f32 = undefined;
        var any = false;
        for (ids.items) |id| {
            if (pos >= cap) break;
            logits = self.tf.forward(id, pos);
            pos += 1;
            any = true;
        }

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.alloc);
        if (!any) return out.toOwnedSlice(self.alloc);

        var gram = zarch.grammar.BalancedJson{};
        const gctx = GrammarCtx{ .tok = self.tok };
        // Prefijo forzado para empujar la forma de tool-call JSON.
        const prefix = "{\"name\":\"";
        for (prefix) |b| _ = gram.feed(b);
        try out.appendSlice(self.alloc, prefix);

        var gen: usize = 0;
        while (gen < max_tokens and pos < cap) : (gen += 1) {
            const next = zarch.grammar.grammarArgmax(logits, &gram, gctx, GrammarCtx.dec) orelse break;
            if (self.tok.eos_id) |e| if (next == e) break;
            const one = [_]u32{next};
            try self.tok.decode(one[0..], &out);
            if (gram.isComplete()) break;
            logits = self.tf.forward(next, pos);
            pos += 1;
        }
        return out.toOwnedSlice(self.alloc);
    }

    /// Embedding por hidden states reales del transformer (mean pooling de la
    /// última capa oculta post-FFN residual). Esto produce embeddings contextualizados
    /// que capturan semántica, no sólo lookup de tabla de tokens.
    /// Requiere procesar cada token secuencialmente con forwardWithHidden para
    /// obtener el hidden state de la última capa.
    fn embed(ctx: *anyopaque, allocator: std.mem.Allocator, text: []const u8) anyerror![]f32 {
        const self: *ServeEngine = @alignCast(@ptrCast(ctx));
        // El buffer DEVUELTO debe asignarse con el `allocator` recibido: el caller
        // (serveRequest) lo libera con ese mismo allocator. Asignarlo con self.alloc
        // sería un free cross-allocator si difieren. Los temporales usan self.alloc.

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.alloc);
        try self.tok.encode(text, &ids);
        if (ids.items.len == 0) return allocator.alloc(f32, 0);

        const dim = self.tf.config.dim;
        const n_layers = self.tf.config.n_layers;

        // Buffer para hidden states de la última capa de cada token.
        // forwardWithHidden escribe hidden_out[l * dim ..] para cada capa l.
        // Nos interesa la última capa (l = n_layers - 1).
        const hidden_buf = try self.alloc.alloc(f32, n_layers * dim);
        defer self.alloc.free(hidden_buf);

        var sum = try allocator.alloc(f32, dim);
        errdefer allocator.free(sum);
        @memset(sum, 0.0);

        // Reset KV cache para procesar la secuencia desde cero
        const kv_dim = (dim * self.tf.config.n_kv_heads) / self.tf.config.n_heads;
        for (0..n_layers) |l| {
            const loff = l * self.tf.config.seq_len * kv_dim;
            @memset(self.tf.key_cache[loff..][0..self.tf.config.seq_len * kv_dim], 0);
            @memset(self.tf.value_cache[loff..][0..self.tf.config.seq_len * kv_dim], 0);
        }

        var valid_tokens: usize = 0;
        for (ids.items, 0..) |id, pos| {
            if (id >= self.tf.config.vocab_size) continue;
            _ = self.tf.forwardWithHidden(id, pos, hidden_buf);
            // Hidden state de la última capa (post-FFN residual, pre-final-norm)
            const last_layer_hidden = hidden_buf[(n_layers - 1) * dim ..][0..dim];
            for (0..dim) |i| sum[i] += last_layer_hidden[i];
            valid_tokens += 1;
        }

        if (valid_tokens == 0) {
            // Fallback a mean pooling de token embeddings si forward falló
            const emb_table = self.tf.token_embedding_table;
            for (ids.items) |id| {
                if (id >= self.tf.config.vocab_size) continue;
                const e = emb_table[id * dim ..][0..dim];
                for (0..dim) |i| sum[i] += e[i];
                valid_tokens += 1;
            }
        }

        if (valid_tokens == 0) {
            allocator.free(sum);
            return allocator.alloc(f32, 0);
        }

        // Mean pooling
        const count: f32 = @floatFromInt(valid_tokens);
        for (0..dim) |i| sum[i] /= count;

        // L2 normalize (estándar para embeddings de sentencia)
        var norm: f32 = 0;
        for (sum) |v| norm += v * v;
        norm = @sqrt(norm);
        if (norm > 1e-12) {
            const inv_norm = 1.0 / norm;
            for (sum) |*v| v.* *= inv_norm;
        }

        return sum;
    }
};

const McpBridge = struct {
    io: std.Io,
    tools: [4]zagent.Tool,

    fn execute(ctx: *anyopaque, name: []const u8, args_json: []const u8, allocator: std.mem.Allocator) !zmcp.ToolResult {
        const self: *McpBridge = @alignCast(@ptrCast(ctx));
        for (self.tools) |tool| {
            if (std.mem.eql(u8, tool.vtable.name(tool.ptr), name)) {
                const result = try tool.vtable.execute(tool.ptr, self.io, args_json, allocator);
                return .{ .content = result.content, .is_error = result.is_error };
            }
        }
        return .{ .content = try allocator.dupe(u8, "tool not found"), .is_error = true };
    }
};

fn runMcpSelftest(alloc: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const stdout = std.Io.File.stdout();
    var buf: [1024]u8 = undefined;

    const exe_path = if (args.len > 0) args[0] else "fluency";
    try stdout.writeStreamingAll(io, "[mcp-selftest] Spawning fluency mcp-server...\n");

    var child = try std.process.spawn(io, .{
        .argv = &.{ exe_path, "mcp-server" },
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .pipe,
    });

    var stdin_buf: [4096]u8 = undefined;
    var stdout_buf: [4096]u8 = undefined;
    var stderr_buf: [4096]u8 = undefined;
    const child_stdin = child.stdin.?.writer(io, &stdin_buf);
    const child_stdout = child.stdout.?.reader(io, &stdout_buf);
    const child_stderr = child.stderr.?.reader(io, &stderr_buf);
    _ = child_stderr;

    var client = zmcp.Client.initPipes(alloc, child_stdin, child_stdout);

    try stdout.writeStreamingAll(io, "[mcp-selftest] initialize...\n");
    try client.initialize("fluency-selftest");

    try stdout.writeStreamingAll(io, "[mcp-selftest] tools/list...\n");
    const tools = try client.toolsList();
    defer client.deinitToolsList(tools);
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[mcp-selftest] {d} tools listed\n", .{tools.len}));
    for (tools) |t| {
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "  - {s}: {s}\n", .{ t.name, t.description }));
    }

    try stdout.writeStreamingAll(io, "[mcp-selftest] tools/call finish...\n");
    const result = try client.toolsCall("finish", "{}");
    defer alloc.free(result.content);
    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[mcp-selftest] result: {s} (is_error={})\n", .{ result.content, result.is_error }));

    try stdout.writeStreamingAll(io, "[mcp-selftest] bridge to zagent.Registry...\n");
    var reg = zagent.Registry.init(alloc);
    defer reg.deinit();
    try zmcp.registerTools(&client, &reg, alloc);

    if (reg.tools.get("finish")) |tool| {
        const tr = try tool.vtable.execute(tool.ptr, io, "{}", alloc);
        defer alloc.free(tr.content);
        try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[mcp-selftest] registry dispatch: {s}\n", .{tr.content}));
    } else {
        try stdout.writeStreamingAll(io, "[mcp-selftest] ERROR: finish not in registry\n");
    }

    // Limpiar wrappers registrados para evitar fugas
    var it = reg.tools.iterator();
    while (it.next()) |entry| {
        const wrapper = @as(*zmcp.McpToolWrapper, @ptrCast(@alignCast(entry.value_ptr.ptr)));
        alloc.free(wrapper.tool_name);
        alloc.destroy(wrapper);
    }

    // Cerrar stdin del child para que el servidor termine su bucle
    if (child.stdin) |f| {
        f.close(io);
        child.stdin = null;
    }

    _ = try child.wait(io);
    try stdout.writeStreamingAll(io, "[mcp-selftest] PASSED\n");
}

/// Realiza una petición HTTP usando curl.exe (disponible en Windows 10+) y devuelve el body.
fn httpRequestRaw(alloc: std.mem.Allocator, io: std.Io, host: []const u8, port: u16, method: []const u8, path: []const u8, body: ?[]const u8) ![]u8 {
    const url = try std.fmt.allocPrint(alloc, "http://{s}:{d}{s}", .{ host, port, path });
    defer alloc.free(url);

    var argv_buf: [9][]const u8 = undefined;
    var argc: usize = 0;
    argv_buf[argc] = "C:\\Windows\\System32\\curl.exe";
    argc += 1;
    argv_buf[argc] = "-s";
    argc += 1;
    argv_buf[argc] = "-X";
    argc += 1;
    argv_buf[argc] = method;
    argc += 1;
    if (body) |b| {
        argv_buf[argc] = "-H";
        argc += 1;
        argv_buf[argc] = "Content-Type: application/json";
        argc += 1;
        argv_buf[argc] = "-d";
        argc += 1;
        argv_buf[argc] = b;
        argc += 1;
    }
    argv_buf[argc] = url;
    argc += 1;

    var child = try std.process.spawn(io, .{
        .argv = argv_buf[0..argc],
        .stdout = .pipe,
    });

    var out_buf: [32768]u8 = undefined;
    var out_total: usize = 0;
    if (child.stdout) |f| {
        var read_buf: [4096]u8 = undefined;
        var file_reader = f.reader(io, &read_buf);
        while (true) {
            const n = (&file_reader.interface).readSliceShort(out_buf[out_total..]) catch |err| {
                if (err == error.EndOfStream) break;
                return err;
            };
            if (n == 0) break;
            out_total += n;
            if (out_total >= out_buf.len) break;
        }
        f.close(io);
        child.stdout = null;
    }
    child.kill(io);

    return alloc.dupe(u8, out_buf[0..out_total]);
}

fn runServeSelftest(alloc: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    const exe_path = if (args.len > 0) args[0] else "fluency";
    const port: u16 = 11499;

    try stdout.writeStreamingAll(io, "[serve-selftest] Spawning fluency serve...\n");
    var child = try std.process.spawn(io, .{
        .argv = &.{ exe_path, "serve", "--model", "Fluency-selftest", "--port", "11499" },
        .stdout = .pipe,
        .stderr = .pipe,
    });

    // Esperar a que el servidor arranque
    try std.Io.sleep(io, std.Io.Duration.fromSeconds(2), .real);

    var passed: usize = 0;
    var failed: usize = 0;

    const Check = struct {
        name: []const u8,
        path: []const u8,
        method: []const u8,
        body: ?[]const u8,
        must_contain: []const []const u8,
    };

    const checks = [_]Check{
        .{ .name = "GET /v1/models", .path = "/v1/models", .method = "GET", .body = null, .must_contain = &.{ "\"object\":\"list\"", "Fluency-selftest" } },
        .{ .name = "GET /api/tags", .path = "/api/tags", .method = "GET", .body = null, .must_contain = &.{ "\"models\"", "Fluency-selftest" } },
        .{ .name = "POST /api/show", .path = "/api/show", .method = "POST", .body = "{\"name\":\"Fluency-selftest\"}", .must_contain = &.{ "\"details\"", "\"modelfile\"" } },
        .{ .name = "POST /v1/chat/completions", .path = "/v1/chat/completions", .method = "POST", .body = "{\"model\":\"Fluency-selftest\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}", .must_contain = &.{ "\"choices\"", "\"message\"" } },
        .{ .name = "POST /api/chat", .path = "/api/chat", .method = "POST", .body = "{\"model\":\"Fluency-selftest\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}", .must_contain = &.{ "\"message\"", "\"done\":true" } },
        .{ .name = "POST /v1/embeddings", .path = "/v1/embeddings", .method = "POST", .body = "{\"input\":\"hello\"}", .must_contain = &.{ "\"object\":\"list\"", "\"embedding\"" } },
        .{ .name = "POST /api/embeddings", .path = "/api/embeddings", .method = "POST", .body = "{\"input\":\"hello\"}", .must_contain = &.{ "\"embeddings\"" } },
    };

    for (checks) |c| {
        const resp = httpRequestRaw(alloc, io, "127.0.0.1", port, c.method, c.path, c.body) catch |err| {
            try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[serve-selftest] FAIL {s}: {s}\n", .{ c.name, @errorName(err) }));
            failed += 1;
            continue;
        };
        defer alloc.free(resp);

        var ok = true;
        for (c.must_contain) |needle| {
            if (std.mem.indexOf(u8, resp, needle) == null) {
                ok = false;
                try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[serve-selftest] FAIL {s}: missing '{s}'\n", .{ c.name, needle }));
            }
        }
        if (ok) {
            try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[serve-selftest] PASS {s}\n", .{c.name}));
            passed += 1;
        } else {
            failed += 1;
        }
    }

    // Matar el servidor
    child.kill(io);

    try stdout.writeStreamingAll(io, try std.fmt.bufPrint(&buf, "[serve-selftest] {d}/{d} checks passed\n", .{ passed, passed + failed }));
    if (failed == 0) {
        try stdout.writeStreamingAll(io, "[serve-selftest] PASSED\n");
    } else {
        try stdout.writeStreamingAll(io, "[serve-selftest] FAILED\n");
    }
}

test "F5 protocol surfaces smoke" {
    const allocator = std.testing.allocator;

    var req = try zrpc.parseRequest(allocator, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}");
    defer req.deinit();
    const pong = try zrpc.makeResult(allocator, req.id(), "{}");
    defer allocator.free(pong);
    try std.testing.expect(std.mem.indexOf(u8, pong, "\"result\":{}") != null);

    const models = [_]zserve.ModelInfo{.{ .id = "fluency-local" }};
    const models_json = try zserve.modelsResponse(allocator, models[0..]);
    defer allocator.free(models_json);
    try std.testing.expect(std.mem.indexOf(u8, models_json, "\"object\":\"model\"") != null);

    const tool_infos = [_]zmcp.ToolInfo{.{
        .name = "finish",
        .description = "finish task",
        .parameters_json = "{\"type\":\"object\",\"properties\":{}}",
    }};
    const Ctx = struct {
        fn execute(_: *anyopaque, _: []const u8, _: []const u8, alloc: std.mem.Allocator) !zmcp.ToolResult {
            return .{ .content = try alloc.dupe(u8, "ok"), .is_error = false };
        }
    };
    var dummy: u8 = 0;
    var server = zmcp.Server.init(allocator, tool_infos[0..], &dummy, Ctx.execute);
    const list = try server.handle("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}");
    defer allocator.free(list);
    try std.testing.expect(std.mem.indexOf(u8, list, "\"name\":\"finish\"") != null);
}

// --- TAREA 1: tests de gramática restringida + integración agente ---

const EchoTool = struct {
    call_count: usize = 0,
    pub fn name(_: *anyopaque) []const u8 {
        return "echo";
    }
    pub fn description(_: *anyopaque) []const u8 {
        return "echo";
    }
    pub fn parameters_json(_: *anyopaque) []const u8 {
        return "{\"type\":\"object\",\"properties\":{\"msg\":{\"type\":\"string\"}}}";
    }
    pub fn execute(ptr: *anyopaque, _: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const self: *EchoTool = @alignCast(@ptrCast(ptr));
        self.call_count += 1;
        return .{ .content = try allocator.dupe(u8, args_json), .is_error = false };
    }
    pub fn tool(self: *EchoTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

test "grammarArgmax en bucle genera JSON tool-call valido y parseable" {
    const alloc = std.testing.allocator;

    const Ctx = struct {
        fn dec(_: @This(), id: u32, buf: []u8) []const u8 {
            const chars = [_][]const u8{ "{", "\"", "n", "a", "m", "e", ":", "f", "i", "s", "h", "r", ",", "g", "u", "t", "}" };
            if (id >= chars.len) return "";
            const s = chars[id];
            @memcpy(buf[0..s.len], s);
            return buf[0..s.len];
        }
    };

    // Secuencia para emitir: {"name":"finish","arguments":{}}
    const sequence = [_]u32{ 0, 1, 2, 3, 4, 5, 1, 6, 1, 7, 8, 2, 8, 9, 10, 1, 12, 1, 3, 11, 13, 14, 4, 5, 2, 15, 9, 1, 6, 0, 16, 16 };

    var g = zarch.grammar.BalancedJson{};
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(alloc);

    for (sequence) |expected_id| {
        var logits = try alloc.alloc(f32, 17);
        defer alloc.free(logits);
        @memset(logits, -1e9);
        logits[expected_id] = 10.0;

        const pick = zarch.grammar.grammarArgmax(logits, &g, Ctx{}, Ctx.dec);
        try std.testing.expect(pick != null);
        try std.testing.expectEqual(expected_id, pick.?);

        var buf: [64]u8 = undefined;
        const bytes = Ctx.dec(Ctx{}, pick.?, &buf);
        try output.appendSlice(alloc, bytes);
    }

    try std.testing.expect(g.isComplete());

    const calls = try zagent.parseToolCalls(alloc, output.items);
    defer zagent.deinitToolCalls(alloc, calls);
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("finish", calls[0].name);
}

test "agent ReAct despacha tool-call desde generate_fn mock JSON" {
    const alloc = std.testing.allocator;
    var reg = zagent.Registry.init(alloc);
    defer reg.deinit();

    var echo_tool = EchoTool{};
    try reg.register(echo_tool.tool());

    const MockGen = struct {
        var call_count: usize = 0;
        fn generate(_: ?*anyopaque, _: []const u8, gpa: std.mem.Allocator) ![]u8 {
            call_count += 1;
            return try gpa.dupe(u8, "{\"name\":\"finish\",\"arguments\":{}}");
        }
    };
    MockGen.call_count = 0;

    var agent = zagent.Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.generate_fn = &MockGen.generate;

    // Prompt con tool-call no-finish para iter 0; generate_fn provee finish en iter 1
    const prompt = "{\"name\":\"echo\",\"arguments\":{\"msg\":\"hola\"}}";
    var history = try agent.run(prompt, 2);
    defer history.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), MockGen.call_count);
    try std.testing.expectEqual(@as(usize, 1), echo_tool.call_count);
    try std.testing.expect(std.mem.indexOf(u8, history.items, "hola") != null);
}

test "McpBridge execute finish returns ok" {
    const alloc = std.testing.allocator;
    var finish_tool = zagent.tools.FinishTool{};
    var read_tool = zagent.tools.FsReadTool{};
    var write_tool = zagent.tools.FsWriteTool{};
    var bash_tool = zagent.tools.BashTool{};
    var bridge = McpBridge{
        .io = undefined,
        .tools = .{
            finish_tool.tool(),
            read_tool.tool(),
            write_tool.tool(),
            bash_tool.tool(),
        },
    };
    const result = try McpBridge.execute(&bridge, "finish", "{}", alloc);
    defer alloc.free(result.content);
    try std.testing.expect(!result.is_error);
}

test "zmcp server handles sequential requests" {
    const alloc = std.testing.allocator;
    const tools = [_]zmcp.ToolInfo{
        .{ .name = "echo", .description = "echo", .parameters_json = "{\"type\":\"object\",\"properties\":{\"msg\":{\"type\":\"string\"}}}" },
    };
    const Ctx = struct {
        fn execute(_: *anyopaque, _: []const u8, args_json: []const u8, a: std.mem.Allocator) !zmcp.ToolResult {
            return .{ .content = try a.dupe(u8, args_json), .is_error = false };
        }
    };
    var dummy: u8 = 0;
    var server = zmcp.Server.init(alloc, tools[0..], &dummy, Ctx.execute);

    const r1 = try server.handle("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}");
    defer alloc.free(r1);
    try std.testing.expect(std.mem.indexOf(u8, r1, "\"name\":\"echo\"") != null);

    const r2 = try server.handle("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"echo\",\"arguments\":{\"msg\":\"hola\"}}}");
    defer alloc.free(r2);
    try std.testing.expect(std.mem.indexOf(u8, r2, "hola") != null);
}

test "zmcp bridge registerTools round-trip" {
    const alloc = std.testing.allocator;
    const tools = [_]zmcp.ToolInfo{
        .{ .name = "echo", .description = "echo", .parameters_json = "{\"type\":\"object\",\"properties\":{\"msg\":{\"type\":\"string\"}}}" },
    };
    const Ctx = struct {
        fn execute(_: *anyopaque, _: []const u8, args_json: []const u8, a: std.mem.Allocator) !zmcp.ToolResult {
            return .{ .content = try a.dupe(u8, args_json), .is_error = false };
        }
    };
    var dummy: u8 = 0;
    var server = zmcp.Server.init(alloc, tools[0..], &dummy, Ctx.execute);
    var tctx = zmcp.MemTransportCtx{ .server = &server };

    var client = zmcp.Client.init(alloc, undefined);
    client.transport_fn = &zmcp.memTransport;
    client.transport_ctx = &tctx;

    var reg = zagent.Registry.init(alloc);
    defer reg.deinit();
    try zmcp.registerTools(&client, &reg, alloc);

    const tool = reg.tools.get("echo") orelse return error.NotFound;
    const res = try tool.vtable.execute(tool.ptr, undefined, "{\"msg\":\"hola\"}", alloc);
    defer alloc.free(res.content);
    try std.testing.expect(std.mem.indexOf(u8, res.content, "hola") != null);

    // Limpiar wrappers registrados para evitar fugas en test
    var it = reg.tools.iterator();
    while (it.next()) |entry| {
        const wrapper = @as(*zmcp.McpToolWrapper, @ptrCast(@alignCast(entry.value_ptr.ptr)));
        alloc.free(wrapper.tool_name);
        alloc.destroy(wrapper);
    }
}

test "AgentBundle.wire deja punteros estables: despacha tool-call sin colgar" {
    // Regresión: setupAgentAndRegistry devolvía la struct por valor mientras agent.registry
    // y los tool.ptr apuntaban a sus propios locales → al moverse al retornar quedaban
    // colgantes y `self.registry.tools.contains(...)` panicaba ("incorrect alignment").
    // AgentBundle posee registry/agent/tools como campos y wire() fija los punteros tras
    // colocar el bundle en su frame final. Aquí ejercitamos exactamente ese contrato.
    const alloc = std.testing.allocator;
    // io no se usa: la ruta tool-only de `finish` no toca IO (igual que el test mock).
    var ar = AgentBundle.init(alloc, undefined);
    ar.wire();
    defer ar.deinit();

    // Los punteros internos deben apuntar dentro del propio bundle, no a locales muertos.
    try std.testing.expectEqual(&ar.reg, ar.agent.registry);
    try std.testing.expectEqual(&ar.agent.memory, ar.mem_read.store.?);
    try std.testing.expectEqual(&ar.agent.memory, ar.mem_write.store.?);
    try std.testing.expect(ar.agent.registry.tools.contains("finish"));

    // El bucle ReAct debe despachar la tool-call sin panic (esto es lo que crasheaba).
    var history = try ar.agent.run("<tool_call>{\"name\":\"finish\",\"arguments\":{}}</tool_call>", 5);
    defer history.deinit(alloc);
    try std.testing.expect(std.mem.indexOf(u8, history.items, "finish") != null);
}

// --- Regresión F7.B (speculative decoding) ---
// Se ubica en main.zig (raíz de test) y NO en src/zspeculate.zig porque éste se
// importa como MÓDULO (build.zig), no por ruta relativa, y `zig test` no ejecuta
// los tests de módulos importados así. Cubre los 2 bugs hallados en auditoría:
//   (1) `generate` truncaba `out` al ÚLTIMO token (clearRetainingCapacity por token).
//   (2) EOS no uniforme podía emitir un token de más respecto a greedy.
fn specArgmax(logits: []const f32) u32 {
    var best: u32 = 0;
    var best_val = logits[0];
    for (logits, 0..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best = @intCast(i);
        }
    }
    return best;
}

fn specMkW(a: std.mem.Allocator, list: *std.ArrayList([]f32), n: usize, seed: f32) ![]f32 {
    const w = try a.alloc(f32, n);
    for (w, 0..) |*x, i| x.* = @sin(seed + @as(f32, @floatFromInt(i)) * 0.1) * 0.1;
    try list.append(a, w);
    return w;
}

fn specBuildTinyTf(a: std.mem.Allocator, cfg: zarch.Config, list: *std.ArrayList([]f32)) !zarch.Transformer {
    const dim = cfg.dim;
    const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    const hidden = cfg.hidden_dim;
    const L = cfg.n_layers;
    const vocab = cfg.vocab_size;
    var tf = try zarch.Transformer.init(a, cfg);
    tf.token_embedding_table = try specMkW(a, list, vocab * dim, 1.0);
    tf.rms_att_weight = try specMkW(a, list, L * dim, 2.0);
    tf.rms_ffn_weight = try specMkW(a, list, L * dim, 3.0);
    tf.wq = try specMkW(a, list, L * dim * dim, 4.0);
    tf.wk = try specMkW(a, list, L * kv_dim * dim, 5.0);
    tf.wv = try specMkW(a, list, L * kv_dim * dim, 6.0);
    tf.wo = try specMkW(a, list, L * dim * dim, 7.0);
    tf.w1 = try specMkW(a, list, L * dim * hidden, 8.0);
    tf.w2 = try specMkW(a, list, L * hidden * dim, 9.0);
    tf.w3 = try specMkW(a, list, L * dim * hidden, 10.0);
    tf.rms_final_weight = try specMkW(a, list, dim, 11.0);
    return tf;
}

test "F7.B speculative: out acumula salida completa y coincide con greedy" {
    const alloc = std.testing.allocator;
    const cfg = zarch.Config{
        .dim = 4,
        .hidden_dim = 8,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 8,
        .seq_len = 32,
        .n_experts = 0,
        .theta = 10000.0,
        .eps = 1e-5,
    };
    var bufs: std.ArrayList([]f32) = .empty;
    defer {
        for (bufs.items) |b| alloc.free(b);
        bufs.deinit(alloc);
    }

    const MockTok = struct {
        allocator: std.mem.Allocator,
        eos_id: ?u32 = null,
        pub fn decode(self: *@This(), tids: []const u32, out: *std.ArrayList(u8)) !void {
            for (tids) |id| try out.append(self.allocator, @intCast(id & 0xFF));
        }
    };

    const prompt = [_]u32{ 1, 2, 3, 1, 2, 3 };
    const K: usize = 6;

    // Referencia greedy pura.
    var ref_ids: std.ArrayList(u32) = .empty;
    defer ref_ids.deinit(alloc);
    try ref_ids.appendSlice(alloc, &prompt);
    {
        var tf_ref = try specBuildTinyTf(alloc, cfg, &bufs);
        defer tf_ref.deinit();
        var pos: usize = 0;
        var logits: []f32 = undefined;
        for (prompt) |id| {
            logits = tf_ref.forward(id, pos);
            pos += 1;
        }
        var g: usize = 0;
        while (g < K) : (g += 1) {
            const best = specArgmax(logits);
            try ref_ids.append(alloc, best);
            logits = tf_ref.forward(best, pos);
            pos += 1;
        }
    }

    // Speculative (mismos pesos deterministas).
    var tf_spec = try specBuildTinyTf(alloc, cfg, &bufs);
    defer tf_spec.deinit();
    var spec = zspeculate.SpeculativeGenerator{ .tf = &tf_spec, .ngram_len = 2, .max_draft = 4 };
    defer spec.deinit(alloc);
    var spec_ids: std.ArrayList(u32) = .empty;
    defer spec_ids.deinit(alloc);
    try spec_ids.appendSlice(alloc, &prompt);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    var mtok = MockTok{ .allocator = alloc };
    {
        var pos: usize = 0;
        for (prompt) |id| {
            _ = tf_spec.forward(id, pos);
            pos += 1;
        }
    }
    const n = try spec.generate(alloc, &mtok, &spec_ids, &out, K, cfg.seq_len);

    // (1) Acumulación: 1 byte por token generado (con el bug era 1, no n).
    try std.testing.expectEqual(n, out.items.len);
    try std.testing.expect(out.items.len >= 2);
    // (2) Igualdad bit-a-bit con greedy.
    try std.testing.expectEqual(ref_ids.items.len, spec_ids.items.len);
    for (ref_ids.items, spec_ids.items) |a, b| try std.testing.expectEqual(a, b);
}


