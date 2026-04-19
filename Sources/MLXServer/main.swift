import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon
import MLXNN

func log(_ msg: String) {
    FileHandle.standardOutput.write(Data("[MLXServer] \(msg)\n".utf8))
}

// MARK: - I/O helpers

/// Loop-guarded write to a socket fd. Returns true if `bytes` was fully written,
/// false if the peer disconnected or an unrecoverable error occurred. Handles
/// EINTR and partial writes so SSE frames and HTTP bodies can't silently truncate.
@discardableResult
func writeAll(_ fd: Int32, _ bytes: UnsafePointer<UInt8>, _ count: Int) -> Bool {
    var remaining = count
    var ptr = bytes
    while remaining > 0 {
        let n = write(fd, ptr, remaining)
        if n > 0 {
            remaining -= n
            ptr = ptr.advanced(by: n)
        } else if n < 0 {
            if errno == EINTR { continue }
            return false  // EPIPE / ECONNRESET / EAGAIN (blocking socket) / etc.
        } else {
            return false  // write returning 0 is effectively EOF
        }
    }
    return true
}

@discardableResult
func writeAll(_ fd: Int32, _ s: String) -> Bool {
    var data = Data(s.utf8)
    return data.withUnsafeMutableBytes { raw -> Bool in
        guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return false }
        return writeAll(fd, base, raw.count)
    }
}

/// Loop-guarded read. Handles EINTR. Returns bytes read, 0 on EOF, -1 on error.
func readAll(_ fd: Int32, _ buf: UnsafeMutablePointer<UInt8>, _ count: Int) -> Int {
    var remaining = count
    var ptr = buf
    var total = 0
    while remaining > 0 {
        let n = read(fd, ptr, remaining)
        if n > 0 {
            total += n
            remaining -= n
            ptr = ptr.advanced(by: n)
        } else if n == 0 {
            return total  // EOF
        } else {
            if errno == EINTR { continue }
            return -1
        }
    }
    return total
}

/// JSON string-escape per RFC 8259 — handles backslash, quotes, all control
/// chars < 0x20 (including \b \f \n \r \t), and leaves other UTF-8 intact.
/// Model output can contain embedded control chars that break hand-rolled escapes.
func jsonEscape(_ s: String) -> String {
    var out = ""
    out.reserveCapacity(s.utf8.count + 2)
    for scalar in s.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\u{08}": out += "\\b"
        case "\u{09}": out += "\\t"
        case "\u{0A}": out += "\\n"
        case "\u{0C}": out += "\\f"
        case "\u{0D}": out += "\\r"
        default:
            if scalar.value < 0x20 {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
    }
    return out
}

/// Configure a freshly-accepted client socket: low-latency SSE (TCP_NODELAY),
/// receive/send timeouts to prevent slow-loris from hogging slots, and keepalive.
func configureClientSocket(_ fd: Int32, recvTimeoutSec: Int = 60, sendTimeoutSec: Int = 30) {
    var one: Int32 = 1
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, socklen_t(MemoryLayout<Int32>.size))
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, socklen_t(MemoryLayout<Int32>.size))

    var rcv = timeval(tv_sec: recvTimeoutSec, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rcv, socklen_t(MemoryLayout<timeval>.size))
    var snd = timeval(tv_sec: sendTimeoutSec, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &snd, socklen_t(MemoryLayout<timeval>.size))
}

// MARK: - OpenAI Types

struct ChatMessage: Codable {
    let role: String
    let content: String?

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        // Content can be string, null, or array — just grab string or nil
        if let str = try? container.decode(String.self, forKey: .content) {
            content = str
        } else {
            content = nil
        }
    }

    enum CodingKeys: String, CodingKey { case role, content }
}

struct ChatRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let max_tokens: Int?
    let temperature: Float?
    let stream: Bool?
    // Accept but ignore extra fields
    let tools: AnyCodable?
    let tool_choice: AnyCodable?
    let top_p: Float?
    let frequency_penalty: Float?
    let presence_penalty: Float?
    let stop: AnyCodable?
    let n: Int?

    enum CodingKeys: String, CodingKey {
        case model, messages, max_tokens, temperature, stream
        case tools, tool_choice, top_p, frequency_penalty, presence_penalty, stop, n
    }
}

// Wraps any JSON value
struct AnyCodable: Codable {
    let value: Any?
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let arr = try? container.decode([JSONValue].self) {
            value = arr.map { $0.toAny() }
        } else if let dict = try? container.decode([String: JSONValue].self) {
            value = dict.mapValues { $0.toAny() }
        } else {
            value = nil
        }
    }
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encodeNil()
    }
}

// Recursive JSON value for preserving tools structure
enum JSONValue: Codable {
    case string(String), int(Int), double(Double), bool(Bool), null
    case array([JSONValue]), object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Int.self) { self = .int(v) }
        else if let v = try? c.decode(Double.self) { self = .double(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
        else if let v = try? c.decode([String: JSONValue].self) { self = .object(v) }
        else if c.decodeNil() { self = .null }
        else { self = .null }
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }
    func toAny() -> Any {
        switch self {
        case .string(let v): return v
        case .int(let v): return v
        case .double(let v): return v
        case .bool(let v): return v
        case .null: return NSNull()
        case .array(let v): return v.map { $0.toAny() }
        case .object(let v): return v.mapValues { $0.toAny() } as [String: Any]
        }
    }
}

struct ChatResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let system_fingerprint: String
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Codable {
        let index: Int
        let message: Message?
        let delta: Message?
        let finish_reason: String?
    }

    struct Message: Codable {
        let role: String?
        let content: String?
        let reasoning_content: String?

        init(role: String? = nil, content: String? = nil, reasoning_content: String? = nil) {
            self.role = role
            self.content = content
            self.reasoning_content = reasoning_content
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encodeIfPresent(role, forKey: .role)
            try container.encodeIfPresent(content, forKey: .content)
            try container.encodeIfPresent(reasoning_content, forKey: .reasoning_content)
        }

        enum CodingKeys: String, CodingKey {
            case role, content, reasoning_content
        }
    }

    struct Usage: Codable {
        let prompt_tokens: Int
        let completion_tokens: Int
        let total_tokens: Int
    }
}

// MARK: - Minimal HTTP Server using URLSession's HTTPServer
// Using a raw socket server via Foundation for zero dependencies

// MARK: - Server Prompt Cache (Multi-Session)

struct CachedSession {
    let id: UUID = UUID()
    var tokenIds: [Int]
    var kvCache: [KVCache]
    var lastUsed: Date
}

/// Multi-session prompt cache with LCP (longest common prefix) matching.
/// Keeps up to `maxSessions` cached KV states. When a new request arrives,
/// finds the session with the longest matching token prefix, trims KV to
/// that prefix, and returns only the new tokens to prefill.
struct CacheMetrics {
    var totalRequests: Int = 0
    var cacheHits: Int = 0
    var cacheMisses: Int = 0
    var trimFailures: Int = 0
    var totalPrefillTokens: Int = 0
    var totalReusedTokens: Int = 0
    var totalPrefillMs: Double = 0
    var totalDecodeMs: Double = 0
    var totalDecodeTokens: Int = 0
    var evictions: Int = 0

    var hitRate: Double { totalRequests > 0 ? Double(cacheHits) / Double(totalRequests) : 0 }
    var avgPrefillTokens: Double { totalRequests > 0 ? Double(totalPrefillTokens) / Double(totalRequests) : 0 }
    var avgPrefillMs: Double { totalRequests > 0 ? totalPrefillMs / Double(totalRequests) : 0 }
    var avgDecodeTokensPerSec: Double { totalDecodeMs > 0 ? Double(totalDecodeTokens) / (totalDecodeMs / 1000) : 0 }
}

actor ServerPromptCache {
    var sessions: [CachedSession] = []
    /// Sessions currently being mutated by an active generate loop. Eviction and
    /// flush skip these; without this guard, a concurrent request could evict the
    /// KV cache that another task is writing into, corrupting memory.
    private var inUse: Set<UUID> = []
    let maxSessions: Int
    var metrics = CacheMetrics()
    let kvScheme: String?

    init(maxSessions: Int = 3, kvScheme: String? = nil) {
        self.maxSessions = maxSessions
        self.kvScheme = kvScheme
    }

    func markInUse(_ id: UUID) { inUse.insert(id) }
    func markIdle(_ id: UUID) { inUse.remove(id) }

    func recordRequest(hit: Bool, prefillTokens: Int, reusedTokens: Int) {
        metrics.totalRequests += 1
        if hit { metrics.cacheHits += 1 } else { metrics.cacheMisses += 1 }
        metrics.totalPrefillTokens += prefillTokens
        metrics.totalReusedTokens += reusedTokens
    }

    func recordTiming(prefillMs: Double, decodeMs: Double, decodeTokens: Int) {
        metrics.totalPrefillMs += prefillMs
        metrics.totalDecodeMs += decodeMs
        metrics.totalDecodeTokens += decodeTokens
    }

    func recordEviction() { metrics.evictions += 1 }
    func recordTrimFailure() { metrics.trimFailures += 1 }
    func getMetrics() -> CacheMetrics { metrics }
    func getSessionCount() -> Int { sessions.count }

    /// Find the session with the longest common prefix match.
    /// Returns (kvCache, newTokensToProcess, cacheStatus, sessionId).
    func fetch(tokens newTokens: [Int], model: any LanguageModel) -> ([KVCache], [Int], CacheStatus, UUID) {
        var bestIdx = -1
        var bestPrefix = 0

        for (i, session) in sessions.enumerated() {
            let prefix = commonPrefix(session.tokenIds, newTokens)
            if prefix > bestPrefix {
                bestPrefix = prefix
                bestIdx = i
            }
        }

        if bestIdx >= 0 && bestPrefix > 0 {
            let session = sessions[bestIdx]
            // Trim based on actual KV cache size, not tokenIds.count.
            // The cache may have extra decode tokens from interrupted generation.
            let actualCacheSize = session.kvCache.first?.offset ?? session.tokenIds.count
            let trimAmount = actualCacheSize - bestPrefix

            // If the new request extends the cached session (same prefix, more tokens),
            // we can trim and use in-place. If it diverges (different suffix), we need
            // to copy so the original stays intact for future reuse.
            let isExtension = (bestPrefix == session.tokenIds.count) || (trimAmount == 0)

            if isExtension {
                // Same conversation continuing — use in-place, no copy needed
                if trimAmount > 0 {
                    for c in session.kvCache {
                        if c.trim(trimAmount) == 0 {
                            metrics.trimFailures += 1
                            return freshCache(tokens: newTokens, model: model)
                        }
                    }
                }
                sessions[bestIdx].lastUsed = Date()
                sessions[bestIdx].tokenIds = Array(newTokens[0..<bestPrefix])
                let remaining = Array(newTokens[bestPrefix...])
                let status = CacheStatus.hit(prefixReused: bestPrefix, totalTokens: newTokens.count, newTokens: remaining.count)
                recordRequest(hit: true, prefillTokens: remaining.count, reusedTokens: bestPrefix)
                return (session.kvCache, remaining, status, session.id)
            } else {
                // Divergent tail: DO NOT reuse. Observed leak with TurboQuantKVCache
                // where trim-in-place leaves compressed K/V state that the attention
                // path reads despite offset having moved down — content from the
                // prior conversation's post-prefix tokens bled into responses to
                // unrelated system prompts. The shared-prefix optimization for
                // divergent conversations isn't worth correctness risk. Start fresh.
                return freshCache(tokens: newTokens, model: model)
            }
        }

        return freshCache(tokens: newTokens, model: model)
    }

    private func freshCache(tokens: [Int], model: any LanguageModel) -> ([KVCache], [Int], CacheStatus, UUID) {
        evictIfNeeded()
        let kvParams = kvScheme.map { GenerateParameters(kvScheme: $0) }
        let cache = model.newCache(parameters: kvParams)
        let session = CachedSession(tokenIds: [], kvCache: cache, lastUsed: Date())
        sessions.append(session)
        let status = CacheStatus.miss(totalTokens: tokens.count, sessionsCount: sessions.count)
        recordRequest(hit: false, prefillTokens: tokens.count, reusedTokens: 0)
        return (cache, tokens, status, session.id)
    }

    /// Save token state after generation completes.
    ///
    /// `promptTokens` is required; `generatedTokens` should hold the token IDs
    /// actually produced by generate() for maximum cache reuse — the next request
    /// will re-tokenize the assistant reply and try to match a longer prefix.
    /// Correctness does not depend on `generatedTokens` being populated, since
    /// `fetch()` reads `actualCacheSize` from the live KV cache offset; passing
    /// `[]` just means the next request won't reuse the KV for the assistant
    /// turn and will trim+reprefill that region.
    func save(sessionId: UUID, promptTokens: [Int], generatedTokens: [Int] = []) {
        if let idx = sessions.firstIndex(where: { $0.id == sessionId }) {
            sessions[idx].tokenIds = promptTokens + generatedTokens
            sessions[idx].lastUsed = Date()
        }
    }

    /// Pick the index of the oldest evictable (idle) session, or nil if none.
    /// In-use sessions are skipped so concurrent requests can't corrupt each
    /// other's KV state.
    private func oldestIdleIndex() -> Int? {
        var bestIdx: Int? = nil
        var bestDate = Date.distantFuture
        for (i, s) in sessions.enumerated() where !inUse.contains(s.id) {
            if s.lastUsed < bestDate {
                bestDate = s.lastUsed
                bestIdx = i
            }
        }
        return bestIdx
    }

    private func evictIfNeeded() {
        while sessions.count >= maxSessions {
            guard let idx = oldestIdleIndex() else {
                // All sessions busy — nothing safe to evict. The new freshCache
                // caller will append past maxSessions; better a brief capacity
                // overshoot than a use-after-free.
                log("evictIfNeeded: all \(sessions.count) sessions in use, skipping")
                return
            }
            let s = sessions[idx]
            log("Evicting session \(idx) (\(s.tokenIds.count) tokens, idle \(Int(-s.lastUsed.timeIntervalSinceNow))s)")
            sessions.remove(at: idx)
            metrics.evictions += 1
        }
    }

    /// Evict idle sessions, keeping at most `keep` sessions. Skips in-use.
    func evictIdle(keep: Int) {
        while sessions.count > keep {
            guard let idx = oldestIdleIndex() else { return }
            let s = sessions[idx]
            log("Memory pressure eviction: session \(idx) (\(s.tokenIds.count) tokens)")
            sessions.remove(at: idx)
            metrics.evictions += 1
        }
    }

    /// Flush idle sessions (e.g., on critical memory pressure). In-use sessions
    /// stay — killing them mid-generate would mutate KV that a live task is
    /// iterating. Once those tasks finish (markIdle), a subsequent flush clears them.
    func flush() {
        let before = sessions.count
        sessions.removeAll(where: { !inUse.contains($0.id) })
        let freed = before - sessions.count
        metrics.evictions += freed
        log("Flushed \(freed) idle cached sessions (\(sessions.count) in-use retained)")
    }

    private func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let maxLen = min(a.count, b.count)
        for i in 0..<maxLen {
            if a[i] != b[i] { return i }
        }
        return maxLen
    }
}

enum CacheStatus {
    case hit(prefixReused: Int, totalTokens: Int, newTokens: Int)
    case miss(totalTokens: Int, sessionsCount: Int)
    case trimFailed

    var logString: String {
        switch self {
        case .hit(let prefix, let total, let new):
            return "cache=hit prefix=\(prefix)/\(total) new=\(new)"
        case .miss(let total, let sessions):
            return "cache=miss tokens=\(total) sessions=\(sessions)"
        case .trimFailed:
            return "cache=trim_failed"
        }
    }
}

// MARK: - Parallel Inference Slots (llama-server style)

enum SlotState: String {
    case idle
    case prefilling
    case generating
}

/// A single inference slot that tracks state for one concurrent request.
/// Each slot gets its own KV cache, enabling fast context switching without
/// cache eviction. The slot pool limits maximum concurrent requests.
final class ServerSlot: @unchecked Sendable {
    let id: Int
    var state: SlotState = .idle
    var requestId: String = ""
    var startTime: CFAbsoluteTime = 0
    var promptTokenCount: Int = 0
    var generationTokenCount: Int = 0

    init(id: Int) {
        self.id = id
    }

    func reset() {
        state = .idle
        requestId = ""
        startTime = 0
        promptTokenCount = 0
        generationTokenCount = 0
    }
}

/// Manages the pool of inference slots. Requests acquire a slot before starting
/// generation and release it when done. If all slots are busy, new requests
/// queue until a slot frees up. This limits concurrent model access to N slots.
///
/// The actual generation still uses the existing `generate()` AsyncStream per slot.
/// True token-level round-robin would require model-level batching changes --
/// instead, we get slot-level concurrency with the GPU command queue serializing
/// the actual compute. Each active slot streams at ~1/N aggregate throughput.
actor SlotManager {
    let slots: [ServerSlot]
    let slotCount: Int
    private var slotWaiters: [CheckedContinuation<ServerSlot, Never>] = []
    // Prefill semaphore: only one slot prefills at a time to avoid GPU contention
    private var prefillBusy = false
    private var prefillWaiters: [CheckedContinuation<Void, Never>] = []

    init(slotCount: Int) {
        self.slotCount = slotCount
        var s: [ServerSlot] = []
        for i in 0..<slotCount {
            s.append(ServerSlot(id: i))
        }
        self.slots = s
    }

    /// Acquire exclusive prefill access. Only one slot prefills at a time.
    func acquirePrefill() async {
        if !prefillBusy {
            prefillBusy = true
            return
        }
        await withCheckedContinuation { cont in
            prefillWaiters.append(cont)
        }
    }

    /// Release prefill access, wake next waiter.
    func releasePrefill() {
        if let next = prefillWaiters.first {
            prefillWaiters.removeFirst()
            next.resume()
        } else {
            prefillBusy = false
        }
    }

    /// Acquire an idle slot. If all slots are busy, the caller suspends until
    /// one becomes available (FIFO queue).
    func acquireSlot() async -> ServerSlot {
        if let slot = slots.first(where: { $0.state == .idle }) {
            slot.state = .prefilling
            return slot
        }
        // All slots busy -- queue the caller
        return await withCheckedContinuation { cont in
            slotWaiters.append(cont)
        }
    }

    /// Try to acquire a slot without waiting. Returns nil if all busy (for 503 responses).
    func tryAcquireSlot() -> ServerSlot? {
        if let slot = slots.first(where: { $0.state == .idle }) {
            slot.state = .prefilling
            return slot
        }
        return nil
    }

    /// Release a slot back to idle and wake the next queued waiter, if any.
    func releaseSlot(_ slot: ServerSlot) {
        slot.reset()
        if let waiter = slotWaiters.first {
            slotWaiters.removeFirst()
            slot.state = .prefilling
            waiter.resume(returning: slot)
        }
    }

    /// Get a snapshot of all slot states for the /slots endpoint.
    func slotStatus() -> [(id: Int, state: String, requestId: String, promptTokens: Int, genTokens: Int, elapsed: Double)] {
        slots.map { slot in
            let elapsed = slot.state == .idle ? 0 : CFAbsoluteTimeGetCurrent() - slot.startTime
            return (id: slot.id, state: slot.state.rawValue, requestId: slot.requestId,
                    promptTokens: slot.promptTokenCount, genTokens: slot.generationTokenCount,
                    elapsed: elapsed)
        }
    }

    /// Count of active (non-idle) slots.
    func activeSlotCount() -> Int {
        slots.filter { $0.state != .idle }.count
    }

    /// Number of requests waiting in the queue.
    func queueDepth() -> Int {
        slotWaiters.count
    }
}

final class SimpleHTTPServer {
    let port: UInt16
    let container: ModelContainer
    let modelId: String
    let promptCache: ServerPromptCache
    let slotManager: SlotManager
    let slotCount: Int
    private var serverSocket: Int32 = -1
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    static let maxBodySize = 10 * 1024 * 1024

    /// Compute max sessions based on available physical memory.
    /// Conservative: assume ~1.5GB per cached session (14K tokens FP16 KV for 30B MoE).
    static func autoMaxSessions() -> Int {
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        // Reserve 20GB for model + OS, then ~1.5GB per session
        let available = max(totalGB - 20, 2)
        let sessions = Int(available / 1.5)
        return max(2, min(sessions, 10))  // clamp 2-10
    }

    init(port: UInt16, container: ModelContainer, modelId: String, slotCount: Int = 4, kvScheme: String? = nil) {
        self.port = port
        self.container = container
        self.modelId = modelId
        self.slotCount = slotCount
        let maxSess = SimpleHTTPServer.autoMaxSessions()
        self.promptCache = ServerPromptCache(maxSessions: maxSess, kvScheme: kvScheme)
        self.slotManager = SlotManager(slotCount: slotCount)
        log("Auto-configured: \(maxSess) max cached sessions (\(ProcessInfo.processInfo.physicalMemory / (1024*1024*1024))GB RAM)")
        log("Parallel inference slots: \(slotCount)")
        if let kv = kvScheme { log("KV cache scheme: \(kv)") }
    }

    func start() throws {
        serverSocket = socket(AF_INET, SOCK_STREAM, 0)
        guard serverSocket >= 0 else { throw ServerError.socketCreation }

        var opt: Int32 = 1
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = INADDR_ANY

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(serverSocket, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bindResult == 0 else { throw ServerError.bind(port) }

        guard listen(serverSocket, 64) == 0 else { throw ServerError.listen }

        log("Listening on http://127.0.0.1:\(port)")
        log("Endpoints: GET /v1/models, POST /v1/chat/completions, GET /tokenizer_info, POST /tokenize, GET /metrics, GET /slots")

        // Monitor macOS memory pressure — evict idle sessions under pressure
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: .global())
        source.setEventHandler { [promptCache] in
            Task {
                let event = source.data
                if event.contains(.critical) {
                    log("MEMORY PRESSURE: critical — flushing all cached sessions")
                    await promptCache.flush()
                } else if event.contains(.warning) {
                    log("MEMORY PRESSURE: warning — evicting idle sessions")
                    await promptCache.evictIdle(keep: 1)
                }
            }
        }
        source.resume()
        memoryPressureSource = source

        while true {
            let client = accept(serverSocket, nil, nil)
            if client < 0 {
                let err = errno
                if err == EINTR { continue }
                // EMFILE / ENFILE / ENOBUFS etc. — don't spin at 100% CPU while
                // the system recovers. Brief sleep lets some fds free up.
                log("accept failed: errno=\(err)")
                usleep(10_000)
                continue
            }
            configureClientSocket(client)
            Task { await handleClient(client) }
        }
    }

    func handleClient(_ fd: Int32) async {
        defer { close(fd) }

        let requestStart = CFAbsoluteTimeGetCurrent()
        var method = "?"
        var path = "?"
        var originHeader: String? = nil

        do {
            // Read headers first
            var headerData = Data()
            var byte: UInt8 = 0
            while headerData.count < 65536 {
                let n = read(fd, &byte, 1)
                if n == 1 {
                    headerData.append(byte)
                    // Detect end of headers: \r\n\r\n
                    if headerData.count >= 4 &&
                       headerData[headerData.count-4] == 0x0D && headerData[headerData.count-3] == 0x0A &&
                       headerData[headerData.count-2] == 0x0D && headerData[headerData.count-1] == 0x0A {
                        break
                    }
                } else if n < 0 && errno == EINTR {
                    continue
                } else {
                    return  // EOF or unrecoverable error
                }
            }

            let headerStr = String(data: headerData, encoding: .utf8) ?? ""
            let lines = headerStr.split(separator: "\r\n", omittingEmptySubsequences: false)
            guard let firstLine = lines.first else { return }

            let parts = firstLine.split(separator: " ")
            guard parts.count >= 2 else { return }
            method = String(parts[0])
            path = String(parts[1]).split(separator: "?").first.map(String.init) ?? String(parts[1])

            // Parse Content-Length, Origin header, and read body
            var contentLength = 0
            for line in lines {
                let lower = line.lowercased()
                if lower.hasPrefix("content-length:") {
                    contentLength = Int(lower.dropFirst(15).trimmingCharacters(in: .whitespaces)) ?? 0
                }
                if lower.hasPrefix("origin:") {
                    originHeader = String(line.dropFirst(7)).trimmingCharacters(in: .whitespaces)
                }
            }
            let corsOrigin = originHeader ?? "*"

            // Enforce max body size (10MB)
            if contentLength > SimpleHTTPServer.maxBodySize {
                log("\(method) \(path) — body too large (\(contentLength) bytes)")
                sendResponse(fd: fd, status: 413,
                           body: "{\"error\":{\"message\":\"request body too large (max \(SimpleHTTPServer.maxBodySize / 1024 / 1024)MB)\",\"type\":\"invalid_request_error\",\"code\":413}}",
                           contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
                return
            }

            var bodyData = Data()
            if contentLength > 0 {
                bodyData.reserveCapacity(contentLength)
                var remaining = contentLength
                var buf = [UInt8](repeating: 0, count: min(remaining, 65536))
                while remaining > 0 {
                    let toRead = min(remaining, buf.count)
                    let n = read(fd, &buf, toRead)
                    if n > 0 {
                        bodyData.append(contentsOf: buf[0..<n])
                        remaining -= n
                    } else if n < 0 && errno == EINTR {
                        continue
                    } else {
                        // Short read on a Content-Length-declared body is a hard error.
                        // Handing a truncated JSON to the decoder would produce a
                        // confusing 400; return a clearer 400 now.
                        sendResponse(fd: fd, status: 400,
                                   body: "{\"error\":{\"message\":\"truncated request body (\(contentLength - remaining)/\(contentLength) bytes)\",\"type\":\"invalid_request_error\",\"code\":400}}",
                                   contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
                        return
                    }
                }
            }
            let bodyStr = String(data: bodyData, encoding: .utf8) ?? ""

            log("\(method) \(path) (\(bodyStr.count) bytes)")

            switch (method, path) {
            case ("GET", "/v1/models"):
                let models = "{\"object\":\"list\",\"data\":[{\"id\":\"\(modelId)\",\"object\":\"model\",\"created\":\(Int(Date().timeIntervalSince1970)),\"owned_by\":\"local\",\"meta\":{\"n_ctx_train\":131072}}]}"
                sendResponse(fd: fd, status: 200, body: models, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)

            case ("POST", "/v1/chat/completions"):
                guard let data = bodyStr.data(using: .utf8),
                      let request = try? JSONDecoder().decode(ChatRequest.self, from: data) else {
                    sendResponse(fd: fd, status: 400,
                               body: "{\"error\":{\"message\":\"invalid request\",\"type\":\"invalid_request_error\",\"code\":400}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
                    return
                }
                await handleChat(fd: fd, request: request, corsOrigin: corsOrigin)

            case ("GET", "/tokenizer_info"), ("GET", "/v1/tokenizer_info"):
                let ctx = await container.perform { ctx in ctx }
                let eos = ctx.tokenizer.eosToken ?? ""
                let bos = ctx.tokenizer.bosToken ?? ""
                let eosId = ctx.tokenizer.eosTokenId ?? -1
                let bosId = ctx.tokenizer.bosTokenId ?? -1
                let info = "{\"eos_token\":\"\(eos)\",\"bos_token\":\"\(bos)\",\"eos_token_id\":\(eosId),\"bos_token_id\":\(bosId),\"model\":\"\(modelId)\"}"
                sendResponse(fd: fd, status: 200, body: info, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)

            case ("POST", "/tokenize"), ("POST", "/v1/tokenize"):
                guard let data = bodyStr.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let prompt = json["prompt"] as? String else {
                    sendResponse(fd: fd, status: 400, body: "{\"error\":{\"message\":\"missing prompt\",\"type\":\"invalid_request_error\",\"code\":400}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
                    return
                }
                let addSpecial = json["add_special_tokens"] as? Bool ?? true
                let ctx = await container.perform { ctx in ctx }
                let tokens = ctx.tokenizer.encode(text: prompt, addSpecialTokens: addSpecial)
                let tokensJson = "[\(tokens.map { String($0) }.joined(separator: ","))]"
                sendResponse(fd: fd, status: 200, body: "{\"tokens\":\(tokensJson)}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)

            case ("POST", "/v1/completions"):
                guard let data = bodyStr.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let prompt = json["prompt"] as? String else {
                    sendResponse(fd: fd, status: 400, body: "{\"error\":{\"message\":\"invalid request\",\"type\":\"invalid_request_error\",\"code\":400}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
                    return
                }
                let maxTokens = json["max_tokens"] as? Int ?? 256
                let temperature = json["temperature"] as? Double ?? 0.0
                let isStream = json["stream"] as? Bool ?? false
                await handleCompletions(fd: fd, prompt: prompt, maxTokens: maxTokens,
                                       temperature: Float(temperature), stream: isStream, corsOrigin: corsOrigin)

            case ("GET", "/health"), ("GET", "/"):
                sendResponse(fd: fd, status: 200, body: "{\"status\":\"ok\"}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)

            case ("GET", "/metrics"):
                let m = await promptCache.getMetrics()
                let sc = await promptCache.getSessionCount()
                let activeSlots = await slotManager.activeSlotCount()
                let queueDepth = await slotManager.queueDepth()
                let body = """
                {"cache":{"requests":\(m.totalRequests),"hits":\(m.cacheHits),"misses":\(m.cacheMisses),"hit_rate":\(String(format:"%.3f",m.hitRate)),"trim_failures":\(m.trimFailures),"evictions":\(m.evictions),"sessions_active":\(sc),"sessions_max":\(await promptCache.maxSessions)},"throughput":{"total_prefill_tokens":\(m.totalPrefillTokens),"total_reused_tokens":\(m.totalReusedTokens),"total_decode_tokens":\(m.totalDecodeTokens),"avg_prefill_tokens_per_request":\(String(format:"%.0f",m.avgPrefillTokens)),"avg_prefill_ms":\(String(format:"%.1f",m.avgPrefillMs)),"avg_decode_tok_per_sec":\(String(format:"%.1f",m.avgDecodeTokensPerSec))},"slots":{"total":\(slotCount),"active":\(activeSlots),"queue_depth":\(queueDepth)}}
                """
                sendResponse(fd: fd, status: 200, body: body.trimmingCharacters(in: .whitespacesAndNewlines), contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)

            case ("GET", "/slots"):
                let status = await slotManager.slotStatus()
                let slotsJSON = status.map { s in
                    "{\"id\":\(s.id),\"state\":\"\(s.state)\",\"request_id\":\"\(s.requestId)\",\"prompt_tokens\":\(s.promptTokens),\"generation_tokens\":\(s.genTokens),\"elapsed_ms\":\(String(format:"%.0f",s.elapsed * 1000))}"
                }.joined(separator: ",")
                let qd = await slotManager.queueDepth()
                let body = "{\"slots\":[\(slotsJSON)],\"queue_depth\":\(qd)}"
                sendResponse(fd: fd, status: 200, body: body, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)

            case ("OPTIONS", _):
                // CORS preflight
                sendResponse(fd: fd, status: 204, body: "", contentType: "text/plain",
                            extraHeaders: "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n", corsOrigin: corsOrigin)

            default:
                sendResponse(fd: fd, status: 404, body: "{\"error\":{\"message\":\"not found\",\"type\":\"not_found_error\",\"code\":404}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
            }
        } catch {
            log("ERROR handling \(method) \(path): \(error)")
            let msg = jsonEscape(error.localizedDescription)
            sendResponse(fd: fd, status: 500,
                       body: "{\"error\":{\"message\":\"\(msg)\",\"type\":\"server_error\",\"code\":500}}",
                       contentType: "application/json; charset=utf-8", corsOrigin: originHeader ?? "*")
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - requestStart) * 1000
        log("\(method) \(path) completed in \(String(format: "%.0f", elapsed))ms")
    }

    func handleChat(fd: Int32, request: ChatRequest, corsOrigin: String = "*") async {
        let isStreaming = request.stream ?? false
        let requestId = "chatcmpl-\(UUID().uuidString.prefix(8))"

        // Acquire an inference slot -- queues if all slots are busy (FIFO)
        let slot = await slotManager.acquireSlot()
        slot.requestId = requestId
        slot.startTime = CFAbsoluteTimeGetCurrent()
        log("slot[\(slot.id)] acquired for \(requestId) (active: \(await slotManager.activeSlotCount())/\(slotCount))")

        // Serialize prefill: only one request prefills at a time to avoid GPU contention.
        // Acquired here (before any model access), released on first generated token.
        await slotManager.acquirePrefill()
        var prefillReleased = false
        func releasePrefillOnce() async {
            if !prefillReleased {
                prefillReleased = true
                await slotManager.releasePrefill()
            }
        }

        // Session ID becomes known after fetch(); track in-use so concurrent
        // eviction/flush can't pull the cache out from under the generate loop.
        var markedSessionId: UUID? = nil
        // Keepalive timer (streaming only). Must be nil-safe at every cleanup point.
        var keepaliveTimer: DispatchSourceTimer? = nil

        // All resources acquired above MUST be released before return, not via
        // `defer { Task { await ... } }` — a detached Task leaks the release
        // past the function return so the next `acquireSlot()` sees the slot
        // still in use. Cleanup is awaited inline in all exit paths.

        do {
            slot.state = .prefilling
            var ctx = await container.perform { ctx in ctx }
            // Convert messages to tokenizer format.
            // For models that don't support "tool" role (e.g., MiniMax),
            // convert tool results to user messages and assistant tool_calls to
            // assistant messages with the call info as text.
            var messages: [[String: String]] = []
            for msg in request.messages {
                if msg.role == "tool" {
                    // Convert tool result to user message
                    let toolContent = msg.content ?? ""
                    messages.append(["role": "user", "content": "[Tool Result]: \(toolContent)"])
                } else {
                    var d: [String: String] = ["role": msg.role]
                    if let c = msg.content { d["content"] = c }
                    messages.append(d)
                }
            }
            // Pass tools if present — the chat template injects tool definitions
            let toolsAny = request.tools?.value as? [Any]
            let tokens: [Int]
            if let toolsAny = toolsAny {
                let toolSpecs: [[String: any Sendable]] = toolsAny.compactMap { $0 as? [String: Any] }.map { tool in
                    // Deep convert to [String: any Sendable]
                    func convert(_ v: Any) -> any Sendable {
                        if let s = v as? String { return s }
                        if let n = v as? Int { return n }
                        if let b = v as? Bool { return b }
                        if let d = v as? Double { return d }
                        if let arr = v as? [Any] { return arr.map { convert($0) } as [any Sendable] }
                        if let dict = v as? [String: Any] {
                            return dict.mapValues { convert($0) } as [String: any Sendable]
                        }
                        return String(describing: v)
                    }
                    return tool.mapValues { convert($0) } as [String: any Sendable]
                }
                // Pass preserve_thinking=true so the Qwen3.6 template emits
                // deterministic <think> blocks for historical assistant turns,
                // avoiding prompt-prefix drift and cache invalidation on tool-heavy
                // agentic workloads. See:
                // https://www.reddit.com/r/LocalLLaMA/comments/1sg076h/
                let templateCtx: [String: any Sendable] = ["preserve_thinking": true]
                // Try with tools first; fall back to without if template doesn't support them
                do {
                    tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs, additionalContext: templateCtx)
                } catch {
                    log("Chat template with tools failed (\(error)), retrying without tools")
                    // Inject tool descriptions into system prompt instead
                    var toolDesc = "Available tools:\n"
                    for tool in toolSpecs {
                        if let fn = tool["function"] as? [String: Any],
                           let name = fn["name"] as? String {
                            let desc = fn["description"] as? String ?? ""
                            toolDesc += "- \(name): \(desc)\n"
                        }
                    }
                    if var sys = messages.first, sys["role"] == "system" {
                        sys["content"] = (sys["content"] ?? "") + "\n\n" + toolDesc
                        var adjusted = messages
                        adjusted[0] = sys
                        tokens = try ctx.tokenizer.applyChatTemplate(messages: adjusted, tools: nil, additionalContext: templateCtx)
                    } else {
                        var adjusted = messages
                        adjusted.insert(["role": "system", "content": toolDesc], at: 0)
                        tokens = try ctx.tokenizer.applyChatTemplate(messages: adjusted, tools: nil, additionalContext: templateCtx)
                    }
                }
            } else {
                // Qwen 3.6: let the chat template's TAG_think prompt handle thinking mode.
                // Passing enable_thinking=true causes double-enable conflict → HTTP 400.
                // preserve_thinking=true stabilizes history rendering across turns for cache reuse.
                let templateCtx: [String: any Sendable] = ["preserve_thinking": true]
                tokens = try ctx.tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: templateCtx)
            }
            // If chat template injected <think> as assistant prefix, model outputs thinking
            // content directly (no opening <think> tag in response, only closing </think>).
            // Decode last 8 tokens to text — reliable regardless of tokenizer vocab layout.
            let lastTokensText = ctx.tokenizer.decode(tokens: Array(tokens.suffix(8)))
            let promptPrefillsThink = lastTokensText.contains("<think>")
            log("  think detection: lastTokensText=\(lastTokensText.debugDescription) prefillsThink=\(promptPrefillsThink)")
            // Prompt caching: reuse KV state from previous requests
            let prefillStart = CFAbsoluteTimeGetCurrent()
            let (reusedCache, fetchedNewTokens, cacheStatus, sessionId) = await promptCache.fetch(tokens: tokens, model: ctx.model)
            await promptCache.markInUse(sessionId)
            markedSessionId = sessionId

            // If the cache fully matched (newTokens empty), trim one slot so the
            // model has a seed token to decode from. Without this we'd pass the
            // entire prompt to generate(), defeating the cache hit entirely.
            var newTokens = fetchedNewTokens
            if newTokens.isEmpty, let last = tokens.last {
                for c in reusedCache { _ = c.trim(1) }
                newTokens = [last]
            }
            let tokenArray = MLXArray(newTokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: request.temperature ?? 0.6)
            if let maxTokens = request.max_tokens {
                params.maxTokens = maxTokens
            }

            // Tool-call format is applied per-request via the ctx copy; generate()
            // uses `context: ctx` so the mutation only affects this request's
            // inference path, not a global shared configuration.
            if toolsAny != nil {
                ctx.configuration.toolCallFormat = .xmlFunction
            }

            slot.promptTokenCount = tokens.count
            slot.state = .generating
            log("slot[\(slot.id)] \(cacheStatus.logString) prefill=\(newTokens.count) stream=\(isStreaming) tools=\(toolsAny?.count ?? 0) think_prefilled=\(promptPrefillsThink)")

            if isStreaming {
                // SSE response headers
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: \(corsOrigin)\r\nAccess-Control-Allow-Credentials: true\r\n\r\n"
                if !writeAll(fd, header) {
                    // Client already gone — bail out, cleanup will run below.
                    throw ServerError.clientDisconnected
                }

                // Keepalive: send an SSE heartbeat every 2s while prefill runs so
                // the client's read timer stays alive. Cancelled as soon as the
                // first real token arrives, and re-cancelled again in cleanup.
                let timer = DispatchSource.makeTimerSource(queue: .global())
                let keepaliveFd = fd
                timer.schedule(deadline: .now() + 2, repeating: 2.0)
                timer.setEventHandler {
                    let chunk = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}]}\n\n"
                    writeAll(keepaliveFd, chunk)
                }
                timer.resume()
                keepaliveTimer = timer
            }

            if isStreaming {
                let reqModel = request.model
                var hadToolCall = false
                // Role chunk announces the assistant role immediately (OpenAI spec).
                writeSSE(fd: fd, requestId: requestId, role: "assistant", content: nil, finishReason: nil, requestModel: reqModel, includeNullContent: true)

                // Streaming state, tracked with String.Index for O(1) advancement.
                // `fullText` is append-only (never mutated) so indices stay valid.
                // `unemittedIdx` is where the next not-yet-emitted text starts.
                // `thinkEmitIdx` tracks emitted reasoning within the think block.
                var fullText = ""
                var unemittedIdx = fullText.startIndex
                var thinkStartIdx = fullText.startIndex  // start of think content (after <think>)
                var thinkEmitIdx = fullText.startIndex   // how far we've emitted reasoning
                var inThinkBlock = promptPrefillsThink
                var thinkBlockResolved = false  // true once we've seen </think> and switched to content

                // Tag prefixes that require holdback while the rest of the tag arrives.
                let tagPrefixes = ["<tool_call", "<function=", "<minimax:", "<invoke", "</think", "<think", "<parameter"]

                func unemitted() -> Substring { fullText[unemittedIdx...] }

                do {
                    var tokenCount = 0
                    for try await generation in try generate(
                        input: input, cache: reusedCache, parameters: params, context: ctx
                    ) {
                        tokenCount += 1

                        switch generation {
                        case .chunk(let text):
                            fullText += text
                            // Prefill is done once a real text token arrives (not .info).
                            // Release the GPU prefill lock and cancel the keepalive timer.
                            if !prefillReleased {
                                keepaliveTimer?.cancel(); keepaliveTimer = nil
                                await releasePrefillOnce()
                                log("  first token arrived, prefill done")
                            }
                            if tokenCount <= 3 || tokenCount % 50 == 0 {
                                log("  chunk[\(tokenCount)]: +\(text.count)ch fullText=\(fullText.count)ch think=\(inThinkBlock)")
                            }

                            // -------- Think-block handling --------
                            // Detect opening <think> at the very start of output, if
                            // the prompt template didn't already prefill it.
                            if !inThinkBlock && !thinkBlockResolved {
                                let lead = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
                                if lead.hasPrefix("<think") || lead.hasPrefix("</think") {
                                    inThinkBlock = true
                                    if let open = fullText.range(of: "<think>") {
                                        thinkStartIdx = open.upperBound
                                        thinkEmitIdx = thinkStartIdx
                                    } else {
                                        thinkStartIdx = fullText.startIndex
                                        thinkEmitIdx = fullText.startIndex
                                    }
                                }
                            }
                            if inThinkBlock {
                                if let close = fullText.range(of: "</think>") {
                                    // End of think block: flush any remaining reasoning, then advance
                                    // to post-think content.
                                    let thinkEnd = close.lowerBound
                                    if thinkEmitIdx < thinkEnd {
                                        let reasoning = String(fullText[thinkEmitIdx..<thinkEnd])
                                        if !reasoning.isEmpty {
                                            writeSSE(fd: fd, requestId: requestId, role: nil, content: nil, finishReason: nil, reasoningContent: reasoning, requestModel: reqModel)
                                        }
                                    }
                                    thinkEmitIdx = thinkEnd
                                    inThinkBlock = false
                                    thinkBlockResolved = true
                                    unemittedIdx = close.upperBound
                                    log("  think block ended, fullText=\(fullText.count)ch")
                                } else {
                                    // Still in think: stream new reasoning, do nothing else.
                                    if thinkEmitIdx < fullText.endIndex {
                                        let reasoning = String(fullText[thinkEmitIdx...])
                                        if !reasoning.isEmpty {
                                            writeSSE(fd: fd, requestId: requestId, role: nil, content: nil, finishReason: nil, reasoningContent: reasoning, requestModel: reqModel)
                                            thinkEmitIdx = fullText.endIndex
                                        }
                                    }
                                    continue
                                }
                            }

                            // Skip leading whitespace once content begins, to avoid
                            // a lone "\n" from the template bleeding into output.
                            if unemittedIdx < fullText.endIndex {
                                let u = unemitted()
                                if let firstNonWS = u.firstIndex(where: { !$0.isWhitespace }) {
                                    if firstNonWS != u.startIndex {
                                        unemittedIdx = firstNonWS
                                    }
                                } else {
                                    // All whitespace so far — wait for more
                                    continue
                                }
                            }

                            // After a tool call has been emitted, swallow any trailing XML
                            // closers so they don't appear as content.
                            if hadToolCall {
                                let u = unemitted()
                                let stripped = String(u)
                                    .replacingOccurrences(of: "</minimax:tool_call>", with: "")
                                    .replacingOccurrences(of: "</invoke>", with: "")
                                    .replacingOccurrences(of: "</tool_call>", with: "")
                                    .trimmingCharacters(in: .whitespacesAndNewlines)
                                if stripped.isEmpty {
                                    unemittedIdx = fullText.endIndex
                                    continue
                                }
                            }

                            // Tool-call detection on unemitted tail.
                            let u = unemitted()
                            let containsToolOpen = u.contains("<tool_call>") || u.contains("<function=") || u.contains("<minimax:tool_call>") || u.contains("<invoke name=")
                            if containsToolOpen {
                                let containsToolClose = u.contains("</tool_call>") || u.contains("</function>") || u.contains("</minimax:tool_call>") || u.contains("</invoke>")
                                if containsToolClose {
                                    let snippet = String(u)
                                    let tcs = parseAllToolCalls(snippet)
                                    if !tcs.isEmpty {
                                        hadToolCall = true
                                        for tc in tcs {
                                            emitToolCallSSE(fd: fd, requestId: requestId, name: tc.name, arguments: tc.arguments, requestModel: reqModel)
                                        }
                                    } else {
                                        writeSSE(fd: fd, requestId: requestId, role: nil, content: snippet, finishReason: nil, requestModel: reqModel)
                                    }
                                    unemittedIdx = fullText.endIndex
                                }
                                // incomplete — wait for </...>
                                continue
                            }

                            // Hold back suspicious-looking tag prefixes in case the
                            // next token completes a real tag (e.g. "<too" then "l_call>").
                            // Only hold if the partial looks like the start of a known tag.
                            if let lt = u.lastIndex(of: "<") {
                                let tail = u[lt...]
                                // A known tag opener is either fully present or a prefix of one.
                                let looksLikeTag = tagPrefixes.contains(where: {
                                    tail.hasPrefix($0.prefix(min($0.count, tail.count))) || $0.hasPrefix(tail)
                                })
                                if looksLikeTag {
                                    // Emit everything before the "<", hold the rest.
                                    if lt != u.startIndex {
                                        let safe = String(u[..<lt])
                                        if !safe.isEmpty {
                                            writeSSE(fd: fd, requestId: requestId, role: nil, content: safe, finishReason: nil, requestModel: reqModel)
                                            unemittedIdx = lt
                                        }
                                    }
                                    continue
                                }
                            }

                            // Nothing suspicious — emit the whole unemitted tail.
                            if !u.isEmpty {
                                writeSSE(fd: fd, requestId: requestId, role: nil, content: String(u), finishReason: nil, requestModel: reqModel)
                                unemittedIdx = fullText.endIndex
                            }

                        case .toolCall(let tc):
                            hadToolCall = true
                            // Keepalive cancel here too in case .toolCall arrives before any .chunk.
                            if !prefillReleased {
                                keepaliveTimer?.cancel(); keepaliveTimer = nil
                                await releasePrefillOnce()
                            }
                            let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                            let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict)) ?? Data()
                            let argsRaw = String(data: argsJSON, encoding: .utf8) ?? "{}"
                            emitToolCallSSE(fd: fd, requestId: requestId, name: tc.function.name, arguments: argsRaw, requestModel: reqModel)

                        case .info(let info):
                            log("  .info: tokens=\(tokenCount) fullText=\(fullText.count)ch hadToolCall=\(hadToolCall) think=\(inThinkBlock)")
                            if fullText.count > 0 { log("  fullText preview: \(String(fullText.prefix(400)))") }

                            // Flush any remaining buffered text not yet emitted.
                            if unemittedIdx < fullText.endIndex {
                                let remaining = String(fullText[unemittedIdx...])
                                let remainTCs = parseAllToolCalls(remaining)
                                if !remainTCs.isEmpty {
                                    hadToolCall = true
                                    for tc in remainTCs {
                                        emitToolCallSSE(fd: fd, requestId: requestId, name: tc.name, arguments: tc.arguments, requestModel: reqModel)
                                    }
                                } else if !remaining.isEmpty {
                                    writeSSE(fd: fd, requestId: requestId, role: nil, content: remaining, finishReason: nil, requestModel: reqModel)
                                }
                                unemittedIdx = fullText.endIndex
                            }

                            // Finish reason: tool_calls if any tool was emitted, else
                            // "length" only when the user capped max_tokens AND we hit it.
                            let maxTok = request.max_tokens ?? Int.max
                            let fr: String
                            if hadToolCall {
                                fr = "tool_calls"
                            } else if request.max_tokens != nil && info.generationTokenCount >= maxTok {
                                fr = "length"
                            } else {
                                fr = "stop"
                            }
                            let responseModel = reqModel ?? modelId
                            let promptTokens = tokens.count
                            let usageJSON = ",\"usage\":{\"prompt_tokens\":\(promptTokens),\"completion_tokens\":\(info.generationTokenCount),\"total_tokens\":\(promptTokens + info.generationTokenCount)}"
                            let finalEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"\(fr)\"}]\(usageJSON)}\n\ndata: [DONE]\n\n"
                            writeAll(fd, finalEvent)
                        }
                    }
                } catch {
                    keepaliveTimer?.cancel(); keepaliveTimer = nil
                    await releasePrefillOnce()
                    log("ERROR during streaming generation: \(error)")
                    let errMsg = jsonEscape(error.localizedDescription)
                    let errEvent = "data: {\"error\":{\"message\":\"\(errMsg)\",\"type\":\"server_error\",\"code\":500}}\n\ndata: [DONE]\n\n"
                    writeAll(fd, errEvent)
                }
                // Record timing and save cache
                let elapsed = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
                await promptCache.recordTiming(prefillMs: 0, decodeMs: elapsed, decodeTokens: 0)
                await promptCache.save(sessionId: sessionId, promptTokens: tokens)
            } else {
                keepaliveTimer?.cancel(); keepaliveTimer = nil

                // Non-streaming: collect all text
                var fullText = ""
                var completionTokens = 0
                var toolCalls: [(name: String, args: String)] = []
                for try await generation in try generate(
                    input: input, cache: reusedCache, parameters: params, context: ctx
                ) {
                    if !prefillReleased { await releasePrefillOnce() }
                    switch generation {
                    case .chunk(let text):
                        fullText += text
                        completionTokens += 1
                    case .toolCall(let tc):
                        let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                        let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict)) ?? Data()
                        toolCalls.append((name: tc.function.name, args: String(data: argsJSON, encoding: .utf8) ?? "{}"))
                    default: break
                    }
                }

                // Strip <think>...</think>, preserving as reasoning_content.
                log("  non-streaming fullText (\(fullText.count)ch): \(String(fullText.prefix(400)))")
                var reasoningContent: String? = nil
                if let thinkEnd = fullText.range(of: "</think>") {
                    var thinkContent = String(fullText[..<thinkEnd.lowerBound])
                    if let tagEnd = thinkContent.range(of: "<think>") {
                        thinkContent = String(thinkContent[tagEnd.upperBound...])
                    }
                    reasoningContent = thinkContent.trimmingCharacters(in: .whitespacesAndNewlines)
                    fullText = String(fullText[thinkEnd.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                } else if fullText.hasPrefix("<think>") {
                    // Incomplete think block — strip it entirely, still save as reasoning
                    var thinkContent = fullText
                    if let tagEnd = thinkContent.range(of: "<think>") {
                        thinkContent = String(thinkContent[tagEnd.upperBound...])
                    }
                    reasoningContent = thinkContent.trimmingCharacters(in: .whitespacesAndNewlines)
                    if reasoningContent?.isEmpty == true { reasoningContent = nil }
                    fullText = ""
                }

                // Parse tool calls from accumulated text (XML formats that don't
                // flow through `.toolCall` events).
                if toolCalls.isEmpty {
                    let parsed = parseAllToolCalls(fullText)
                    if !parsed.isEmpty {
                        for tc in parsed {
                            toolCalls.append((name: tc.name, args: tc.arguments))
                        }
                        fullText = ""
                    }
                }

                let responseModel = request.model ?? modelId
                let maxTok = request.max_tokens ?? Int.max
                let finishReason: String
                if !toolCalls.isEmpty {
                    finishReason = "tool_calls"
                } else if request.max_tokens != nil && completionTokens >= maxTok {
                    finishReason = "length"
                } else {
                    finishReason = "stop"
                }

                var responseBody: String
                if toolCalls.isEmpty {
                    let response = ChatResponse(
                        id: requestId, object: "chat.completion",
                        created: Int(Date().timeIntervalSince1970), model: responseModel,
                        system_fingerprint: "mlx-swift-v1",
                        choices: [.init(index: 0, message: .init(role: "assistant", content: fullText, reasoning_content: reasoningContent),
                                       delta: nil, finish_reason: finishReason)],
                        usage: .init(prompt_tokens: tokens.count, completion_tokens: completionTokens,
                                    total_tokens: tokens.count + completionTokens))
                    responseBody = String(data: try JSONEncoder().encode(response), encoding: .utf8)!
                } else {
                    // OpenAI spec: tool_calls[].function.arguments must be a
                    // JSON-ENCODED STRING, not a raw object. We escape the
                    // JSON payload and wrap in quotes so both streaming and
                    // non-streaming responses have the same shape.
                    let tcJSON = toolCalls.map { tc in
                        let tcId = "call_\(UUID().uuidString.prefix(8).lowercased())"
                        let argsEscaped = jsonEscape(tc.args)
                        let nameEscaped = jsonEscape(tc.name)
                        return "{\"id\":\"\(tcId)\",\"type\":\"function\",\"function\":{\"name\":\"\(nameEscaped)\",\"arguments\":\"\(argsEscaped)\"}}"
                    }.joined(separator: ",")
                    let contentEscaped = jsonEscape(fullText)
                    let rcField: String
                    if let rc = reasoningContent {
                        rcField = ",\"reasoning_content\":\"\(jsonEscape(rc))\""
                    } else {
                        rcField = ""
                    }
                    responseBody = "{\"id\":\"\(requestId)\",\"object\":\"chat.completion\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\(contentEscaped)\"\(rcField),\"tool_calls\":[\(tcJSON)]},\"finish_reason\":\"\(finishReason)\"}],\"usage\":{\"prompt_tokens\":\(tokens.count),\"completion_tokens\":\(completionTokens),\"total_tokens\":\(tokens.count + completionTokens)}}"
                }
                sendResponse(fd: fd, status: 200, body: responseBody,
                           contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
                let elapsed = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000
                await promptCache.recordTiming(prefillMs: 0, decodeMs: elapsed, decodeTokens: 0)
                await promptCache.save(sessionId: sessionId, promptTokens: tokens)
            }
        } catch {
            log("ERROR in handleChat: \(error)")
            let errMsg = jsonEscape(error.localizedDescription)
            let err = "{\"error\":{\"message\":\"\(errMsg)\",\"type\":\"server_error\",\"code\":500}}"
            // If streaming headers were already sent, there's nothing safe to
            // do but close. sendResponse here is best-effort for the non-stream
            // and early-error paths.
            if !isStreaming {
                sendResponse(fd: fd, status: 500, body: err, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
            }
        }

        // ---------- Single-path cleanup ----------
        // Everything that was acquired is released here, awaited inline so the
        // next request sees correct state the moment handleChat returns.
        keepaliveTimer?.cancel()
        await releasePrefillOnce()
        if let sid = markedSessionId {
            await promptCache.markIdle(sid)
        }
        await slotManager.releaseSlot(slot)
    }

    func handleCompletions(fd: Int32, prompt: String, maxTokens: Int, temperature: Float, stream: Bool, corsOrigin: String = "*") async {
        let requestId = "cmpl-\(UUID().uuidString.prefix(8))"

        // Acquire an inference slot (shared pool with chat completions)
        let slot = await slotManager.acquireSlot()
        slot.requestId = requestId
        slot.startTime = CFAbsoluteTimeGetCurrent()
        slot.state = .generating
        log("slot[\(slot.id)] acquired for \(requestId) (completions)")

        do {
            let ctx = await container.perform { ctx in ctx }
            let tokens = ctx.tokenizer.encode(text: prompt)
            let tokenArray = MLXArray(tokens)
            let input = LMInput(text: LMInput.Text(tokens: tokenArray))

            var params = GenerateParameters(temperature: temperature)
            params.maxTokens = maxTokens

            log("completions: \(tokens.count) prompt tokens, max_tokens=\(maxTokens), stream=\(stream)")

            if stream {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: \(corsOrigin)\r\nAccess-Control-Allow-Credentials: true\r\n\r\n"
                writeAll(fd, header)

                let result = try await container.generate(input: input, parameters: params)
                var completionTokens = 0
                for try await event in result {
                    if let chunk = event.chunk {
                        completionTokens += 1
                        let escaped = jsonEscape(chunk)
                        let created = Int(Date().timeIntervalSince1970)
                        let sse = "data: {\"id\":\"\(requestId)\",\"object\":\"text_completion\",\"created\":\(created),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"text\":\"\(escaped)\",\"finish_reason\":null}]}\n\n"
                        writeAll(fd, sse)
                    }
                    if event.info != nil {
                        let created = Int(Date().timeIntervalSince1970)
                        let usage = "\"usage\":{\"prompt_tokens\":\(tokens.count),\"completion_tokens\":\(completionTokens),\"total_tokens\":\(tokens.count + completionTokens)}"
                        let sse = "data: {\"id\":\"\(requestId)\",\"object\":\"text_completion\",\"created\":\(created),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}],\(usage)}\n\ndata: [DONE]\n\n"
                        writeAll(fd, sse)
                    }
                }
            } else {
                var fullText = ""
                let result = try await container.generate(input: input, parameters: params)
                var usage: (prompt: Int, completion: Int) = (tokens.count, 0)
                for try await event in result {
                    if let chunk = event.chunk { fullText += chunk; usage.completion += 1 }
                }
                let escaped = jsonEscape(fullText)
                let body = "{\"id\":\"\(requestId)\",\"object\":\"text_completion\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(modelId)\",\"choices\":[{\"index\":0,\"text\":\"\(escaped)\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":\(usage.prompt),\"completion_tokens\":\(usage.completion),\"total_tokens\":\(usage.prompt + usage.completion)}}"
                sendResponse(fd: fd, status: 200, body: body, contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
            }
        } catch {
            let errMsg = jsonEscape(String(describing: error))
            sendResponse(fd: fd, status: 500, body: "{\"error\":{\"message\":\"\(errMsg)\",\"type\":\"server_error\",\"code\":500}}", contentType: "application/json; charset=utf-8", corsOrigin: corsOrigin)
        }

        // Release slot inline — never via detached Task, which would return before
        // release completes and the next acquireSlot() would see it still busy.
        await slotManager.releaseSlot(slot)
    }

    /// Parse XML tool call from text: <function=name><parameter=key>value</parameter></function>
    /// Handles both <tool_call>...<function=...>...</tool_call> and bare <function=...>
    /// Parse ALL tool calls from text. Returns array of (name, arguments).
    func parseAllToolCalls(_ text: String) -> [(name: String, arguments: String)] {
        // Try MiniMax format first (can have multiple <invoke> blocks)
        let minimax = parseAllMiniMaxToolCalls(text)
        if !minimax.isEmpty { return minimax }
        // Fall back to single generic parse
        if let tc = parseToolCallXML(text) { return [tc] }
        return []
    }

    func parseToolCallXML(_ text: String) -> (name: String, arguments: String)? {
        // Try MiniMax format: <minimax:tool_call><invoke name="..."><parameter name="...">value</parameter></invoke></minimax:tool_call>
        let all = parseAllMiniMaxToolCalls(text)
        if let first = all.first { return first }

        // Try generic format: <function=name><parameter=key>value</parameter></function>
        guard let funcStart = text.range(of: "<function=") else { return nil }
        guard let nameEnd = text.range(of: ">", range: funcStart.upperBound..<text.endIndex) else { return nil }

        let funcName = String(text[funcStart.upperBound..<nameEnd.lowerBound])

        // Extract parameters
        var args: [String: String] = [:]
        var search = nameEnd.upperBound
        while let paramStart = text.range(of: "<parameter=", range: search..<text.endIndex) {
            guard let pNameEnd = text.range(of: ">", range: paramStart.upperBound..<text.endIndex) else { break }
            let paramName = String(text[paramStart.upperBound..<pNameEnd.lowerBound])
            guard let paramEnd = text.range(of: "</parameter>", range: pNameEnd.upperBound..<text.endIndex) else { break }
            var value = String(text[pNameEnd.upperBound..<paramEnd.lowerBound])
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }
            args[paramName] = value
            search = paramEnd.upperBound
        }

        let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
        return (name: funcName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}")
    }

    /// Parse MiniMax-specific tool call XML format:
    /// <minimax:tool_call><invoke name="terminal"><parameter name="command">ls -la</parameter></invoke></minimax:tool_call>
    /// Parse ALL <invoke> blocks from a MiniMax tool call (supports multiple tools in one response)
    func parseAllMiniMaxToolCalls(_ text: String) -> [(name: String, arguments: String)] {
        guard text.contains("<invoke name=") || text.contains("<parameter name=") else { return [] }
        var results: [(name: String, arguments: String)] = []

        // Normal path: find <invoke name="...">...</invoke> blocks
        var searchStart = text.startIndex
        while let invokeStart = text.range(of: "<invoke name=\"", range: searchStart..<text.endIndex) {
            guard let invokeEnd = text.range(of: "</invoke>", range: invokeStart.upperBound..<text.endIndex) else { break }
            let invokeBlock = String(text[invokeStart.lowerBound..<invokeEnd.upperBound])
            if let tc = parseSingleMiniMaxInvoke(invokeBlock) {
                results.append(tc)
            }
            searchStart = invokeEnd.upperBound
        }

        // Fallback: malformed XML with <parameter> but no <invoke name=">
        // MiniMax sometimes omits <invoke name="..."> and outputs parameters directly
        if results.isEmpty && text.contains("<parameter name=") && !text.contains("<invoke name=") {
            // Extract parameters and infer tool name from parameter names
            var args: [String: String] = [:]
            var paramSearch = text.startIndex
            while let paramStart = text.range(of: "<parameter name=\"", range: paramSearch..<text.endIndex) {
                let afterParam = text[paramStart.upperBound...]
                guard let pNameEnd = afterParam.range(of: "\">") else { break }
                let paramName = String(afterParam[afterParam.startIndex..<pNameEnd.lowerBound])
                let valueStart = pNameEnd.upperBound
                guard let paramEnd = text.range(of: "</parameter>", range: valueStart..<text.endIndex) else { break }
                var value = String(text[valueStart..<paramEnd.lowerBound])
                if value.hasPrefix("\n") { value = String(value.dropFirst()) }
                if value.hasSuffix("\n") { value = String(value.dropLast()) }
                args[paramName] = value
                paramSearch = paramEnd.upperBound
            }
            if !args.isEmpty {
                // Infer tool name from parameter names
                let toolName: String
                if args["command"] != nil { toolName = "terminal" }
                else if args["content"] != nil && args["path"] != nil { toolName = "write_file" }
                else if args["path"] != nil { toolName = "read_file" }
                else if args["query"] != nil { toolName = "grep" }
                else { toolName = "terminal" }  // default fallback
                let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
                results.append((name: toolName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}"))
                log("  inferred tool '\(toolName)' from malformed MiniMax XML (no <invoke> tag)")
            }
        }

        return results
    }

    /// Parse a single <invoke name="...">...</invoke> block
    func parseSingleMiniMaxInvoke(_ text: String) -> (name: String, arguments: String)? {
        guard let invokeStart = text.range(of: "<invoke name=\"") else { return nil }
        let afterName = text[invokeStart.upperBound...]
        guard let nameEnd = afterName.range(of: "\"") else { return nil }
        let funcName = String(afterName[afterName.startIndex..<nameEnd.lowerBound])

        var args: [String: String] = [:]
        var search = nameEnd.upperBound
        while let paramStart = text.range(of: "<parameter name=\"", range: search..<text.endIndex) {
            let afterParam = text[paramStart.upperBound...]
            guard let pNameEnd = afterParam.range(of: "\">") else { break }
            let paramName = String(afterParam[afterParam.startIndex..<pNameEnd.lowerBound])
            let valueStart = pNameEnd.upperBound
            guard let paramEnd = text.range(of: "</parameter>", range: valueStart..<text.endIndex) else { break }
            var value = String(text[valueStart..<paramEnd.lowerBound])
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }
            args[paramName] = value
            search = paramEnd.upperBound
        }

        let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
        return (name: funcName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}")
    }

    func parseMiniMaxToolCall(_ text: String) -> (name: String, arguments: String)? {
        guard text.contains("<minimax:tool_call>") || text.contains("<invoke name=") else { return nil }

        // Extract function name from <invoke name="...">
        guard let invokeStart = text.range(of: "<invoke name=\"") else { return nil }
        let afterName = text[invokeStart.upperBound...]
        guard let nameEnd = afterName.range(of: "\"") else { return nil }
        let funcName = String(afterName[afterName.startIndex..<nameEnd.lowerBound])

        // Extract parameters: <parameter name="key">value</parameter>
        var args: [String: String] = [:]
        var search = nameEnd.upperBound
        while let paramStart = text.range(of: "<parameter name=\"", range: search..<text.endIndex) {
            let afterParam = text[paramStart.upperBound...]
            guard let pNameEnd = afterParam.range(of: "\">") else { break }
            let paramName = String(afterParam[afterParam.startIndex..<pNameEnd.lowerBound])
            let valueStart = pNameEnd.upperBound
            guard let paramEnd = text.range(of: "</parameter>", range: valueStart..<text.endIndex) else { break }
            var value = String(text[valueStart..<paramEnd.lowerBound])
            // Trim only one leading/trailing newline (preserve internal newlines in code)
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }
            args[paramName] = value
            search = paramEnd.upperBound
        }

        let argsJSON = (try? JSONSerialization.data(withJSONObject: args)) ?? Data()
        return (name: funcName, arguments: String(data: argsJSON, encoding: .utf8) ?? "{}")
    }

    /// Emit a tool call as an SSE event
    func emitToolCallSSE(fd: Int32, requestId: String, name: String, arguments: String, requestModel: String? = nil) {
        // OpenAI spec: tool_calls[].function.arguments is a JSON-ENCODED STRING,
        // so the raw JSON payload must be JSON-string-escaped before embedding.
        let argsEscaped = jsonEscape(arguments)
        let nameEscaped = jsonEscape(name)
        let tcId = "call_\(UUID().uuidString.prefix(8).lowercased())"
        let responseModel = requestModel ?? modelId
        let tcEvent = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"finish_reason\":null,\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"index\":0,\"id\":\"\(tcId)\",\"type\":\"function\",\"function\":{\"name\":\"\(nameEscaped)\",\"arguments\":\"\(argsEscaped)\"}}]}}]}\n\n"
        writeAll(fd, tcEvent)
    }

    func writeSSE(fd: Int32, requestId: String, role: String?, content: String?, finishReason: String?, reasoningContent: String? = nil, requestModel: String? = nil, includeNullContent: Bool = false) {
        var parts: [String] = []
        if let role = role { parts.append("\"role\":\"\(jsonEscape(role))\"") }
        if let content = content {
            parts.append("\"content\":\"\(jsonEscape(content))\"")
        } else if includeNullContent {
            parts.append("\"content\":null")
        }
        if let rc = reasoningContent { parts.append("\"reasoning_content\":\"\(jsonEscape(rc))\"") }
        let deltaJson = "{\(parts.joined(separator: ","))}"

        let fr = finishReason.map { "\"\(jsonEscape($0))\"" } ?? "null"
        let responseModel = requestModel ?? modelId
        let event = "data: {\"id\":\"\(requestId)\",\"object\":\"chat.completion.chunk\",\"created\":\(Int(Date().timeIntervalSince1970)),\"model\":\"\(responseModel)\",\"system_fingerprint\":\"mlx-swift-v1\",\"choices\":[{\"index\":0,\"delta\":\(deltaJson),\"finish_reason\":\(fr)}]}\n\n"
        writeAll(fd, event)
    }

    func sendResponse(fd: Int32, status: Int, body: String, contentType: String, extraHeaders: String = "", corsOrigin: String = "*") {
        let statusText: String
        switch status {
        case 200: statusText = "OK"
        case 204: statusText = "No Content"
        case 400: statusText = "Bad Request"
        case 413: statusText = "Payload Too Large"
        case 404: statusText = "Not Found"
        case 500: statusText = "Internal Server Error"
        case 503: statusText = "Service Unavailable"
        default: statusText = "Unknown"
        }
        // Connection: close — we close(fd) immediately after this write anyway,
        // and keepalive without a proper request loop leaves clients hanging.
        let headers = "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\nAccess-Control-Allow-Origin: \(corsOrigin)\r\nAccess-Control-Allow-Credentials: true\r\n\(extraHeaders)\r\n"
        if !writeAll(fd, headers) { return }
        if !body.isEmpty { writeAll(fd, body) }
    }

    enum ServerError: Error {
        case socketCreation, bind(UInt16), listen, clientDisconnected
    }
}

// MARK: - Entry Point

@main
struct MLXServerApp {
    @MainActor
    static func main() async throws {
        // Ignore SIGPIPE -- client disconnects during streaming should not crash the server
        signal(SIGPIPE, SIG_IGN)

        let args = CommandLine.arguments
        let model = args.firstIndex(of: "--model").flatMap { i in
            i + 1 < args.count ? args[i + 1] : nil
        } ?? "mlx-community/gemma-4-e2b-it-4bit"

        let port = args.firstIndex(of: "--port").flatMap { i in
            i + 1 < args.count ? UInt16(args[i + 1]) : nil
        } ?? 8080

        let slots = args.firstIndex(of: "--slots").flatMap { i in
            i + 1 < args.count ? Int(args[i + 1]) : nil
        } ?? 4  // Default to 4 parallel slots (like llama-server)

        let kvScheme = args.firstIndex(of: "--kv").flatMap { i in
            i + 1 < args.count ? args[i + 1] : nil
        }  // e.g. "turbo4v2", "turbo4", "affine4" — nil = FP16 default

        log("Loading model: \(model)")
        let config: ModelConfiguration
        if model.hasPrefix("/") || model.hasPrefix("~") || model.hasPrefix(".") {
            // Local path
            let expandedPath = NSString(string: model).expandingTildeInPath
            config = ModelConfiguration(directory: URL(fileURLWithPath: expandedPath))
        } else {
            // HuggingFace model ID
            config = ModelConfiguration(id: model)
        }
        // Try LLM first, fall back to VLM (for Qwen3-VL, Gemma4-VL, etc.)
        let container: ModelContainer
        do {
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { log("Model loaded (LLM)") }
            }
        } catch {
            log("LLM load failed (\(error)), trying VLM...")
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { log("Model loaded (VLM)") }
            }
        }

        let server = SimpleHTTPServer(port: port, container: container, modelId: model, slotCount: slots, kvScheme: kvScheme)
        try server.start()
    }
}
