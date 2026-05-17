import type {
    LlmMessage,
    NormalizedToolCall,
    NormalizedToolResult,
    OpenAIToolSchema,
    StreamChatParams,
    StreamChatResult,
} from "./types";

const AZURE_API_VERSION = "2024-08-01-preview";
const MAX_OUTPUT_TOKENS = 16384;

function getConfig(apiKeyOverride?: string | null): { endpoint: string; apiKey: string } {
    const endpoint = process.env.AZURE_OPENAI_ENDPOINT?.trim() ?? "";
    const apiKey = apiKeyOverride?.trim() || process.env.AZURE_OPENAI_API_KEY?.trim() || "";
    if (!endpoint || !apiKey) throw new Error("Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.");
    return { endpoint, apiKey };
}

function deploymentUrl(endpoint: string, model: string): string {
    return `${endpoint}/openai/deployments/${model}/chat/completions?api-version=${AZURE_API_VERSION}`;
}

function toMessages(messages: LlmMessage[], systemPrompt?: string) {
    const result: { role: string; content: string }[] = [];
    if (systemPrompt) result.push({ role: "system", content: systemPrompt });
    for (const m of messages) result.push({ role: m.role, content: m.content });
    return result;
}

function toTools(tools: OpenAIToolSchema[]) {
    return tools.map((t) => ({ type: "function", function: t.function }));
}

function extractSseJson(buffer: string): { events: unknown[]; rest: string } {
    const events: unknown[] = [];
    const chunks = buffer.split(/\n\n/);
    const rest = chunks.pop() ?? "";
    for (const chunk of chunks) {
        for (const line of chunk.split("\n")) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;
            const data = trimmed.slice(5).trim();
            if (!data || data === "[DONE]") continue;
            try { events.push(JSON.parse(data)); } catch { /* incomplete */ }
        }
    }
    return { events, rest };
}

type ChunkChoice = {
    delta?: {
        content?: string;
        tool_calls?: { index: number; id?: string; function?: { name?: string; arguments?: string } }[];
    };
    finish_reason?: string;
};

async function chatRequest(
    endpoint: string,
    apiKey: string,
    model: string,
    messages: { role: string; content: unknown }[],
    opts: { tools?: unknown[]; stream?: boolean; maxTokens?: number },
): Promise<Response> {
    const res = await fetch(deploymentUrl(endpoint, model), {
        method: "POST",
        headers: { "api-key": apiKey, "Content-Type": "application/json" },
        body: JSON.stringify({
            messages,
            tools: opts.tools?.length ? opts.tools : undefined,
            stream: opts.stream ?? false,
            max_tokens: opts.maxTokens ?? MAX_OUTPUT_TOKENS,
        }),
    });
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        const err = new Error(`Azure OpenAI request failed (${res.status}): ${text || res.statusText}`);
        (err as { status?: number }).status = res.status;
        throw err;
    }
    return res;
}

export async function streamOpenAI(params: StreamChatParams): Promise<StreamChatResult> {
    const { model, systemPrompt, tools = [], callbacks = {}, runTools, apiKeys } = params;
    const maxIter = params.maxIterations ?? 10;
    const { endpoint, apiKey } = getConfig(apiKeys?.openai);
    const azureTools = toTools(tools);
    let messages = toMessages(params.messages, systemPrompt);
    let fullText = "";

    for (let iter = 0; iter < maxIter; iter++) {
        const response = await chatRequest(endpoint, apiKey, model, messages, {
            tools: azureTools,
            stream: true,
        });
        if (!response.body) throw new Error("Azure OpenAI response had no body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let pendingText = "";
        // accumulate tool call arguments across chunks
        const toolCallAccum: Record<number, { id: string; name: string; arguments: string }> = {};

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const { events, rest } = extractSseJson(buffer);
            buffer = rest;

            for (const event of events as { choices?: ChunkChoice[] }[]) {
                const choice = event.choices?.[0];
                if (!choice) continue;
                const delta = choice.delta ?? {};

                // text delta
                if (delta.content) {
                    if (azureTools.length) {
                        pendingText += delta.content;
                    } else {
                        fullText += delta.content;
                        callbacks.onContentDelta?.(delta.content);
                    }
                }

                // tool call streaming
                for (const tc of delta.tool_calls ?? []) {
                    const idx = tc.index;
                    if (!toolCallAccum[idx]) {
                        toolCallAccum[idx] = { id: tc.id ?? "", name: tc.function?.name ?? "", arguments: "" };
                        const partial: NormalizedToolCall = { id: tc.id ?? String(idx), name: tc.function?.name ?? "", input: {} };
                        callbacks.onToolCallStart?.(partial);
                    }
                    if (tc.id) toolCallAccum[idx].id = tc.id;
                    if (tc.function?.name) toolCallAccum[idx].name = tc.function.name;
                    if (tc.function?.arguments) toolCallAccum[idx].arguments += tc.function.arguments;
                }
            }
        }

        // finalise tool calls
        const toolCalls: NormalizedToolCall[] = Object.values(toolCallAccum).map((tc) => {
            let input: Record<string, unknown> = {};
            try {
                const parsed = JSON.parse(tc.arguments || "{}");
                if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) input = parsed;
            } catch { /* keep empty */ }
            return { id: tc.id, name: tc.name, input };
        });

        if (!toolCalls.length || !runTools) {
            if (pendingText) { fullText += pendingText; callbacks.onContentDelta?.(pendingText); }
            break;
        }

        const results = await runTools(toolCalls);

        // build next turn with assistant tool_calls + tool results
        const assistantToolCallsMsg = {
            role: "assistant",
            content: null as unknown as string,
            tool_calls: toolCalls.map((tc) => ({
                id: tc.id,
                type: "function",
                function: { name: tc.name, arguments: JSON.stringify(tc.input) },
            })),
        };
        const toolResultMsgs = results.map((r) => ({
            role: "tool",
            tool_call_id: r.tool_use_id,
            content: r.content,
        }));
        messages = [...messages, assistantToolCallsMsg, ...toolResultMsgs];
    }

    return { fullText };
}

export async function completeOpenAIText(params: {
    model: string;
    systemPrompt?: string;
    user: string;
    maxTokens?: number;
    apiKeys?: { openai?: string | null };
}): Promise<string> {
    const { endpoint, apiKey } = getConfig(params.apiKeys?.openai);
    const messages = toMessages([{ role: "user", content: params.user }], params.systemPrompt);
    const res = await chatRequest(endpoint, apiKey, params.model, messages, { maxTokens: params.maxTokens ?? 512 });
    const json = await res.json() as { choices?: { message?: { content?: string } }[] };
    return json.choices?.[0]?.message?.content ?? "";
}

export type { NormalizedToolResult };
